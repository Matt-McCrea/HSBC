from einops import rearrange
import numpy as np
from torch import nn
import os
import time
from lightning import LightningModule
import torch
import constants as cst
from constants import LearningHyperParameter
import matplotlib.pyplot as plt

import wandb
from models.diffusers.gaussian_diffusion import GaussianDiffusion
from utils.utils_models import pick_augmenter
from lion_pytorch import Lion
from torch_ema import ExponentialMovingAverage
from models.diffusers.TRADES.Sampler import LossSecondMomentResampler


class DiffusionEngine(LightningModule):
    
    def __init__(self, config):
        super().__init__()
        self.conditional_dropout = config.HYPER_PARAMETERS[LearningHyperParameter.CONDITIONAL_DROPOUT]
        self.IS_AUGMENTATION = config.IS_AUGMENTATION
        self.IS_WANDB = config.IS_WANDB
        self.augment_dim = config.HYPER_PARAMETERS[LearningHyperParameter.AUGMENT_DIM]
        self.cond_type = config.COND_TYPE
        self.cond_method = config.COND_METHOD
        self.cond_seq_size = config.HYPER_PARAMETERS[LearningHyperParameter.SEQ_SIZE] - config.HYPER_PARAMETERS[LearningHyperParameter.MASKED_SEQ_SIZE]
        self.reg_term_weight = config.HYPER_PARAMETERS[LearningHyperParameter.REG_TERM_WEIGHT]
        self.num_diffusionsteps = config.HYPER_PARAMETERS[LearningHyperParameter.NUM_DIFFUSIONSTEPS]
        self.size_type_emb = config.HYPER_PARAMETERS[LearningHyperParameter.SIZE_TYPE_EMB]
        self.size_order_emb = config.HYPER_PARAMETERS[LearningHyperParameter.SIZE_ORDER_EMB]
        self.chosen_model = config.CHOSEN_MODEL.value
        self.betas = config.BETAS
        self.training = config.IS_TRAINING
        self.test_batch_size = config.HYPER_PARAMETERS[LearningHyperParameter.TEST_BATCH_SIZE]
        self.epochs = config.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS]
        self.seq_size = config.HYPER_PARAMETERS[LearningHyperParameter.SEQ_SIZE]
        self.train_losses, self.vlb_train_losses, self.simple_train_losses = [], [], []
        self.val_ema_losses, self.test_ema_losses = [], []
        self.min_loss_ema = np.inf
        self.min_train_loss = np.inf
        self.filename_ckpt = config.FILENAME_CKPT
        self.last_path_ckpt_ema = None
        self.optimizer = config.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER]
        self.lr = config.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE]
        self.cond_size = config.COND_SIZE
        if self.IS_AUGMENTATION:
            self.feature_augmenter = pick_augmenter(config.CHOSEN_AUGMENTER, self.size_order_emb, self.augment_dim, self.cond_size, self.cond_type, config.CHOSEN_COND_AUGMENTER, self.cond_method, self.chosen_model)
            self.diffuser = GaussianDiffusion(config, self.feature_augmenter).to(cst.DEVICE, non_blocking=True)
        else:
            self.diffuser = GaussianDiffusion(config, None).to(cst.DEVICE, non_blocking=True)
            
        self.type_embedder = nn.Embedding(3, self.size_type_emb, dtype=torch.float32)
        self.type_embedder.requires_grad_(False)
        self.type_embedder.weight.data = torch.tensor([[ 0.4438, -0.2984,  0.2888], [ 0.8249,  0.5847,  0.1448], [ 1.5600, -1.2847,  1.0294]], device=cst.DEVICE, dtype=torch.float32)
        if self.IS_WANDB:
            wandb.log({"type_embedder": self.type_embedder.weight.data}, step=0)
            
        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)
        self.ema.to(cst.DEVICE)
        self.sampler = LossSecondMomentResampler(self.num_diffusionsteps)
        self.vlb_sampler = LossSecondMomentResampler(self.num_diffusionsteps)
        self.simple_sampler = LossSecondMomentResampler(self.num_diffusionsteps)
        # per-phase timing accumulators (seconds); reset each epoch
        self._t = {"forward": 0.0, "loss": 0.0, "ema": 0.0, "sampler": 0.0, "total": 0.0}
        self._t_steps = 0
        self._t_val_total = 0.0
        self._t_val_steps = 0
        # ── scheduled sampling (v1, stop-gradient, k=1) ──
        self._ss_on = cst.SCHEDULED_SAMPLING
        self._ss_p_max = cst.SS_P_MAX
        self._ss_ramp_frac = cst.SS_RAMP_FRAC
        self._ss_gen_seq = config.HYPER_PARAMETERS[LearningHyperParameter.MASKED_SEQ_SIZE]
        self._ss_used, self._ss_total = 0, 0
        # prior-corrected nearest-anchor type decode, matching --type-decode prior at inference
        self._ss_log_prior = torch.log(torch.tensor([0.49, 0.48, 0.03], device=cst.DEVICE))
        if self._ss_on:
            steps = len(getattr(self.diffuser, "t", [])) or self.num_diffusionsteps
            print(f"[scheduled-sampling] ON  p_max={self._ss_p_max} ramp_frac={self._ss_ramp_frac} "
                  f"rollout_steps={steps} sampler={getattr(self.diffuser, 'sampling_type', '?')}")
            if steps > 20:
                print(f"[scheduled-sampling] WARNING: rollout uses {steps} sampler steps — this makes each "
                      f"self-conditioned training step expensive. Launch with DDIM + small DDIM_NSTEPS (~10).")
        self.save_hyperparameters()
        

    def forward(self, cond_orders, x_0, cond_lob, is_train, batch_idx=None):
        # x_0 shape is (batch_size, seq_size=1, cst.LEN_ORDER=8)
        x_0, cond_orders = self.type_embedding(x_0, cond_orders)
        if is_train:
            self.t, _ = self.sampler.sample(x_0.shape[0])
            recon = self.single_step(cond_orders, x_0, cond_lob, batch_idx)
        else:
            self.t = torch.full(size=(x_0.shape[0],), fill_value=self.num_diffusionsteps-1, device=cst.DEVICE, dtype=torch.int64)
            for i in range(self.num_diffusionsteps-1, -1, -1):
                recon = self.single_step(cond_orders, x_0, cond_lob)
                self.t -= 1
        return recon

    def sample(self, **kwargs) -> torch.Tensor:
        cond_orders: torch.Tensor = kwargs['cond_orders']
        x_0: torch.Tensor = kwargs['x']
        cond_lob: torch.Tensor = kwargs['cond_lob']
        x_0, cond_orders = self.type_embedding(x_0, cond_orders)
        x_0 = torch.zeros_like(x_0)
        weights = self.sampler.weights()
        x_t = self.diffuser.sample(x_0, cond_orders, cond_lob, weights)
        return x_t


    def single_step(self, cond_orders, x_0, cond_lob, batch_idx=None):
        # forward process
        x_t, noise = self.diffuser.forward_reparametrized(x_0, self.t)
        if torch.isnan(x_t).any():
            print("before aug:", x_t.max())
        # augment
        x_t_aug, cond_orders, cond_lob = self.diffuser.augment(x_t, cond_orders, cond_lob)
        if torch.isnan(x_t_aug).any():
            print("after aug:", x_t_aug.max())
        weights = self.sampler.weights()
        x_recon = self.diffuser.ddpm_single_step(x_0, x_t_aug, x_t, self.t, cond_orders, noise, weights, cond_lob, batch_idx)
        # return the deaugmented denoised input and the reverse context
        return x_recon
    

    def type_embedding(self, x_0, cond):
        order_type = x_0[:, :, 1]
        order_type_emb = self.type_embedder(order_type.long())
        x_0 = torch.cat((x_0[:, :, :1], order_type_emb, x_0[:, :, 2:]), dim=2)
        cond_type = cond[:, :, 1]
        cond_depth_emb = self.type_embedder(cond_type.long())
        cond = torch.cat((cond[:, :, :1], cond_depth_emb, cond[:, :, 2:]), dim=2)
        return x_0, cond

    def _ss_prob(self):
        """Scheduled-sampling probability for the current epoch: 0 early (teacher-forced), ramping to
        SS_P_MAX over SS_RAMP_FRAC of training."""
        if not self._ss_on:
            return 0.0
        max_ep = getattr(self.trainer, "max_epochs", None) or 1
        ramp = max(1, int(self._ss_ramp_frac * max_ep))
        return self._ss_p_max * min(1.0, self.current_epoch / ramp)

    def _decode_type(self, g_emb):
        """Turn the model's embedded generated block back into RAW orders usable as conditioning:
        replace the embedded type sub-vector (columns 1:1+size_type_emb) with a prior-corrected
        nearest-anchor type index (same decode as --type-decode prior), and keep the model's continuous
        outputs for the other channels. Output width matches cst.LEN_ORDER."""
        ste = self.size_type_emb
        type_sub = g_emb[:, :, 1:1 + ste]                                  # (B, G, ste)
        W = self.type_embedder.weight.to(g_emb.dtype)                      # (num_types, ste)
        d2 = ((type_sub.unsqueeze(2) - W.view(1, 1, -1, ste)) ** 2).sum(-1)  # (B, G, num_types)
        score = 0.5 * d2 - self._ss_log_prior.to(g_emb.dtype)             # prior-corrected
        type_idx = score.argmin(-1, keepdim=True).to(g_emb.dtype)         # (B, G, 1)
        return torch.cat([g_emb[:, :, :1], type_idx, g_emb[:, :, 1 + ste:]], dim=2)
    
    
    def loss(self):
        # regularization term to avoid order with negative size
        L_hybrid, L_simple, L_vlb = self.diffuser.loss()
        #print(f"hybrid loss: {L_hybrid.mean()}")
        #print(f"simple loss: {L_simple.mean()}")
        #print(f"vlb loss: {L_vlb.mean()}")
        return L_hybrid, L_simple, L_vlb


    def training_step(self, input, batch_idx):
        if self.global_step == 0 and self.IS_WANDB:
            self._define_log_metrics()
        if self._ss_on and len(input) == 5:
            # scheduled-sampling batch: (cond, x_0, lob, x_0_next, lob_shift)
            cond_orders, x_0, cond_lob, x_0_next, lob_shift = input
            self._ss_total += 1
            if torch.rand(1).item() < self._ss_prob():
                self._ss_used += 1
                with torch.no_grad():                              # stop-gradient rollout
                    lob_in = cond_lob.contiguous() if self.cond_type == 'full' else None
                    g_emb = self.sample(cond_orders=cond_orders.contiguous(),
                                        x_0=x_0.contiguous(), cond_lob=lob_in)
                    g_raw = self._decode_type(g_emb)               # embedded -> raw orders
                    # slide the window: drop the oldest generated-block worth of real orders, append
                    # the model's own generated block -> conditioning is now partly self-generated
                    cond_orders = torch.cat([cond_orders[:, self._ss_gen_seq:, :], g_raw], dim=1)
                x_0, cond_lob = x_0_next, lob_shift               # score the REAL next block, real shifted book
            if batch_idx % 1000 == 0:
                print(f"DIAG scheduled_sampling: p={self._ss_prob():.3f} used={self._ss_used}/{self._ss_total}")
            x_0 = x_0.contiguous()
            cond_orders = cond_orders.contiguous()
            cond_lob = cond_lob.contiguous()
        else:
            x_0 = input[1].contiguous()
            cond_orders = input[0].contiguous()
            cond_lob = input[2].contiguous()
        x_0.requires_grad_(True)
        cond_orders.requires_grad_(True)
        cond_lob.requires_grad_(True)
        if self.cond_type != 'full':
            cond_lob = None

        t0 = time.perf_counter()
        recon = self.forward(cond_orders, x_0, cond_lob, is_train=True, batch_idx=batch_idx)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        batch_loss, L_simple, L_vlb = self.loss()
        t2 = time.perf_counter()

        self.simple_train_losses.append(torch.mean(L_simple).item())
        self.vlb_train_losses.append(torch.mean(L_vlb).item())
        batch_loss_mean = torch.mean(batch_loss)
        self.train_losses.append(batch_loss_mean.item())
        self.sampler.update_losses(self.t, batch_loss[0])
        self.vlb_sampler.update_losses(self.t, L_vlb[0])
        self.simple_sampler.update_losses(self.t, L_simple[0])
        t3 = time.perf_counter()

        self.diffuser.init_losses()
        self.ema.update()
        t4 = time.perf_counter()

        self._t["forward"]  += t1 - t0
        self._t["loss"]     += t2 - t1
        self._t["sampler"]  += t3 - t2
        self._t["ema"]      += t4 - t3
        self._t["total"]    += t4 - t0
        self._t_steps       += 1

        if batch_idx % 1000 == 0:
            print(f"batch loss: {batch_loss_mean}")
        return batch_loss_mean

    def on_train_epoch_start(self) -> None:
        print(f'learning rate: {self.optimizer.param_groups[0]["lr"]}')

    def on_validation_start(self) -> None:
        loss = sum(self.train_losses) / len(self.train_losses)
        if isinstance(self.diffuser, GaussianDiffusion):
            L_simple = sum(self.simple_train_losses) / len(self.simple_train_losses)
            L_vlb = sum(self.vlb_train_losses) / len(self.vlb_train_losses)
            if self.IS_WANDB:
                # Use global_step instead of current_epoch for logging
                wandb.log({
                    'train loss simple': L_simple,
                    'train loss vlb': L_vlb,
                    'train_loss': loss,
                }, step=self.global_step)
                
                #Simple loss plot
                plt.figure()
                plt.plot(range(self.num_diffusionsteps), np.mean(self.simple_sampler._loss_history, axis=-1))
                plt.xlabel('num_diffusionsteps')
                plt.ylabel('Simple')
                wandb.log({"simple_loss": wandb.Image(plt)}, step=self.global_step)
                plt.close()
                
                # VLB loss plot
                plt.figure()
                plt.plot(range(self.num_diffusionsteps), np.mean(self.vlb_sampler._loss_history, axis=-1))
                plt.xlabel('num_diffusionsteps')
                plt.ylabel('VLB')
                wandb.log({"vlb_loss": wandb.Image(plt)}, step=self.global_step)
                plt.close()
                
                print(f'\ntrain loss simple on step {self.global_step} is {round(L_simple, 3)}')
                print(f'\ntrain loss vlb on step {self.global_step} is {round(L_vlb, 3)}')
                print(f'\ntrain loss on step {self.global_step} is {round(loss, 3)}')
        self.train_losses = []
        self.simple_train_losses = []
        self.vlb_train_losses = []
        self.val_ema_losses = []
        self.simple_val_losses = []
        self.vlb_val_losses = []
    

    def validation_step(self, input, batch_idx):
        x_0 = input[1]
        cond_orders = input[0]
        cond_lob = input[2]
        if self.cond_type != 'full':
            cond_lob = None
        t0 = time.perf_counter()
        with self.ema.average_parameters():
            recon = self.forward(cond_orders, x_0, cond_lob, is_train=False)
            batch_loss, L_simple, L_vlb = self.loss()
            self.simple_val_losses.append(torch.mean(L_simple).item())
            self.vlb_val_losses.append(torch.mean(L_vlb).item())
            batch_loss_mean = torch.mean(batch_loss)
            self.val_ema_losses.append(batch_loss_mean.item())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t_val_total += time.perf_counter() - t0
        self._t_val_steps += 1
        self.diffuser.init_losses()
        return batch_loss_mean


    def on_validation_epoch_end(self) -> None:
        loss_ema = sum(self.val_ema_losses) / len(self.val_ema_losses)

        if self._t_steps > 0:
            n = self._t_steps
            t = self._t
            tot = t["total"] if t["total"] > 0 else 1e-9
            print(f"\n[Timing] Avg per training step (N={n}):")
            print(f"  NN forward+aug : {1000*t['forward']/n:6.1f} ms  ({100*t['forward']/tot:4.1f}%)")
            print(f"  Loss (MSE+VLB) : {1000*t['loss']/n:6.1f} ms  ({100*t['loss']/tot:4.1f}%)")
            print(f"  Sampler update : {1000*t['sampler']/n:6.1f} ms  ({100*t['sampler']/tot:4.1f}%)")
            print(f"  EMA update     : {1000*t['ema']/n:6.1f} ms  ({100*t['ema']/tot:4.1f}%)")
            print(f"  Total          : {1000*t['total']/n:6.1f} ms")
            for k in self._t:
                self._t[k] = 0.0
            self._t_steps = 0

        if self._t_val_steps > 0:
            avg_val_ms = 1000 * self._t_val_total / self._t_val_steps
            print(f"[Timing] Avg per validation step: {avg_val_ms:.1f} ms")
            self._t_val_total = 0.0
            self._t_val_steps = 0

        # model checkpointing
        if loss_ema < self.min_loss_ema:
            self.min_loss_ema = loss_ema
            self.model_checkpointing(loss_ema)

        if isinstance(self.diffuser, GaussianDiffusion):
            L_simple = sum(self.simple_val_losses) / len(self.simple_val_losses)
            L_vlb = sum(self.vlb_val_losses) / len(self.vlb_val_losses)
            if self.IS_WANDB:
                wandb.log({'val_loss_simple': L_simple}, step=self.current_epoch + 1)
                wandb.log({'val_loss_vlb': L_vlb}, step=self.current_epoch + 1)
            print(f'\nval loss simple on epoch {self.current_epoch} is {round(L_simple, 3)}')
            print(f'\nval loss vlb on epoch {self.current_epoch} is {round(L_vlb, 3)}')

        self.log('val_ema_loss', loss_ema)
        print(f"\n val ema loss on epoch {self.current_epoch} is {round(loss_ema, 3)}")
        

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                [
                    {'params': self.diffuser.parameters()},
                    {'params': self.type_embedder.parameters(), "lr": 0.01},
                ],
                lr=self.lr,
                weight_decay=1e-4,
            )
        elif self.optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer == 'LION':
            self.optimizer = Lion(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5, threshold=0.01
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_ema_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _define_log_metrics(self):
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("val_ema_loss", summary="min")

    def model_checkpointing(self, loss):
        if self.last_path_ckpt_ema is not None:
            os.remove(self.last_path_ckpt_ema)
        filename_ckpt_ema = ("val_ema=" + str(round(loss, 3)) +
                             "_epoch=" + str(self.current_epoch) +
                             "_" + self.filename_ckpt +
                             ".ckpt"
                             )
        path_ckpt_ema = cst.DIR_SAVED_MODEL + "/" + str(self.chosen_model) + "/" + filename_ckpt_ema
        with self.ema.average_parameters():
            self.trainer.save_checkpoint(path_ckpt_ema)
        self.last_path_ckpt_ema = path_ckpt_ema


