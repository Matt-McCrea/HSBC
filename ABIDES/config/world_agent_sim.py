import argparse
from datetime import datetime
import os
import time
import warnings

import numpy as np
import pandas as pd
import sys
import datetime as dt

import torch
from dateutil.parser import parse
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

import constants as cst
from Kernel import Kernel
from agent.WorldAgent import WorldAgent
from util.order import LimitOrder
from util import util
from utils.utils_data import load_compute_normalization_terms
from agent.ExchangeAgent import ExchangeAgent
from agent.execution.POVExecutionAgent import POVExecutionAgent
from pathlib import Path

import configuration
from models.diffusers.diffusion_engine import DiffusionEngine
from models.gan.gan_engine import GANEngine

torch.serialization.add_safe_globals([configuration.Configuration, cst.Models, 
cst.LearningHyperParameter, cst.Stocks, cst.Engine])

########################################################################################################################
############################################### GENERAL CONFIG #########################################################

parser = argparse.ArgumentParser(description='Detailed options for RMSC03 config.')

parser.add_argument('-c',
                    '--config',
                    required=True,
                    help='Name of config file to execute')
parser.add_argument('-t',
                    '--ticker',
                    required=True,
                    help='Ticker (symbol) to use for simulation')
parser.add_argument('-date',
                    '--historical-date',
                    required=True,
                    type=parse,
                    help='historical date being simulated in format YYYYMMDD.')
parser.add_argument('-st',
                    '--start-time',
                    default='09:30:00',
                    type=parse,
                    help='Starting time of simulation.'
                    )
parser.add_argument('-et',
                    '--end-time',
                    default='11:00:00',
                    type=parse,
                    help='Ending time of simulation.'
                    )
parser.add_argument('--config_help',
                    action='store_true',
                    help='Print argument options for this config file')
# Execution agent config
parser.add_argument('-e',
                    '--execution-agents',
                    type=bool,
                    default=False,
                    help='Flag to allow the execution agent to trade.')
parser.add_argument('-m',
                    '--chosen-model',
                    type=str,
                    default='TRADES')
parser.add_argument('-p',
                    '--execution-pov',
                    type=float,
                    default=0.1,
                    help='Participation of Volume level for execution agent')
parser.add_argument('-d',
                    '--diffusion',
                    type=bool,
                    default=False,
                    help='Using diffusion')
#add a parser argument that takes in nput a float value for the proportion of volume
# that the agent will trade
parser.add_argument('-id',
                    '--id',
                    type=float,
                    default=None,
                    help='diffusion-id-which-is-best-val-loss')
parser.add_argument('-seed',
                    '--seed',
                    type=int,
                    default=cst.SEED,
                    help='seed for random number generation')
parser.add_argument('-type',
                    '--sampling-type',
                    type=str,
                    default='DDIM',
                    help='Sampling type for diffusion')
parser.add_argument('-eta',
                    '--ddim-eta',
                    type=float,
                    default=0.0,
                    help='eta for DDIM')
parser.add_argument('-nsteps',
                    '--ddim-nsteps',
                    type=int,
                    default=1,
                    help='nsteps for DDIM')
parser.add_argument('--tail-steps',
                    type=int,
                    default=2,
                    help='For HYBRID_PP_DDIM: number of DDIM steps appended after DPM-Solver++ steps')
parser.add_argument('--guidance-scale',
                    type=float,
                    default=1.0,
                    help='Classifier-free guidance weight (1.0 = off). w<1 blends toward the marginal '
                         'distribution, w>1 sharpens conditioning. Requires a checkpoint trained with '
                         'CONDITIONAL_DROPOUT > 0. Doubles NN evaluations per step when != 1.0.')
parser.add_argument('--churn-steps',
                    type=int,
                    default=3,
                    help='CHURN sampler: number of early (high-noise) steps to apply EDM renoise on')
parser.add_argument('--churn-strength',
                    type=float,
                    default=0.3,
                    help='CHURN sampler: EDM renoise strength kappa in [0,0.9] (0 = pure DPM-Solver++). '
                         'Re-injects entropy on the early steps our HYBRID_DDPM_PP result showed set '
                         'the marketable-order diversity, at few-step (DPM++) accuracy on the tail.')
parser.add_argument('--real-data-path',
                    type=str,
                    default=None,
                    help='Path to real market-replay processed_orders.csv for KL divergence report')
# ── Hypothesis-testing flags (all default off = original behavior) ──────────────
parser.add_argument('--fix-time', action='store_true',
                    help='H1: feed generated inter-arrival times back into the conditioning history '
                         '(fixes the frozen time channel during generation)')
parser.add_argument('--type-decode', type=str, default='l1', choices=['l1', 'l2', 'prior'],
                    help='H2: type-embedding decode: l1 (original), l2, or prior '
                         '(Bayes prior-corrected — penalizes the geometrically oversized MARKET region)')
parser.add_argument('--fix-cancel-bind', action='store_true',
                    help='H3: bind generated cancels to the nearest same-side resting order instead of '
                         'dropping them when no exact-price match exists')
parser.add_argument('--fix-lob-pad', action='store_true',
                    help='H5: pad empty LOB levels with LOBSTER sentinels before z-scoring (match training)')
parser.add_argument('--drop-type2-cond', action='store_true',
                    help='H7: exclude type-2 partial cancels from the conditioning history (match training)')
parser.add_argument('--depth-temp', type=float, default=1.0,
                    help='Scale the decoded depth z-score by this factor (1.0 = off). kappa>1 widens the '
                         'depth distribution so some orders spill into negative (marketable) depth, '
                         'restoring the execution-driving tail that deterministic few-step sampling collapses.')
parser.add_argument('--depth-reshape', nargs='?', const='data/quantile_targets/real_depth_limit.npy',
                    default=None, metavar='NPY',
                    help='Quantile-reshape decoded LIMIT depth: rank each raw z within the model\'s own '
                         'recent outputs (rolling buffer, midrank ties), read the same quantile off the '
                         'REAL signed-depth marginal (build with scripts/build_quantile_targets.py). '
                         'Nonlinear per-sample map — restores the real crossing tail with real MAGNITUDES, '
                         'which the unclamp retrain alone could not (DDIM10 B_crossing_limit stayed 0). '
                         'Optional value = target .npy path.')
parser.add_argument('--size-reshape', nargs='?', const='data/quantile_targets',
                    default=None, metavar='DIR',
                    help='Quantile-reshape decoded size onto the REAL per-type size marginals '
                         '(real_size_{limit,cancel,market}.npy in DIR). Also eliminates the ~30-40%% '
                         'negative-size decode waste (42%% of batches resampled under DDPM).')
parser.add_argument('--depth-noise', type=float, default=0.0,
                    help='Per-sample N(0,sigma) added to z_depth at decode (LIMIT only; 0 = off). The '
                         '"dumb variance fix" comparator to --depth-reshape: acts per-sample (can split '
                         'the collapsed atom, unlike --depth-temp) and never enters the sampler or other '
                         'channels (cannot destabilize, unlike CHURN). sigma~0.15 predicts ~0.5%% crossing.')
parser.add_argument('--dn-target-exec', type=float, default=0.0,
                    help='Execution-rate feedback controller for --depth-noise (0 = off, fixed sigma). '
                         'Holds the realized exec share (Channel A + B, last 1000 placed orders) at this '
                         'target by scaling sigma in [0.25x, 4x]. Fixes the 75-min liquidity death '
                         'spiral: fixed sigma over-executes 2-3x real, drains the book dry by ~min 45 '
                         '(events 8k->219/bucket, spread 1->41 ticks, mid teleports -5%%). Real exec '
                         'share ~0.07 (09:45-10:00) / ~0.045 (09:45-11:00).')
# --- DIRECTION A levers (impact + volatility). Both default 0 = OFF = winning config unchanged. ---
parser.add_argument('--cancel-boost', type=float, default=0.0,
                    help='Bias the type decode toward CANCEL (subtract this from the cancel anchor '
                         'score). More cancels drain resting liquidity -> thinner book -> larger price '
                         'impact per marketable order. Targets the under-cancel/thick-book gap that '
                         'LOB-Bench does not penalise but that makes impact too cheap. 0 = off.')
parser.add_argument('--depth-drift', type=float, default=0.0,
                    help='AR(1) directional-persistence amplitude on the depth channel (LIMIT only). '
                         'Creates short runs of same-side aggression -> transient mid excursions that '
                         'mean-revert (E[drift]=0, so NO net trend), raising realized volatility and '
                         'reducing the over-mean-reversion. 0 = off. Try 0.1-0.3.')
parser.add_argument('--depth-drift-phi', type=float, default=0.995,
                    help='AR(1) persistence for --depth-drift (0..1). Higher = longer directional runs. '
                         'Default 0.995.')
# --- LONG-HORIZON STABILITY levers. Both default 0 = OFF = winning config unchanged. ---
parser.add_argument('--book-target-thick', type=float, default=0.0,
                    help='Book-balancing spontaneous cancellation (0 = off). When a side top-of-book size '
                         'exceeds this multiple of the real mean level size (normalization_terms["lob"][0]), '
                         'cancel our own resting orders at the touch to thin it. Recreates the cancel churn '
                         'real markets have -> targets BOTH the under-cancel gap and the 90-min lopsided-book '
                         'divergence (they are the same problem). Try 1.5-3.0. Needs --book-cancel-rate.')
parser.add_argument('--book-cancel-rate', type=float, default=0.5,
                    help='Fraction of the per-side excess touch size removed each generation step by '
                         '--book-target-thick (0..1). Only active when --book-target-thick > 0. Default 0.5.')
parser.add_argument('--cond-clip', type=float, default=0.0,
                    help='Clip the z-scored order-book SIZE conditioning to [-C, C] before it enters the '
                         'model (0 = off). Keeps the fed-back book state inside training support over long '
                         'horizons, arresting the closed-loop drift that grows the touch sizes OOD. Applies '
                         'to sizes only (prices are handled by PRICE_REANCHOR; missing-level sentinels '
                         'untouched). Try 4-6.')
parser.add_argument('--flow-balance', type=float, default=0.0,
                    help='Adaptive directional bias that counters one-sided limit-order FLOW — the '
                         'CROSS-DAY drift driver (limOFI one-sided while exec/B-S stay balanced; drift '
                         'persists at every --depth-noise, so it is directional not a variance problem). '
                         'The flow-side twin of the book-balancing cancel: tracks a rolling limit-side '
                         'imbalance and nudges the decoded direction of LIMIT orders toward the thin side '
                         '(limit-only; biasing a cancel/market side would thin support). Try 0.5-2.0. 0 = off.')
parser.add_argument('--flow-balance-window', type=int, default=500,
                    help='Rolling window (number of recent limit orders) for the --flow-balance imbalance.')
parser.add_argument('--ckpt-path', type=str, default=None,
                    help='Use this exact checkpoint file, bypassing -id val_ema matching. Lets you trial '
                         'per-epoch checkpoints that share a rounded val_ema (which -id cannot disambiguate).')

args, remaining_args = parser.parse_known_args()

if args.config_help:
    parser.print_help()
    sys.exit()

seed = args.seed  # Random seed specification on the command line.
torch.manual_seed(seed)
np.random.seed(seed)
exchange_log_orders = True
log_orders = True
warnings.filterwarnings("ignore")
simulation_start_time = dt.datetime.now()
print("Simulation Start Time: {}".format(simulation_start_time))
print("Configuration seed: {}\n".format(seed))
########################################################################################################################
############################################### AGENTS CONFIG ##########################################################

# Historical date to simulate.
historical_date = pd.to_datetime(args.historical_date)
mkt_open = historical_date + pd.to_timedelta(args.start_time.strftime('%H:%M:%S'))
mkt_close = historical_date + pd.to_timedelta(args.end_time.strftime('%H:%M:%S'))
agent_count, agents, agent_types = 0, [], []

# Hyperparameters
symbol = args.ticker
#check if INTC is zip or unzip
path = "{}/{}/{}_{}_{}".format(
            cst.DATA_DIR,
            symbol,
            symbol,
            cst.DATE_TRADING_DAYS[0],
            cst.DATE_TRADING_DAYS[-1]
        )
if symbol == "INTC" and not Path(path).exists():
    print("INTC is not unzipped, unzipping...")
    import zipfile
    with zipfile.ZipFile(cst.DATA_DIR + f"/{symbol}/{symbol}.zip", 'r') as zip_ref:
        zip_ref.extractall(cst.DATA_DIR + f"/{symbol}")
    print("INTC unzipped")

if args.chosen_model == "TRADES":
    chosen_model = cst.Models.TRADES
elif args.chosen_model == "CGAN":
    chosen_model = cst.Models.CGAN

#check if there are the checkpoints in data/checkpoints
if args.diffusion:
    dir_path = Path(cst.DIR_SAVED_MODEL + "/" + str(chosen_model.value))
    if not dir_path.exists():
        print("Checkpoints not found, downloading...")
        try:
            import gdown
            
            # Create the directory if it doesn't exist
            dir_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Google Drive folder ID
            folder_id = '1fg5G9KzmzC6E4FUYSCjObJ7sCEdjo43W'
            
            # Use gdown's download_folder functionality
            gdown.download_folder(
                id=folder_id,
                output=str(dir_path),
                quiet=False,
                use_cookies=False
            )
            print("Checkpoints downloaded successfully")
            
        except Exception as e:
            print(f"Error downloading checkpoints: {str(e)}")
            print("Please ensure you have a working internet connection")
            sys.exit(1)

normalization_terms = load_compute_normalization_terms(symbol, cst.DATA_DIR, chosen_model, n_lob_levels=10)
starting_cash = 100000000000  # Cash in this simulator is always in CENTS.

# 1) Exchange Agent

#  How many orders in the past to store for transacted volume computation
# stream_history_length = int(pd.to_timedelta(args.mm_wake_up_freq).total_seconds() * 100)
stream_history_length = 2500000

agents.extend([ExchangeAgent(id=0,
                             name="EXCHANGE_AGENT",
                             type="ExchangeAgent",
                             mkt_open=mkt_open,
                             mkt_close=mkt_close,
                             symbols=[symbol],
                             log_orders=exchange_log_orders,
                             pipeline_delay=0,
                             computation_delay=0,
                             stream_history=stream_history_length,
                             book_freq=0,
                             wide_book=True,
                             random_state=np.random.RandomState(
                                 seed=seed))
               ])
agent_types.extend("ExchangeAgent")
agent_count += 1

# 2) World Agent
if args.diffusion:
    dir_path = Path(cst.DIR_SAVED_MODEL + "/" + str(chosen_model.value))
    best_val_loss = np.inf
    if args.ckpt_path is not None:
        checkpoint_reference = Path(args.ckpt_path)   # exact file, bypasses -id val_ema matching
    elif args.id is None:
        for file in dir_path.iterdir():
            if symbol in file.name:
                try:
                    val_loss = float(file.name.split("=")[1].split("_")[0])
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        checkpoint_reference = file
                except:
                    continue
    else:
        for file in dir_path.iterdir():
            try:
                val_loss = float(file.name.split("=")[1].split("_")[0])
                if val_loss == args.id:
                    checkpoint_reference = file
            except:
                continue
    print("checkpoint used: ", checkpoint_reference)
    checkpoint = torch.load(checkpoint_reference, map_location=cst.DEVICE, weights_only=False)
    checkpoint["hyper_parameters"]["chosen_model"] = chosen_model
    config = checkpoint["hyper_parameters"]["config"]
    config.IS_WANDB = False
    config.CHOSEN_MODEL = chosen_model
    config.SAMPLING_TYPE = args.sampling_type
    config.HYPER_PARAMETERS[cst.LearningHyperParameter.DDIM_ETA] = args.ddim_eta
    config.HYPER_PARAMETERS[cst.LearningHyperParameter.DDIM_NSTEPS] = args.ddim_nsteps
    config.HYPER_PARAMETERS[cst.LearningHyperParameter.DDIM_TAIL_STEPS] = args.tail_steps
    config.HYPER_PARAMETERS[cst.LearningHyperParameter.GUIDANCE_SCALE] = args.guidance_scale
    config.HYPER_PARAMETERS[cst.LearningHyperParameter.CHURN_STEPS] = args.churn_steps
    config.HYPER_PARAMETERS[cst.LearningHyperParameter.CHURN_STRENGTH] = args.churn_strength
    if config.CHOSEN_MODEL == cst.Models.TRADES:
        # load checkpoint
        model = DiffusionEngine.load_from_checkpoint(checkpoint_reference, config=config, map_location=cst.DEVICE)
        agents.extend([WorldAgent(id=1,
                          name="WORLD_AGENT",
                          type="WorldAgent",
                          symbol=symbol,
                          date=str(historical_date.date()),
                          date_trading_days=cst.DATE_TRADING_DAYS,
                          model=model,
                          data_dir=cst.DATA_DIR,
                          cond_type=config.COND_TYPE,
                          cond_seq_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE] - config.HYPER_PARAMETERS[cst.LearningHyperParameter.MASKED_SEQ_SIZE],
                          size_type_emb=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SIZE_TYPE_EMB],
                          log_orders=log_orders,
                          random_state=np.random.RandomState(
                              seed=args.seed),
                          normalization_terms=normalization_terms,
                          using_diffusion=args.diffusion,
                            chosen_model=args.chosen_model,
                            gen_seq_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MASKED_SEQ_SIZE],
                            fix_time=args.fix_time,
                            type_decode=args.type_decode,
                            fix_cancel_bind=args.fix_cancel_bind,
                            fix_lob_pad=args.fix_lob_pad,
                            drop_type2_cond=args.drop_type2_cond,
                            depth_temp=args.depth_temp,
                            depth_reshape=args.depth_reshape,
                            size_reshape=args.size_reshape,
                            depth_noise=args.depth_noise,
                            dn_target_exec=args.dn_target_exec,
                            cancel_boost=args.cancel_boost,
                            depth_drift=args.depth_drift,
                            depth_drift_phi=args.depth_drift_phi,
                            book_target_thick=args.book_target_thick,
                            book_cancel_rate=args.book_cancel_rate,
                            cond_clip=args.cond_clip,
                            flow_balance=args.flow_balance,
                            flow_balance_window=args.flow_balance_window,
                          )
               ])
    elif config.CHOSEN_MODEL == cst.Models.CGAN:
        import torch.serialization
        from configuration import Configuration
        torch.serialization.add_safe_globals([Configuration])
        model = GANEngine.load_from_checkpoint(checkpoint_reference, config=config, map_location=cst.DEVICE, weights_only=False)
        agents.extend([WorldAgent(id=1,
                          name="WORLD_AGENT",
                          type="WorldAgent",
                          symbol=symbol,
                          date=str(historical_date.date()),
                          date_trading_days=cst.DATE_TRADING_DAYS,
                          model=model,
                          data_dir=cst.DATA_DIR,
                          log_orders=log_orders,
                          random_state=np.random.RandomState(
                              seed=args.seed),
                          normalization_terms=normalization_terms,
                          using_diffusion=args.diffusion,
                            chosen_model=args.chosen_model,
                            seq_len=config.HYPER_PARAMETERS[cst.LearningHyperParameter.SEQ_SIZE],
                          )
               ])
            
    # we freeze the model
    for param in model.parameters():
        param.requires_grad = False
else:
    agents.extend([WorldAgent(id=1,
                          name="WORLD_AGENT",
                          type="WorldAgent",
                          symbol=symbol,
                          date=str(historical_date.date()),
                          date_trading_days=cst.DATE_TRADING_DAYS,
                          model=None,
                          data_dir=cst.DATA_DIR,
                          cond_type=None,
                          cond_seq_size=None,
                          size_type_emb=None,
                          log_orders=log_orders,
                          random_state=np.random.RandomState(
                              seed=args.seed),
                          normalization_terms=normalization_terms,
                          using_diffusion=args.diffusion,
                            chosen_model=args.chosen_model if args.diffusion else None,
                          )
               ])



agent_types.extend("WorldAgent")
agent_count += 1

# 3) Execution Agent
trade_pov = True if args.execution_agents else False

#### Participation of Volume Agent parameters
# POV agent start one hour after market open and ends 30 minutes after 
pov_agent_start_time = mkt_open + pd.to_timedelta('0:15:00')
pov_agent_end_time = mkt_open + pd.to_timedelta('01:00:00')
pov_proportion_of_volume = args.execution_pov
pov_quantity = 1e5
pov_frequency = '1min'
pov_direction = "BUY"

pov_agent = POVExecutionAgent(id=agent_count,
                              name='POV_EXECUTION_AGENT',
                              type='ExecutionAgent',
                              symbol=symbol,
                              starting_cash=starting_cash,
                              start_time=pov_agent_start_time,
                              end_time=pov_agent_end_time,
                              freq=pov_frequency,
                              lookback_period=pov_frequency,
                              pov=pov_proportion_of_volume,
                              direction=pov_direction,
                              quantity=pov_quantity,
                              trade=trade_pov,
                              log_orders=True,  # needed for plots so conflicts with others
                              random_state=np.random.RandomState(seed=seed))
if trade_pov:
    execution_agents = [pov_agent]
    agents.extend(execution_agents)
    agent_types.extend("ExecutionAgent")
    agent_count += 1

########################################### KERNEL AND OTHER CONFIG ####################################################

kernel = Kernel("World Agent Kernel", random_state=np.random.RandomState(seed=seed))
kernelStartTime = mkt_open
kernelStopTime = mkt_close + pd.to_timedelta('00:00:01')

# parse the string into a datetime object
tmp = datetime.strptime(str(mkt_close), "%Y-%m-%d %H:%M:%S")

# extract the date and time components
date = tmp.date()
time_mkt_close = str(tmp.time()).replace(':', '-')

# Distinguishing suffix for the hypothesis-testing flags: without this, two runs that
# only differ by --fix-time / --type-decode / etc. (same sampler, eta, nsteps, date,
# window, checkpoint) produce the IDENTICAL log_dir name and silently overwrite each
# other's processed_orders.csv. Empty string when no flags are set, so unaffected runs
# keep their original directory names.
_flag_suffix = ""
if args.diffusion:
    if args.fix_time:
        _flag_suffix += "_ft"
    if args.type_decode != "l1":
        _flag_suffix += "_td" + args.type_decode
    if args.fix_cancel_bind:
        _flag_suffix += "_fcb"
    if args.fix_lob_pad:
        _flag_suffix += "_flp"
    if args.drop_type2_cond:
        _flag_suffix += "_dt2"
    if args.guidance_scale != 1.0:
        _flag_suffix += "_gs{}".format(args.guidance_scale)
    if args.depth_temp != 1.0:
        _flag_suffix += "_dtemp{}".format(args.depth_temp)
    if args.depth_reshape:
        _flag_suffix += "_dr"
    if args.size_reshape:
        _flag_suffix += "_sr"
    if args.depth_noise > 0.0:
        _flag_suffix += "_dn{}".format(args.depth_noise)
    if args.dn_target_exec > 0.0:
        _flag_suffix += "_te{}".format(args.dn_target_exec)
    if args.cancel_boost != 0.0:
        _flag_suffix += "_cb{}".format(args.cancel_boost)
    if args.depth_drift > 0.0:
        _flag_suffix += "_dd{}".format(args.depth_drift)
    if args.book_target_thick > 0.0:
        _flag_suffix += "_bt{}r{}".format(args.book_target_thick, args.book_cancel_rate)
    if args.cond_clip > 0.0:
        _flag_suffix += "_cc{}".format(args.cond_clip)
    if args.flow_balance > 0.0:
        _flag_suffix += "_fb{}".format(args.flow_balance)

if trade_pov:
    if args.diffusion:
        log_dir = "world_agent_{}_{}_{}_pov_{}_{}_{}_{}_{}_".format(symbol, date, time_mkt_close, pov_proportion_of_volume, seed, args.sampling_type, args.ddim_eta, args.ddim_nsteps) + checkpoint_reference.name[:13] + _flag_suffix
    else:
        log_dir = "market_replay_{}_{}_{}_pov_{}_{}".format(symbol, date, time_mkt_close, pov_proportion_of_volume, seed)
else:
    if args.diffusion:
        log_dir = "world_agent_{}_{}_{}_{}_{}_{}_{}_".format(symbol, date, time_mkt_close, seed, args.sampling_type, args.ddim_eta, args.ddim_nsteps) + checkpoint_reference.name[:13] + _flag_suffix
    else:
        log_dir = "market_replay_{}_{}_{}_{}".format(symbol, date, time_mkt_close, seed)

defaultComputationDelay = 0  # 50 nanoseconds
kernel.runner(agents=agents,
              startTime=kernelStartTime,
              stopTime=kernelStopTime,
              defaultComputationDelay=defaultComputationDelay,
              log_dir=log_dir)

simulation_end_time = dt.datetime.now()
print("Simulation End Time: {}".format(simulation_end_time))
print("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))

# Sampler timing summary (accumulated across all generated orders)
if args.diffusion and 'model' in dir() and hasattr(model, 'diffuser'):
    summary = model.diffuser.timing_summary()
    print("\n" + summary)
    timing_path = os.path.join("ABIDES", "log", log_dir, "timing_summary.txt")
    try:
        with open(timing_path, "w") as _f:
            _f.write(summary + "\n")
    except Exception:
        pass

# KL divergence report: compare generated orders to real data
if args.real_data_path:
    generated_csv = os.path.join("ABIDES", "log", log_dir, "processed_orders.csv")
    if os.path.exists(generated_csv):
        try:
            import sys as _sys
            _sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
            from evaluation.quantitative_eval.kl_divergence import compute_distribution_distances
            compute_distribution_distances(args.real_data_path, generated_csv)
        except Exception as e:
            print(f"[KL divergence] Could not compute: {e}")
    else:
        print(f"[KL divergence] Generated CSV not found at {generated_csv}")
