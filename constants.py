from enum import Enum
import os
import torch

# Keep signed (negative = marketable, spread-crossing) depths in BOTH training preprocessing and
# simulation conditioning. Default off = original behaviour (depth clamped at 0, so the model never
# sees a marketable order as a target and can only emit them as sampling noise).
#
# Two ways to turn this on — checked in order:
#   1. env var UNCLAMP_DEPTH=1 (works when your launcher actually propagates env vars to the python
#      subprocess — some HPC job launchers/srun/sbatch configs do NOT, silently).
#   2. a FILE named UNCLAMP_DEPTH_FLAG in the current working directory (repo root). This is the
#      robust option: a file on disk survives any launcher, since it doesn't depend on process
#      environment propagation at all. scripts/unclamp_retrain.sh creates/removes it for you.
# Whichever is on, it must be on identically at BOTH training and simulation time so the depth
# conditioning matches what the model was trained on.
UNCLAMP_DEPTH = (os.environ.get("UNCLAMP_DEPTH", "0") == "1") or os.path.exists("UNCLAMP_DEPTH_FLAG")

# Anchor all prices to each day's opening mid before normalization (training preprocessing AND
# simulation conditioning — must match, same discipline as UNCLAMP_DEPTH). WHY: TRADES z-scores
# ABSOLUTE prices with a global training mean (mean_price≈36.21, σ≈0.67), so the whole Jan-30
# test day sits at −3.4..−3.9σ, and the moment the (genuinely real) intraday decline pushes the
# mid through z≈−4.0 ($33.50) the model itself degenerates: the time channel explodes (event
# rate collapses ~35x), spreads gap, generation dies — observed at the SAME price threshold in
# both the fixed-σ and σ-controlled 75-min runs, with a healthy book. Anchoring to the session
# open makes the price channel a bounded intraday deviation (±~1%), removing the OOD boundary.
# File flag, not just env (env vars silently failed twice on this remote).
PRICE_REANCHOR = (os.environ.get("PRICE_REANCHOR", "0") == "1") or os.path.exists("PRICE_REANCHOR_FLAG")

# Scheduled-sampling retrain (Stage 3, v1). During training, with a scheduled probability, replace the
# real conditioning order-history with the model's OWN generated block (self-generated cond_orders),
# keep the real book state, and train against the real NEXT block. This exposes the model to its own
# drifted order-flow — the failure Stage 1 isolated (one-sided limit flow) — so it learns to recover.
# Stop-gradient rollout (self-generated conditioning is augmented input, not backpropped through).
# File-flag gated (env vars unreliable on this remote); default OFF => the training loop is unchanged.
# NOTE: launch the retrain with a FAST sampler (DDIM, small DDIM_NSTEPS ~10) so the per-step rollout is
# cheap; the engine prints the rollout step count and warns if it is large.
SCHEDULED_SAMPLING = (os.environ.get("SCHEDULED_SAMPLING", "0") == "1") or os.path.exists("SCHEDULED_SAMPLING_FLAG")
SS_P_MAX = 0.5        # max fraction of training steps that condition on self-generated order-history
SS_RAMP_FRAC = 0.4    # ramp p from 0 -> SS_P_MAX over this fraction of max_epochs (teacher-forced early)

# Keep a checkpoint per epoch during the retrain, instead of only the single best-by-val-loss (which
# deletes the rest). WHY: val loss is a poor proxy for sim stability — the scheduled-sampling
# checkpoint that fixed the cross-day drift (val_ema=0.715) was deleted by the best-val logic before we
# could keep it. With this on we retain one checkpoint per epoch so they can be trialled on stability.
# File-flag, default off => normal training keeps only the best checkpoint as before.
KEEP_EPOCH_CHECKPOINTS = (os.environ.get("KEEP_EPOCH_CHECKPOINTS", "0") == "1") or os.path.exists("KEEP_EPOCH_CHECKPOINTS_FLAG")


class LearningHyperParameter(str, Enum):
    NUM_DIFFUSIONSTEPS = "num_diffusionsteps"
    OPTIMIZER = "optimizer_name"
    LEARNING_RATE = "lr"
    EPOCHS = "epochs"
    BATCH_SIZE = "batch_size"
    CONDITIONAL_DROPOUT = "conditional_dropout"
    DROPOUT = "dropout"
    SEQ_SIZE = "seq_size"          #it's the sequence length
    MASKED_SEQ_SIZE = "masked_seq_size"
    AUGMENT_DIM = "augment_dim"
    SIZE_TYPE_EMB = "size_type_emb"
    SIZE_ORDER_EMB = "size_order_emb"
    LAMBDA = "lambda"
    CDT_DEPTH = "CDT_depth"
    CDT_MLP_RATIO = "CDT_mlp_ratio"
    CDT_NUM_HEADS = "CDT_num_heads"
    TEST_BATCH_SIZE = "test_batch_size"
    REG_TERM_WEIGHT = "reg_term_weight"
    P_NORM = "p_norm"
    DDIM_ETA = "ddim_eta"
    DDIM_NSTEPS = "ddim_nsteps"
    DDIM_TAIL_STEPS = "ddim_tail_steps"  # for HYBRID_PP_DDIM: number of DDIM steps at the end
    GUIDANCE_SCALE = "guidance_scale"    # classifier-free guidance weight (1.0 = off; needs checkpoint trained with CONDITIONAL_DROPOUT > 0)
    CHURN_STEPS = "churn_steps"          # CHURN sampler: number of early (high-noise) steps to renoise
    CHURN_STRENGTH = "churn_strength"    # CHURN sampler: EDM renoise strength κ (0 = pure DPM-Solver++)
    ONE_HOT_ENCODING_TYPE = "one_hot_encoding_type"
    CSDI_SIDE_DIM = "CSDI_side_dim"
    CSDI_CHANNELS = "CSDI_channels"
    CSDI_DIFFUSION_STEP_EMB_DIM = "CSDI_diffusion_step_emb_dim"
    CSDI_EMBEDDING_TIME_DIM = "CSDI_embedding_time_dim"
    CSDI_EMBEDDING_FEATURE_DIM = "CSDI_embedding_feature_dim"
    CSDI_LAYERS = "CSDI_layers"
    CSDI_N_HEADS = "CSDI_n_heads"
    MARKET_FEATURES_DIM = "market_features_dim"
    ORDER_FEATURES_DIM = "order_features_dim"
    GENERATOR_CHANNELS = "gen_channels"
    GENERATOR_LSTM_INPUT_DIM = "gen_LSTM_input_dimensions"
    GENERATOR_LSTM_HIDDEN_STATE_DIM = "gen_LSTM_hidden_state_dim"
    GENERATOR_NUM_FC_LAYERS = "gen_num_fc_layers"
    GENERATOR_FC_HIDDEN_DIM = "gen_fc_hidden_dim"
    GENERATOR_NUM_CONV_LAYERS = "gen_num_conv_layers"
    GENERATOR_KERNEL_SIZE = "gen_kernel_size"
    GENERATOR_STRIDE = "gen_stride"
    DISCRIMINATOR_LSTM_INPUT_DIM = "disc_LSTM_input_dimensions"
    DISCRIMINATOR_LSTM_HIDDEN_STATE_DIM = "disc_LSTM_hidden_state_dim"
    DISCRIMINATOR_NUM_FC_LAYERS = "disc_num_fc_layers"
    DISCRIMINATOR_FC_HIDDEN_DIM = "disc_fc_hidden_dim"
    DISCRIMINATOR_NUM_CONV_LAYERS = "disc_num_conv_layers"
    DISCRIMINATOR_KERNEL_SIZE = "disc_kernel_size"
    DISCRIMINATOR_STRIDE = "disc_stride"
    DISCRIMINATOR_CHANNELS = "disc_channels"
    
    


class Optimizers(Enum):
    ADAM = "Adam"
    RMSPROP = "RMSprop"
    SGD = "SGD"
    LION = "LION"


class Metrics(Enum):      #Quantitative evaluation
    test_loss = 'test_loss'
    pred_score = 'pred_score'
    disc_score = 'disc_score'
    js_shannon = 'js_shannon'
    kolmogorov_smirnov = 'kolmogorov_smirnov'

class Models(str, Enum):
    TRADES = "TRADES"
    CGAN = "CGAN"
    CDT = "CDT"

class LOB_Charts(Enum):      #Qualitative evaluation

    #real vs generated distribution
    t_sne = 't_sne'
    density_volume = 'density_volume'
    density_price = 'density_price'
    histogram_direction = 'density_direction'
    density_interarrival = 'density_interarrival'
    histogram_type = 'density_type'
    volume_first_time = 'volume_first_time'
    in_volume_min_time = 'in_volume_min_time'
    depth_time = 'depth_time'
    spread_time = 'spread_time'

    #market_experiment charts
    market_experiment_mid_price_time = 'market_experiment_mid_price_time'
    market_experiment_mid_price_difference_time = 'market_experiment_mid_price_difference_time'

    #stylized facts
    minutely_log_returns = 'minutely_log_returns'
    volume_correlation = 'volume_correlation'
    autocorrelation =  'autocorrelation'
    volatility_clustering = 'volatility_clustering'
    agregation_normality = 'agregation_normality'
    order_volume = 'order_volume'
    quoote_interarrival_time = 'quoote_interarrival_time'
    time_to_first_fill = 'time_to_first_fill'
    num_lim_orders_time_SEQ = 'num_lim_orders_time_SEQ'


class Stocks(Enum):
    APPL = "AAPL"
    INTC = "INTC"
    TSLA = "TSLA"
    AVXL = "AVXL"
    GOOG = "GOOG"
    AAME = "AAME"


class OrderEvent(Enum):
    """ The possible kind of orders in the lob """
    SUBMISSION = 1
    CANCELLATION = 2
    DELETION = 3
    EXECUTION = 4


class DatasetType(Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "val"
    

class Engine(str, Enum):    
    """NN_ENGINE = "NNEngine"
    GAN_ENGINE = "GANEngine"""
    DIFFUSION_ENGINE = "models.diffusers.DiffusionEngine"
    GAN_ENGINE = "models.gan.GANEngine"

    

# DEPRECATED: hardcoded normalization statistics used as fallback when
# data/{stock}/normalization_stats.json has not yet been generated by preprocessing.
# Run preprocessing (IS_DATA_PREPROCESSED=False) to produce the JSON, after which
# these constants are no longer used and can be removed.

# for 15 days of TSLA
TSLA_LOB_MEAN_SIZE_10 = 165.44670902537212
TSLA_LOB_STD_SIZE_10 = 481.7127061897184
TSLA_LOB_MEAN_PRICE_10 = 20180.439318660694
TSLA_LOB_STD_PRICE_10 = 814.8782058033195

TSLA_EVENT_MEAN_SIZE = 88.09459295373463
TSLA_EVENT_STD_SIZE = 86.55913199110894
TSLA_EVENT_MEAN_PRICE = 20178.610720500274
TSLA_EVENT_STD_PRICE = 813.8188032145645
TSLA_EVENT_MEAN_TIME = 0.08644932804905886
TSLA_EVENT_STD_TIME = 0.3512181506722207
TSLA_EVENT_MEAN_DEPTH = 7.365325300819055
TSLA_EVENT_STD_DEPTH = 8.59342838063813

# these are the values for the market features used by CGAN
TSLA_MEAN_SPREAD = 1628.1331238445746
TSLA_STD_SPREAD = 823.685980941235
TSLA_MEAN_RETURN = 2.471099866467089e-07
TSLA_STD_RETURN = 0.00020927952921847475
TSLA_MEAN_VOL_IMB = 0.5036961437566201
TSLA_STD_VOL_IMB = 0.18250211511475767
TSLA_MEAN_ABS_VOL = 965.1632447653776
TSLA_STD_ABS_VOL = 1285.16124777206
TSLA_MEAN_CANCEL_DEPTH = 1.2893666222896607
TSLA_STD_CANCEL_DEPTH = 2.1555155776464994
TSLA_MEAN_SIZE_100 = 0.6347363685292531
TSLA_STD_SIZE_100 = 0.8520664436360541


# for 15 days of INTC
INTC_LOB_MEAN_SIZE_10 = 6222.424274871972
INTC_LOB_STD_SIZE_10 = 7538.341086370264
INTC_LOB_MEAN_PRICE_10 = 3635.766219937785
INTC_LOB_STD_PRICE_10 = 44.15649995373795

INTC_EVENT_MEAN_SIZE = 324.6800802006092
INTC_EVENT_STD_SIZE = 574.5781447696605
INTC_EVENT_MEAN_PRICE = 3635.78165265669
INTC_EVENT_STD_PRICE = 43.872407609651184
INTC_EVENT_MEAN_TIME = 0.025201754040915927
INTC_EVENT_STD_TIME = 0.11013627432323592
INTC_EVENT_MEAN_DEPTH = 1.3685517399834501
INTC_EVENT_STD_DEPTH = 2.333747222206966

INTC_MEAN_SPREAD = 116.59695974561068
INTC_STD_SPREAD = 39.33230591185948
INTC_MEAN_RETURN = -5.2581575805820944e-08
INTC_STD_RETURN = 8.171171316578973e-05
INTC_MEAN_VOL_IMB = 0.5005629676888042
INTC_STD_VOL_IMB = 0.1838374952729647
INTC_MEAN_ABS_VOL = 40100.45630062603
INTC_STD_ABS_VOL = 43213.292109848255
INTC_MEAN_CANCEL_DEPTH = 0.649548691430768
INTC_STD_CANCEL_DEPTH = 1.6964303084449814
INTC_MEAN_SIZE_100 = 3.040093999614961
INTC_STD_SIZE_100 = 5.5826348200688924



SEED = 30

PRECISION = 32
N_LOB_LEVELS = 10
LEN_LEVEL = 4
LEN_ORDER = 6
LEN_ORDER_CGAN = 7

DATE_TRADING_DAYS = ["2015-01-02", "2015-01-30"]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DIR_EXPERIMENTS = "data/experiments"
DIR_SAVED_MODEL = "data/checkpoints"
DATA_DIR = "data"
RECON_DIR = "data/reconstructions"
PROJECT_NAME = ""
SPLIT_RATES = (.85, .05, .10)



