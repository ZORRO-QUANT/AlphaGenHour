from typing import Type

from data.expression import *

# ----------------------------
# training related
BATCH_SIZE = 128
STEPS = 20
POOL_CAPACITY = 5
SEED = 10

# ----------------------------
# target
TARGET = 5

# ----------------------------
# feature extraction model related
N_LAYERS = 2
D_MODEL = 32
DROPOUT = 0.3

# ----------------------------
# policy model related
GAMMA = 1.0
ENT_COEF = 0.1

# ----------------------------
# token generation related

MAX_EXPR_LENGTH = 15
MAX_EPISODE_LENGTH = 256


OPERATORS: List[Type[Operator]] = [
    # Unary
    Abs,
    WinsorizeStandardize,
    Log,
    # Binary
    Add,
    Sub,
    Mul,
    Div,
    # Rolling
    Delay,
    Mean,
    Sum,
    Std,
    Var,  # Skew, Kurt,
    Max,
    Min,
    Med,
    Mad,
    Rank,
    Delta,
    Wma,
    Ema,
    Product,
    BollingerBandWidth,
    Rsi,
    MaDeviation,
    Argmax,
    Argmin,
    # # Pair rolling
    Cov,
    Corr,
    Beta,
    ResidualMomentum,
    RightWeightedKurt,
    # If_Else,
]

DELTA_TIMES = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]

CONSTANTS = [
    -30.0,
    -10.0,
    -5.0,
    -2.0,
    -1.0,
    -0.5,
    -0.01,
    0.01,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    30.0,
]

REWARD_PER_STEP = 0.0
