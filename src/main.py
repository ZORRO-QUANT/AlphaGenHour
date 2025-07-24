import datetime
import os
import warnings
from pathlib import Path

# Set PyTorch MPS fallback BEFORE any PyTorch imports
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import logging
from typing import List, Optional, Tuple, Union, Dict

import pandas as pd
import torch
from sb3_contrib.ppo_mask import MaskablePPO

from config import *
from data.calculator import StockDataCalculator
from data.expression import *
from models.linear_alpha_pool import LinearAlphaPool, MseAlphaPool
from rl.callbacks import CustomCallback
from rl.env.wrapper import AlphaEnv
from rl.policy import Decoder, LSTMSharedNet, TransformerSharedNet
from utils import load_config, reseed_everything, setup_logger
from utils.constants import *

warnings.filterwarnings("ignore")

path_general = Path(load_config("path.yaml")["general"])
save_path = Path(load_config("path.yaml")["save_path"])
# Create logs directory if it doesn't exist
PROJECT_ROOT = Path(__file__).parents[1]  # Go up one level from src to project root
logs_dir = PROJECT_ROOT / "logs"

# Setup root logger with date-based filename
log_file = logs_dir / f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logger = setup_logger("root", log_file=log_file)

logger = logging.getLogger(__name__)


def save_factors(
    data: StockData,
    pool: LinearAlphaPool,
    final_cal: StockDataCalculator,
    group: Group,
    batch_size: int = BATCH_SIZE,
    policy_model: str = "all",
):

    experiment_folder = f"group_{group.name}_batch_{batch_size}_policy_{policy_model}"

    # todo: modify this path to change the folder of alphas

    path_factor = (
        save_path
        / data.data_sources.kline.exchange.name
        / data.data_sources.kline.universe.name
        / "Alphas"
        / data.data_sources.kline.freq
        / experiment_folder
    )

    if not path_factor.exists():
        path_factor.mkdir(exist_ok=True, parents=True)

    dates, symbol = data._dates, data._stock_ids

    q = data.max_backtrack_days
    h = -data.max_future_days
    for expre in pool.state["exprs"]:

        data = final_cal.evaluate_alpha(expre)

        df_alpha = pd.DataFrame(data.cpu(), index=dates[q:h], columns=symbol)

        df_reset = df_alpha.reset_index()
        df_long = df_reset.melt(
            id_vars="time", var_name="symbol", value_name=f"{expre}"
        )
        df_long.dropna(subset=f"{expre}", inplace=True)
        df_long.to_csv(f"{path_factor}/{expre}.csv", index=False, encoding="utf-8-sig")


def run_single_experiment(
    data_sources: DataSources,
    spans: TrainTestSpans,
    groupby: GroupBy,
    group: Group,
    alphas: Union[List[Alpha], None] = None,
    seed: int = SEED,
    pool_capacity: int = POOL_CAPACITY,
    steps: int = STEPS,
    last_policy: Union[Dict,None] = None,
):

    reseed_everything(seed)

    logger.info(
        f"""[Main] Starting training process
        Seed: {seed}
        Instruments:
        Pool capacity: {pool_capacity}
        Total Iteration Steps: {steps}"""
    )

    name_prefix = f"{group.name}_{pool_capacity}_{seed}"
    save_folder = (
        save_path
        / data_sources.kline.exchange.name
        / data_sources.kline.universe.name
        / "Alphas"
        / data_sources.kline.freq
        / name_prefix
    )

    if not save_folder.exists():
        save_folder.mkdir(parents=True, exist_ok=True)

    device: torch.device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    close_ = Feature(Features.CLOSE)

    data_train = StockData(
        start_time=spans.train_start,
        end_time=spans.train_end,
        data_sources=data_sources,
        alphas=alphas,
        device=device,
        groupby=groupby,
        group=group,
    )

    data_valid = StockData(
        start_time=spans.valid_start,
        end_time=spans.valid_end,
        data_sources=data_sources,
        alphas=alphas,
        device=device,
        groupby=groupby,
        group=group,
    )

    data_test = StockData(
        start_time=spans.test_start,
        end_time=spans.test_end,
        data_sources=data_sources,
        alphas=alphas,
        device=device,
        groupby=groupby,
        group=group,
    )

    target = Delay(close_, -TARGET) / close_ - 1

    calculator_train = StockDataCalculator(data_train, target)
    calculator_valid = StockDataCalculator(data_valid, target)
    calculator_test = StockDataCalculator(data_test, target)

    def build_pool(exprs: List[Expression]) -> LinearAlphaPool:
        pool = MseAlphaPool(
            capacity=pool_capacity,
            calculator=calculator_train,
            ic_lower_bound=None,
            l1_alpha=5e-3,
            device=device,
        )
        if len(exprs) != 0:
            pool.force_load_exprs(exprs)
        return pool

    pool = build_pool([])

    env = AlphaEnv(pool=pool, device=device, print_expr=True)

    checkpoint_callback = CustomCallback(
        save_path=save_folder,
        verbose=1,
        group=group,
        data=data_train,
        pool=pool,
        train_calculator=calculator_train,
        valid_calculator=calculator_valid,
        test_calculator=calculator_test,
        policy="LSTM",
        data_sources=data_sources,
        target=target,
        alphas=alphas
    )
    
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=N_LAYERS,
                d_model=D_MODEL,
                dropout=DROPOUT,
                device=device,
            ),
        ),
        gamma=GAMMA,
        ent_coef=ENT_COEF,
        batch_size=BATCH_SIZE,
        device=device,
        verbose=1,
    )
    if last_policy is not None:
            model.policy.load_state_dict(last_policy)

    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=name_prefix,
    )

    save_factors(
        data=data_train,
        pool=pool,
        final_cal=calculator_train,
        group=groupby,
        batch_size=model.batch_size,
        policy_model="LSTM",
    )
    current_policy = model.policy.state_dict()
    return current_policy


def main(
    start: datetime.datetime,
    end: datetime.datetime,
    batch: str,
    data_sources: DataSources,
    alphas: List[Alpha],
    groupby: GroupBy,
    group: Group,
    pool_capacity: int = POOL_CAPACITY,
    steps: Optional[int] = STEPS,
):
    """
    :param pool_capacity: Maximum size of the alpha pool
    :param steps: Total iteration steps
    """

    # make the span according to the start / end / batch
    last_policy=None

    for batch_start in pd.date_range(
        start=start, end=end, freq=batch, inclusive="left"
    ):

        logger.info(
            f"-> Now training date from {start.strftime('%Y-%m-%d %H:%M:%S')} to {(start + pd.Timedelta(batch)).strftime('%Y-%m-%d %H:%M:%S')}"
        )

        batch_end = batch_start + pd.Timedelta(batch)

        batch_spans = TrainTestSpans(
            train_start=batch_start,
            train_end=batch_end,
            valid_start=batch_start,
            valid_end=batch_end,
            test_start=batch_start,
            test_end=batch_end,
        )
        

        current_policy = run_single_experiment(
            data_sources=data_sources,
            alphas=alphas,
            spans=batch_spans,
            groupby=groupby,
            group=group,
            pool_capacity=pool_capacity,
            steps=steps,
            last_policy = last_policy
        )
        last_policy = current_policy


if __name__ == "__main__":

    # ------------------------------------
    # state the data sources
    data_sources = DataSources(
        kline=DataSource(exchange=Exchange.Binance, universe=Universe.spot, freq="1h"),
        group=DataSource(exchange=Exchange.Binance, universe=Universe.spot, freq="1h"),
    )

    # ------------------------------------
    # state all the alpha categories we want to compute
    alphas = [
        Alpha(
            category=Category.liquidity_1d,
            alpha="liq_nonliquidity_10",
        ),
        Alpha(
            category=Category.pv_1d,
            alpha="corr_retvd_30",
        ),
    ]

    # ------------------------------------
    # state the group
    groupby = GroupBy.amount_quarter_spot_3
    group = Group.MIDDLE

    # ------------------------------------
    # state the batch / start / end
    batch = "10d"
    start = datetime.datetime(2021, 1, 1, 8, 0, 0)
    end = datetime.datetime(2025, 6, 1, 8, 0, 0)

    main(
        start=start,
        end=end,
        batch=batch,
        data_sources=data_sources,
        alphas=alphas,
        groupby=groupby,
        group=group,
    )
