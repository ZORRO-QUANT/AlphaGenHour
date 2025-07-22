import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from data.calculator import StockDataCalculator
from data.stock_data import StockData
from models.linear_alpha_pool import LinearAlphaPool
from rl.env.core import AlphaEnvCore
from utils.constants import Group

logger = logging.getLogger(__name__)


class CustomCallback(BaseCallback):
    def __init__(
        self,
        data: StockData,
        pool: LinearAlphaPool,
        save_path: Path,
        train_calculator: StockDataCalculator,
        valid_calculator: StockDataCalculator,
        test_calculator: StockDataCalculator,
        group: Group,
        policy: str = "LSTM",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_path = save_path

        self.group = group
        self.policy = policy

        self.valid_calculator = valid_calculator
        self.test_calculator = test_calculator

        self.last_time = time.time()
        self.fisrt_time = time.time()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:

        assert self.logger is not None
        ic_valid, rank_ic_valid = self.pool.test_ensemble(self.valid_calculator)
        ic_test, rank_ic_test = self.pool.test_ensemble(self.test_calculator)

        now = time.time()
        epoch_time = now - self.last_time
        total_time = (now - self.fisrt_time) / 60
        self.last_time = now
        self.logger.record("group", self.group.name)
        self.logger.record("policy", self.policy)
        self.logger.record("epoch/time (s)", epoch_time)
        self.logger.record("total/time (min)", total_time)
        self.logger.record("pool/size", self.pool.size)
        self.logger.record(
            "pool/significant",
            (np.abs(self.pool.weights[: self.pool.size]) > 1e-4).sum(),
        )
        self.logger.record("pool/best_ic_ret", self.pool.best_ic_ret)
        self.logger.record("pool/eval_cnt", self.pool.eval_cnt)

        self.logger.record("valid/ic", ic_valid)
        self.logger.record("valid/rank_ic", rank_ic_valid)
        self.logger.record("test/ic", ic_test)
        self.logger.record("test/rank_ic", rank_ic_test)

        self.show_pool_state()
        self.save_checkpoint()

    def save_checkpoint(self):
        path = os.path.join(self.save_path, f"{self.num_timesteps}_steps")
        self.model.save(path)  # type: ignore
        if self.verbose > 1:
            print(f"Saving model checkpoint to {path}")
        with open(f"{path}_pool.json", "w") as f:
            json.dump(self.pool.to_json_dict(), f)

    def show_pool_state(self):
        state = self.pool.state
        logger.info("---------------------------------------------")
        for i in range(self.pool.size):
            weight = state["weights"][i]
            expr_str = str(state["exprs"][i])
            ic_ret = state["ics_ret"][i]
            print(f"> Alpha #{i}: weight {weight}, expr {expr_str}, IC {ic_ret}")
        print(f'>> Ensemble ic_ret: {state["best_ic_ret"]}')
        print("---------------------------------------------")

    @property
    def pool(self) -> LinearAlphaPool:
        assert isinstance(self.env_core.pool, LinearAlphaPool)
        return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore
