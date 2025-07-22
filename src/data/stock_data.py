import datetime
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import polars as pl
import torch

from utils.constants import *
from utils.utils import load_config

path_general = Path(load_config("path.yaml")["general"])


class Features(IntEnum):
    OPEN = 0
    HIGH = 1
    LOW = 2
    CLOSE = 3
    VOLUME = 4
    AMOUNT = 5
    TRADES = 6
    TAKERBUYVOLUME = 7
    TAKERBUYAMOUNT = 8


class StockData:

    def __init__(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        groupby: GroupBy,
        group: Group,
        data_sources: DataSources,
        alphas: Union[List[Alpha], None] = None,
        max_backtrack_days: int = 100,
        max_future_days: int = 30,
        preloaded_data: Optional[Tuple[torch.Tensor, pd.Index, pd.Index]] = None,
        device: torch.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        ),
    ) -> None:

        self._start_time = start_time
        self._end_time = end_time
        self.alphas = alphas
        self.data_sources = data_sources
        self.device = device
        self.groupby = groupby
        self.group = group

        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days

        self._features = self._get_features() if alphas else list(Features)

        self.data, self._dates, self._stock_ids = (
            preloaded_data if preloaded_data is not None else self._get_data()
        )

    def _get_features(self):
        """get the features based on the kline and the alphas

        Returns:
            _type_: IntEnum
        """
        features = [
            "OPEN",
            "HIGH",
            "LOW",
            "CLOSE",
            "VOLUME",
            "AMOUNT",
            "TRADES",
            "TAKERBUYVOLUME",
            "TAKERBUYAMOUNT",
        ] + [alpha.name for alpha in self.alphas]

        return IntEnum("Features", features, module=__name__, start=0)

    def _get_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:

        features = ["$" + f.name.lower() for f in self._features]

        # compute the real start and end datetime
        real_start = self._start_time - datetime.timedelta(days=self.max_backtrack_days)
        real_end = self._end_time + datetime.timedelta(days=self.max_future_days)

        # ----------------------------------------------------
        # 1) read the kline data and alpha data
        # 1.1) read the kline data

        path_kline = (
            path_general
            / self.data_sources.kline.exchange.name
            / self.data_sources.kline.universe.name
            / "Klines"
            / "1h"
        )

        files = sorted(
            [
                path_
                for path_ in list(path_kline.glob("*.csv"))
                if path_.stem >= real_start.strftime("%Y-%m-%d %H:%M:%S")[:10]
                and path_.stem < real_end.strftime("%Y-%m-%d %H:%M:%S")[:10]
            ]
        )

        dfs = []

        for file in files:
            df_temp = pl.read_csv(file)
            dfs.append(df_temp)

        if dfs:

            df_kline = pl.concat(dfs, how="vertical")
            df_kline = df_kline.sort(["time", "symbol"])

        else:

            return None, None, None

        # convert the time
        df_kline = df_kline.with_columns(
            pl.col("time")
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
            .alias("time")
        )

        # add the time offset
        df_kline = df_kline.with_columns(
            (pl.col("time") + pl.duration(days=1)).alias("time")
        )

        dfs_alphas = []

        # ------------------------------------
        # 1.2) loop over and read each alpha
        for _alpha in self.alphas:

            alpha_path = (
                path_general / _alpha.exchange.name / _alpha.universe.name / _alpha.path
            )

            # read the alpha file
            df_alpha = pl.read_csv(alpha_path)

            # ------------------------------------
            # 2) do the data processing

            # convert the time
            df_alpha = df_alpha.with_columns(
                pl.col("time")
                .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
                .alias("time")
            )

            # sort data according to the (`time`, `symbol`)
            df_alpha = df_alpha.sort(["time", "symbol"])

            # Calculate the requested aggregations
            for agg_method, window in _alpha.aggregations.items():

                if agg_method == "STD":
                    df_alpha = df_alpha.with_columns(
                        pl.col(_alpha.alpha)
                        .rolling_std(window, min_periods=window)
                        .over("symbol")
                        .alias(f"{_alpha.alpha}_{agg_method}{window}")
                    )

                elif agg_method == "MA":
                    df_alpha = df_alpha.with_columns(
                        pl.col(_alpha.alpha)
                        .rolling_mean(window, min_periods=window)
                        .over("symbol")
                        .alias(f"{_alpha.alpha}_{agg_method}{window}")
                    )

                else:
                    raise ValueError("agg_method should be one of `MA` and `STD`")

                # drop the original alpha / drop all the nans
                df_alpha = df_alpha.drop(_alpha.alpha)
                df_alpha = df_alpha.drop_nulls()

            # ----------------------------------
            # convert the symbol name
            df_alpha = self.convert_symbols(
                df=df_alpha,
                exchange=_alpha.exchange.name,
                universe=_alpha.universe.name,
            )

            dfs_alphas.append(df_alpha)

        # ------------------------------------
        # Inner merge all alpha DataFrames
        if dfs_alphas:
            # Start with the first DataFrame
            df_alphas = dfs_alphas[0]

            # Inner join the rest sequentially
            for df in dfs_alphas[1:]:
                df_alphas = df_alphas.join(df, on=["time", "symbol"], how="inner")
        else:
            # if no dfs, raise an error
            raise

        # ----------------------------------------------------
        # 2) merge the kline data and alpha data / filter the datetime
        df = df_kline.join(df_alphas, on=["time", "symbol"], how="inner")

        df = df.filter(
            pl.col("time").is_between(
                real_start,
                real_end,
                closed="both",
            )
        )

        # ----------------------------------------------------
        # 3) read the group file if self.Group != Group.ALL
        if self.group != Group.ALL:

            df_group = self._read_group()

            df = df.join(df_group, on=["time", "symbol"], how="inner")
            df = df.filter(pl.col("class") == self.group.name)
            df = df.drop("class")

        # Drop id column and convert to pandas for reshaping operations
        df = df.drop("id")
        df_pandas = df.to_pandas()
        df_pandas = df_pandas.set_index(["time", "symbol"]).sort_index()
        df_pandas = df_pandas.stack().unstack(level=1)

        dates = df_pandas.index.levels[0]  # type: ignore
        stock_ids = df_pandas.columns
        values = df_pandas.values
        values = values.reshape((-1, len(features), values.shape[-1]))  # type: ignore

        return (
            torch.tensor(values, dtype=torch.float32, device=self.device),
            dates,
            stock_ids,
        )

    def _read_group(self):
        """
        Returns None
        -------

        This method is used for reading the kline file and format the symbols according to the Binance symbol format
        """
        exchange = self.data_sources.group.exchange.name
        universe = self.data_sources.group.universe.name
        freq = self.data_sources.group.freq

        path_group = (
            path_general
            / exchange
            / universe
            / "Groups"
            / freq
            / f"{self.groupby.name}.csv"
        )

        df_group = pl.scan_csv(path_group).collect()
        df_group = df_group.drop("id")
        df_group = df_group.with_columns(
            pl.col("time")
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
            .alias("time")
        )

        # rename the symbols, eventually all the symbols are in the format of `BTCUSDT`
        if (
            self.data_sources.group.exchange == Exchange.Okx
            and self.data_sources.group.universe == Universe.perp
        ):
            df_group = df_group.with_columns(
                pl.col("symbol").map_elements(lambda x: "".join(x.split("-")[:2]))
            )
        elif (
            self.data_sources.group.exchange == Exchange.Okx
            and self.data_sources.group.universe == Universe.spot
        ):
            df_group = df_group.with_columns(
                pl.col("symbol").map_elements(lambda x: x.replace("-", ""))
            )
        elif (
            self.data_sources.group.exchange == Exchange.Binance
            and self.data_sources.group.universe == Universe.perp
        ):
            df_group = df_group.with_columns(
                pl.col("symbol").map_elements(
                    lambda x: x.replace("1000000", "")
                    .replace("1000", "")
                    .replace("1MBABYDOGE", "BABYDOGE")
                )
            )
        elif (
            self.data_sources.group.exchange == Exchange.Binance
            and self.data_sources.group.universe == Universe.spot
        ):
            df_group = df_group.with_columns(
                pl.col("symbol").map_elements(
                    lambda x: x.replace("1000", "").replace("1MBABYDOGE", "BABYDOGE")
                )
            )
        else:
            pass

        # add the time offset
        df_group = df_group.with_columns(
            (pl.col("time") + pl.duration(days=1)).alias("time")
        )

        return df_group

    def convert_symbols(
        self,
        df: pl.DataFrame,
        exchange: Exchange = Exchange.Binance,
        universe: Universe = Universe.spot,
    ):
        """_summary_

        Args:
            df (pl.DataFrame): the input df we wanna convert
            exchange (Exchange): the exchange of the df
            universe (Universe): the universe of the df
        """

        # ----------------------------------
        # convert the symbols
        if exchange == Exchange.Binance and universe == Universe.perp:
            df = df.with_columns(
                pl.col("symbol").map_elements(
                    lambda x: x.replace("1000000", "")
                    .replace("1000", "")
                    .replace("1MBABYDOGE", "BABYDOGE")
                )
            )

        elif exchange == Exchange.Binance and universe == Universe.spot:
            df = df.with_columns(
                pl.col("symbol").map_elements(
                    lambda x: x.replace("1000", "").replace("1MBABYDOGE", "BABYDOGE")
                )
            )
        else:
            pass

        return df

    def __getitem__(self, slc: slice) -> "StockData":
        "Get a subview of the data given a date slice or an index slice."
        if slc.step is not None:
            raise ValueError("Only support slice with step=None")
        if isinstance(slc.start, str):
            return self[self.find_date_slice(slc.start, slc.stop)]
        start, stop = slc.start, slc.stop
        start = start if start is not None else 0
        stop = (
            (stop if stop is not None else self.n_days)
            + self.max_future_days
            + self.max_backtrack_days
        )
        start = max(0, start)
        stop = min(self.data.shape[0], stop)
        idx_range = slice(start, stop)
        data = self.data[idx_range]
        remaining = (
            data.isnan()
            .reshape(-1, data.shape[-1])
            .all(dim=0)
            .logical_not()
            .nonzero()
            .flatten()
        )
        data = data[:, :, remaining]
        return StockData(
            start_time=self._dates[start + self.max_backtrack_days].strftime(
                "%Y-%m-%d"
            ),
            end_time=self._dates[stop - 1 - +self.max_future_days].strftime("%Y-%m-%d"),
            max_backtrack_days=self.max_backtrack_days,
            max_future_days=self.max_future_days,
            features=self._features,
            device=self.device,
            preloaded_data=(
                data,
                self._dates[idx_range],
                self._stock_ids[remaining.tolist()],
            ),
        )

    def find_date_index(self, date: str, exclusive: bool = False) -> int:
        ts = pd.Timestamp(date)
        idx: int = self._dates.searchsorted(ts)  # type: ignore
        if exclusive and self._dates[idx] == ts:
            idx += 1
        idx -= self.max_backtrack_days
        if idx < 0 or idx > self.n_days:
            raise ValueError(
                f"Date {date} is out of range: available [{self._start_time}, {self._end_time}]"
            )
        return idx

    def find_date_slice(
        self, start_time: Optional[str] = None, end_time: Optional[str] = None
    ) -> slice:
        """
        Find a slice of indices corresponding to the given date range.
        For the input, both ends are inclusive. The output is a normal left-closed right-open slice.
        """
        start = None if start_time is None else self.find_date_index(start_time)
        stop = (
            None
            if end_time is None
            else self.find_date_index(end_time, exclusive=False)
        )
        return slice(start, stop)

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    @property
    def stock_ids(self) -> pd.Index:
        return self._stock_ids

    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Parameters:
        - `data`: a tensor of size `(n_days, n_stocks[, n_columns])`, or
        a list of tensors of size `(n_days, n_stocks)`
        - `columns`: an optional list of column names
        """
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.n_days != n_days:
            raise ValueError(
                f"number of days in the provided tensor ({n_days}) doesn't "
                f"match that of the current StockData ({self.n_days})"
            )
        if self.n_stocks != n_stocks:
            raise ValueError(
                f"number of stocks in the provided tensor ({n_stocks}) doesn't "
                f"match that of the current StockData ({self.n_stocks})"
            )
        if len(columns) != n_columns:
            raise ValueError(
                f"size of columns ({len(columns)}) doesn't match with "
                f"tensor feature count ({data.shape[2]})"
            )
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days :]
        else:
            date_index = self._dates[self.max_backtrack_days : -self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)
