from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Type, Union

import torch
from torch import Tensor

from data.stock_data import Features, StockData
from utils.maybe import Maybe, none, some

_ExprOrFloat = Union["Expression", float]
_DTimeOrInt = Union["DeltaTime", int]


class OutOfDataRangeError(IndexError):
    pass


class Expression(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor: ...

    def __repr__(self) -> str:
        return str(self)

    def __add__(self, other: _ExprOrFloat) -> "Add":
        return Add(self, other)

    def __radd__(self, other: float) -> "Add":
        return Add(other, self)

    def __sub__(self, other: _ExprOrFloat) -> "Sub":
        return Sub(self, other)

    def __rsub__(self, other: float) -> "Sub":
        return Sub(other, self)

    def __mul__(self, other: _ExprOrFloat) -> "Mul":
        return Mul(self, other)

    def __rmul__(self, other: float) -> "Mul":
        return Mul(other, self)

    def __truediv__(self, other: _ExprOrFloat) -> "Div":
        return Div(self, other)

    def __rtruediv__(self, other: float) -> "Div":
        return Div(other, self)

    def __pow__(self, other: _ExprOrFloat) -> "Pow":
        return Pow(self, other)

    def __rpow__(self, other: float) -> "Pow":
        return Pow(other, self)

    def __pos__(self) -> "Expression":
        return self

    def __neg__(self) -> "Sub":
        return Sub(0.0, self)

    def __abs__(self) -> "Abs":
        return Abs(self)

    @property
    @abstractmethod
    def is_featured(self) -> bool: ...


class Feature(Expression):
    def __init__(self, feature: Features) -> None:
        self._feature = feature

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        assert period.step == 1 or period.step is None
        if (
            period.start < -data.max_backtrack_days
            or period.stop - 1 > data.max_future_days
        ):
            raise OutOfDataRangeError()
        start = period.start + data.max_backtrack_days
        stop = period.stop + data.max_backtrack_days + data.n_days - 1
        return data.data[start:stop, int(self._feature), :]

    def __str__(self) -> str:
        return "$" + self._feature.name.lower()

    @property
    def is_featured(self):
        return True

    @property
    def only_TF(self):
        return False


class Constant(Expression):
    def __init__(self, value: float) -> None:
        self.value = value

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        assert period.step == 1 or period.step is None
        if (
            period.start < -data.max_backtrack_days
            or period.stop - 1 > data.max_future_days
        ):
            raise OutOfDataRangeError()
        device = data.data.device
        dtype = data.data.dtype
        days = period.stop - period.start - 1 + data.n_days
        return torch.full(
            size=(days, data.n_stocks),
            fill_value=self.value,
            dtype=dtype,
            device=device,
        )

    def __str__(self) -> str:
        return str(self.value)

    @property
    def is_featured(self):
        return False

    @property
    def only_TF(self):
        return False


class DeltaTime(Expression):
    # This is not something that should be in the final expression
    # It is only here for simplicity in the implementation of the tree builder
    def __init__(self, delta_time: int) -> None:
        self._delta_time = delta_time

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        assert False, "Should not call evaluate on delta time"

    def __str__(self) -> str:
        return f"{self._delta_time}d"

    @property
    def is_featured(self):
        return False

    @property
    def only_TF(self):
        return False


def _into_expr(value: _ExprOrFloat) -> "Expression":
    return value if isinstance(value, Expression) else Constant(value)


def _into_delta_time(value: Union[int, DeltaTime]) -> DeltaTime:
    return value if isinstance(value, DeltaTime) else DeltaTime(value)


# Operator base classes


class Operator(Expression):
    @classmethod
    @abstractmethod
    def n_args(cls) -> int: ...

    @classmethod
    @abstractmethod
    def category_type(cls) -> Type["Operator"]: ...

    @classmethod
    @abstractmethod
    def validate_parameters(cls, *args) -> Maybe[str]: ...

    @classmethod
    def _check_arity(cls, *args) -> Maybe[str]:
        arity = cls.n_args()
        if len(args) == arity:
            return none(str)
        else:
            return some(
                f"{cls.__name__} expects {arity} operand(s), but received {len(args)}"
            )

    @classmethod
    def _check_exprs_featured(cls, args: list) -> Maybe[str]:
        any_is_featured: bool = False
        for i, arg in enumerate(args):
            if not isinstance(arg, (Expression, float)):
                return some(f"{arg} is not a valid expression")
            if isinstance(arg, DeltaTime):
                return some(
                    f"{cls.__name__} expects a normal expression for operand {i + 1}, "
                    f"but got {arg} (a DeltaTime)"
                )
            any_is_featured = any_is_featured or (
                isinstance(arg, Expression) and arg.is_featured
            )
        if not any_is_featured:
            if len(args) == 1:
                return some(
                    f"{cls.__name__} expects a featured expression for its operand, "
                    f"but {args[0]} is not featured"
                )
            else:
                return some(
                    f"{cls.__name__} expects at least one featured expression for its operands, "
                    f"but none of {args} is featured"
                )
        return none(str)

    @classmethod
    def _check_delta_time(cls, arg) -> Maybe[str]:
        if not isinstance(arg, (DeltaTime, int)):
            return some(
                f"{cls.__name__} expects a DeltaTime as its last operand, but {arg} is not"
            )
        return none(str)

    @property
    @abstractmethod
    def operands(self) -> Tuple[Expression, ...]: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({','.join(str(op) for op in self.operands)})"

    @property
    def only_TF(self):
        return False  # 其他的算子都不返回TF


class UnaryOperator(Operator):
    def __init__(self, operand: _ExprOrFloat) -> None:
        self._operand = _into_expr(operand)

    @classmethod
    def n_args(cls) -> int:
        return 1

    @classmethod
    def category_type(cls):
        return UnaryOperator

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        return cls._check_arity(*args).or_else(
            lambda: cls._check_exprs_featured([args[0]])
        )

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        return self._apply(self._operand.evaluate(data, period))

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor: ...

    @property
    def operands(self):
        return (self._operand,)

    @property
    def is_featured(self):
        return self._operand.is_featured


class BinaryOperator(Operator):
    def __init__(self, lhs: _ExprOrFloat, rhs: _ExprOrFloat) -> None:
        self._lhs = _into_expr(lhs)
        self._rhs = _into_expr(rhs)

    @classmethod
    def n_args(cls) -> int:
        return 2

    @classmethod
    def category_type(cls):
        return BinaryOperator

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        return cls._check_arity(*args).or_else(
            lambda: cls._check_exprs_featured([args[0], args[1]])
        )

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        return self._apply(
            self._lhs.evaluate(data, period), self._rhs.evaluate(data, period)
        )

    @abstractmethod
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._lhs},{self._rhs})"

    @property
    def operands(self):
        return self._lhs, self._rhs

    @property
    def is_featured(self):
        return self._lhs.is_featured or self._rhs.is_featured


class ISOperator(Operator):
    def __init__(self, lhs: _ExprOrFloat, rhs: _ExprOrFloat) -> None:
        self._lhs = _into_expr(lhs)
        self._rhs = _into_expr(rhs)

    @classmethod
    def n_args(cls) -> int:
        return 2

    @classmethod
    def category_type(cls):
        return ISOperator

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        return cls._check_arity(*args).or_else(
            lambda: cls._check_exprs_featured([args[0], args[1]])
        )

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        return self._apply(
            self._lhs.evaluate(data, period), self._rhs.evaluate(data, period)
        )

    @abstractmethod
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._lhs},{self._rhs})"

    @property
    def operands(self):
        return self._lhs, self._rhs

    @property
    def is_featured(self):
        return False

    # self._lhs.is_featured or self._rhs.is_featured


class IfElseOperator(Operator):
    def __init__(
        self,
        judgement: Expression,
        lhs: Union[Expression, float],
        rhs: Union[Expression, float],
    ) -> None:
        self._judgement = judgement
        self._lhs = lhs if isinstance(lhs, Expression) else Constant(lhs)
        self._rhs = rhs if isinstance(rhs, Expression) else Constant(rhs)

    @classmethod
    def n_args(cls) -> int:
        return 3

    @classmethod
    def category_type(cls) -> Type["Operator"]:
        return IfElseOperator

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        return self._apply(
            self._judgement.evaluate(data, period),
            self._lhs.evaluate(data, period),
            self._rhs.evaluate(data, period),
        )

    @abstractmethod
    def _apply(self, judgement: Tensor, lhs: Tensor, rhs: Tensor) -> Tensor: ...

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        return none(str)

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._judgement},{self._lhs},{self._rhs})"

    @property
    def operands(self):
        return self._judgement, self._lhs, self._rhs

    @property
    def is_featured(self):
        return self._lhs.is_featured or self._rhs.is_featured


class RollingOperator(Operator):
    def __init__(self, operand: _ExprOrFloat, delta_time: _DTimeOrInt) -> None:
        self._operand = _into_expr(operand)
        self._delta_time = _into_delta_time(delta_time)._delta_time

    @classmethod
    def n_args(cls) -> int:
        return 2

    @classmethod
    def category_type(cls):
        return RollingOperator

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        return (
            cls._check_arity(*args)
            .or_else(lambda: cls._check_exprs_featured([args[0]]))
            .or_else(lambda: cls._check_delta_time(args[1]))
        )

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time + 1
        stop = period.stop
        # L: period length (requested time window length)
        # W: window length (dt for rolling)
        # S: stock count
        values = self._operand.evaluate(data, slice(start, stop))  # (L+W-1, S)
        values = values.unfold(0, self._delta_time, 1)
        # print(values.shape)           # (L, S, W)
        return self._apply(values)  # (L, S)

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor: ...

    @property
    def operands(self):
        return self._operand, DeltaTime(self._delta_time)

    @property
    def is_featured(self):
        return self._operand.is_featured


class PairRollingOperator(Operator):
    def __init__(
        self, lhs: _ExprOrFloat, rhs: _ExprOrFloat, delta_time: _DTimeOrInt
    ) -> None:
        self._lhs = _into_expr(lhs)
        self._rhs = _into_expr(rhs)
        self._delta_time = _into_delta_time(delta_time)._delta_time

    @classmethod
    def n_args(cls) -> int:
        return 3

    @classmethod
    def category_type(cls):
        return PairRollingOperator

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        return (
            cls._check_arity(*args)
            .or_else(lambda: cls._check_exprs_featured([args[0], args[1]]))
            .or_else(lambda: cls._check_delta_time(args[2]))
        )

    def _unfold_one(
        self, expr: Expression, data: StockData, period: slice = slice(0, 1)
    ) -> Tensor:
        start = period.start - self._delta_time + 1
        stop = period.stop
        # L: period length (requested time window length)
        # W: window length (dt for rolling)
        # S: stock count
        values = expr.evaluate(data, slice(start, stop))  # (L+W-1, S)
        return values.unfold(0, self._delta_time, 1)  # (L, S, W)

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        lhs = self._unfold_one(self._lhs, data, period)
        rhs = self._unfold_one(self._rhs, data, period)
        return self._apply(lhs, rhs)  # (L, S)

    @abstractmethod
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: ...

    @property
    def operands(self):
        return self._lhs, self._rhs, DeltaTime(self._delta_time)

    @property
    def is_featured(self):
        return self._lhs.is_featured or self._rhs.is_featured


# Operator implementations


class Abs(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.abs()


class Sign(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.sign()


class Log(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.log()


class CSRank(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        nan_mask = operand.isnan()
        n = (~nan_mask).sum(dim=1, keepdim=True)
        rank = operand.argsort().argsort() / n
        rank[nan_mask] = torch.nan
        return rank


class Is_Greater(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:

        # -------------------------
        judgement = lhs > rhs

        return judgement

    @property
    def is_featured(self):
        return False

    @property
    def is_valid_ifelse(self):
        return self._lhs.is_featured and self._rhs.is_featured

    @property
    def only_TF(self):
        return True


class Is_Lesser(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:

        # -------------------------
        judgement = lhs < rhs

        return judgement

    @property
    def is_featured(self):
        return False

    # self._lhs.is_featured and self._rhs.is_featured
    @property
    def only_TF(self):
        return True

    @property
    def is_valid_ifelse(self):
        return self._lhs.is_featured or self._rhs.is_featured


class Is_Equal(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        judgement = lhs == rhs
        return judgement

    @property
    def is_featured(self):
        return False

    # self._lhs.is_featured and self._rhs.is_featured
    @property
    def is_valid_ifelse(self):
        return self._lhs.is_featured and self._rhs.is_featured

    @property
    def only_TF(self):
        return True


class If_Else(IfElseOperator):
    def _apply(self, judgement: Tensor, lhs: Tensor, rhs: Tensor) -> Tensor:
        return torch.where(judgement, lhs, rhs)


class Scale(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand / operand.abs().nansum(dim=-1, keepdim=True)


class WinsorizeStandardize(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        lower_bound = torch.nanquantile(operand, 0.03, dim=-1, keepdim=True)
        upper_bound = torch.nanquantile(operand, 0.97, dim=-1, keepdim=True)
        # # 缩尾处理
        w = torch.clamp(operand, lower_bound, upper_bound)
        mask = torch.isnan(w)
        n = (~mask).sum(dim=1)
        w[mask] = 0.0
        mean = w.sum(dim=1) / n
        std = ((((w - mean[:, None]) * ~mask) ** 2).sum(dim=1) / n).sqrt()
        w = (w - mean[:, None]) / std[:, None]
        w[mask] = torch.nan
        # 归一
        return w


class Add(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return lhs + rhs


class Sub(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return lhs - rhs


class Mul(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return lhs * rhs


class Div(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return lhs / rhs


class Pow(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return lhs**rhs


class Greater(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return lhs.max(rhs)


class Less(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return lhs.min(rhs)


class Delay(RollingOperator):
    # Ref is not *really* a rolling operator, in that other rolling operators
    # deal with the values in (-dt, 0], while Ref only deal with the values
    # at -dt. Nonetheless, it should be classified as rolling since it modifies
    # the time window.

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time
        stop = period.stop - self._delta_time
        return self._operand.evaluate(data, slice(start, stop))

    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the RollingOperator interface
        ...


class Mean(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)


class Sum(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.sum(dim=-1)


class Std(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.std(dim=-1)


class Var(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.var(dim=-1)


class Skew(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # skew = m3 / m2^(3/2)
        central = operand - operand.mean(dim=-1, keepdim=True)
        m3 = (central**3).mean(dim=-1)
        m2 = (central**2).mean(dim=-1)
        return m3 / m2**1.5


class Kurt(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # kurt = m4 / var^2 - 3
        central = operand - operand.mean(dim=-1, keepdim=True)
        m4 = (central**4).mean(dim=-1)
        var = operand.var(dim=-1)
        return m4 / var**2 - 3


class Max(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.max(dim=-1)[0]


class Min(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.min(dim=-1)[0]


class Med(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.median(dim=-1)[0]


class Mad(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        central = operand - operand.mean(dim=-1, keepdim=True)
        return central.abs().mean(dim=-1)


class Rank(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        last = operand[:, :, -1, None]
        left = (last < operand).count_nonzero(dim=-1)
        right = (last <= operand).count_nonzero(dim=-1)
        result = (right + left + (right > left)) / (2 * n)
        return result


class Delta(RollingOperator):
    # Delta is not *really* a rolling operator, in that other rolling operators
    # deal with the values in (-dt, 0], while Delta only deal with the values
    # at -dt and 0. Nonetheless, it should be classified as rolling since it
    # modifies the time window.

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time
        stop = period.stop
        values = self._operand.evaluate(data, slice(start, stop))
        return values[self._delta_time :] - values[: -self._delta_time]

    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the RollingOperator interface
        ...


class Wma(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        weights = torch.arange(n, dtype=operand.dtype, device=operand.device)
        weights /= weights.sum()
        return (weights * operand).sum(dim=-1)


class Ema(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        alpha = 1 - 2 / (1 + n)
        power = torch.arange(n, 0, -1, dtype=operand.dtype, device=operand.device)
        weights = alpha**power
        weights /= weights.sum()
        return (weights * operand).sum(dim=-1)


class DecayLinear(RollingOperator):

    # 创建线性递增权重：[1, 2, 3, ..., n]
    # 最近的数据权重最大，最早的数据权重最小
    # decay linear
    def _apply(self, operand: Tensor) -> Tensor:
        # [..., window]
        n = operand.shape[-1]
        weights = torch.arange(1, n + 1, 1, dtype=operand.dtype, device=operand.device)
        weights = weights / weights.sum()  # 归一化
        # 加权平均，最后一维是window
        return (weights * operand).sum(dim=-1)


class Argmax(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # operand: shape [..., window]
        # 返回最大值出现的相对天数（0表示最后一天，d-1表示最早）
        argmax_idx = operand.argmax(dim=-1)  # shape: [...]
        # 反向计数：0表示今天，d-1表示最早
        n = operand.shape[-1]
        return n - 1 - argmax_idx.float()


class Argmin(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # operand: shape [..., window]
        # 返回最大值出现的相对天数（0表示最后一天，d-1表示最早）
        argmin_idx = operand.argmin(dim=-1)  # shape: [...]
        # 反向计数：0表示今天，d-1表示最早
        n = operand.shape[-1]
        return n - 1 - argmin_idx.float()


class Product(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.prod(dim=-1)


class BollingerBandWidth(RollingOperator):
    # 布林带宽度 (Bollinger Band Width)
    def _apply(self, operand: Tensor) -> Tensor:

        sma = operand.mean(dim=-1)
        std = operand.std(dim=-1)
        if std.isnan().all():
            std = operand.std(dim=-1, correction=0)
        upper = sma + 2 * std
        lower = sma - 2 * std
        close = operand[:, :, -1]
        # 取当前价格（最后一个）
        return (close - lower) / (upper - lower + 1e-8)


class Rsi(RollingOperator):
    # 相对强弱指数 (RSI)
    def _apply(self, operand: Tensor) -> Tensor:
        delta = operand[:, :, 1:] - operand[:, :, :-1]

        avg_gain = torch.clamp(delta, min=0).mean(dim=-1)
        avg_loss = torch.clamp(-delta, min=0).mean(dim=-1)
        rs = avg_gain / avg_loss

        return 100 - (100 / (1 + rs))


class TSMOM(RollingOperator):
    # 最后一天price除以第一天price
    def _apply(self, operand: Tensor) -> Tensor:
        return operand[:, :, -1] / operand[:, :, 0] - 1


class MaDeviation(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        mean = operand.float().mean(dim=-1)
        return operand[:, :, -1] / mean - 1


# 价格偏离移动平均 (Price Deviation from MA)


class Cov(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        n = lhs.shape[-1]
        clhs = lhs - lhs.mean(dim=-1, keepdim=True)
        crhs = rhs - rhs.mean(dim=-1, keepdim=True)
        return (clhs * crhs).sum(dim=-1) / (n - 1)


class Corr(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        clhs = lhs - lhs.mean(dim=-1, keepdim=True)
        crhs = rhs - rhs.mean(dim=-1, keepdim=True)
        ncov = (clhs * crhs).sum(dim=-1)
        nlvar = (clhs**2).sum(dim=-1)
        nrvar = (crhs**2).sum(dim=-1)
        stdmul = (nlvar * nrvar).sqrt()
        stdmul[(nlvar < 1e-6) | (nrvar < 1e-6)] = 1
        return ncov / stdmul


class Beta(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        n = lhs.shape[-1]
        clhs = lhs - lhs.mean(dim=-1, keepdim=True)
        crhs = rhs - rhs.mean(dim=-1, keepdim=True)
        ncov = (clhs * crhs).sum(dim=-1) / (n - 1)
        nlvar = (clhs**2).sum(dim=-1) / (n - 1)
        return ncov / nlvar


class ResidualMomentum(PairRollingOperator):
    # open:
    # close:
    def _apply(self, open: Tensor, close: Tensor) -> Tensor:
        # lhs and rhs are of shape (L,s,w)->(length of days, num of stocks, window size)
        # Step 1:Compute the daily return of each stock
        stock_return = (close - open) / open  # (L,S, W)
        # Step 2:Compute the market return based on the daily return
        market_return = stock_return.nanmean(dim=1, keepdim=True)  # (L, 1, W)
        # Step 3: Compute means of stock return and market return over the window size W
        stock_return_mean = stock_return.mean(dim=-1, keepdim=True)
        market_return_mean = market_return.mean(dim=-1, keepdim=True)
        # Step 4:Compute covariance and variance of rhs(lowest prices)
        cov = (
            (stock_return - stock_return_mean) * (market_return - market_return_mean)
        ).sum(
            dim=-1, keepdim=True
        )  # .(L.S..1)
        var_rhs = ((market_return - market_return_mean) ** 2).sum(
            dim=-1, keepdim=True
        )  # (L，S.1)
        # step 5:Calculate beta coefficient(slope)
        beta = cov / var_rhs  # (L,S，1)
        # Step 7: Compute the residual
        residual = stock_return - beta * market_return  # (L,S, W)

        return (residual / residual.std(dim=-1, keepdim=True)).nanmean(dim=-1)


class RightWeightedKurt(PairRollingOperator):
    # use the right hand side tensor to compute the weighted average of the left hand side
    def _apply(
        self, lhs: Tensor, rhs: Tensor
    ) -> Tensor:  # here, we use the rhs tensor as the threshold decider
        lhs = (lhs - lhs[:, :, -1].unsqueeze(-1)) / lhs[:, :, -1].unsqueeze(-1)
        weights_ = rhs / rhs.sum(dim=-1, keepdim=True)
        operand = lhs * weights_
        central = operand - operand.mean(dim=-1, keepdim=True)
        m4 = (central**4).mean(dim=-1)
        var = operand.var(dim=-1)
        return m4 / var**2 - 3


Operators: List[Type[Operator]] = [
    # Unary
    Abs,
    Sign,
    Log,
    CSRank,
    Scale,
    WinsorizeStandardize,
    # Binary
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Greater,
    Less,
    Is_Greater,
    Is_Lesser,
    Is_Greater,
    Is_Equal,
    # Rolling
    Delay,
    Mean,
    Sum,
    Std,
    Var,
    Skew,
    Kurt,
    Max,
    Min,
    Med,
    Mad,
    Rank,
    Delta,
    Wma,
    Ema,
    DecayLinear,
    Argmax,
    Argmin,
    Product,
    BollingerBandWidth,
    Rsi,
    TSMOM,
    MaDeviation,
    # Pair rolling
    Cov,
    Corr,
    Beta,
    ResidualMomentum,
    RightWeightedKurt,
    If_Else,
]
