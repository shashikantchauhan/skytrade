from abc import ABC, abstractmethod
from collections.abc import Sequence

from trading_scanner.domain.models import Candle, Signal


class Strategy(ABC):
    """Contract for an independently testable trading strategy."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def evaluate(self, candles: Sequence[Candle]) -> Signal | None:
        """Return a signal only when this strategy's criteria are met."""
