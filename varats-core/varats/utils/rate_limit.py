"""Rate-limiting function decorator."""
import logging
import typing as tp
from functools import wraps
from time import time, sleep

LOG = logging.getLogger(__name__)

RetType = tp.TypeVar("RetType")


class _RateLimiter():

    def __init__(self, max_invocations: int, timeframe_seconds: float):
        self.__max_invocations = max_invocations
        self.__timeframe = timeframe_seconds

        self.__num_invocations = 0
        self.__first_invocation: tp.Optional[float] = None

    def __call__(self, func: tp.Callable[...,
                                         RetType]) -> tp.Callable[..., RetType]:

        @wraps(func)
        def wrapper(*args: tp.Any, **kwargs: tp.Any) -> RetType:
            now = time()
            if not self.__first_invocation:
                self.__first_invocation = now
            if self.__num_invocations >= self.__max_invocations:
                self.__num_invocations = 0
                delta = now - self.__first_invocation
                if delta < self.__timeframe:
                    wait_time = self.__timeframe - delta
                    LOG.warning(
                        f"You hit the rate limit ({self.__max_invocations} "
                        f"calls/{self.__timeframe}s). Waiting {wait_time}s."
                    )
                    sleep(wait_time)
                    self.__first_invocation = None

            self.__num_invocations += 1
            return func(*args, **kwargs)

        return wrapper


def rate_limit(max_invocations: int, timeframe_seconds: float) -> _RateLimiter:
    """
    Rate-limit calls to a function. If more than ``max_invocations`` occur in
    the ``timeframe_seconds``, the decorator will sleep until the end of
    ``timeframe_seconds``.

    Args:
        max_invocations: the number of invocations that are allowed in the given
                         timeframs
        timeframe_seconds: the timeframe used to calculate the rate limit

    Returns:
        the rate-limited function
    """
    return _RateLimiter(max_invocations, timeframe_seconds)
