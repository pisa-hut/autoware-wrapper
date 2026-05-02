from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto

from rclpy.time import Duration, Time


class PublishMode(Enum):
    FIXED_RATE = auto()
    ALWAYS = auto()
    DISABLED = auto()


@dataclass
class TopicPublisher:
    name: str
    rate_hz: float
    mode: PublishMode
    enabled: bool
    publish_fn: Callable[[Time], None]

    last_pub_time: Time | None = field(default=None, init=False, repr=False)
    _period: Duration | None = field(default=None, init=False, repr=False)

    def setup(self):
        self._period = None if self.rate_hz <= 0.0 else Duration(seconds=1.0 / self.rate_hz)

    def should_publish(self, now: Time) -> bool:
        if not self.enabled:
            return False

        if self.mode == PublishMode.DISABLED:
            return False

        if self.mode == PublishMode.ALWAYS:
            return True

        # FIXED_RATE
        if self._period is None:
            return False

        if self.last_pub_time is None:
            return True

        if now < self.last_pub_time:
            return True

        return (now - self.last_pub_time) >= self._period

    def try_publish(self, now: Time) -> bool:
        if not self.should_publish(now):
            return False
        self.publish_fn(now)
        self.last_pub_time = now
        return True

    def force_publish(self, now: Time):
        if self.enabled:
            self.publish_fn(now)
            self.last_pub_time = now


class PublishManager:
    def __init__(self):
        self._items: dict[str, TopicPublisher] = {}

    def add(self, item: TopicPublisher):
        item.setup()
        self._items[item.name] = item

    def publish_all(self, now: Time) -> dict[str, bool]:
        result = {}
        for name, item in self._items.items():
            result[name] = item.try_publish(now)
        return result

    def enable(self, name: str):
        self._items[name].enabled = True

    def disable(self, name: str):
        self._items[name].enabled = False

    def force_publish(self, name: str, now: Time):
        self._items[name].force_publish(now)
