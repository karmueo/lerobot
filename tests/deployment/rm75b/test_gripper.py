"""用真实 DDS 字段形状验证开度、逐侧保护和指令隔离。"""

from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from lerobot.deployment.rm75b.config import GripperConfig
from lerobot.deployment.rm75b.gripper import DdsBindings, GripperDriver


class FakeDds:
    """可控单调时钟及可独立掉线的左右电机。"""

    def __init__(self, positions: dict[str, float]) -> None:
        self.now = 0.0
        self.positions = positions
        self.callbacks: dict[str, Callable[[Any], None]] = {}
        self.writes: list[tuple[str, float]] = []
        self.publishers: list[str] = []
        self.closed: list[str] = []

    def advance(self, duration: float) -> None:
        self.now += duration
        for side, position in self.positions.items():
            if side in self.callbacks:
                self.callbacks[side](SimpleNamespace(states=[SimpleNamespace(q=position)]))

    def driver(self, *, motion_enabled: bool = True, **kwargs: Any) -> GripperDriver:
        config = replace(GripperConfig(), discovery_s=0.05, **kwargs)
        bindings = DdsBindings(
            initialize=lambda *_: None,
            publisher=self.publisher,
            subscriber=self.subscriber,
            motor_cmd=lambda mode, q, *_: SimpleNamespace(q=q, mode=mode),
            motor_cmds=SimpleNamespace,
            motor_states=object,
        )
        return GripperDriver(
            config, dds=bindings, motion_enabled=motion_enabled, clock=lambda: self.now, sleep=self.advance
        )

    def subscriber(self, topic: str, _: Any) -> SimpleNamespace:
        side = topic.split("/")[2]

        def initialize(receiver: Callable[[Any], None]) -> None:
            self.callbacks[side] = receiver
            self.advance(0)

        return SimpleNamespace(Init=initialize, Close=lambda: self.closed.append(topic))

    def publisher(self, topic: str, _: Any) -> SimpleNamespace:
        self.publishers.append(topic)

        def write(message: Any) -> bool:
            self.writes.append((topic.split("/")[2], float(message.cmds[0].q)))
            return True

        return SimpleNamespace(Init=lambda: None, Close=lambda: self.closed.append(topic), Write=write)


def test_dry_run_reads_actual_mapping_without_opening_publishers():
    vendor = FakeDds({"right": 0.02 + 0.25 * 4.98})
    driver = vendor.driver(motion_enabled=False, mapping=((0, 0), (0.5, 0.25), (1, 1)))
    driver.connect()
    assert driver.online_sides == ("right",)
    state = driver.read_state()
    assert state is not None
    assert state.openness == pytest.approx(0.5)
    assert state.raw_openness == pytest.approx(0.25)
    assert state.motor_positions == vendor.positions
    for action in (lambda: driver.set_target(0.2), lambda: driver.tick(vendor.now), driver.hold):
        with pytest.raises(RuntimeError, match="未启用"):
            action()
    driver.close()
    assert not vendor.publishers and not vendor.writes


def test_target_check_requires_both_calibrated_sides() -> None:
    """两侧偏差相互抵消时，平均开度达标仍不足以结束动作."""
    vendor = FakeDds({"left": 0.02 + 0.20 * 4.98, "right": 0.02 + 0.30 * 4.98})
    driver = vendor.driver(mapping=((0, 0), (0.5, 0.25), (1, 1)))
    driver.connect()
    state = driver.read_state()
    assert state is not None and state.openness == pytest.approx(0.5)
    assert not driver.at_target(state, 0.5)
    vendor.positions["left"] = 0.02 + 0.24 * 4.98
    vendor.positions["right"] = 0.02 + 0.26 * 4.98
    vendor.advance(0.02)
    state = driver.read_state()
    assert state is not None and driver.at_target(state, 0.5)
    driver.close()


def test_first_command_and_later_side_do_not_open_automatically():
    vendor = FakeDds({"right": 1.0})
    driver = vendor.driver()
    driver.connect()
    driver.tick(vendor.now)
    assert not vendor.writes
    driver.hold()
    driver.tick(vendor.now)
    assert vendor.writes == [("right", 1.0)]
    vendor.positions["left"] = 4.0
    vendor.advance(0.02)
    driver.set_target(1.0)
    driver.tick(vendor.now)
    assert driver.online_sides == ("right",)
    assert vendor.writes[-1][0] == "right"
    assert vendor.writes[-1][1] == pytest.approx(1.12)
    driver.hold()
    vendor.advance(0.02)
    driver.tick(vendor.now)
    assert vendor.writes[-1][1] == pytest.approx(1.12)
    driver.close()


def test_discovery_and_stale_side_fail_closed():
    vendor = FakeDds({})
    driver = vendor.driver()
    with pytest.raises(RuntimeError, match="未收到"):
        driver.connect()
    assert len(vendor.closed) == 2 and not vendor.publishers

    vendor = FakeDds({"left": 2.0, "right": 2.0})
    driver = vendor.driver()
    driver.connect()
    driver.set_target(0.0)
    del vendor.positions["right"]
    vendor.advance(0.26)
    assert driver.read_state() is None
    with pytest.raises(RuntimeError, match="反馈失效"):
        driver.tick(vendor.now)
    assert not vendor.writes
    driver.close()


def test_slew_and_tracking_error_caps_apply_to_both_sides():
    vendor = FakeDds({"left": 1.0, "right": 3.0})
    driver = vendor.driver()
    driver.connect()
    driver.set_target(0.0)
    for _ in range(4):
        vendor.advance(0.02)
        driver.tick(vendor.now)
    assert dict(vendor.writes[-2:]) == pytest.approx({"left": 0.70, "right": 2.70})
    assert dict(vendor.writes[:2]) == pytest.approx({"left": 0.88, "right": 2.88})
    driver.close()
