"""覆盖 RM SDK 的单位、反馈故障、设置恢复和运动授权。"""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from lerobot.deployment.rm75b import arm
from lerobot.deployment.rm75b.config import ArmConfig


class FakeArm:
    """模拟 SDK 查询字典和真实函数签名，记录全部执行器调用。"""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.callback: Callable[[Any], None] | None = None
        self.udp = {
            "cycle": 2,
            "enable": True,
            "port": 8090,
            "force_coordinate": 0,
            "ip": "192.168.1.100",
            "custom_config": {"joint_speed": 0},
        }
        self.model = "RM_75"
        self.reject_canfd = False

    def rm_set_timeout(self, timeout: int) -> None:
        self.calls.append(("timeout", timeout))

    def rm_create_robot_arm(self, ip: str, port: int) -> SimpleNamespace:
        self.calls.append(("connect", ip, port))
        return SimpleNamespace(id=1)

    def rm_get_robot_info(self) -> tuple[int, dict[str, Any]]:
        return 0, {"arm_dof": 7, "arm_model": self.model}

    def rm_get_realtime_push(self) -> tuple[int, dict[str, Any]]:
        return 0, self.udp

    def rm_set_realtime_push(self, config: dict[str, Any]) -> int:
        self.calls.append(("udp", config))
        self.udp = config
        return 0

    def rm_realtime_arm_state_call_back(self, callback: Callable[[Any], None]) -> None:
        self.callback = callback

    def rm_movej_canfd(self, joints: list[float], follow: bool) -> int:
        self.calls.append(("canfd", joints, follow))
        return -1 if self.reject_canfd else 0

    def rm_movej(self, joints: list[float], speed: int, radius: int, connect: int, block: int) -> int:
        self.calls.append(("home", joints, speed, radius, connect, block))
        return 0

    def rm_set_arm_stop(self) -> int:
        self.calls.append(("stop",))
        return 0

    def rm_delete_robot_arm(self) -> int:
        self.calls.append(("delete",))
        return 0

    def rm_destroy(self) -> int:
        self.calls.append(("destroy",))
        return 0

    def publish(self, *, ip: bytes = b"192.168.1.18", parse_error: int = 0, joint_error: int = 0) -> None:
        assert self.callback is not None
        self.callback(
            SimpleNamespace(
                arm_ip=ip,
                errCode=parse_error,
                joint_status=SimpleNamespace(
                    joint_position=[90.0] * 7,
                    joint_speed=[180.0] * 7,
                    joint_err_code=[joint_error] + [0] * 6,
                    joint_en_flag=[True] * 7,
                ),
                err=SimpleNamespace(err=[0] * 24, err_len=0),
            )
        )


@pytest.fixture
def mock_arm(monkeypatch):
    """只模拟本地端口探测及厂商 SDK，驱动代码保持真实."""
    monkeypatch.setattr(arm, "check_udp_endpoint", lambda *_: None)
    vendor = FakeArm()
    sdk = SimpleNamespace(
        RoboticArm=lambda _: vendor,
        rm_thread_mode_e=SimpleNamespace(RM_TRIPLE_MODE_E=2),
        rm_udp_custom_config_t=dict,
        rm_realtime_push_config_t=dict,
        rm_realtime_arm_state_callback_ptr=lambda callback: callback,
    )
    return vendor, sdk


def test_dry_run_rejects_all_motion_at_sdk_boundary(mock_arm):
    vendor, sdk = mock_arm
    driver = arm.ArmDriver(ArmConfig(), sdk=sdk)
    driver.connect()
    assert vendor.udp["cycle"] == 1
    assert vendor.udp["custom_config"]["joint_speed"] == 1
    assert driver.read_state() is None
    vendor.publish(ip=b"192.168.1.99")
    assert driver.read_state() is None
    vendor.publish()
    state = driver.read_state()
    assert state is not None
    np.testing.assert_allclose(state.joints, np.pi / 2)
    np.testing.assert_allclose(state.velocities, np.pi)
    for action in (
        lambda: driver.home(np.zeros(7)),
        lambda: driver.send_joint_target(np.zeros(7)),
        driver.stop,
    ):
        with pytest.raises(RuntimeError, match="未启用"):
            action()
    driver.close()
    assert vendor.udp["port"] == 8090
    assert vendor.udp["custom_config"]["joint_speed"] == 0
    assert [record[0] for record in vendor.calls] == ["timeout", "connect", "udp", "udp", "delete", "destroy"]


def test_motion_is_low_follow_nonblocking_and_stop_is_idempotent(mock_arm):
    vendor, sdk = mock_arm
    driver = arm.ArmDriver(ArmConfig(), sdk=sdk, motion_enabled=True)
    driver.connect()
    with pytest.raises(RuntimeError, match="新鲜"):
        driver.send_joint_target(np.zeros(7))
    vendor.publish()
    driver.home(np.full(7, np.pi / 2))
    assert vendor.calls[-1] == ("home", [90.0] * 7, 10, 0, 0, 0)
    driver.send_joint_target(np.full(7, np.pi / 2))
    assert vendor.calls[-1] == ("canfd", [90.0] * 7, False)
    driver.stop()
    driver.stop()
    assert vendor.calls.count(("stop",)) == 1
    driver.close()


def test_faults_latch_and_failed_send_requires_stop(mock_arm):
    vendor, sdk = mock_arm
    driver = arm.ArmDriver(ArmConfig(), sdk=sdk, motion_enabled=True)
    driver.connect()
    vendor.publish()
    vendor.reject_canfd = True
    with pytest.raises(RuntimeError, match="返回 -1"):
        driver.send_joint_target(np.zeros(7))
    vendor.publish(joint_error=123)
    vendor.publish()
    state = driver.read_state()
    assert state is not None and state.errors == (123,)
    with pytest.raises(RuntimeError, match="无故障"):
        driver.home(np.zeros(7))
    vendor.publish(parse_error=-3)
    vendor.publish()
    with pytest.raises(RuntimeError, match="解析失败"):
        driver.read_state()
    driver.stop()
    assert vendor.calls[-1] == ("stop",)
    driver.close()


def test_non_rm75_releases_connection(mock_arm):
    vendor, sdk = mock_arm
    vendor.model = "GEN_72"
    driver = arm.ArmDriver(ArmConfig(), sdk=sdk)
    with pytest.raises(RuntimeError, match="不是七轴 RM_75"):
        driver.connect()
    assert vendor.calls[-2:] == [("delete",), ("destroy",)]
