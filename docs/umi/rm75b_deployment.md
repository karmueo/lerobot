# RM75B 串行实机部署

在 Jetson AGX Orin 上使用本地 EMA 检查点控制 RM75B、Unitree DDS 夹爪和腕部 UVC 相机。

## 环境准备

1. 在 LeRobot 仓库根目录打开交互终端，使用现有 Python 3.12 虚拟环境和 Jetson 定制 Torch、torchvision。将下列占位符换成本机目录，后续部署命令沿用该终端。

   ```bash
   # 绑定测试机器的路径，实际使用中要根据情况修改。
   # <CycloneDDS 安装目录> = /home/scl/work/FastUMI_Data/ros2_ws/.deps/unitree_gripper/install/cyclonedds
   export CYCLONEDDS_HOME="<CycloneDDS 安装目录>"
   bash scripts/setup_rm75b_deployment.sh --check
   ```

   看到「硬件依赖导入检查通过」后继续。缺少 Python 硬件依赖时，改用 `--install`，以本地 Unitree 源码和 CycloneDDS 0.10.2 C 库安装绑定及 UVC 包。若没有适配 Python 3.12/aarch64 的 UVC 轮子，将 `RM75B_UVC_PACKAGE` 设为本机已编译的轮子路径。[依赖准备脚本](../../scripts/setup_rm75b_deployment.sh)

   | 参数 | 作用 | 必填/可选 | 默认值 | 取值约束或示例 |
   | --- | --- | --- | --- | --- |
   | 模式（位置参数） | 检查或安装硬件依赖 | 可选 | `--check` | `--check`、`--install` |
   | `CYCLONEDDS_HOME` | Python DDS 客户端使用的 C 库安装目录 | 脚本可选，部署终端需设置 | 脚本内置测试机路径 | 目录内须有 `lib/libddsc.so.0.10.2` 才能安装绑定 |
   | `RM75B_SDK_ROOT` | 导入检查使用的机械臂 Python SDK 目录 | 可选 | 脚本内置测试机路径 | SDK 不在默认位置时设置 |
   | `RM75B_UNITREE_SDK_SOURCE` | 安装绑定时使用的 Unitree Python 源码目录 | 可选 | 脚本内置测试机路径 | 目录内含 `unitree_sdk2py` |
   | `RM75B_UVC_PACKAGE` | 安装的 UVC 包 | 可选 | `pupil-labs-uvc==1.0.4` | 包版本或本机轮子路径 |

2. 核对 [部署配置](rm75b_deployment.yaml) 中的 EMA 检查点、对应数据集、机械臂 SDK、URDF、初始关节位文件、控制器地址、本机 UDP 反馈地址和 DDS 网卡。配置默认使用控制器 `192.168.1.18:8080`、本机反馈 `192.168.1.100:8089`。腕部相机按 `/dev/v4l/by-path/*-video-index0` 对应的实际物理端口填写，要求 1920×1080、30 Hz、MJPEG。确认串口访问权限，退出占用设备的 ROS 2 驱动和遥操端。

3. 确认夹爪服务配置。默认 `gripper.server_mode: auto`，实机启动时复用同网卡的单个服务，或从 `gripper.server_vendor_dir` 启动厂商服务。等待上限由 `gripper.server_start_timeout_s` 设置，默认 12 秒。厂商服务使用系统中的 CycloneDDS 0.10.2 C/C++ 库和 ROS 2 Humble 的 Iceoryx 库。部署会话退出时关闭它启动的服务。

   若配置为 `gripper.server_mode: external`，先在另一个终端从仓库根目录启动服务。将两个占位符换为实际厂商目录和配置中的 DDS 网卡，看到 `Dex1-1 Gripper Server started` 后保持该终端运行。[夹爪服务脚本](../../scripts/run_rm75b_gripper_server.sh)

   ```bash
   RM75B_GRIPPER_VENDOR="<厂商服务目录>" RM75B_GRIPPER_NETWORK="<DDS 网卡>" \
     bash scripts/run_rm75b_gripper_server.sh --run
   ```

   | 参数 | 作用 | 必填/可选 | 默认值 | 取值约束或示例 |
   | --- | --- | --- | --- | --- |
   | 模式（位置参数） | 检查动态库或前台启动服务 | 可选 | `--check` | `--check`、`--run` |
   | `RM75B_GRIPPER_VENDOR` | 厂商服务目录 | 可选 | 脚本内置测试机路径 | 目录内含可执行文件 `dex1_1_gripper_server` 和 `lib/` |
   | `RM75B_GRIPPER_NETWORK` | 服务绑定的网卡 | 可选 | `wlP1p1s0` | 与 `gripper.network_interface` 一致 |

## 实机执行

<a id="分步基础验收"></a>

1. 运动前确认操作者在场、运动范围已清空且急停可立即操作。首次闭环前，先核验回位、夹爪开合和 TCP 小位移，并用同一部署配置完成单段空跑。[基础验收记录](rm75b_deployment_acceptance.md#实机5mm基础验收结果) · [空跑记录与命令](rm75b_deployment_acceptance.md#夹爪服务与完整-dry-run-验收)
2. 从单段执行开始。若虚拟环境没有 `lerobot-deploy-rm75b` 入口，将其换为 `python -m lerobot.deployment.rm75b.cli`，保留其余命令和参数。[部署入口](../../src/lerobot/deployment/rm75b/cli.py)

   ```bash
   uv run --no-sync lerobot-deploy-rm75b --mode run --max-chunks 1 \
     --config docs/umi/rm75b_deployment.yaml
   ```

   | 参数 | 作用 | 必填/可选 | 默认值 | 取值约束或示例 |
   | --- | --- | --- | --- | --- |
   | `--config` | 部署 YAML | 可选 | `docs/umi/rm75b_deployment.yaml` | 选择与设备及资产匹配的配置 |
   | `--mode` | 选择运行模式 | 可选 | `dry-run` | 实机执行用 `run`。另有 `inspect`、`benchmark`、`dry-run` |
   | `--max-chunks` | 每回合完整动作段上限 | 可选 | 无上限 | 正整数，仅用于 `run` 或 `dry-run`。首次用 `1` |
   | `--save-inference` | 保存每次推理的原始相机帧、观测状态和动作输出 | 可选 | 关闭 | 仅用于 `run` 或 `dry-run` |
   | `--num-inference-steps` | 覆盖采样步数 | 可选 | 沿用 YAML，默认配置沿用检查点的 100 步 | 1–100。调整后需重新测速和验证 |
   | `--verify-native` | 测速时对照原始实现 | 可选 | 关闭 | 仅用于 `benchmark` |
   | `--warmup` | 测速预热次数 | 可选 | 2 | 非负整数，仅测速时生效 |
   | `--samples` | 测速样本数 | 可选 | 20 | 正整数，仅测速时生效 |

3. 等终端提示「已连接」，按空格回到初始关节位并张开夹爪。看到 `ready` 后按回车开始。程序确认起始反馈稳定后执行，完成一段后回到 `ready`，此时按 `q` 退出。

   | 按键 | 行为 |
   | --- | --- |
   | 空格 | 预检回位路径，通过后限速回位并张开夹爪 |
   | 回车 | 核验起始关节角和开度，稳定后开始回合 |
   | 退格 | 停止当前回合并清除待执行动作，夹爪保持最后的安全目标，会话保留 |
   | `q` 或 Ctrl+C | 停臂并关闭会话 |

   保持终端焦点。按键即时生效，重复使用空格或回车前松开按键并等待至少 1 秒。

   回位沿预检的关节插值路径执行，速度同时受部署限速和 URDF 关节速度的 10% 约束。实机动作段只有在机械臂和各在线夹爪均到达末点并保持稳定后才计为完成。

   实机运行中，模型动作点超出基座系 TCP 工作空间时，程序拒绝整段动作、停止当前回合并回到 `ready`。核对现场和日志后，按空格回位，再按回车启动新回合。停臂失败或其他故障会结束会话。

4. 在终端打印的运行目录（默认位于 `outputs/deploy/rm75b`）查看 `manifest.json` 核对配置和资产，查看 `events.jsonl` 确认 `chunk_completed` 及 `chunk_limit` 停止原因，查看 `summary.json` 核对回合和故障。自动启动的夹爪服务日志为 `gripper-server.log`。将现场观察和任务结果记入 [验收记录](rm75b_deployment_acceptance.md)。

   运行目录按上海时间命名，例如 `2026_1110__1921_11-run`。同秒重名时添加 `_02` 等序号。

   在启动命令中加 `--save-inference` 后，同一目录会增加两个子目录，按回合号和请求号配对。

   - `inference_inputs/` 为每次请求保存三张原始 JPEG 和 `observation.json`。JSON 包含帧时间、插值后的七轴关节角（弧度）、模型观测状态和回合起始 TCP。
   - `inference_outputs/` 保存八步动作和推理耗时。已取回但被作废的结果标记为 `discarded: true`，规划越界拒绝可对照事件日志。推理失败或会话退出时尚未取回的结果没有输出文件。
