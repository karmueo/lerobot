# RM75B 本地串行部署设计与任务清单

本文冻结 2026-11-10 已确认方案。目标为当前 Jetson AGX Orin 上的原生 LeRobot
Diffusion EMA 模型、RM_API2 机械臂、Unitree DDS 夹爪及 FastUMI UVC 相机。
不依赖 ROS2，不更改训练权重或原有用户修改。第一阶段仅实现串行闭环。

## 模型与几何契约

- 检查点：`outputs/train/dp_handumi_tcp_yaml_2026-11-09_18-24-13/checkpoints/best/pretrained_model_ema`。
- 本地数据：`data/handumi_data/umi-demo-tcp-dp-240_134`，30 Hz；3 帧 RGB 134×240，state/action 为 7 维。
- 状态与动作：回合起始 TCP 系的 xyz（m）、rotvec（rad）、归一化夹爪开度。
- 模型 horizon=16，原生 `predict_action_chunk` 已返回切片 `[2:10]` 的 8 个目标；禁止再次切片、逐步累加或误当关节角。
- FP32、DDPM100，加载检查点全部前后处理器和统计量；不联网下载骨干权重，不调用 `select_action`。
- `T_base_TCP=FK(q) @ T_Link7_TCP`；回合原点冻结为 `T_base_start`。
- 观测 `inv(T_base_start) @ T_base_TCP`；目标 `T_base_start @ T_start_TCP_target @ inv(T_Link7_TCP)` 送 IK。
- RGB 中心裁剪、INTER_AREA 缩放至宽240高134，CHW float32 /255；输入 `[1,3,3,134,240]` 和 `[1,3,7]`。

## 模块分工与冻结接口

共享类型见 `src/lerobot/deployment/rm75b/types.py`，所有模块内部使用单调时钟、弧度、米。

| 包 | 所有权 | 交付 |
| --- | --- | --- |
| A | config.py、geometry.py、observations.py、policy.py、对应测试 | 配置/资产校验、同步观测、模型子进程、离线一致性与测速 |
| B | arm.py、gripper.py、camera.py、kinematics.py、trajectory.py、对应测试 | 硬件驱动、低跟随轨迹及指令隔离 |
| C | runtime.py、cli.py、telemetry.py、运行层测试、操作文档和准备脚本 | 状态机、CLI、日志、故障注入和运行文档 |
| Lead | types.py、公共依赖/入口、部署 YAML、设计/验收文档 | 接口冻结、集成和独立验收 |

构造约定：`ArmDriver(cfg.arm, motion_enabled=False)`、`GripperDriver(cfg.gripper, motion_enabled=False)`、
`CameraSource(cfg.camera)`、`ArmKinematics.from_urdf(path)`、
`TrajectoryExecutor(kinematics, tcp_offset, cfg.execution, cfg.safety)`。
轨迹接口：`plan(actions, anchor, current_joints, current_gripper)->JointTrajectory`，
`sample(trajectory, elapsed)->(joints, gripper)`，不在规划/采样中下发命令。
运动学：`forward(q)`、`solve_ik(target, seed)`、`lower`、`upper`、`velocities`。

观测：`ObservationAssembler(kinematics, tcp_offset, cfg.execution)`，
`add_arm(ArmState)`、`add_gripper(GripperState)`、`add_frame(CameraFrame)`、`clear()`，
`build(anchor, episode_id, request_id, not_before, now)->ObservationWindow | None`。
几何：`pose_to_transform(pose6)`、`transform_to_pose(T)`、`invert(T)`、`rotation_distance(R1,R2)`、`validate_transform(T)`。
推理：`PolicyRunner(cfg.policy)`，`predict(window)->ActionChunk`；
`InferenceWorker(cfg.policy)` 实现共享 `InferenceInterface`。模型启动/预热可阻塞，但必须在运动开始前完成。
资产：`load_assets(cfg)->DeploymentAssets`，字段 `tcp_offset`、`initial_joints`、`model_config`、`metadata`、`hashes`。
配置：`load_config(path)->DeploymentConfig`、`config_to_dict(cfg)`。YAML 相对路径统一相对于仓库根目录。

### 配置字段（默认值是实现契约）

- policy：checkpoint(Path)、dataset_root(Path)、device='cuda:0'、num_inference_steps=None、startup_timeout_s=120、seed=0。
- arm：ip='192.168.1.18'、port=8080、udp_ip='192.168.1.100'、udp_port=8089、udp_cycle_ms=5、sdk_root(Path，默认 RM_API2/Python)、urdf(Path)、initial_pose(Path)。
- gripper：network_interface='wlP1p1s0'、domain_id=0、sides=('left','right')（启动仅冻结有反馈侧）、discovery_s=2、pos_close_rad=.02、pos_open_rad=5、kp=5、kd=.05、max_speed_rad_s=6、max_hold_error_rad=.30、feedback_timeout_s=.25、mapping=((0,0),(1,1))。
- camera：video_device=现有 by-path USB4.2 端口、width=1920、height=1080、fps=30、frame_timeout_s=.2、history_size=16。
- execution：control_hz=50、fps=30、n_obs_steps=3、n_action_steps=8、velocity_scale=.25、sync_slop_s=.05、history_tolerance_s=.015、input_timeout_s=.2、inference_timeout_s=30、settle_s=.3、settle_timeout_s=5、start_settle_s=.5、start_joint_tolerance_rad=.035、start_gripper_min=.95、stationary_velocity_rad_s=.02、home_timeout_s=120、mode='sequential'。
- safety：workspace_min=(.05,-.70,.05)、workspace_max=(.90,.70,1)、max_translation_m=.02、max_rotation_rad=10°、max_joint_step_rad=10°、feedback_timeout_s=.25、max_send_gap_s=.1、drift_translation_m=.002、drift_rotation_rad=1°。
- logging：output_dir=outputs/deploy/rm75b、lock_path=/tmp/lerobot-rm75b.lock。

未知配置字段、非有限值、无效时间/频率/映射、不同模型与数据契约均应明确失败。
模型/归一化资产必须是本地完整文件；核对 URDF 指纹、TCP 矩阵、固定初始位。

## 控制与时序

控制主进程独占硬件，spawn 子进程独占模型；有界队列仅允许一个推理请求。
READY → 起点与静止检查 → 冻结回合原点 → 三帧观测 → 推理 → 整段IK/检查 → 执行 → 稳定 → 再观测。
暂停、故障、停止使旧 episode/request 失效；迟到结果永不驱动机械臂。
停止不自动回位或张爪，重启不自动恢复运动。显式 home 先预检完整关节插值路径，再以不高于 URDF 关节速度 10% 和部署限速的速度回位并张开夹爪。

三帧时距为 -2/30,-1/30,0（容差15ms），全部在本段稳定完成之后。
反馈按图像接收时间线性插值（两侧均在50ms内），不得外推或重复帧。
推理期间维持最后目标并监测硬件；漂移超过2mm/1°或反馈年龄超限则故障。
串行动作从推理及预检完成后重新安排时间，不按旧观测时间裁剪。
八个路点全部先求IK，种子为前一解；URDF限位与25%速度约束、10°关节路点跳变、20mm/10° TCP步长和base系TCP工作空间保护。
50Hz低跟随，统一延长动作段使各路点均满足速度限制，并向上对齐控制节拍；机械臂和夹爪同一时间轴，不跳点追赶。
整段插值预检最多500个控制周期（约10秒），超出则拒绝整段，避免极低速度配置导致规划线程长期无法退出。
每次下发前再检查实际采样点的 TCP。末点保持至少 0.3 秒，机械臂的跟踪误差及速度均须达标，实机各在线夹爪侧的开度误差须不超过 0.05。5 秒内未稳定则停止。
控制间隔超过100ms、设备错误、传感器失联、进程退出均故障；不自动恢复。

## 夹爪、相机与生命周期

夹爪DDS域0、50Hz；仅真实在线电机有资格接收命令，在线集合启动后冻结。
真实反馈为观测，q=.02+g*(5-.02)；kp=5,kd=.05,速度6rad/s,误差钳位.30rad。
双向分段线性映射默认为恒等；不加入闭合锁定与夹空重试。
相机移植无ROS的UVC采集和物理端口定位，MJPEG完整帧主机接收时间戳、最新待解码缓存、短历史。
厂商夹爪服务外部管理，不自动杀进程；本地独占锁、重复控制进程检查。
inspect/benchmark/dry-run禁止底层运动、home、stop及夹爪发送，连接不自动移动。

## 交付与验收

CLI模式inspect/benchmark/dry-run/run，默认dry-run；交互home/start/stop/quit。
日志保存配置/依赖/源码与资产指纹、原点、观测、动作、轨迹、反馈、阶段时延、故障与摘要。
保留定制torch/torchvision，不执行会替换它们的整体uv sync；优先uv run --no-sync。
并行仅预留接口，明确拒绝并行配置。

详细验收记录见 `rm75b_deployment_acceptance.md`；实施子代理负责本模块测试，Lead独立审查和复测。

## 实施进度（2026-11-10）

- [x] 冻结共享接口、部署配置及验收标准。
- [x] A：模型、几何、三帧观测、原生一致性与测速。
- [x] B：硬件驱动、运动学、整段规划及底层指令隔离。
- [x] C：串行状态机、CLI、日志、环境脚本与操作文档。
- [x] 主代理：独立差异审查、故障返工、92项软件测试及真实GPU子进程测试。
- [x] 原生GPU一致性：10个验证窗口×3种子，最大误差0；DDPM100完整推理P95为4.384秒。
- [x] 原生只读硬件：相机实际采集、机械臂真实反馈、资源关闭；未发送执行器命令。
- [x] 夹爪服务在线及真实开度反馈、真实模型完整单段dry-run；主代理独立复连和日志验收通过，未下发执行器指令。
- [x] 人工监督下回位、夹爪开合、TCP三轴5mm检查与50mm往返；用户逐轴确认现场方向和幅度正常。
- [x] 基础验收预检改为独立spawn进程，保留100ms保护；修复后Y/Z按20%档实机通过。
- [ ] 小角度旋转、专门的现场停止响应和传感器中断测试。
- [ ] 单段、多段、完整Pick and place及10次效果统计。

实现沿用[官方部署文档](https://huggingface.co/docs/lerobot/main/inference)的本地模型与处理器组织方式。
当前部署入口为专用串行CLI，默认不开放并行执行。官方main文档会随版本更新，复现以当前仓库源码和运行清单为准。
