# RM75B 部署验收记录

日期：2026-11-10。软件、真实模型、夹爪反馈与完整单段 dry-run 验收通过。
用户确认到现场监督后，已执行回位、夹爪闭合/张开及三轴各5mm往返；反馈验收通过，用户确认夹爪开合正常。
用户认为5mm幅度太小，要求扩大至50mm并在每轴开始前单独确认；现已逐轴执行完成，
用户分别确认X、Y、Z方向、约50mm幅度和返回均正常。X使用原5%档，Y/Z使用用户要求的20%档。
已完成20%速度上限下的模型八步单段实机执行，用户确认平稳、机械臂与夹爪无异常。
尚未执行多段模型闭环或Pick and place，不能标记“实机任务已达成”。

## 软件与模型验收

| 项目 | 结果与证据 |
| --- | --- |
| 契约与几何 | 非单位原点往返、TCP/Link7外参、零/接近π旋转、单位转换、原生八步不重复切片通过 |
| 观测与夹爪 | RGB中心裁剪、三帧时距、重复/缺失/过期输入、禁止外推、双向映射、单侧反馈及掉线测试通过 |
| 轨迹与状态机 | 整段IK拒绝、限位/跳变/工作空间、极低限速拒绝、0.1/1/5秒推理延迟、迟到结果丢弃、新回合原点、控制截止时刻和关闭失败测试通过 |
| 指令隔离 | inspect/benchmark不创建执行器；dry-run底层禁止运动、回位、停臂及夹爪发送，mock测试通过 |
| 软件回归 | 新部署76项测试与现有训练/评估脚本16项测试，共92项通过；2项GPU开关测试在普通测试运行中跳过 |
| 原生模型一致性 | 同一Orin、FP32、DDPM100、10个真实验证窗口×3种子共30组对照；包含各去噪步RNG，最大绝对误差0，要求rtol=1e-5/atol=1e-6 |
| 真实推理子进程 | 单独启用GPU测试，实际spawn加载/预热、旧请求作废排空、新请求返回8×7 FP32、限时关闭通过，1项测试用时23.96秒 |
| Ruff | 新部署源文件、测试及import_utils的check与format检查通过 |
| Mypy | 新部署15个源文件用仓库固定版本Mypy检查通过；全仓钩子仍有27个既有错误，与干净HEAD逐项相同，本次未新增 |
| 环境 | RM SDK 1.1.7可加载；CycloneDDS 0.10.2、Unitree SDK 1.0.1、pupil-labs-uvc 1.0.4已安装并导入成功；Torch 2.11.0、torchvision 0.26.0+336d36e保持不变 |

独立验收纠正了控制周期内阻塞后发送迟到指令、旧请求排空占用新回合超时、同时间戳故障漏检、
规划验证次数无界、将自身uv启动器误判为冲突以及停臂失败仍返回成功等问题，并增加回归覆盖。

全仓Mypy的既有问题位于Diffusion配置/导出、compute_stats与export模块，共6个文件。
基线为`76b7ff9120ed127df54719ea62c857aeb45db607`，使用同一Mypy 2.3.1钩子对干净HEAD快照检查。
相关错误列表保存在[本次日志](../../outputs/deploy/rm75b/acceptance/mypy.log)和[基线日志](../../outputs/deploy/rm75b/acceptance/mypy_baseline.log)。

## 性能

真实EMA检查点、cuda:0、FP32、DDPM100；预热2次，测量20个真实验证窗口。

| 阶段 | P50 | P95 | 最大 |
| --- | ---: | ---: | ---: |
| 本地验证窗口读取 | 0.013061 s | 0.016897 s | 0.020242 s |
| 原生处理器预处理 | 0.002473 s | 0.002597 s | 0.002932 s |
| 模型推理 | 4.378439 s | 4.380551 s | 4.381124 s |
| 原生后处理 | 0.001204 s | 0.001351 s | 0.001582 s |
| 完整推理（预处理至返回动作） | 4.382146 s | 4.384309 s | 4.385013 s |

进程峰值RSS为4,399,521,792字节，CUDA峰值allocated为1,162,246,656字节，reserved为1,258,291,200字节。
Jetson使用统一内存，这些指标不能简单相加为物理内存用量。

50Hz是配置的执行器发送节拍；上述约4.38秒是一次八步预测耗时。实际闭环还包括重新采集、IK、
轨迹播放和至少0.3秒稳定等待，本轮尚未测得完整实机闭环频率。当前证据支持串行部署，不支持开放并行执行。

完整测量与资产摘要：[native_benchmark.json](../../outputs/deploy/rm75b/acceptance/native_benchmark.json)。

## 早前实机只读检查

| 检查 | 实际结果 |
| --- | --- |
| 本机静态预检 | SDK、UDP绑定、网卡、UVC、相机USB端口全部通过；检查当时未发现冲突控制进程 |
| UVC采集 | 约3秒内收到91个不同帧，RGB uint8、1920×1080，最后一帧接收年龄约39ms，关闭成功 |
| RM75B反馈 | 连通192.168.1.18，读取149次有效七轴状态，无控制器故障；与initial_pose最大误差约0.0000673 rad，关闭并还原UDP配置成功 |
| Unitree反馈 | 发现阶段未收到真实DDS反馈，按要求拒绝连接；未以零值或指令代替实测开度 |
| 厂商服务与串口 | 检查当时服务端未运行，主机可见/dev/ttyUSB0至/dev/ttyUSB3；串口存在不代表电机通信已通过 |
| 服务端动态库 | 独立脚本--check通过，成对CycloneDDS C/C++与Iceoryx符号解析成功；未自动启动服务端 |

原`hardware.launch.py`的jpeg模式通过UVC打开同一USB物理端口，即使`/dev/video*`链接缺失也能定位设备。
`start_recorder:=false`仍会启动机械臂驱动、回位和夹爪节点。其Humble环境还提供了厂商夹爪服务需要的
Iceoryx库路径；独立服务脚本显式设置这些原生动态库，不需要ROS节点。

只读设备证据：[passive_hardware.json](../../outputs/deploy/rm75b/acceptance/passive_hardware.json)、
[host_processes.json](../../outputs/deploy/rm75b/acceptance/host_processes.json)。
静态预检的`ready=true`表示本机依赖、地址和路径可用，不代表夹爪服务已在线或整个任务可以运动。

以上“服务未运行、无夹爪反馈”为早前检查结果，后续授权启动及验收结果如下。
原ROS硬件launch与新部署会竞争机械臂、夹爪和相机，应先在原终端退出；程序只报告冲突，不自动终止进程。

## 夹爪服务与完整 dry-run 验收

2026-11-10 14:07（北京时间），实施子代理按用户授权执行
`bash scripts/run_rm75b_gripper_server.sh --run`。启动前未发现冲突控制进程或已有厂商服务。
服务发现左侧电机 ID 1，串口 `/dev/ttyUSB0`，反馈话题 `rt/dex1/left/state`；未发现右侧电机，
本会话冻结在线集合为 `left`。验收结束时独立厂商服务 PID `1993068` 仍运行，部署及采样进程均已退出。
无需为后续会话重复启动同一服务。

| 验收项 | 实际结果 |
| --- | --- |
| 子代理只读夹爪采样 | 3.003秒、149条不同接收时间戳的反馈，0缺失；最大年龄2.172ms；开度0.996809～0.996816，实测电机q约4.984126rad |
| 主代理独立DDS复连 | 2秒、100条不同接收时间戳的反馈，0缺失；最大年龄2.187ms；未创建命令发布器，关闭成功，厂商服务仍存活 |
| 真实观测窗口 | 相机帧502、503、504，间隔36.006ms、31.994ms；均在回合原点冻结之后，最新帧进入推理时年龄17.569ms |
| 原生模型 | 指定EMA、CUDA、FP32、DDPM100，返回8×7动作；模型推理4.384100秒，含前后处理共4.388305秒 |
| 整段轨迹 | 全部八个TCP目标通过IK与预检，含实测起点共九个关节路点；主代理重算坐标转换，最大末端位置残差0.1574mm、旋转残差0.000341rad |
| 空跑状态顺序 | starting → observing → inferring → planning → executing → settling → ready；完成1段后按chunk_limit结束回合 |
| 指令隔离 | 34条预定指令全部applied=false，九个路点按序覆盖；主代理另复跑机械臂、夹爪和状态机三项隔离测试，全部通过 |
| 时序与实际反馈 | 规划时间轴0.32秒，冻结原点至完成空跑5.242160秒；预定指令最大记录间隔20.272ms；全程七轴反馈最大峰峰变化0.010003° |
| 日志及资源 | 无fault、关闭errors为空、CLI退出0；主代理核对运行模块和资产SHA256与当前文件一致 |

上述时间轴和指令间隔来自空跑，不能视为真实执行器发送性能、运动跟踪或实机闭环频率。
退出时终端出现一次厂商SDK的`[Reader] take sample error`。该提示来自DDS回调的
`reader.take(1)`异常分支，SDK未保留异常细节，尚不能确定精确原因；本次退出和后续独立复连均成功，
没有残留部署进程或已记录的关闭失败。未修改第三方SDK。

本轮未修改运行代码或降低任何起始、安全阈值。完整复现命令（交互输入`start`，返回ready后输入`quit`）：

```bash
env -u PYTHONPATH \
  CYCLONEDDS_HOME=/home/scl/work/FastUMI_Data/ros2_ws/.deps/unitree_gripper/install/cyclonedds \
  UV_CACHE_DIR=/tmp/lerobot-rm75b-uv-cache \
  uv run --no-sync lerobot-deploy-rm75b --mode dry-run --max-chunks 1 \
  --config docs/umi/rm75b_deployment.yaml
```

证据目录：[夹爪服务与反馈](../../outputs/deploy/rm75b/acceptance/gripper-passive-20261110_140745/)、
[主代理完整空跑审核](../../outputs/deploy/rm75b/acceptance/gripper-passive-20261110_140745/root-dry-run-review.json)、
[主代理独立复连](../../outputs/deploy/rm75b/acceptance/gripper-passive-20261110_140745/root-feedback-review.json)。
完整空跑的[配置及指纹](../../outputs/deploy/rm75b/20261110T060839.026969Z-dry-run/manifest.json)、
[事件](../../outputs/deploy/rm75b/20261110T060839.026969Z-dry-run/events.jsonl)、
[摘要](../../outputs/deploy/rm75b/20261110T060839.026969Z-dry-run/summary.json)保留在原运行目录。

## 基础验收准备与现场确认

用户要求运动前先确认已到现场监督。准备阶段没有启用`--execute`，随后收到用户明确确认：
“已在现场，范围已清空，可以开始基础验收”，才由实施子代理启动实机步骤。
独立[分步入口](../../src/lerobot/deployment/rm75b/acceptance.py)不加载模型；
操作步骤见[部署手册](rm75b_deployment.md#分步基础验收)。

实施子代理完成准备，主代理审查后修正了规划阻塞、发送截止时刻、未发送指令误记为已发送、
故障停止失败漏记，以及末点漂移后先纠偏再报错等问题。主代理独立部署回归为84项通过、2项GPU开关测试跳过；
最终修正后的基础验收模块单独复跑10项mock全部通过，Ruff和仓库固定版本Mypy检查通过。
这些结果只证明工具的软件保护与离线计划，不代表实机基础动作已完成。

| 准备项 | 结果 |
| --- | --- |
| 只读设备预检 | 无冲突；机械臂无故障，距配置初始位最大关节差0.002871°；左夹爪开度0.996812，服务PID1993068在线 |
| 三轴目标 | 沿基座系X/Y/Z正方向各5mm，固定姿态；由回位后冻结的同一实测基线出发，每轴独立返回 |
| 离线主轴FK位移 | X为4.993mm、Y为4.976mm、Z为4.990mm；主代理独立重算通过，返程关节目标与基线精确一致 |
| 低速与稳定 | 本次每个去/返程均2.08秒，关节速度不超过URDF上限5%；末点位置误差要求≤0.5mm、姿态误差≤0.2° |
| 现场观察 | 到位后保持至少2秒，再接收独立back命令；真实方向和幅度仍须外部观察确认 |

[只读预检](../../outputs/deploy/rm75b/acceptance/basic-plan-20261110_142215/passive-preflight.log)、
[主代理离线轨迹核验](../../outputs/deploy/rm75b/acceptance/basic-axis-plans-review.json)、
[部署回归JUnit](../../outputs/deploy/rm75b/acceptance/basic-preparation-tests.xml)、
[最终基础模块JUnit](../../outputs/deploy/rm75b/acceptance/basic-acceptance-tests.xml)均已保存。

## 实机5mm基础验收结果

北京时间14:31～14:36，实施子代理逐项执行，主代理核对阶段日志后发出下一步指令。
回位以控制器速度5%受理并完成稳定判定；机械臂原本就在初始位附近，本次没有验证大范围回位路径。
夹爪闭合与张开分别约1.13秒，实际开度为0和0.9985，用户现场确认开合正常。
闭合时原始电机q为−0.01692rad，归一化开度按既有端点限制为0；未修改标定映射。

下表均由真实七轴反馈计算TCP，不能代替独立的物理位移测量。

| 方向 | 去程主轴位移 | 相对5mm目标的位置误差 | 返回基线位置残差 | 返回姿态残差 |
| --- | ---: | ---: | ---: | ---: |
| +X | 4.842mm | 0.322mm | 0.314mm | 0.0503° |
| +Y | 4.778mm | 0.253mm | 0.193mm | 0.0426° |
| +Z | 4.851mm | 0.264mm | 0.197mm | 0.0656° |

主代理独立检查了6条轨迹的全部路点覆盖、反向返程、限位/限速及已下发目标的FK工作空间，均通过。
5756条机械臂指令的连续发送间隔中位数20.064ms、最大20.719ms；夹爪控制tick最大间隔33.516ms，
均未触及100ms限制。16654次反馈记录没有控制器故障；验收进程退出0，日志faults和关闭errors均为空。
末次夹爪开度0.998536；独立厂商服务PID1993068保留运行，未执行模型推理或模型闭环。

用户现场回复：“夹爪开合正常，三轴方向运动幅度太小，可以运动50mm，每一轴开始运动前让我确认。”
因此本轮只将三轴的反馈控制记录标记通过，5mm物理方向/幅度观察结论仍不充分。

[主代理独立审核](../../outputs/deploy/rm75b/acceptance/20261110T063120.310785Z-basic-acceptance/root-review.json)、
[完整事件](../../outputs/deploy/rm75b/acceptance/20261110T063120.310785Z-basic-acceptance/events.jsonl)、
[运行摘要](../../outputs/deploy/rm75b/acceptance/20261110T063120.310785Z-basic-acceptance/summary.json)已保存。
实际运行的全部部署模块已另存到同目录的`source_snapshot`，其SHA256与运行清单逐项一致。

## 50mm扩展验收准备

新增受限参数`--axis-distance-mm {5,50}`，默认仍为5mm。50mm由十段5mm轨迹完整预检后拼接为81个路点；
不改变原工作空间、相邻步长、关节限位和反馈保护。去返程各20.8秒，返程反转真实时间轴，保持非均匀分段的速度限制。
主代理独立复跑13项测试、Ruff、Mypy，并检查每轴1041个插值FK点及反向时间、关节速度，全部通过。

基于5mm回合实测基线，+X/+Y/+Z的离线末点残差分别为0.019、0.086、0.030mm；这些是离线结果。
每轴运动仍须取得该轴单独的现场确认，不能把一次确认用于后续轴。
[完整路径与边界](../../outputs/deploy/rm75b/acceptance/basic-plan-50mm-20261110_144114/offline-summary.json)、
[主代理独立预检](../../outputs/deploy/rm75b/acceptance/basic-plan-50mm-20261110_144114/root-review.json)、
[最终测试JUnit](../../outputs/deploy/rm75b/acceptance/basic-50mm-tests.xml)均已保存。

### X轴50mm：反馈与现场观察通过

用户单独确认“确认开始X轴50mm往返”后，执行5%档的X轴去返。实际反馈位移
`[49.828, 0.187, -0.083] mm`，相对目标位置误差0.267mm；返回基线残差0.323mm、姿态残差0.0585°。
全部81个路点按序下发，连续发送最大间隔20.644ms，无故障，退出0且资源关闭正常。
主代理独立核对源码快照、轨迹反向时间轴、路点覆盖及目标限位/工作空间，均通过。
用户现场明确回复：“方向和幅度符合预期，返回正常”，因此X轴50mm外部观察通过；没有单独尺测数据。

[X轴独立验收](../../outputs/deploy/rm75b/acceptance/20261110T064438.191451Z-basic-acceptance/root-review.json)、
[完整日志](../../outputs/deploy/rm75b/acceptance/20261110T064438.191451Z-basic-acceptance/events.jsonl)、
[摘要](../../outputs/deploy/rm75b/acceptance/20261110T064438.191451Z-basic-acceptance/summary.json)已保存。

### 20%档与Y轴首次尝试：预检超时，未发生轴向运动

用户要求运行速度设置为20%。新增`--speed-percent {5,20}`，20%档同时调整关节速度上限、回位速度
及验收时间轴；控制仍为50Hz，50mm单程规划6.4秒。默认仍为5%，本次20%通过显式参数启用并记录在manifest。
主代理独立复跑16项mock、Ruff、Mypy通过；所有原保护阈值保持不变。

用户单独确认20%档Y轴50mm往返后，控制器20%回位成功，Y轴在轨迹预检阶段触发
“基础验收控制周期超时”。故障与上一反馈周期相距158.672ms，与进入planning相距143.072ms。
主代理复核日志确认：零trajectory、零joint_command和零夹爪目标；Y轴尚未开始，Z轴未执行。
停臂调用没有报错，退出码1，关闭errors为空；没有自动重试。

纯离线对照复现了同进程线程预检可使控制循环停顿约140ms，独立spawn进程对照则保持约20ms。
现已将基础验收预检移至独立常驻CPU进程，在连接硬件前握手及预热；控制线程只提交小请求和检查已完成结果，
完整轨迹由专用读线程接收。子进程数值线程均为1，不加载Torch；每周期检查进程健康，停止后旧结果失效，
退出采用有界回收。100ms控制/发送保护、反馈时限和原有几何阈值均保持不变。

主代理独立复跑部署测试98项通过、2项GPU开关测试跳过；包含22项基础验收测试，覆盖启动超时、
进程崩溃、卡死回收、执行中进程失效、过期结果及主线程故意超期时拒发。Ruff和17个部署源文件的Mypy通过。
主代理再运行12次生产规划进程：首次提交0.210ms、后续最大0.180ms，50Hz循环最大间隔20.076ms，
启动232.6ms（硬件连接前）、关闭46.8ms；全部轨迹与直接离线结果一致。
这证明本次离线调度修复，不构成硬实时保证；实机错误仍须由保护捕获。

修复复测通过后再次询问用户，收到“仍在现场，确认重新开始Y轴50mm往返”才允许重新运动，Z轴仍需单独确认。
[原始隔离对照](../../outputs/deploy/rm75b/acceptance/offline-planner-isolation-20261110/results.txt)、
[主代理生产进程复测](../../outputs/deploy/rm75b/acceptance/spawn-planner-20pct-20261110_150201/root-results.json)、
[部署回归JUnit](../../outputs/deploy/rm75b/acceptance/spawn-planner-regression.xml)保留在outputs中。

[Y轴失败尝试的独立审核](../../outputs/deploy/rm75b/acceptance/20261110T065242.237443Z-basic-acceptance/root-review.json)、
[故障摘要](../../outputs/deploy/rm75b/acceptance/20261110T065242.237443Z-basic-acceptance/summary.json)、
[20%离线预检](../../outputs/deploy/rm75b/acceptance/basic-plan-50mm-20pct-20261110_144953/offline-summary.json)、
[16项测试JUnit](../../outputs/deploy/rm75b/acceptance/basic-20pct-tests.xml)已保存；失败会话源码也已按清单SHA归档。

### 修复后Y/Z轴50mm：20%档反馈与现场观察通过

修复后，用户重新确认Y轴，并在Y轴返回后单独确认Z轴，实施子代理按顺序执行。
两轴使用同一回位后冻结的基线、20%速度档、50Hz控制和独立spawn规划进程，原0.5mm末点位置、
0.2°末点姿态及100ms控制/发送保护均未放宽。

| 方向 | 反馈主轴位移 | 去程目标位置误差 | 返回基线位置残差 | 返回姿态残差 |
| --- | ---: | ---: | ---: | ---: |
| +Y，20%档 | 49.845mm | 0.248mm | 0.232mm | 0.0447° |
| +Z，20%档 | 49.783mm | 0.237mm | 0.193mm | 0.0524° |

Y/Z规划耗时分别约155ms和143ms，期间主循环最大间隔20.081ms。
单程规划6.4秒，实测从首条关节指令至末点稳定约7.95～8.17秒，包括全部路点播放与稳定等待。
主代理检查7053条实际下发记录、四条81路点轨迹、完整反向时间轴、关节限位/限速、
目标FK工作空间和源码快照SHA，均通过。整个会话（含等待现场观察）最大连续发送间隔22.356ms，
无fault、退出0、关闭errors为空，CLI与规划子进程均已退出。

动态反馈仍存在控制器跟随滞后：相对最近已发指令的最大关节差，Y为2.271°、Z为1.111°，均发生于返程；
实际TCP相对轴线的最大横向偏离分别0.578mm和0.492mm，最大姿态偏离分别0.148°和0.112°。
这些是运动中的指标，与上表稳定后的末点容差分别记录；没有用末点通过结论代替整条动态路径精度评估。

用户现场回复Y轴“观察正常”，Z轴“方向和幅度符合预期，返回正常”。没有单独尺测数据，
表中的精确位移/误差仍来自真实关节反馈及FK，现场观察用于核对物理方向与约50mm量级。
最终夹爪开度0.998536，保持张开；独立厂商服务PID1993068继续运行。未执行模型闭环或额外的开合动作。

[Y/Z主代理独立验收](../../outputs/deploy/rm75b/acceptance/20261110T070721.325188Z-basic-acceptance/root-review.json)、
[完整事件](../../outputs/deploy/rm75b/acceptance/20261110T070721.325188Z-basic-acceptance/events.jsonl)、
[摘要](../../outputs/deploy/rm75b/acceptance/20261110T070721.325188Z-basic-acceptance/summary.json)、
[进程退出复核](../../outputs/deploy/rm75b/acceptance/20261110T070721.325188Z-basic-acceptance/implementation-exit-review.json)已保存。

### 20%速度上限，模型单段实机执行通过

用户现场授权模型控制测试。使用原EMA检查点、DDPM100、FP32、20%关节速度上限和
`--max-chunks 1`，只发送一次 `start`，从已满足起点容差的现位置开始，没有回位指令。
模型运行也复用独立CPU规划进程，在硬件连接前预热；控制线程按实际发送时刻检查机械臂
和夹爪的100ms间隔，并避免同一周期重复发送机械臂保持目标。修复后主代理独立复跑
99项测试通过、2项GPU开关测试跳过，Ruff和Mypy通过。

先以相同配置完成真实传感器与CUDA的单段dry-run。规划提交0.697ms，预检阶段最大反馈
间隔20.104ms，8步动作及起点依序覆盖，40条模拟指令均为 `applied=false`，退出无错误。

本次实机推理耗时4.383秒，返回8×7动作。规划进程提交耗时0.559ms，预检通过后执行
0.32秒计划，8个动作目标与起点依序下发。包含插值和末点保持共85条实际机械臂指令，
最大发送间隔24.143ms，从首条指令到稳定用时1.701秒。末点稳定时，按关节反馈及FK
计算的TCP相对起点位移为 `[+8.328, -9.854, +5.292]` mm，位置目标误差0.407mm，
姿态目标误差0.136°。模型该段夹爪目标开度为0.99647～0.99879，退出反馈开度0.99854。

到达单段上限后按 `chunk_limit` 停止并回到待命状态，随后执行 `quit`。无fault，
退出码0，关闭errors为空。控制CLI及规划子进程退出，厂商夹爪服务PID1993068保留。
用户现场确认运动平稳，机械臂和夹爪无异常。该段验证模型驱动的实机运动，未尝试完整抓取任务。

[本轮独立配置](../../outputs/deploy/rm75b/acceptance/model-single-20pct-20261110_152317/deployment.yaml)、
[同配置dry-run复核](../../outputs/deploy/rm75b/acceptance/model-single-20pct-20261110_152317/dry-run-spawn-review.json)、
[实机复核](../../outputs/deploy/rm75b/acceptance/model-single-20pct-20261110_152317/run-review.json)、
[主代理独立验收](../../outputs/deploy/rm75b/20261110T073108.714181Z-run/root-review.json)、
[实机终端日志](../../outputs/deploy/rm75b/acceptance/model-single-20pct-20261110_152317/run-terminal.log)、
[完整事件](../../outputs/deploy/rm75b/20261110T073108.714181Z-run/events.jsonl)和
[运行源码快照](../../outputs/deploy/rm75b/20261110T073108.714181Z-run/source_snapshot)已保存。
16个模块的快照SHA与实机运行清单一致。

## 待完成的实机验收

- [x] 夹爪真实反馈、只读重连和完整单段dry-run，由主代理独立验收。
- [x] 现场监督下回位、夹爪闭合/张开；用户确认夹爪开合正常。
- [ ] 夹爪连续双向标定；本轮只验证既有映射的端点开合。
- [x] 三个TCP轴各5mm反馈检查，以及逐轴确认后的50mm往返与外部方向/幅度观察。
- [ ] 小角度旋转的物理核验，本轮未请求或执行。
- [x] 模型单段执行，核对全部路点、真实发送间隔、末点反馈和单段限额停止。
- [ ] 模型多段闭环，核对相邻动作段观测时序及实际闭环周期。
- [ ] 现场停止响应、反馈中断、推理超时，退出后确认最终夹爪状态。
- [ ] 至少一次完整抓取—搬运—释放，并在固定场景完成至少10次尝试。

已有URDF、初始位与TCP外参的三个基座轴5mm离线轨迹全部通过；末点FK位置误差分别约0.121、
0.287、0.162mm。该计算结果不能替代物理TCP标定验证。

任务效果尚无实测尝试，成功率未统计。后续每次记录回合日志路径、物体与场景、是否完成抓取/搬运/释放、
失败阶段及初步归因；保留失败证据，不自动加入旧夹空重试或闭合锁定。

## 复跑命令与证据

从仓库根目录执行；GPU测试需正常访问Orin设备。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync pytest \
  tests/deployment/rm75b tests/scripts/test_eval_dp_handumi_tcp.py \
  tests/scripts/test_train_dp_handumi_tcp.py -q
LEROBOT_RM75B_RUN_WORKER=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync pytest \
  tests/deployment/rm75b/test_policy.py::test_real_checkpoint_worker_lifecycle -q -s
uv run --no-sync lerobot-deploy-rm75b --mode benchmark --verify-native
uv run --no-sync ruff check src/lerobot/deployment tests/deployment src/lerobot/utils/import_utils.py
uv run --no-sync pre-commit run mypy --all-files
```

[软件测试日志](../../outputs/deploy/rm75b/acceptance/pytest.log)、
[软件JUnit](../../outputs/deploy/rm75b/acceptance/software_junit.xml)、
[真实GPU子进程JUnit](../../outputs/deploy/rm75b/acceptance/worker_junit.xml)、
[运行配置及本机预检](../../outputs/deploy/rm75b/acceptance/inspect.json)均保存在本机outputs目录。
该目录沿用仓库忽略规则，交付运行证据时应单独归档。
