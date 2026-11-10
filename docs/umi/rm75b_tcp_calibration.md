# RM75B 双 Tag 到抓取中心的手测与复核

本流程在机械臂静止时用量具测量抓取中心，并用腕部相机复核。建议准备卡尺或钢直尺、角尺、深度尺、等高测量块，以及一个已知尺寸、带止挡和中心标记的抓取夹具。

## 建立测量基准

1. 确认标签 ID 为 0 和 1。沿黑框外缘测量两个标签的边长，确认尺寸一致。以各自黑框对角线交点定位中心，再量两中心距离。中心距用于独立核对视觉尺度。
2. 取两个中心的中点为原点，+X 从 ID 0 指向 ID 1。+Z 朝标签正面，取两枚标签的平均法向在垂直于 X 的平面上的投影方向。+Y = Z × X，按安装约定朝指尖。先用直尺或平面桥检查两标签是否共面。若明显不共面，记录两平面的差异，避免用单侧标签平面代替共同基准。
3. 记录与任务对应的夹爪开度及接触深度。使用带止挡的圆柱或块规，使两指的预定接触面和抓取深度可重复。抓取中心取两指在该深度的接触中心中点。夹具可用十字标记指示中心，若标记位于夹具表面，须实测其相对三维中心的偏置。按现场流程隔离执行器，确认夹爪不会意外动作且工装已支撑，再靠近测量。相机可独立供电采图。

FastUMI 的 T_d_tag_tcp.yaml 中 [0, +120, -18] mm 及同向旋转均为初始几何估计。旧 ROS 命令 keyboard_tag_tcp_capture 读取该矩阵并可自动随机走位。本轮不要运行该工具。

## 独立重复测量三个分量

| 分量 | 实体测量方法 |
| --- | --- |
| X | 将抓取中心投影到双 Tag 基准平面。用角尺把投影点引到两中心连线，量交点沿 ID 0 到 ID 1 方向相对中点的有符号距离。靠 ID 1 为正。 |
| Y | 量该投影点到两中心连线的垂直距离。朝指尖为正。 |
| Z | 用平面桥、等高块和深度尺量基准面至夹具可接触表面，再按夹具半径或厚度换算到中心，并扣除平面桥及量块高度。中心在标签正面侧为正，在背面侧为负。 |

重新摆放量具并重新就位夹具，至少独立测三次。记录量具精度、夹具尺寸、止挡位置、中心补偿、开度来源和任何偏斜。开度以 [0, 1] 表示闭合到张开。多个开度应分别建表，同一表保持开度和接触深度一致。标签若装在活动指组件上，双 Tag 基准也会随开度变化。

单个抓取中心只能确定平移。求解保持训练约定的同向轴与单位旋转。要独立评估实际夹持方向，可在夹具上设置三个非共线参考点或两个已知正交方向，另行测量物理轴偏差。

## 生成模板与候选

在 LeRobot 仓库根目录运行，每条命令使用新的输出目录。

```bash
evidence=outputs/deploy/rm75b/acceptance/tcp-survey-$(date +%Y%m%d-%H%M%S)
env -u PYTHONPATH uv run --no-sync python -m lerobot.deployment.rm75b.calibration_cli \
  template --output-dir "$evidence/template"
```

编辑 measurements.yaml。填写实测黑框边长 marker_size_mm、开度 gripper_openness 和三次以上独立测得的 x_mm、y_mm、z_mm。有实测中心距时填写 measured_tag_distance_mm，量具和夹具信息写在 measurement_notes。

```bash
env -u PYTHONPATH uv run --no-sync python -m lerobot.deployment.rm75b.calibration_cli \
  solve --measurements "$evidence/template/measurements.yaml" --output-dir "$evidence/solve"
```

candidate.yaml 中的 T_d_tag_tcp 将 TCP 系点变换到双 Tag 系，平移由重复测值求均值并转为米，旋转为单位矩阵。报告保存每轴样本标准差、范围、样本数和手测文件 SHA-256。标准差反映重复性，实际精度还需结合量具、夹具装配和独立复核。候选保留未验收状态。

## 用腕部原图复核

使用匹配当前相机和采集分辨率的鱼眼内参。夹具就位、开度与手测一致、设备静止且腕部相机空闲时，采一张原图。已有合适的原始帧时可直接用于复核。

```bash
env -u PYTHONPATH uv run --no-sync python -m lerobot.deployment.rm75b.calibration_cli \
  capture --config docs/umi/rm75b_deployment.yaml --output-dir "$evidence/capture"
```

先设置 shell 变量 marker_mm 为实测黑框外沿毫米数。可选参数 --measured-tag-distance-mm 接收实测中心距。若原图中可独立辨认出抓取中心的实体标记，可添加 --tcp-pixel U V，坐标从原图左上角起算。夹具表面的偏置标记只有在其偏置经过独立处理后才可用作抓取中心的像素观测。

```bash
env -u PYTHONPATH uv run --no-sync python -m lerobot.deployment.rm75b.calibration_cli \
  image --image "$evidence/capture/raw_rgb.png" \
  --camera-calibration /home/scl/work/FastUMI_Data/ros2_ws/src/fastumi_usb_camera/config/EndEffectorCalib.yaml \
  --marker-size-mm "${marker_mm:?请设置实测黑框外沿毫米数}" \
  --candidate "$evidence/solve/candidate.yaml" --output-dir "$evidence/image"
```

查看 overlay.png 和 report.json，核对 ID、轴方向、中心距、角点重投影误差，以及提供了独立 TCP 像素时的像素残差。双 Tag 重投影误差只反映标签角点拟合，叠加的候选十字是预测位置。一个像素无法独立确认 TCP 深度和三维方向。

如需三维复核，在夹具上固定第三个已知尺寸的标靶，独立测定它到抓取中心的变换。用同一张原图分别求相机到双 Tag 和相机到夹具靶的变换，再比较两条变换链算出的相机系 TCP。对重新放置夹具后拍摄的多张原图重复。两条变换链仍共用相机内参，可另用覆盖抓取区域、尺寸已知的参考靶检验近场鱼眼投影。Link7 到抓取中心的独立复核还需 Link7 实体基准或外部量测。

## 候选与部署标定的关系

候选保存在独立输出目录。生成候选不会替换活动的 T_d_tag_tcp、T_d_cam_tcp 或检查点元数据。训练数据和部署检查点沿用原有外参及摘要，应用新标定前应核对训练和部署坐标是否兼容，必要时重新导出训练数据并训练，或明确验证补偿变换。保持原有摘要校验。
