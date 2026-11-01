# 未来预测、缺失模态重建与历史缓存接口（v1）

本约定用于后续 RGB-T 模态缺失跟踪。当前只实现特征接口、冻结参考编码器、
历史缓存及监督目标构建，没有启动训练，也没有实现预测器、重建器或决策策略。

## 1. 固定三种特征的含义

`backbone.encode_feature_bundle(crops, ...)` 返回 `TrackingFeatureBundle`：

| 字段 | 定义 | 用途 |
| --- | --- | --- |
| `independent` | 前 6 层后，尚未和其他图交互的归一化 patch 特征 | 中间特征分析或后续辅助分支；不是本版参考监督空间 |
| `isolated` | 每张图分别走完整 12 层，最终 LayerNorm 后的 patch 特征 | 本版真实观测、历史缓存、未来外观和重建监督的共同空间 |
| `joint` | 前 6 层独立、后 6 层联合交互后的 patch 特征 | 当前定位头和重建残差融合的目标空间 |

所有字段均排除 CLS/register token，网格顺序为行优先；`grids[name]` 使用 `(H,W)`。
`isolated` 和 `joint` 虽然都是 768 维，注意力上下文不同，不能直接视为相同语义。
重建器输出参考空间中的缺失模态候选，再通过可训练的 `Phi` 转到定位空间：

```text
joint_available + fuse_gate * Phi(joint_available, reconstructed_isolated_missing)
    -> tracker.predict_from_features(...) -> boxes
```

`Phi` 和融合门控属于后续模块，本次没有增加这些可训练参数。

默认 RGB `forward` / `forward_features` 仍只走原来的联合路径；新出口不改变原
参数名、架构标记、检查点或输出。`extract_feature_bundle(..., include_isolated=True)`
才会启用额外的单图后半段计算，原 GOT-10k 预训练不会自动承担这部分成本。

## 2. “冻结参考编码器”是什么

首版采用同款 DINOv3 + LoRA 的**固定快照**，没有另外引入更大的教师模型：

1. 完成 RGB 预训练，以及后续 RGB-T 可用性训练，使 TIR 适配器也获得真实训练。
2. 建立 `FrozenReferenceEncoder.from_tracker(rgbt_tracker)`。
3. 首版未来/重建模块训练期间，固定这个表示空间；建议先冻结 backbone，只训练
   新模块、融合映射和需要适配的预测头。本版不采用 EMA，避免缓存/标签随教师变化。
4. 若以后要继续更新 backbone 或改归一化，建立新版本参考空间并清空、重建缓存。

RGB-only 快照现在可以用于验证 RGB 接口；复制出的 TIR LoRA 在真正适配前，
不能据此宣称已经获得有效的热红外教师或重建标签。

参考空间固定为 `dinov3-isolated-final-ln-search-v1`：

- 输入是单个真实模态的**搜索视图**，默认 256×256，对应 16×16 patch。
- 使用 `x` 角色编码；初始缓存也从初始帧的搜索视图提取，不能把 `z/d` 角色特征混入。
- 使用最终 LayerNorm 输出，未额外做 L2 归一化。监督默认采用带有效掩码的余弦损失。
- 没有其他当前模态、模板或未来图像参与这张图的注意力。
- 参数完全冻结，保持 eval 模式，关闭 LoRA dropout 与训练时 RoPE 随机扰动。
- 不使用 `inference_mode` 产生缓存，确保其输出可安全作为后续可训练注意力层的 K/V。
- 特征空间 ID 包含协议、搜索网格、输入归一化标识、权重、LoRA scaling、权重 dtype
  等内容的指纹。换参数/配置后不能继续写入旧缓存。

本版参考视图只做轴对齐 crop/resize、padding 和确定性归一化；不对这些视图做独立
随机翻转、旋转或颜色增强。原 GOT-10k 预训练的增强不受影响。未来阶段若需要几何
增强，必须同步变换所有时刻、框和裁剪元数据，或扩展为完整仿射映射，不能只改图像。

输入已由数据层归一化，不在参考类内部再次归一化。默认标识
`rgb-imagenet_tir-mm-v1` 指 RGB 使用 ImageNet 均值/标准差，TIR 三通道使用
均值 0.449、标准差 0.226（对应本仓库 `mm` 配置）。若 RGB-T 适配使用其他参数，
构造快照时必须指定对应的 `normalization_profile`，并在监督和部署时一致使用。

方法按数据用途区分：

| 方法 | 输出类型 | 可否进入历史缓存 |
| --- | --- | --- |
| `observe` | `FeatureObservation` | 可，但必须通过时间、可靠性与几何检查 |
| `supervise` | `FeatureSupervision` | 不可；只用于训练标签 |
| `encode_context` + `observation_from_context` | 当前真实搜索图的 `FeatureObservation` | 可；复用同一次编码的单图出口 |

用于生成监督时，`supervise` 可读取学生被遮掉的真实当前模态或训练集真实未来帧。
推理时只使用 `observe` / `encode_context` 读取可用真实观测，不读取缺失模态或未来。
监督快照定义的表示也可作为部署时的冻结表示；不能训练时使用该空间，部署时却
把联合特征当作同一空间。固定 backbone 后可复用一份编码器；`encode_context` 共享
浅层计算，同时返回单图、联合出口，但额外的后半段计算不是免费的。

## 3. 历史缓存确定为：真实目标 ROI + 时间/几何/可靠性

每个轨迹、每个模态独立缓存，默认设置如下；这些是起始协议，并非调优结果：

- 每条观测的目标 ROI 采样成 4×4，即 `[16,768]`。
- 每个模态最多 8 条，初始给定目标的真实锚点保留，优先淘汰最旧的近期项。
- 普通写入要求跟踪置信度大于 0.84；初始化给定框可跳过置信度门限。
- 普通项读取时最多保留 64 帧年龄；初始锚点可更老，但显式返回 `age`。
- 写入可靠性门限是下限。后续模板决策还应先决定是否调用 `record`，本类不会代替
  未来收益判断，也不会因为置信度高就自动更新图像模板。
- 只保存 CPU 上脱离计算图的独立副本，同时保留原图归一化框、帧号、模态、置信度、
  初始标记和有效 token 掩码。接口读取结果也不与内部存储共享可变张量。

严格执行：

```text
开始轨迹：start_track(track_id, initialization_frame_index=0)
帧 t 开始：read(track_id, now_frame_index=t) 只返回 frame_index < t
获取 A_t / joint -> 直通定位临时框 -> 预测未来 -> 决定预算 -> 重建/融合 -> 最终定位 -> 模板决策
帧 t 结束：获准后 record(真实可用观测, 当前预测框, now_frame_index=t)
结束轨迹：reset(track_id)
```

`record` 拒绝监督标签、普通生成张量、未来帧写入、特征空间混用、重复/倒序写入。
初始化 GT 仅可用于 `start_track` 声明的初始帧，后续框来源必须是预测。
缺失模态不编码、不写入，原有历史保留；恢复时按正常规则追加。查询返回明确的
`modalities / frame_indices / ages / confidences / initial / valid_mask`，预测器自行
编码模态和时间差。空历史是合法输入，不能补造缺失模态历史。

历史特征缓存与原始图像模板库是不同对象；重建特征不进入任何真实观测库。

## 4. 当前重建标签与未来标签分别定义

所有框为**原图归一化 XYXY**。源裁剪范围可以超出 `[0,1]`，表示 padding。
特征网格使用 `(H,W)`；图像尺寸和裁剪工具的参数使用 `(W,H)`。

### 当前重建

`reconstruction_target(supervision, anchor_frame_indices=t, anchor_crop_boxes=C_t)`：

- 使用当前真实缺失模态的单图最终特征，默认 `[B,256,768]`。
- 学生看到的只是可用模态；这个当前缺失模态仅进入独立参考监督调用。
- 参考帧号必须恰好是 `t`，裁剪范围必须和当前搜索范围 `C_t` 一致。
- 真实图像之外的 padding token 不参加损失。
- 此目标不来自双模态联合注意力输出，也不是未来目标。

### 未来预测

默认规划预测 `t+1、t+2`，`future_target` 每次构建一个明确 horizon 的标签：

1. 先用直通定位头对 `joint_available` 产生临时框 `b_base_t`；未来预测以这个框和当前
   搜索范围 `C_t` 为锚点。初始化时可以使用给定初始框。不要使用依赖重建结果的最终框。
2. 在未来真实帧上沿用同一个原图归一化范围 `C_t`，不根据未来 GT 重定位搜索图。
3. 参考编码器分别编码未来真实模态；未来 GT 只用于抽取监督 ROI 和计算运动标签。
4. 外观目标为未来 GT 框内部固定 4×4 的目标相对 ROI 特征。
5. 运动目标为：

   `( (cx_future-cx_base)/w_base, (cy_future-cy_base)/h_base,
       log(w_future/w_base), log(h_future/h_base) )`。

这使“目标移动到别处”不被错误地当成特征重建误差，也保留了真实运动信息。
`future_target` 检查帧号精确等于 `t+horizon`，以及未来裁剪确实沿用当前范围。
`anchor_box_source` 明确标为 `baseline_prediction`（默认）或 `initial_ground_truth`；
最终融合框、真实未来框均不能作为预测时的坐标锚点。最终融合框仍可用于当前输出和
帧结束时的真实历史更新。ROI 与未来运动标签的生成均不参加反向传播。

ROI 使用中心采样、双线性插值和 `align_corners=False`；采样点在原图或裁剪范围之外
会被掩码排除。实现语义可对照
[PyTorch grid_sample 文档](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.grid_sample.html)。
默认要求至少一半 ROI 采样点有效。目标移出搜索范围时，外观标签失效，但已知且可见的
目标框仍可提供运动监督；不可见/无有效框时运动与外观均屏蔽。未来该模态没有真实图像
时，不调用编码器制造伪标签；数据层应跳过该模态/时刻的外观监督。

`masked_cosine_loss` 对全无效批次返回可反传的零损失，且屏蔽无效位置的 NaN。

## 5. 与现有裁剪工具对接

`crop_boxes_from_siamfc` 把已有跟踪器的裁剪参数变为上述原图归一化范围。
必须使用 `apply_siamfc_cropping` **返回的 adjusted 参数**，而不是整数裁剪调整前的参数：

```python
crop_boxes = crop_boxes_from_siamfc(adjusted_params, image_sizes_wh, (256, 256))
```

`siamfc_from_crop_boxes(C_t, future_image_sizes_wh, (256,256))` 可在未来帧重建相同范围。
未来监督裁剪建议使用现有 `apply_siamfc_cropping_subpixel`，保持同一仿射范围，避免
逐帧整数裁剪产生边界漂移。当前预测框也必须转成原图归一化 XYXY 后才能做 ROI 对齐。

## 6. 最小接入示例

```python
from trackit.models.methods.dinov3_rgb import (
    FrozenReferenceEncoder, CausalFeatureHistory, reconstruction_target, future_target)

reference = FrozenReferenceEncoder.from_tracker(rgbt_tracker)
history = CausalFeatureHistory(reference.space_id, reference.encoder.embed_dim)
history.start_track(track_id, initialization_frame_index=0)

# 当前帧：所有 images 都是真实可用、确定性预处理后的图像；缺失搜索图不传入。
context = reference.encode_context(real_crops)
past = history.read(track_id, now_frame_index=t, device='cuda')
current_rgb = reference.observation_from_context(
    context, 'x_rgb', [t], current_crop_boxes)

# 后续预测器读取 current_rgb + past，不能读取下面训练标签。
# 在预测未来之前，从 context.features.joint 得到 baseline_box_normalized 作为固定锚点。
# ... future predictor / reconstruction / fusion / head / template decision ...

# 得到当前可靠预测后，决定写入；若被模板策略拒绝，则不调用 record。
history.record(track_id, current_rgb, predicted_box_normalized,
               now_frame_index=t, confidence=score)

# 单独的训练监督调用，不与 real_crops 拼接，也不写进 history。
missing_label_features = reference.supervise(
    true_current_tir, 'tir', [t], current_crop_boxes)
rec_label = reconstruction_target(missing_label_features, [t], current_crop_boxes)
future_label_features = reference.supervise(
    true_future_tir_in_current_crop, 'tir', [t + 1], current_crop_boxes)
fut_label = future_target(
    future_label_features, anchor_frame_indices=[t], anchor_crop_boxes=current_crop_boxes,
    anchor_boxes=baseline_box_normalized, future_boxes=future_gt_box_normalized,
    horizon=1, visible=[True])
```

本示例展示数据边界，不代表预测/重建模块已经实现。训练未来模块必须使用按时间排序的
序列与真实过去缓存，不能直接复用 GOT-10k 阶段的无序 interval 三图采样。真实未来
可以评价动作收益或产生标签，但不能进入当前的 `context`、`history` 或决策输入。

## 7. 验证与兼容性

测试覆盖单图特征不受其他模态污染、冻结/确定性、快照与输入协议指纹、原预测/检查点
兼容、裁剪坐标往返、目标运动与 ROI 外观分离、越界/不可见掩码、当前/未来监督区分、
缓存容量/老化/初始锚点/轨迹隔离，以及对监督标签和未来写入的拒绝。

26 项完整回归测试通过（其中 11 项为本次时序特征接口测试）。还使用官方 DINOv3
权重和 GOT-10k 同一序列的真实第 0、1 帧，在 RTX 5090 上检查了 adjusted 裁剪坐标、
固定未来搜索范围、真实历史写入与标签生成：观测/重建目标为 `[1,256,768]`，历史/未来
ROI 为 `[1,16,768]`。该真实帧检查仅前向，没有反向传播或优化器更新。
这些检查不代表已经得到可用的 RGB-T 适配权重或未来模块训练结果。

```sh
python -m unittest discover -s tests -p test_temporal_feature_contract.py -v
```

只验证接口与数据约定；不说明未来预测精度、重建收益或门控策略已经得到验证。
