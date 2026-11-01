# DINOv3 RGB-T 模态缺失微调 v1

本版本把 RGB 预训练权重接到 RGB-T 适配、未来预测、缺失模态特征重建和逐帧测试。
默认命令仅做预检；只有显式 `--execute` 才会训练。当前没有提交训练。

实现位置是独立分支 `codex/dinov3-rgbt-temporal-v1`，继承完整续训修复 `27833e2`。
原 RGB 训练分支、最佳权重和测试提交包均保留。使用本目录里的独立训练/评估入口，
而不是原先只接受 RGB 输入的 `main.py DINOv3RGB ...`。

## 初始化与数据

- 来源：RGB GOT-10k 第 20 轮最佳模型，训练提交 `c139965`；用户提供的官方测试结果
  为 AO 0.711 / SR50 0.840 / SR75 0.656。
- 完整初始化文件：`dinov3_rgbt_init_e20.safetensors`，SHA-256 为
  `819b43d675d93017c58e344b74097796db1189f16f3542eca26642d7bed6bc75`。
  RGB 参数、共享深层 LoRA、角色编码与定位头保留；浅层 TIR Q/V LoRA 和 TIR
  模态嵌入由 RGB 复制并独立存储。任务 LoRA 没有合并进 DINOv3。
- DINOv3 ViT-B/16，768 维，12 层，前 6 层模态独立、后 6 层共享，Q/V LoRA
  rank32 / alpha64 / dropout0.05；模板128、搜索256。
- 按用户指定，LasHeR **979 训练 / 245 测试**。只读官方
  `trainingsetList.txt` / `testingsetList.txt`；不使用 `lasher_train.txt` / `lasher_val.txt`。
- 两个 split 的名字必须互斥。预检核对全部序列的 RGB/TIR/标注数量，并检查每段
  首尾图像的配对尺寸。每次采样还检查窗口内尺寸，尺寸不一致直接报错。
- RGB 使用 ImageNet mean/std；TIR 三通道使用 mean=.449、std=.226，训练、参考编码
  和测试完全一致。所有预测/对齐元数据使用原图归一化 XYXY，结果文件保存原像素 XYWH。
- 测试集不参与轮次、超参数和权重选择；阶段切换使用固定末轮权重。

## 两阶段训练

配置文件：`config/DINOv3Missing/temporal_v1.yaml`。以下预算是提交前的初版配置。
epoch 指随机采样预算，不是完整遍历每个视频一次。

| 阶段 | 更新内容 | 轮次 | clip/轮 | batch | 学习率 | 可训练参数 |
|---|---|---:|---:|---:|---:|---:|
| adapt | RGB/TIR 浅层 LoRA、共享深层 LoRA、角色/模态嵌入、定位头 | 10 | 32768 | 16 | 1e-4 | 4,141,061 |
| temporal | P、共享重建器、Phi、融合门控及定位头；整个骨干和参考快照冻结 | 10 | 8192 | 8 | 新模块3e-4，定位头3e-5 | 7,194,381 |

均使用 AdamW、cosine、1轮 warmup、WD .05、梯度裁剪1.0、BF16、8个数据进程、
4个 PyTorch CPU 线程，关闭 compile。1D 参数和角色/模态嵌入不做 weight decay。
默认总预算为 adapt 20,480 次、temporal 10,240 次优化器更新。

阶段 B 只能从实际执行过优化器更新的 adapt 检查点开始。RGB-only 模型复制出的 TIR
分支不能直接作为训练监督空间。阶段交接先严格加载适配权重，再建立固定参考快照。
不使用 EMA；参考空间指纹同时覆盖参数、LoRA 标度、网格、归一化与计算精度。
参考编码器显式管理 autocast，不随调用方上下文变化。

## 因果数据和监督

本实现沿用 `docs/temporal_feature_contract.md` 的空间、坐标和缓存契约。

阶段 A 从一个视频中取初始化帧和之后最多100帧间隔的当前帧。阶段 B 取连续窗口：

```text
t-3（本 clip 初始化） -> t-2 -> t-1 -> t | t+1 -> t+2
          因果输入与真实历史               | 仅训练监督
```

每个 clip 允许使用其初始化帧的 GT。搜索区域 C 由该初始框确定，clip 内所有观测和
未来监督沿用同一 C；当前/未来 GT 不参与搜索裁剪或预测器输入。在线测试时 C 由上帧
预测框确定。短 clip 训练仍存在与长序列在线跟踪的分布差异，不能据此宣称已解决长时漂移。

`inputs` 只有模板、真实可用的过去/当前搜索图、可用性、帧号、裁剪范围和初始框。
`targets` 单独保存当前真实完整图、未来完整图及 GT。隐藏输入槽填 NaN，模型按可用性
分组后才选择对应图像，不能把填充当成模态观测。训练时三种非空状态均参与采样，并以
0.8概率保持缺失状态；测试支持真实 SCDT JSON，或明确标记为 synthetic 的模拟状态。

历史缓存只存固定参考编码器的 isolated、最终 LayerNorm、search 角色特征。
初始化 ROI 可用初始 GT；后续 ROI 来自预测框，并通过置信度>.84和有效 ROI 检查。
缓存每模态最多8项，保留初始锚点，近期条目最多64帧；按 track/modality/space_id 隔离。
当前帧的真实观测在预测和模板决策之后才能写入。生成特征和监督特征没有缓存写入入口。
训练 clip 的历史/在线模板在批次间不延续，因而检查点不依赖预取中的临时缓存。

无效初始/当前 GT 会重新采样；无效未来 GT 保留并通过监督有效掩码排除，不强行重建或
回归它。出现完整模态、RGB-only、TIR-only 的混合批次时，模型保留样本原顺序。

## 未来预测与重建

未来预测器读取当前真实 isolated 特征、历史目标 ROI 和当前直通定位的临时框。
输入保持 detached；预测器本身可训练。输出每个模态在 t+1/t+2 的：

- 4×4 目标 ROI 外观特征，通道768；
- 相对临时框的中心位移与对数尺度变化，共4维；
- 未来目标 ROI 在固定搜索视图中是否可用于外观监督的概率。

当前重建器维持 `256 tokens × 256 channels` 的内部状态。查询由状态 LN 与迭代步数
编码构成，三路 cross-attention 分别读取当前真实观测、有效历史、预测未来：

```text
Z_j = Z_(j-1) + O_current + gate_history * O_history + gate_future * O_future
Z_j = Z_j + FFN(LN(Z_j))
F_missing = Linear_256_to_768(LN(Z_j))
```

1、3、5步共享同一单元与 Readout，训练时执行5步并监督三个出口。
0步跳过整个重建器。候选在 isolated 参考空间产生，经 Phi cross-attention 转到
joint 定位空间：`joint_available + fuse_gate * Phi(joint_available, reconstructed)`。
Phi 的末层零初始化，以维持接入瞬间的直通定位输出。完整双模态默认均值读出两个
联合搜索特征，无需生成当前缺失候选。

损失：BCE + GIoU + .5×当前重建余弦 + .5×未来外观余弦 + .1×未来运动 SmoothL1
 + .05×未来 ROI 可用性 BCE。无效空间位置不计入余弦，越出搜索区域但仍有有效 GT 的
未来目标保留运动监督。监督编码按模态/horizon 批量执行，参考标签与当前推理分开计算。

在线时，只有真实可用模态能更新相应在线模板与历史。缺失槽保留此前可靠模板。
RGB/TIR 都缺失时保持上一预测，不读缺失图、不编码、不重建、不更新真实模板。

### 本版策略范围

本轮实现 P、重建器和可训练的软融合门控。在线默认固定3步；可配置0/1/3/5步消融。
模板更新先沿用置信度阈值。**预算和模板动作尚未加入基于 rollout 收益的策略训练**，
不把固定策略宣称为学到的三个决策头。该范围对应先完成两类特征模块的微调接入。

## 执行方式

在 `/home/projects/GOLAv2-rgbt-temporal` 中运行。以下第一条仅预检，无优化器更新：

```bash
python finetune_dinov3_rgbt.py --output output/dinov3_rgbt_temporal/v1
```

下面是待用户确认后的训练入口，当前没有执行：

```bash
python finetune_dinov3_rgbt.py --execute --stage all \
  --output output/dinov3_rgbt_temporal/v1
```

也可单独执行 `--stage adapt`，完成后用 `--stage temporal --adaptation-checkpoint
.../adapt/last.pt` 接第二阶段。训练前强制要求代码已提交，输出目录中已有权重时拒绝
覆盖。环境路径只读配置文件，不从 shell 中获取或输出 W&B API key。

恢复同一阶段时指定同一输出根目录及完整检查点：

```bash
python finetune_dinov3_rgbt.py --execute --stage all \
  --output output/dinov3_rgbt_temporal/v1 \
  --resume output/dinov3_rgbt_temporal/v1/temporal/last.pt
```

`last.pt` 原子保存完整模型、冻结参考快照、AdamW、LR scheduler、AMP scaler、
Python/NumPy/Torch/CUDA RNG、epoch/下一batch、真实更新数和epoch统计。采样按
seed/epoch/index确定，预取进程不会消耗模型 RNG；轮中恢复从下一样本索引重建加载器。
恢复检查数据、结构、代码版本、目标函数和调度签名。旧 RGB v1 的不完整 `state.pth`
不能用来恢复本流程；它的模型权重可作为初始化。

W&B 默认项目 `DINOv3RGBT`，entity沿用 `mas-38-anhui-university`，两阶段独立run并分组。
同目录续训复用该阶段 W&B ID。`--disable-wandb` 可关闭云端日志。
每阶段保存 `metrics.jsonl`（定期批次采样）、`epoch_summary.jsonl`（所有批次均值）、
`progress.json`、`run_info.json`。覆盖各项损失、定位IoU、梯度范数、LR范围、三种可用性
数量、融合门值、监督有效token数、数据等待时间、耗时及显存。异常会写明失败原因。
批次日志保存序列名和帧号，异常记录保留最后一个已接收批次，方便定位数据与模型问题。

```bash
python analyze_dinov3_rgbt_training.py output/dinov3_rgbt_temporal/v1
```

训练后的245序列测试同样需要显式执行：

```bash
python evaluate_dinov3_rgbt.py --execute \
  --checkpoint output/dinov3_rgbt_temporal/v1/temporal/last.pt \
  --output output/dinov3_rgbt_temporal/v1_test
```

默认分别测试 complete、rgb_only、tir_only、synthetic_missing。官方缺失标记文件使用
`--mode schedule --missing-mask-file /path/to/markers.json`。初始帧两模态均真实可用，
沿用现有 SCDT 初始化约定；后续缺失图像不解码。每模态单独维护在线模板。
保存原像素XYWH预测、真实逐帧耗时、可用性/重建步数/门值/历史写入诊断。
评估复用仓库 OPE：序列均值、21点成功率曲线、PR@20px与NPR@0.2，包含初始化帧。
模拟缺失结果不能标注成官方 SCDT 协议结果。

## 已做验证

CPU 回归覆盖：NaN缺失槽隔离、混合批次顺序、参考空间隔离、未来/当前帧历史拒绝、
三路重建梯度、1/3/5共享单元、0级旁路、监督不影响当前推理、双缺失旁路、单模态模板
保留、无效未来标签掩码、显式参考精度、两阶段完整状态恢复、真实训练循环的合成数据
轮中断恢复、阶段交接和测试文件导出。训练循环测试只用合成小模型，不是 LasHeR 训练。

另外使用实际 RGB 最佳权重、完整 ViT-B/16 和真实 LasHeR 图像在5090上做前向预检，
以及 adapt batch16、temporal batch8 的无优化器前后向检查。实际批量下预测器29/29、
重建器42/42、融合模块14/14个参数张量都有有限梯度。
检查中发现并修复了“同一 autocast 上下文中的无梯度历史预热缓存了 detached 权重
cast”的问题；预热期间禁用该缓存，并在正式更新前检查关键投影梯度。
复现检查命令为 `python benchmarks/check_dinov3_rgbt.py`，不创建优化器或更新权重。
结构检查时 TIR 尚未训练，检查结果只证明可执行性。没有生成“已适配参考”检查点，
没有据此报告重建效果或 RGB-T 测试成绩。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
```
