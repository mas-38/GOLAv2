# DINOv3 + LoRA 的 GOT-10k RGB 跟踪预训练配置审查

审查日期：2026-11-01。对象：`DINOv3RGB/dinov3` 当前工作区配置和实际调用代码。本文是调研与实验设计，**没有修改模型或训练配置，没有启动训练，也没有运行真实模型的 optimizer.step**。仓库及其父目录未发现适用的 `AGENTS.md`。工作区已有未提交改动；下文以读取到的工作区文件为准，而不是仅以 Git HEAD 为准。

## 1. 结论与证据强度

**当前配置适合作为可运行的第一版 RGB 任务预训练基线；现有证据还不足以支持直接把一次无验证的 100-epoch、约 35–50 小时训练当作默认下一步。** 优先补齐验证和检查点选择，再用有限预算检查学习率及 LoRA 容量。不能仅凭 epoch 数、短时吞吐量或单次有限梯度，就判断它已能收敛或能提高后续 RGB-T 跟踪性能。

这不是“DINOv3 比 DINOv2 强，因此旧配方直接有效”的已验证替换。相对 LoRAT，当前同时改了骨干、位置编码、两图/三图输入、前后层的交互方式、LoRA 注入范围、rank、缩放、初始化、dropout、batch 和 weight decay。它有清楚的后续迁移动机，但这些改变合在一起构成一个新基线。[当前模型配置](/home/projects/GOLAv2/config/DINOv3RGB/dinov3/config.yaml:18)、[实际骨干](/home/projects/GOLAv2/trackit/models/methods/dinov3_rgb/backbone.py:35)、[LoRAT 官方配置](https://github.com/LitingLin/LoRAT/blob/5260744ff0a65207a73289bbda1788c377d23bcc/config/LoRAT/B-224/config.yaml)。

| 优先级 | 判断 | 证据与边界 |
|---|---|---|
| P1：长训前处理 | 默认没有训练中的验证；只留最近三个 checkpoint，不能保证保留最适合迁移的模型 | 配置事实。已有独立 GOT-val 闭环评估入口，但它替换训练任务，不能直接当作训练时附加 val 开关。见 §5.1。 |
| P1：优先小试 | `QV-only/r8` 的适配容量与覆盖范围是主要建模风险 | LoRAT 有直接消融支持检查这个方向；但实验在 **DINOv2-L、RGB 混合训练、LaSOT 等评估**，不能证明 DINOv3-B/GOT-only 的 r8 必然不足。见 §4.1。 |
| P1：优先小试 | batch16/lr1e-4/100epoch 尚未建立合适的更新预算 | 样本预算与 LoRAT GOT-only 相同，更新次数却为其 8 倍。不能据此直接称为过训练，也不能假定同一个 LR 可无条件迁移。见 §3、§4.2。 |
| P1：启用前修复 | 不能只把梯度累积改成 8 来得到“等价 batch128” | 当前累积路径没有除 loss，且 scheduler 用 optimizer-update 长度却接收 microbatch 索引；已用 CPU、无参数更新复现调度提前结束。**当前 accum=1 不受这两项问题影响。**见 §5.3。 |
| P2：分布偏差 | 在线模板 `d` 是同一视频的另一次独立随机抽帧，不代表真实过去的可靠预测模板 | 代码事实。对 RGB 外观匹配预训练不构成推理时未来泄漏，但与在线部署分布有差异，且不能复用为未来预测训练数据。见 §5.2。 |
| P2：记录并小试 | 冻结 DINOv3 权重后，训练态 RoPE 坐标随机缩放仍生效；当前采样频率也不同于官方 forward | 代码事实，不直接定性为 bug。见 §5.4。 |
| P2：工程修复 | 共享可变容器、未适配的负样本分支是遗留健壮性风险 | 可复现别名/构造异常；没有证据证明当前 6-worker CUDA 正样本训练已发生跨批次污染。见 §5.5。 |

本文使用三种证据：**代码事实**表示可由所引文件直接核对；**论文证据**表示原论文或作者官方实现给出的观察；**建议/推断**表示适用于本项目的待实验判断。所有拟议实验都需要另行告知用户后再提交；本报告没有提交任何训练。

## 2. 当前配置实际做了什么

### 2.1 三个训练阶段必须分开

1. **图像自监督预训练**：加载官方 DINOv3 ViT-B/16 的 `LVD1689M` 权重。该计算已经由上游完成，不属于本项目 GOT-10k 预算。
2. **RGB 跟踪任务训练（当前阶段）**：冻结原始 DINOv3，学习模板—搜索关系、定位头、角色编码和 LoRA。它是面向后续 RGB-T 的任务底座。
3. **RGB-T 与缺失模态适配（后续阶段）**：复制浅层 RGB adapter 初始化 TIR 分支，再学习双模态融合、缺失鲁棒性与后续预测/重建/决策。当前 RGB `forward` 不执行未来预测器或重建器，不能把其未来开销计入本阶段。

这一区分有公开先例：GOLA 的论文 Implementation Details 明确使用 LoRAT 的已训练权重初始化，再开展 RGB-T 适配；其“10 epochs”不是从 DINO 图像权重开始获得完整跟踪能力的总成本。[GOLA 论文](https://arxiv.org/html/2512.05359v1#S4.SS1)、[官方训练配置](https://github.com/MelanTech/GOLA/blob/339c737cda6a24be667b6e5abdc721e8d6046f05/config/GOLA/run.yaml)。当前迁移接口也只复制 adapter，并未把导出结果变成已训练 RGB-T 模型。[迁移函数](/home/projects/GOLAv2/trackit/models/methods/dinov3_rgb/backbone.py:58)、[阶段说明](/home/projects/GOLAv2/docs/dinov3_rgb_pretraining.md:38)。

### 2.2 架构、数据与优化快照

| 项目 | 实际配置/行为 | 本地证据 |
|---|---|---|
| 骨干 | 官方 ViT-B/16，12 层、768 维、12 heads，原权重全部冻结；严格加载官方本地 checkpoint | [loader](/home/projects/GOLAv2/trackit/models/backbone/dinov3/builder.py:40)、[官方 B/16 定义](/home/projects/dinov3/dinov3/hub/backbones.py:277) |
| 输入 | `z=128²,d=128²,x=256²`，对应 `64+64+256=384` patch tokens；每图另保留 CLS+4 register，联合段共 399 tokens | [配置](/home/projects/GOLAv2/config/DINOv3RGB/dinov3/config.yaml:7)、[crop 准备与拼接](/home/projects/GOLAv2/trackit/models/methods/dinov3_rgb/backbone.py:72) |
| 交互 | 前 6 层逐图处理，三图共用 RGB adapter；后 6 层三图联合注意力，共用 shared adapter。这里的浅层“RGB-specific”是模态专属，不是 z/d/x 各有一套 | [构造](/home/projects/GOLAv2/trackit/models/methods/dinov3_rgb/backbone.py:49)、[forward](/home/projects/GOLAv2/trackit/models/methods/dinov3_rgb/backbone.py:112) |
| 位置与类型 | 复用每图自身网格的 RoPE；所有 prefix 放在联合序列前部；5×768 角色/前景背景编码和 RGB 模态编码可训练。没有加入 DINOv2 的绝对位置表 | [backbone](/home/projects/GOLAv2/trackit/models/methods/dinov3_rgb/backbone.py:53) |
| LoRA | 仅 Q/V；每个低秩投影 r8、alpha16、缩放 alpha/r=2、dropout .05；A Kaiming、B=0；K、attention output、FFN 不注入 | [LoRA 实现](/home/projects/GOLAv2/trackit/models/methods/dinov3_rgb/lora.py:8) |
| 可训练参数 | Q/V LoRA **294,912**；总可训练 **2,665,733**，因此其中约 89% 是头和编码，而不是骨干 adapter | [head](/home/projects/GOLAv2/trackit/models/methods/dinov2_missing/head.py:37)、[参数公式与预算](#3-预算对照不能只看-epoch) |
| 数据 | 配置仅 GOT-10k train，无混合数据或 `data_specs` 子集；本机 manifest 有 9,335 条 train、180 条 val。正式加载后的清洗保留量应随日志记录 | [数据源](/home/projects/GOLAv2/config/_dataset/train-got10k.yaml:1)；本次只读 manifest 核对 |
| 采样 | 每 epoch 131,072 三图样本；z/x interval gap100；d 从另一次独立 pair 的首帧取得；默认全部正 pair、全部正 online template | [run 配置](/home/projects/GOLAv2/config/DINOv3RGB/run.yaml:32)、[worker](/home/projects/GOLAv2/trackit/data/methods/siamese_tracker_train/siamese_training_pair_sampling/worker.py:64) |
| 裁剪 | 模板 factor2，无 scale/translation jitter；搜索 factor4，scale jitter .25、translation jitter3、输出最小目标10像素 | [run 配置](/home/projects/GOLAv2/config/DINOv3RGB/run.yaml:42)、[processor](/home/projects/GOLAv2/trackit/data/methods/siamese_tracker_train/transform/default/processor.py:208) |
| 图像增强 | 三图各自 p=.5 翻转；brightness/contrast/saturation .4 独立抖动；随后三图共同随机选择灰度、solarization、Gaussian blur **三者之一且必执行** | [augmentation pipeline](/home/projects/GOLAv2/trackit/data/methods/siamese_tracker_train/transform/default/augmentation/pipeline.py:35)、[DeiT3 实现](/home/projects/GOLAv2/trackit/data/methods/siamese_tracker_train/transform/default/augmentation/deit_3_augmentation.py:65) |
| 归一化 | 三通道 RGB ImageNet mean/std。模板框及搜索框随翻转变换，标签在增强后生成 | [processor](/home/projects/GOLAv2/trackit/data/methods/siamese_tracker_train/transform/default/processor.py:108) |
| 优化 | AdamW，统一 lr1e-4，wd .05；类型编码与一维参数 wd0；batch16，accum1，clip norm1；BF16；compile 关闭；固定 RGB 图适用 static_graph | [run 配置](/home/projects/GOLAv2/config/DINOv3RGB/run.yaml:70)、[optimizer 参数筛选](/home/projects/GOLAv2/trackit/runner/training/common/optimization/optimizer/per_parameter_options/apply.py:24) |
| 调度 | 100 epochs，per-step cosine 至1e-6；warmup2epochs 从1e-7开始；`warmup_prefix=false` | [scheduler 配置](/home/projects/GOLAv2/config/DINOv3RGB/run.yaml:86)、[scheduler builder](/home/projects/GOLAv2/trackit/runner/training/common/optimization/lr_scheduler/timm_scheduler/builder.py:4) |
| 损失 | IoU-aware BCE + GIoU，各权重1。分类/回归是两套分别可训练的三层 MLP；各自在空间位置共享权重，**不是分类与回归共用同一套 MLP 参数** | [head](/home/projects/GOLAv2/trackit/models/methods/dinov2_missing/head.py:37)、[criterion](/home/projects/GOLAv2/trackit/criteria/methods/box_with_score_map/__init__.py:23) |

当前保留原始 masked K bias、LayerScale、FFN、norm 的方式合理；prefix 与 patch RoPE 的排序也与官方 attention 通过 `N-rope_length` 确定 prefix 的接口相容。这是实现一致性结论，不能替代对“三图联合 RoPE 是否最优”的实验。[本地封装](/home/projects/GOLAv2/trackit/models/methods/dinov3_rgb/lora.py:48)、[官方 attention](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/layers/attention.py#L66)。

## 3. 预算对照：不能只看 epoch

本阶段一个 sample 是 `(z,d,x)` 三个裁剪，不是一张图，也不是一个 optimizer step：

```text
当前：100 × 131,072 = 13,107,200 triplets
每轮更新：131,072 / 16 = 8,192
总更新：819,200（accum=1）
总裁剪调用数：39,321,600（抽到同一帧时，底图解码可能复用）
warmup：16,384 updates = 262,144 triplets

LoRAT GOT-only：100 × 131,072 = 13,107,200 pairs
每轮更新：131,072 / 128 = 1,024
总更新：102,400
warmup：2,048 updates = 262,144 pairs
```

所以当前与 LoRAT **样本暴露数相同、更新数 8 倍、每样本图像数不同、token 处理方式不同**。不能把 100 epochs 简化成“对齐论文”，也不能把更新数 8 倍直接翻译成“过训练 8 倍”。[当前 run](/home/projects/GOLAv2/config/DINOv3RGB/run.yaml:2)、[LoRAT run](https://github.com/LitingLin/LoRAT/blob/5260744ff0a65207a73289bbda1788c377d23bcc/config/LoRAT/run.yaml)、[GOT-only mixin](https://github.com/LitingLin/LoRAT/blob/5260744ff0a65207a73289bbda1788c377d23bcc/config/LoRAT/_mixin/got10k.yaml)。

### 3.1 同阶段/同协议的公开参照

下表的步数由官方配置计算，不声称是作者日志实测。`M` 表示百万个训练样本单元。两图、三图、五图和 RGB-T 六裁剪的 sample 不能直接当作等量计算；多 GPU 的本地 batch 与 global batch 也已区分。

| 方法、已核实身份 | 图像预训练 / 任务阶段 | 跟踪训练数据与预算 | global batch / 更新数 | 输入与模板机制 | 一手来源 |
|---|---|---|---|---|---|
| 当前 DINOv3RGB，项目首版 | DINOv3 B/16；RGB task training | GOT train；100×131,072=13.1072M 三图 | 16 / 819,200 | 128/128/256；前6独立、后6联合；未来模块未运行 | [本地配置](/home/projects/GOLAv2/config/DINOv3RGB/run.yaml:2) |
| LoRAT，ECCV 2024 | DINOv2 B/L/g，patch14；RGB task training | GOT-only：100×131,072=13.1072M pairs；通用混合：170×131,072=22.28224M pairs | 128 / 102,400（GOT）；174,080（混合） | B-224：112/224，8²/16² patch；单初始模板、全层联合 | [论文 §4.1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00113.pdf)、[run](https://github.com/LitingLin/LoRAT/blob/5260744ff0a65207a73289bbda1788c377d23bcc/config/LoRAT/run.yaml)、[GOT mixin](https://github.com/LitingLin/LoRAT/blob/5260744ff0a65207a73289bbda1788c377d23bcc/config/LoRAT/_mixin/got10k.yaml) |
| LoRATv2，**NeurIPS 2025 Main Conference** | DINOv2 B/L reg4，patch14；两阶段 RGB task training | phase1、phase2 的 run 均继承170 epochs；分别应用 GOT mixin 则各100，每阶段13.1072M units；完整 GOT 两阶段合计26.2144M units | 128 / 每阶段102,400，合计204,800（按两阶段均应用 GOT mixin 计算） | p1：z224/x224；p2：z224/历史x224/当前x378；frame-wise causal、KV cache、stream-specific LoRA | [正式 proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/ad7e42e7b1f638e991d822724969be45-Abstract-Conference.html)、[p1](https://github.com/LitingLin/LoRAT/blob/5260744ff0a65207a73289bbda1788c377d23bcc/config/LoRATv2/p1-B-224/run.yaml)、[p2继承](https://github.com/LitingLin/LoRAT/blob/5260744ff0a65207a73289bbda1788c377d23bcc/config/LoRATv2/p2-B-378/config.yaml)、[GOT mixin](https://github.com/LitingLin/LoRAT/blob/5260744ff0a65207a73289bbda1788c377d23bcc/config/LoRATv2/_mixin/got10k.yaml) |
| OSTrack-256，ECCV 2022 | MAE ViT-B/16；骨干全量微调 | 官方 GOT 配置100×60,000=6M pairs；混合配方300×60,000=18M | 32/GPU×4=128；GOT 约46.8k实际完整batch更新 | 128/256，两图；联合编码、候选消除、无在线模板 | [论文 §4.1](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820332.pdf)、[GOT配置](https://github.com/botaoye/OSTrack/blob/main/experiments/ostrack/vitb_256_mae_ce_32x4_got10k_ep100.yaml)、[loader](https://github.com/botaoye/OSTrack/blob/main/lib/train/base_functions.py) |
| MixFormer-CvT，CVPR 2022；此处比原始 CvT 分支 | CvT-21 ImageNet-22k；stage1微调跟踪器、stage2训练 SPM | 官方 GOT配置：500×60,000=30M，另40×60,000=2.4M score-training samples | 官方 GOT脚本8GPU×32=256；约117,000 + 9,360完整batch更新 | 两个128模板、320搜索；asymmetric mixed attention；SPM选择在线模板 | [论文+附录](https://arxiv.org/pdf/2203.11082)、[stage1](https://github.com/MCG-NJU/MixFormer/blob/8b87088f45af02c614a277176c63181261e93de3/experiments/mixformer_cvt/baseline_got.yaml)、[stage2](https://github.com/MCG-NJU/MixFormer/blob/8b87088f45af02c614a277176c63181261e93de3/experiments/mixformer_cvt_online/baseline_got.yaml)、[8GPU脚本](https://github.com/MCG-NJU/MixFormer/blob/8b87088f45af02c614a277176c63181261e93de3/tracking/train_mixformer_cvt.sh) |
| GOLA，AAAI 2026 | 已训练 LoRAT 的 DINOv2 B/L + LoRA priors；**RGB-T adaptation** | 官方代码 LasHeR train；10×131,072=1.31072M 多模态训练单元，另有之前的RGB成本 | 128 / 10,240 | 每模态 z/d112、x224，共六裁剪；rank分组和部分冻结 | [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/37843)、[论文 Implementation Details](https://arxiv.org/html/2512.05359v1#S4.SS1)、[run](https://github.com/MelanTech/GOLA/blob/339c737cda6a24be667b6e5abdc721e8d6046f05/config/GOLA/run.yaml) |
| MMLoRAT，作者官方工程，**未把它认定为独立已发表顶会论文** | DINOv2；README要求外部预训练权重，具体训练来源需额外核对 | 默认 LasHeR：10×60,000=.6M；另提供 LasHeR+RGB联合数据配置 | 默认64；名义 .6M/64=9,375，实际取整/丢尾以loader为准 | 多模态在线模板；RGB SOT在该工程中按双模态输入处理 | [README](https://github.com/MelanTech/MMLoRAT/blob/ebe1d0406ef28e525719f0a39d9eb1626d94ddba/README.md)、[run](https://github.com/MelanTech/MMLoRAT/blob/ebe1d0406ef28e525719f0a39d9eb1626d94ddba/config/MMLoRAT/run.yaml)、[联合数据](https://github.com/MelanTech/MMLoRAT/blob/ebe1d0406ef28e525719f0a39d9eb1626d94ddba/config/_dataset/train_joint.yaml) |

步数细节：OSTrack/MixFormer 官方 loader 使用 DistributedSampler 和 `drop_last=True`。按上述4/8GPU启动，每轮实际消费59,904样本，分别468/234步；表中的6M、30M是配置请求数，不能当成精确消费量。LoRAT 的131,072可被128整除，没有这项尾批差异。[OSTrack loader](https://github.com/botaoye/OSTrack/blob/main/lib/train/base_functions.py#L124)、[MixFormer loader](https://github.com/MCG-NJU/MixFormer/blob/8b87088f45af02c614a277176c63181261e93de3/lib/train/base_functions.py#L124)。

LoRATv2 论文 §4.1 的 “131,072 iterations/epoch” 应结合官方 run 的 `samples_per_epoch:131072` 和 batch128解释为采样量，而不能机械计成131,072个 optimizer updates。[论文 PDF p8](https://papers.nips.cc/paper_files/paper/2025/file/ad7e42e7b1f638e991d822724969be45-Paper-Conference.pdf)、[官方 run](https://github.com/LitingLin/LoRAT/blob/5260744ff0a65207a73289bbda1788c377d23bcc/config/LoRATv2/p1-B-224/run.yaml)。

混合数据配方与 GOT-only 必须另列：LoRAT/OSTrack 的通用配方使用 LaSOT、TrackingNet、GOT-10k、COCO，且混合配方中的 GOT 有排除约1k序列的约定；其 GOT-only 配置则直接使用 GOT train。不能把混合训练模型的总预算或外域结果当成 GOT-only 的同协议对照。[LoRAT 混合数据](https://github.com/LitingLin/LoRAT/blob/5260744ff0a65207a73289bbda1788c377d23bcc/config/_dataset/train.yaml)、[GOT-only 数据](https://github.com/LitingLin/LoRAT/blob/5260744ff0a65207a73289bbda1788c377d23bcc/config/_dataset/train-got10k.yaml)、[OSTrack 论文 §4.1](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820332.pdf)。

### 3.2 优化和适配容量对照

| 设置 | 当前 | LoRAT 官方 | LoRATv2 官方 | OSTrack/MixFormer 对照的限制 |
|---|---|---|---|---|
| 可训练骨干范围 | 仅 Q/V LoRA | 每层 Q/K/V、attention proj、FFN fc1/fc2 的 LoRA | attention与FFN均有按stream路由的LoRA | 是全量微调（MixFormer冻结BN，SPM阶段另冻结跟踪器），不能照搬LR与正则 |
| rank / alpha / dropout | 8 / 16 / .05 | 64 / 64 / 0 | 64 / 64 / 0 | 无LoRA |
| 初始化 | A Kaiming、B零 | 默认A、B均truncated normal(.02) | 此表不把未逐项核对的初始化当作与v1相同 | 不适用 |
| LR / WD | 1e-4 / .05 | 1e-4 / .1 | 1e-4 / .1 | OSTrack head4e-4/backbone4e-5、wd1e-4；MixFormer head1e-4/backbone1e-5、wd1e-4 |
| cosine / warmup | min1e-6；2epoch；prefix=false | min1e-6；2epoch；prefix=true | min5e-6；2epoch；prefix=true | step schedule，训练阶段也不同 |
| loss | IoU-aware BCE+GIoU，1:1 | 相同基本目标 | 相同基本目标 | OSTrack还有L1与focal；MixFormer stage1 GIoU+L1，SPM有单独score训练 |

证据：[LoRAT 注入工具](https://github.com/LitingLin/LoRAT/blob/5260744ff0a65207a73289bbda1788c377d23bcc/trackit/models/methods/LoRAT/funcs/vit_lora_utils.py)、[LoRAT 初始化](https://github.com/LitingLin/LoRAT/blob/5260744ff0a65207a73289bbda1788c377d23bcc/trackit/models/methods/LoRAT/modules/lora/__init__.py)、[LoRATv2 attention](https://github.com/LitingLin/LoRAT/blob/5260744ff0a65207a73289bbda1788c377d23bcc/trackit/models/methods/LoRATv2/modules/attention.py)、[LoRATv2 MLP](https://github.com/LitingLin/LoRAT/blob/5260744ff0a65207a73289bbda1788c377d23bcc/trackit/models/methods/LoRATv2/modules/mlp.py)，以及上表各方法run/config。LoRATv2的stream特化贯穿Transformer，与当前“浅层按模态、深层共享”不是同一设计。

参数公式提供容量尺度，不提供性能预测：当前每层两个 d×d 投影，参数为 `12×2×(2×768×8)=294,912`；LoRAT-B 的 Q/K/V/O 四个 d×d 加两个 d↔4d FFN，共 `12×(8d+10d)×64=10,616,832` 个 adapter 参数。当前 adapter 是后者的 **1/36**，而不是只少一个 rank8/64=1/8 的比例；头与角色编码还会改变总 trainable 比例。[当前 QVAdapter](/home/projects/GOLAv2/trackit/models/methods/dinov3_rgb/lora.py:23)、[官方逐线性层注入](https://github.com/LitingLin/LoRAT/blob/5260744ff0a65207a73289bbda1788c377d23bcc/trackit/models/methods/LoRAT/funcs/vit_lora_utils.py)。

### 3.3 近期工作的适用边界

LoRATv2 已由 NeurIPS 官方 proceedings 确认为2025主会论文；其单帧到多帧分阶段训练是可信的“先学跟踪、再扩时间建模”参照，但不是当前三图双向联合后6层架构的直接验证。[正式来源](https://papers.nips.cc/paper_files/paper/2025/hash/ad7e42e7b1f638e991d822724969be45-Abstract-Conference.html)。

另查到作者官方 DTPTrack 的可用代码，会议身份由主会话进一步核对为 **CVPR2026，pp6847–6856**：[CVF正式页面](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_Drift-Resilient_Temporal_Priors_for_Visual_Tracking_CVPR_2026_paper.html)。其DINOv2-B/14使用五帧、all-linear rank64，官方Base配置仍为batch128、每轮131,072、GOT100epoch。这支持把历史可靠性单独处理，不构成“现在就加未来预测模块”的理由。技术细节以作者arXiv及官方代码核对：[作者论文 Appendix A](https://arxiv.org/html/2604.02654v1#A1)、[官方README](https://github.com/NorahGreen/DTPTrack/blob/8a6d04c2b918667f1717f73555dca8198d3b11a0/README.md)、[Base配置](https://github.com/NorahGreen/DTPTrack/blob/8a6d04c2b918667f1717f73555dca8198d3b11a0/config/DTPTrack/Base/config.yaml)、[run](https://github.com/NorahGreen/DTPTrack/blob/8a6d04c2b918667f1717f73555dca8198d3b11a0/config/DTPTrack/Base/run.yaml)。

DINOv3 官方报告中直接展示的 tracking 证据是 DAVIS/YouTube-VOS/MOSE 上基于patch相似度的 **视频分割标签传播**。它不是GOT-10k框跟踪的LoRA训练配方，也不验证本项目的rank、LR或35小时收敛。本文没有取得可完整复核、与当前结构同条件的DINOv3 GOT-only官方配方；这表示本次证据缺口，不是宣称所有DINOv3 SOT研究都不存在。[DINOv3 §6.1.5、§D.5](https://arxiv.org/html/2508.10104v1#S6.SS1.SSS5)、[官方仓库](https://github.com/facebookresearch/dinov3)。

## 4. 哪些设计应保留，哪些应优先试验

### 4.1 LoRA容量：先做受控对照，不预设r64是最优

最相关的直接证据是 LoRAT 补充材料 **p5、Table6**。该表在DINOv2-L-224上比较：

| 注入范围 | rank | LaSOT SUC |
|---|---:|---:|
| Q/V | 4 | 66.2 |
| Q/V | 64 | 71.9 |
| 全部Transformer线性层 | 16 | 73.0 |
| 全部Transformer线性层 | 64 | 74.2 |

这些数据支持把“仅QV且极低rank”列为优先消融项，**不支持把上述百分点差距预测到DINOv3-B/GOT-only**。rank与注入范围应分别观察；更大的总trainable数也不等同于更多有效adapter容量。[原始补充材料 §C.4、Table6](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00113-supp.pdf)。

建议保留当前B=0初始化为基线。它使初始adapter贡献为零，便于确认官方骨干行为；A初始梯度为零是B=0的正常结果，不能当作unused。LoRAT确实比较过初始化，但其truncated normal优势来自另一骨干/配方，不足以要求本项目立即改变初始化。[当前初始化](/home/projects/GOLAv2/trackit/models/methods/dinov3_rgb/lora.py:13)、[LoRAT官方初始化与可选分支](https://github.com/LitingLin/LoRAT/blob/5260744ff0a65207a73289bbda1788c377d23bcc/trackit/models/methods/LoRAT/modules/lora/__init__.py)。

首轮优先比较QV r8和QV r32（alpha随rank从16变64，保持alpha/r=2）。另设all-linear r16/alpha32候选，保持dropout .05、A/B初始化、采样、layer split、头、优化器和样本进度一致。后者需要后续显式实现K/O/FFN注入，本次没有修改。它回答“扩大适配范围是否值得”，不是在参数量完全匹配条件下证明FFN单独有效；若获益，再做QV+FFN或参数量匹配消融。

### 4.2 batch、LR、WD与训练时长应一起解释

当前1e-4不是荒谬的学习率：LoRAT、LoRATv2均用该量级，但它们默认global batch128，且alpha/r和适配参数不同。也没有一手结果证明当前batch16必须按线性规则缩到1.25e-5。**推荐把1e-4与3e-5当作两点稳定性筛选，而不是宣布3e-5为最优。**这是针对batch差异的实验建议，证据强度低于同结构实测。[当前优化器](/home/projects/GOLAv2/config/DINOv3RGB/run.yaml:74)、[LoRAT run](https://github.com/LitingLin/LoRAT/blob/5260744ff0a65207a73289bbda1788c377d23bcc/config/LoRAT/run.yaml)。

AdamW的decoupled decay按**每次更新**生效，所以不能只比较`.05 < .1`就说当前正则更弱。忽略两条cosine曲线的细微差别，保持相同LR而更新数8倍、WD减半，会使累计decay指数规模约为LoRAT的4倍；这是零梯度衰减的量级分析，不是实际权重必定缩小到某值的预测。[本地AdamW构造](/home/projects/GOLAv2/trackit/runner/training/common/optimization/optimizer/builder.py:40)、[PyTorch官方AdamW算法](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)。因此先固定WD进行LR筛选，若日志显示adapter范数被持续压低、训练和验证都欠拟合，再单独试WD，而非同时改多个超参数。

100epoch可以作为最大预算；是否在20、50还是100epoch最适合迁移，应看闭环val和后续小规模RGB-T迁移。按更新数直接改成12.5epoch虽然接近LoRAT的102,400步，却只暴露其1/8样本，也不是公平等价配方。

主会话追加的无参数更新探测已证明：**batch16不是当前r8结构在这张32GB卡上的显存硬上限**。真实数据管线、相同确定性设置下，batch64峰值allocated约8.81GiB，端到端均值.5789秒/step；按相同总样本量外推32.93小时，与batch16的34.75小时接近。较大的batch显著减少更新次数并提高纯GPU计算吞吐，但当前管线/等待开销抵消了大部分端到端收益，不能承诺成倍加速。batch128另有16.73GiB的初步可运行探测，但未做同设置完整管线复核，也未包括optimizer.step。详见§6.3。可把**真实batch64、accum1**纳入样本预算相同的优化对照；不应以此认定它的收敛或较大rank显存也已验证。

### 4.3 可保留的部分

- **官方权重、冻结基础层、角色/前景编码、MLP头、IoU-aware BCE+GIoU**：与已有PEFT跟踪思路一致，且当前实现保留官方DINOv3接口特性。没有证据需要先改成卷积头、解冻全部骨干或更换loss。[LoRAT模型与头](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00113.pdf)、[本地head](/home/projects/GOLAv2/trackit/models/methods/dinov2_missing/head.py:37)。
- **128/256分辨率**：DINOv3 patch16下分别是8²/16²，和LoRAT-B-224的patch14网格相同；它是合理的首个计算档位。更高像素分辨率不自动意味着更多token或更强跟踪能力，先不扩大到384。[当前配置](/home/projects/GOLAv2/config/DINOv3RGB/dinov3/config.yaml:7)、[LoRAT-B配置](https://github.com/LitingLin/LoRAT/blob/5260744ff0a65207a73289bbda1788c377d23bcc/config/LoRAT/B-224/config.yaml)。
- **裁剪factor2/4、搜索jitter .25/3、flip/color/3-Augment**：直接接近LoRAT官方配方。独立flip是有意的匹配增强，不应仅因z/x不一致就称为标签错配；bbox确实同步变换。三图3-Augment共用操作的强度值得在训练失败时查，但不是已证实必须删除。[LoRAT run](https://github.com/LitingLin/LoRAT/blob/5260744ff0a65207a73289bbda1788c377d23bcc/config/LoRAT/run.yaml)、[本地flip](/home/projects/GOLAv2/trackit/data/methods/siamese_tracker_train/transform/default/augmentation/horizontal_flip.py:13)。
- **BF16、clip1、完整可恢复checkpoint**：适合当前工程基线；BF16与float32 loss/head输出已明确。后续只有图路径开始因模态缺失/迭代次数动态变化时，才需重新审查DDP static_graph，而不是现在因为未来设计而禁用它。[AMP builder](/home/projects/GOLAv2/trackit/runner/training/common/optimization/amp/builder.py:15)、[head输出](/home/projects/GOLAv2/trackit/models/methods/dinov2_missing/head.py:70)。

前6独立/后6联合是合理的迁移接口选择，但没有上述论文证明“6”是最优。首轮先固定；若容量调整后仍有明显欠拟合，再比较独立深度3/6或0/6，确保未来特征契约同步记录。保留独立特征接口不等于后6层必须恰好6层。

## 5. 代码审查：确定问题与条件性风险

### 5.1 长训缺少模型选择，已有evaluation不是训练val插件

`run.yaml:12`只有train；checkpoint每epoch保留最近3个。`mixin/evaluation.yaml:3、16、49`分别替换整个task/data/runner，构成独立GOT-val闭环评估，使用180条val序列、模板更新阈值.84、Hann penalty .45。直接把此mixin加进正式训练命令会变成评估模式设计，不能理解成“训练同时增加验证”。[默认run](/home/projects/GOLAv2/config/DINOv3RGB/run.yaml:5)、[evaluation mixin](/home/projects/GOLAv2/config/DINOv3RGB/dinov3/mixin/evaluation.yaml:1)。

**长训前必须确定**：按epoch2/5/10/20及后续每5或10epoch独立评估；保存best-val、阶段里程碑和latest恢复状态。只保留最后3轮会丢失较早的迁移候选。早停可以用作预算决策，但不是必须引入自动早停算法；至少要使曲线可见、候选可回溯。

应记录闭环AO、SR0.5、SR0.75和每序列结果，辅以固定裁剪诊断集的IoU、分类/回归损失、grad norm、clip频率、LoRA增量范数和模板更新频率。IoU-aware classification score不是已经校准的模板安全概率，.84来自别的模型背景，第一轮不要同时大量扫阈值。对相同checkpoint可比较在线更新开启/关闭的结果以判断d是否带来收益；这属于推理诊断，不需要再训练一遍。

已有评估还采用dynamic scheduling及AMP默认dtype；官方MMLoRAT近期改用static scheduling并说明并行任务次序可能影响最终结果。这不是当前已复现的错误，但在解释小幅val差异前，应固定batch/dtype/顺序并复跑同一checkpoint估计数值差异。[本地eval设置](/home/projects/GOLAv2/config/DINOv3RGB/dinov3/mixin/evaluation.yaml:35)、[MMLoRAT官方变更说明](https://github.com/MelanTech/MMLoRAT/blob/ebe1d0406ef28e525719f0a39d9eb1626d94ddba/README.md)。

### 5.2 d采样和模板增强的实际语义

`worker.py:71–76`先采z/x；`84–89`再次对整条track采一个独立pair；`97`只用第二pair的首帧做d。`_algos.py:144`的首帧由全部有效帧抽取。因此：

- `|z-x|≤100`是正常成功采样时的约束；缺少候选时可以退化为同帧。当前builder默认auto_extend步长/次数都为0。
- **d与x没有100或200帧的界限，也没有d≤x保证**；同一track足够长时，其跨度原则上可达track长度减1。第二pair内部的gap限制并不能约束被拿来使用的d与第一个x。
- `negative_online_sample_weight=0`使d始终positive；模板自身没有平移/尺度jitter。配置中的`online_bbox_shift`只在`not is_online_positive`执行，当前没有被激活，不能算作已经加入的在线模板噪声增强。

证据：[worker](/home/projects/GOLAv2/trackit/data/methods/siamese_tracker_train/siamese_training_pair_sampling/worker.py:64)、[frame算法](/home/projects/GOLAv2/trackit/data/methods/siamese_tracker_train/siamese_training_pair_sampling/_algos.py:120)、[builder默认值](/home/projects/GOLAv2/trackit/data/methods/siamese_tracker_train/siamese_training_pair_sampling/builder.py:15)、[shift门控](/home/projects/GOLAv2/trackit/data/methods/siamese_tracker_train/transform/default/processor.py:75)。

对当前“从同一目标多视图学习定位”的预训练，这不是自动构成数据泄漏；真实推理能否保持因果取决于推理输入。**风险推断**是d过于干净且无时序语义，训练中更像第二个GT模板，与部署时有定位误差的历史模板不同。可在第二轮单独测试d取真实过去、限定与x的间隔、温和平移/尺度噪声；不建议未经验证就加入大量错误目标模板。

后续t+1/t+2预测训练则必须换成有明确frame index、按时间排序的序列、实际过去缓存和统一裁剪坐标。当前随机interval triplet不能用于证明未来预测因果性。独立flip/color增强也必须在时空监督坐标里做一致处理。[现有契约说明](/home/projects/GOLAv2/docs/temporal_feature_contract.md:124)。

### 5.3 梯度累积存在两个确定问题，当前accum1不触发

**问题A：未归一化loss。** `DefaultTrainer.run:168–172`把完整criterion loss传给backward；`ParameterUpdater_WithAMPSupport:37–43`直接backward和clip，没有除以accumulation steps。设各microbatch loss为L_i，当前得到的是`Σ∇L_i`，不是平均梯度。Adam可能抵消部分全局尺度变化，但clip、epsilon、decay与不同批次归一化不会因此自动等价。[trainer](/home/projects/GOLAv2/trackit/runner/training/default/__init__.py:156)、[parameter updater](/home/projects/GOLAv2/trackit/runner/training/common/optimization/amp/parameter_updater.py:37)。

**问题B：scheduler计数单位不匹配。** builder先用`num_iterations_per_epoch // grad_accumulation_steps`设置warmup和cosine长度；trainer却在每次更新时传入原始`self._iteration`。以当前每epoch8192 microsteps、100epochs、accum8为例，cosine周期是102,400 update units，却会在约第102,400个microstep（12.5epoch）结束，剩余训练长期停在lr_min；warmup也在约.25epoch结束。[builder:21–40](/home/projects/GOLAv2/trackit/runner/training/common/optimization/lr_scheduler/timm_scheduler/builder.py:21)、[调用:194–200](/home/projects/GOLAv2/trackit/runner/training/default/__init__.py:194)。

本次CPU只读构造scheduler、查询其函数，没有optimizer.step，结果为：

| accum | t_initial | warmup_t | lr(micro-index=102400) |
|---:|---:|---:|---:|
| 1 | 819200 | 16384 | 9.6232e-5 |
| 8 | 102400 | 2048 | 1e-6 |

修复方向应同时包括loss语义、独立optimizer-update计数、恢复状态和尾批处理。并且本loss按正patch数P归一化：即使简单除以k，`mean(L_i)`也不严格等于大batch的`Σnumerator_i / ΣP_i`；若目标是真正等价大batch，要明确处理窗口内P的权重。**在上述语义校验前，实验矩阵固定accum1，不建议直接改成8。**

### 5.4 DINOv3 RoPE随机缩放应显式记录

官方B/16设置`rescale_coords=2`；RopePositionEmbedding在training=True时按log-uniform `[1/2,2]`缩放坐标。`requires_grad_(False)`只冻结参数，trainer的`model.train(True)`仍使它生效。[官方 B/16 定义](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/hub/backbones.py#L277)、[RoPE实现](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/layers/rope_position_encoding.py#L57)、[本地train调用](/home/projects/GOLAv2/trackit/runner/training/default/__init__.py:96)。

当前`_prepare_crop`每crop调用一次RoPE，三个crop独立抽样，随后所有block复用；官方VisionTransformer则在block循环内重新调用RoPE。两者在eval态没有这项随机差异，训练态的增强粒度不同。[当前准备](/home/projects/GOLAv2/trackit/models/methods/dinov3_rgb/backbone.py:97)、[官方forward](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/models/vision_transformer.py#L220)。

这不应未经实验定性为错误：逐crop固定扰动可以是一种合法增强。但报告与checkpoint元数据应记录它；如出现定位不稳定，可在固定其他因素时试“保留当前随机缩放”对“只关闭RoPE随机缩放”。不要为关掉这一项直接把全部adapter改成eval，从而同时关闭LoRA dropout。未来冻结参考空间采用eval确定性是另一阶段的问题。

### 5.5 遗留数据容器与负样本路径

1. `TrainData`的`input/target/miscellanies`默认值是三个类级共享dict；两个`TrainData()`的input确实是同一个对象。collator用`.update()`写入这些对象。[定义](/home/projects/GOLAv2/trackit/data/protocol/train_input.py:4)、[collator](/home/projects/GOLAv2/trackit/data/methods/siamese_tracker_train/transform/default/processor.py:285)。应改成每批独立容器。但CUDA streamer重建dict/NamedTuple、多进程传输也会反序列化；**不能只凭别名现象就说当前6-worker训练已污染**。[重建函数](/home/projects/GOLAv2/trackit/miscellanies/torch/tensor_move_helper.py:23)。
2. `PluginRegistrationHelper`三个`extra_*`是未注解的类属性list，实例之间累积。当前单独进程只build一个train分支未必触发，但在同进程重复build或添加train/val后会出现重复注册风险；应在正式启用这种流程前修成实例列表。[定义](/home/projects/GOLAv2/trackit/data/methods/siamese_tracker_train/transform/default/plugin/builder.py:23)。
3. **不要把“补负样本”当成可直接打开的配置优化。** worker负样本分支仍用三个位置参数构造已变成五字段的`SiameseTrainingPairSamplingResult`；本次直接调用相同形状构造，稳定报缺少`is_positive/is_online_positive`。此外全负batch时IoU-aware criterion仍进入IoU赋值，synthetic输入已触发AssertionError；混合正负batch的bbox collate也需重新核对。[worker负分支](/home/projects/GOLAv2/trackit/data/methods/siamese_tracker_train/siamese_training_pair_sampling/worker.py:101)、[五字段定义](/home/projects/GOLAv2/trackit/data/methods/siamese_tracker_train/siamese_training_pair_sampling/_types.py:25)、[criterion门控](/home/projects/GOLAv2/trackit/criteria/methods/box_with_score_map/__init__.py:41)。这些路径默认未启用，不否定当前正样本训练。

### 5.6 loss归一化本身没有发现当前正样本路径的实质错误

正patch集合来自增强后目标框投影到16×16响应图、栅格化得到的区域；不只是单个中心点。令P是batch中正patch数，q是在no_grad中计算的预测框IoU，则当前是：

```text
L_cls = sum_all_batch_and_positions BCEWithLogits(score, positive ? q : 0) / P
L_reg = sum_positive_positions GIoU(pred_box, gt_box) / P
L = L_cls + L_reg
```

DDP时P先跨rank取mean，再由DDP平均梯度，符合整体按正位置数归一化的意图。P为1以下时clamp1。当前全正pair保证每个search有有效框和至少一个正位置；并不意味着响应图只有正分类样本，框外位置仍是负标签。[标签生成](/home/projects/GOLAv2/trackit/data/methods/siamese_tracker_train/transform/default/plugin/box_with_score_map_label_gen/__init__.py:9)、[criterion](/home/projects/GOLAv2/trackit/criteria/methods/box_with_score_map/__init__.py:23)、[LoRAT补充 §A.2](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00113-supp.pdf)。

因此BCE数值大于GIoU是可能的，并不能单凭量级断言loss权重1:1错误；尤其框面积改变P，也会改变分类总量的相对尺度。应先看定位IoU、梯度、score分布及val轨迹，再决定是否重加权。没有负pair也不等于没有背景学习，只是缺乏整幅search无目标/错目标监督。

## 6. 最小实验矩阵与计算上限（提案，未执行）

### 6.1 共同约束

首轮固定GOT train、输入网格、split6/6、三图采样、图像增强、RoPE策略、BF16、loss、accum1、WD.05、seed和验证集；除E外均batch16。每个候选使用**按样本进度定义的同一条100epoch参考LR曲线、同样2epoch warmup**，在预定样本量截断比较；不要把某候选的cosine压缩成5epoch再与100epoch曲线前5epoch比较。恢复训练时保留原scheduler状态。

所有运行都从相同官方DINOv3起点建立模型，不从已经看过更多样本的另一个候选checkpoint开始。只比较LR的A/B可使用相同初始化；rank/注入形状不同会改变RNG消耗量，单纯同seed不能保证之后创建的head/编码逐元素一致。因此容量对照应显式复制一份公共初始head、role/modality embedding状态，仅让不同形状的LoRA按各自记录的seed初始化。当前head和编码约占trainable89%，这一控制尤其有价值。固定验证裁剪集与闭环轨迹，关注效果随**累计samples与updates**的变化。

| 候选 | 唯一主要变化/用途 | LoRA设置 | 第一阶段预算 |
|---|---|---|---|
| A | 当前配方的参考曲线 | QV r8/alpha16/dropout.05，lr1e-4 | 5 epochs=655,360 triplets=40,960 updates |
| B | 观察batch16下较低LR是否更稳 | 与A相同，仅lr3e-5 | 同A |
| C | 检查仅增加rank是否有效 | QV r32/alpha64/dropout.05，使用A/B中选定LR | 同A |
| E | 检查真实大batch与更少更新是否合适；与A只改batch | 与A同为QV r8/alpha16/dropout.05、lr1e-4；真实batch64、accum1 | 5epochs=655,360 triplets=10,240 updates |
| D（备选） | 检查更广注入是否值得；**需先实现并验证新注入，当前不可直接改YAML获得** | Q/K/V/O+FFN fc1/fc2，r16/alpha32/dropout.05，使用同一选定LR；仍A Kaiming/B零 | 同A；替换一个候选或留待下一轮，不默认追加 |

这里C/D的LR选择必须先于其训练记录下来；若A/B在5epoch尚无法可靠区分，就先统一使用A的LR并明确不确定性。C/D不是严格参数量匹配：QV r32约1.180M adapter参数，all-linear r16约2.654M。它们是预算下的候选筛选，不能用一次胜负宣称K、O或FFN单独产生收益。

**分阶段预算建议**：先A/B各5epoch，再做C/E各5epoch；四条共20个“当前epoch等价”的样本量。按既有短时测量每epoch约20.85分钟，名义计算约6.95小时；给初始化、验证、较大adapter开销留余量，第一轮设置10 GPU小时硬预算。E的更新数只有A的1/4，因此它是相同样本暴露下的batch/更新预算比较，并非同更新数比较。D在实现验证后可替换一个候选，或者留到下一轮；不为凑矩阵跳过正确性检查，也不无条件增加第五条。

从第一轮最多留两个候选，各继续到10epoch（再10个epoch等价）；胜者继续到20epoch（再10个）。整个筛选阶段名义约13.9小时，建议上限16 GPU小时；若新注入实际吞吐下降，应按实测降低候选数/追加轮数而不是突破预算。这些时长使用现有r8的无optimizer.step短测外推，**不是对C/D的已测速度承诺**。

5epoch只用于排除不稳定或明显无效设置，不能确定最终最优。若排名接近，不强行宣布赢家：延长最相关两条或为最终候选补一个seed，并从同一预算中分配。优先看跨多个checkpoint的一致趋势，而不是单点AO最高。

### 6.2 何时值得继续长训

进入20→100epoch之前至少取得以下实际信息：能稳定下降的定位误差/训练损失、无非有限梯度、闭环GOT-val的持续进展、模板更新不导致明显性能退化、可恢复checkpoint与best候选保留。若训练改善而val明显退化，停止追加并检查过拟合/模板分布；若训练和val都低且容量更大候选改善，优先改容量；若所有候选均停滞，先排查数据与优化，不直接把预算加倍。

如果最终仍选择100epoch，胜者在20epoch的有效checkpoint可以继续；其他候选的成本是额外筛选成本。不能把12–16小时筛选再加完整100epoch重训，仍声称总预算只有35小时。是否继续、继续多久应由阶段结果决定，并在提交前告知用户。

### 6.3 35–50小时估算的正确用法

主会话已做的RTX5090短时测量均为BF16、6 workers、32条真实序列，没有optimizer.step。batch16测得forward+backward+clip约.11065秒/step、32个端到端测量batch均值.15272秒/step。随后在相同确定性模式下复核batch64，25次纯GPU计算和32次端到端测量得到：

| batch | 100epoch更新数 | 计算均值秒/step | 端到端均值秒/step | 端到端外推100epoch | 峰值allocated |
|---:|---:|---:|---:|---:|---:|
| 16 | 819,200 | .11065 | .15272 | 34.75h | 2.77GiB |
| 64 | 204,800 | .17631 | .57887 | 32.93h | 8.81GiB |

batch64端到端median=.21180秒、p95=1.71342秒，表明短测中有明显等待长尾；其纯GPU外推仅10.03小时。当前证据支持优先检查数据供给/流水线等待，尚不能把所有差额归因于某一磁盘或worker数。batch128的16.73GiB初测采用了不同确定性设置、重复真实裁剪，不能当作同条件端到端结果。原始记录见 [batch16 测量JSON](dinov3_rgb_timing_b16.json) 与 [batch64 测量JSON](dinov3_rgb_timing_b64.json)。

这些测量尚未覆盖全数据IO、真实参数更新、验证和checkpoint开销，35–50小时应被视为当前batch16的资源规划范围，不能作为收敛承诺或完成时间保证。也不能根据计算时间的改善，就将正式batch64预算改称10小时。

本次审查没有重复GPU吞吐测量；使用的是主会话记录。与文献的多GPU/full-finetune/torch.compile计时不作直接速度排名。未来预测、迭代重建0/1/3/5步和决策器尚未启用，也不加入当前预算。

## 7. 对后续RGB-T迁移的具体含义

**可以保留的资产**是：学到的目标定位头、角色前景背景编码、RGB浅层adapter、深层共享跟踪adapter、以及官方DINOv3的稠密特征先验。当前`add_modality('tir')`把RGB浅层adapter复制给TIR，提供一致起点；但热红外分布、双模态融合和缺失策略需要真实RGB-T训练验证，不能由复制行为证明迁移成功。[迁移实现](/home/projects/GOLAv2/trackit/models/methods/dinov3_rgb/backbone.py:58)。

GOT-only是清楚、可复现的首站，适合验证新的RGB入口；通用跟踪底座是否需要LaSOT/TrackingNet/COCO，应根据下游迁移表现与数据预算决定。若以后引入这些任务数据，要将其与GOT-only结果分开命名和报告，不将混合训练结果放进仅GOT训练的同协议对照。[LoRAT官方数据区分](https://github.com/LitingLin/LoRAT/blob/5260744ff0a65207a73289bbda1788c377d23bcc/config/LoRAT/_mixin/got10k.yaml)。

RGB-val最高checkpoint也未必是RGB-T迁移最佳点。可保留20/50/best/final等少量里程碑，在同一个后续RGB-T短适配预算下比较；这项预算应计在RGB-T阶段。GOLA说明强RGB LoRA priors可用于下游结构化适配，但其rank64分组方案不应直接套到只有8rank的QV adapter上，更不能在缺少其训练priors时假定收益成立。[GOLA“Discussion on Training without Priors”与Limitation](https://arxiv.org/html/2512.05359v1)。

未来模块应继续遵守现有契约：当前可用真实输入和过去缓存可进入推理；真实未来/被遮当前模态只用于训练标签。联合特征已经受其他crop影响，不能当成纯历史单图参考或缺失模态真值；当前的isolated/reference接口为避免这个混淆提供了基础，但尚未证明预测、重建或策略有效。[feature bundle](/home/projects/GOLAv2/trackit/models/methods/dinov3_rgb/backbone.py:99)、[时间特征契约](/home/projects/GOLAv2/docs/temporal_feature_contract.md:1)。

## 8. 审查记录与可复核范围

读取代码覆盖：模型构建/官方依赖、Q/V LoRA、token/RoPE、三图采样、裁剪/增强、前景mask与响应图标签、head/loss、optimizer分组、AMP/clip、梯度累积/scheduler、checkpoint与独立evaluation配置。没有发现足够证据证明默认正样本、accum1路径必然不能训练；发现的条件性缺陷已标出触发条件。

本次执行仅包括只读源码/manifest、CPU构造scheduler并查询LR、synthetic全负loss调用和数据类构造异常检查。未写测试替代真实收敛证据。主会话反馈累计已有26项测试通过，本子代理没有重新运行这26项；[较早的文档验证记录](/home/projects/GOLAv2/docs/dinov3_rgb_pretraining.md:153)列出其中8项focused tests、7项回归测试及真实帧GPU前后向检查。这些结果验证兼容性和梯度可达，不等于已经完成优化/收敛验证。

本地Git HEAD：`7fa2380af780a42976bb1f2a2dc71d7233816583`；DINOv3官方固定commit：`6876159a11b4df116f30f667f8c9888617df0751`。读取时关键工作区文件SHA-256：

```text
config/DINOv3RGB/dinov3/config.yaml
ef312cc1bbe6c2d5ccff50dd76a9e48c125831aa5580636a6144c4fbd22def2a
config/DINOv3RGB/run.yaml
e06e039e2115657e79f19c17480718762c2187422709c323c6c9ede99137a162
trackit/models/methods/dinov3_rgb/backbone.py
a805e59b1cffea5e8586227e5cae1cb514d635ea0dfc6306b199562b703596ae
trackit/models/methods/dinov3_rgb/lora.py
7038b12b765f4172c89ef450cbfb1fdabd78045d7129b54cd141946b36945b56
trackit/data/methods/siamese_tracker_train/siamese_training_pair_sampling/worker.py
d329665136687d7ba504dda69aaf26618154ef1b88bb0aa78314180567e702a7
trackit/runner/training/default/__init__.py
eec3861acebf6591e3f9199b6106c085615bddfe0bbbc9b0b337ac531c8cf37c
```

外部源码读取固定版本：LoRAT/LoRATv2 `5260744ff0a65207a73289bbda1788c377d23bcc`；MixFormer `8b87088f45af02c614a277176c63181261e93de3`；GOLA `339c737cda6a24be667b6e5abdc721e8d6046f05`；MMLoRAT `ebe1d0406ef28e525719f0a39d9eb1626d94ddba`；DTPTrack `8a6d04c2b918667f1717f73555dca8198d3b11a0`。OSTrack引用为本次读取的main文件，未把main当作永久不变版本。

尚未获得的证据明确保留为未知：当前模型的训练曲线/最终GOT AO；当前r8与更大rank/all-linear的同条件结果；全数据长期吞吐；各RGB checkpoint的下游RGB-T迁移排序；MMLoRAT下载权重的完整训练数据来源。以上未知不应以DINOv3品牌、上游论文指标或测试通过来填补。
