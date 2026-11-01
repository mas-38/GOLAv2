# DINOv3 + LoRA RGB tracking pretraining on GOT-10k

This is the RGB task-pretraining stage for a later RGB-T missing-modality tracker.
It is a separate `DINOv3RGB` entry point; existing GOLA and DINOv2Missing settings
remain available. The experiment is versioned in
[`experiments/dinov3_rgb_got10k/v1/manifest.yaml`](../experiments/dinov3_rgb_got10k/v1/manifest.yaml).
The official run starts after the code commit and records its full Git revision.

The implemented history/reference-space/future-alignment contract and additional
feature exports are specified in [temporal_feature_contract.md](temporal_feature_contract.md).

## Short run for RGB-T transfer

For the current priority of obtaining an RGB tracking initialization quickly, use
the `fast_transfer` mixin. It defines one run rather than a parameter search:

| Setting | Fast transfer run |
| --- | --- |
| Backbone / attention | Frozen official DINOv3 ViT-B/16; independent first 6 blocks, joint last 6 |
| Trainable parameters | Q/V LoRA rank 32, alpha 64, dropout 0.05; embeddings and tracking head; 3,550,469 total |
| Inputs | Initial/online templates 128 x 128; search 256 x 256 |
| Training data | GOT-10k train only; 131,072 triplets per epoch |
| Budget | 20 epochs; batch 16; accumulation 1; 163,840 optimizer updates |
| Optimization | AdamW LR 1e-4, weight decay 0.05; one warmup epoch, cosine to 1e-6; BF16; clip 1 |
| Validation | All 180 GOT-10k val sequences after epochs 2, 5, 10, 15 and 20; online templates enabled |
| Selection | Highest validation AO; retain all 20 epoch checkpoints and their resume states |

Rank 32 is an engineering choice to increase adapter capacity over the rank-8
baseline without changing the transfer interfaces. It is not a measured optimum.
The 20-epoch run is a task-adaptation budget; validation must establish
whether its best checkpoint is a useful RGB-T initialization.

On the RTX 5090, reserve approximately **8-12 hours** including validation and data
overhead. This extrapolates the measured rank-8 batch-16 pipeline (about 21 minutes
per epoch); the full rank-32 run has not been timed. The first validation checkpoint
should become available in roughly one hour, subject to initial dataset indexing.
A candidate at epoch 10 is expected after roughly 4-6 hours. The cosine schedule
spans all 20 epochs; at epoch 10 it has not reached the minimum learning rate.

The following command **starts training** and has not been submitted:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 python main.py DINOv3RGB dinov3 \
  --mixin_config fast_transfer --device cuda \
  --run_id fast_transfer_r32_e20 \
  --output_dir ./output/dinov3_rgb_got10k
```

The framework appends `run_id` to `output_dir`. Its selected full RGB checkpoint
will be:

```text
output/dinov3_rgb_got10k/fast_transfer_r32_e20/checkpoint/best@valaverage_overlap_got-10k-val/model.bin
```

Epoch directories use zero-based indexes (`epoch_00` through `epoch_19`). Export
the selected checkpoint with `export_dinov3_rgb_checkpoint.py --rgbt` to copy the
learned RGB shallow adapters into the TIR branch. Keep rank 32 / alpha 64 in the
receiving RGB-T configuration; the architecture tensor rejects rank mismatches.
For standalone evaluation of these RGB weights, retain the architecture mixin:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 python main.py DINOv3RGB dinov3 \
  --mixin_config fast_transfer --mixin_config evaluation --eval \
  --device cuda --disable_wandb --weight_path /path/to/model.bin \
  --output_dir ./output/dinov3_rgb_got10k_val
```

No history, future-prediction or reconstruction modules execute in this RGB stage;
their feature interfaces remain available for the later RGB-T implementation.

Validation of this route: all 36 focused/model regression tests pass, including
an application-level test that verifies the best checkpoint belongs to the best
validation epoch rather than the last epoch. A real 60-frame GOT-10k validation
sequence passes the rank-32 closed-loop evaluator and emits the exact AO key used
for selection. This inference check performs zero optimizer updates and does not
establish tracking accuracy. All 180 validation sequences / 21,007 frames pass
file/count checks.

The checkpoint path required two framework fixes: retain metric-based dumpers
when their input iterable is a generator, and provide model weights after
validation even when regular weights were saved before validation.

## Versioned RTX 5090 run

The submitted run applies `fast_transfer` followed by `rtx5090`, plus
`--pin_memory`. First commit the experiment code, then submit with:

```sh
python launch_dinov3_rgb_training.py
```

The launcher derives run ID `dinov3rgb_v1_<commit8>_b64_e20` and tmux session
`dinov3_rgb_v1_<commit8>`. Output is under `output/dinov3_rgb_got10k/<run_id>`.
It fixes both seed and instance ID to 42 so changing the run name does not change
the instance RNG. Checkpoints capture Python, NumPy, CPU and initialized CUDA RNG
states for resumption.
It rejects uncommitted experiment changes and existing run directories. Local
`consts.yaml` and the independent pre-existing MMLoRAT modifications are preserved;
their exact patch and source fingerprints accompany the run. The old
`fast_transfer_r32_e20_b64_5090` startup was interrupted after some updates when the
code-version requirement arrived. It is retained as an excluded run. The versioned
run initializes from the official DINOv3 checkpoint.

The overlay uses batch 64, 20 training loader workers, four PyTorch CPU threads,
single-thread OpenCV, fused AdamW and validation batch 16. BF16 and PyTorch SDPA
Flash Attention were already active in the reference run. The attention backend
was verified with a profiler, including Flash Attention backward. Do not attribute
the measured speedup to newly enabling BF16 or Flash Attention.

Short measurements on an isolated 128-sequence real-data fixture, with rank 32 and
64 timed batches per case:

| Batch | Workers | Pinned async transfer | Triplets / second | Peak allocated GiB |
| --- | --- | --- | --- | --- |
| 16 | 6 | No | 97.2 | 2.78 |
| 32 | 16 | Yes | 243.0 | 4.82 |
| 64 | 16 | Yes | 247.1 | 8.91 |
| 128 | 20 | Yes | 335.1 | 17.07 |
| **64** | **20** | **Yes** | **345.4** | **8.91** |

These are forward/backward/clip measurements, excluding optimizer updates and
full-dataset variation. Fused AdamW separately passes a CUDA update sanity check.
The batch-64/20-worker case reduces mean data wait to about 1 ms and reaches about
3.55 times the reference pipeline throughput. Batch 64 and 128 are close to the
best measured throughput; 64 retains twice as many updates under the fixed sample
budget. Full measurement details are in
[the JSON report](research/dinov3_rgb_rtx5090_tuning_2026-11-01.json).

The batch change alters optimization: both runs see 20 x 131,072 = 2,621,440
triplets, but batch 64 performs 40,960 updates instead of batch 16's 163,840.
The learning rate remains 1e-4 and its warmup/cosine follow the 20-epoch sample
budget. Validation, rather than equal epoch counts alone, must establish quality.
Allow roughly 3-4 hours for this run pending actual full-dataset throughput.

`launch.json` records the exact Git commit, branch, command and environment;
`bootstrap.log` captures startup, and `exit_status.json` is written when
the process exits. The W&B run ID includes the same code revision. Local
diagnostics and the analysis command below apply to its output directory.

Reproduce the throughput measurements without optimizer updates:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 \
python benchmarks/benchmark_dinov3_rgb.py --batch 64 --workers 20 --pin \
  --report ./output/5090_b64_w20_performance.json
```

The previous unversioned job ran in tmux session `dinov3_rgb_5090_20`; its `launch.json` records the exact
command, `bootstrap.log` captures startup, and `exit_status.json` is written when
the process exits normally. Its excluded online run is
[W&B](https://wandb.ai/mas-38-anhui-university/DINOv3RGB/runs/fast_transfer_r32_e20_b64_5090).

Before enabling the larger prefetch queue, batch containers were changed to use
independent dictionaries, retaining tuple/pickle/device-streaming compatibility;
plugin registration lists are also independent per build. All 37 tests versioned
with this experiment pass, including batch isolation and the 5090 overlay
configuration. Two additional existing MMLoRAT workspace regressions also pass.

## Local metrics, W&B and analysis

The fast-transfer mixin enables JSONL statistics independent of W&B. Every 50
batches, it records the mean/min/max/last/finite-count/nonfinite-count for **all**
metrics in that window. The last partial window is flushed at epoch end, and epoch
means include every batch rather than averaging window means. Each completed JSONL
append is closed. A hard process kill can lose the current unfinished window of
up to 49 batches; completed records remain readable.

Recorded training metrics include:

- Total loss, weighted/unweighted BCE and GIoU (`Loss/iou` is the existing display
  name for the configured **GIoU** loss, not a measured IoU).
- IoU at labeled positive patch locations, foreground/background confidence and
  positive patch count. These diagnose training batches; they are not closed-loop
  GOT-10k validation AO. The BCE targets depend on IoU, so BCE need not decrease
  monotonically as localization improves.
- Learning rate and weight decay (group 0 plus min/max across optimizer groups),
  gradient norm **before clipping**, and loss scale when reported by the optimizer
  helper. Embedding/no-decay groups can have zero weight decay even when the
  decayed groups use 0.05.
- Host batch wall time, data wait, throughput computed as total samples / total
  batch time per window, allocated GPU memory and process-lifetime peak allocation.
  Timing introduces no extra CUDA synchronization and is not a GPU kernel timer.

Validation records AO, SR@0.5, SR@0.75, success/precision scores and FPS, with full
per-sequence reports in the existing evaluation archives. Best weights are selected
by `val/average_overlap_GOT-10k-val`.

Relative to the final run directory (the directory ending in `fast_transfer_r32_e20`):

```text
diagnostics/resolved_config.json   # Complete configuration after applying mixins
diagnostics/run_info.json          # Launch arguments, seed, versions, GPU, outcome, W&B URL
diagnostics/provenance.json        # Commit, branch, weights/data fingerprints, source manifest hash
diagnostics/source_manifest.json   # SHA256 of source/configuration/documentation files
diagnostics/working_tree.patch     # Exact tracked local edits outside the experiment commit
diagnostics/host_consts.yaml       # Local dataset paths and constants used by this run
diagnostics/console.log            # Console output and exception tracebacks
train/metrics.jsonl               # Window statistics and explicit nonfinite events
train/epoch_summary.jsonl         # Full training epoch statistics
eval/metrics.jsonl                # Validation runtime windows
eval/epoch_summary.jsonl          # Validation AO/SR and epoch statistics
```

`epoch` is one-based and `epoch_index` is zero-based. `sample_step` is the global
training sample counter **before** the last batch in a record, not an optimizer
update count; evaluation shares this counter without incrementing it. For this
batch-16, accumulation-1 run there are 8,192 optimizer updates per complete epoch.
The submitted RTX 5090 batch-64 overlay has 2,048 updates per epoch, 40,960 total.

W&B config includes `provenance/git_commit`, `provenance/git_branch`, the source
manifest and resolved-configuration SHA256 values, and experiment ID/version.
The local provenance additionally captures the initial checkpoint SHA256,
official DINOv3 source revision/status and GOT-10k split list hashes. No weights,
dataset images, generated caches, run outputs or authentication credentials are
included in the Git commit.

For W&B, register an account, create a key in the account settings and execute
`wandb login` on the training server. The existing default project is `DINOv3RGB`;
use `WANDB_ENTITY` to select a team if needed. The launch above enables W&B. To use
only local JSONL, add `--disable_wandb`; to record W&B offline as well, add
`--wandb_run_offline`. These modes require an explicit choice; no training has been
submitted during setup.

One publisher mirrors the same window means into W&B. Training metrics use the
sample counter as their horizontal axis; validation metrics use one-based epochs.
Validation summaries are logged to history as well as the run summary, so AO/SR
curves remain visible. W&B's internal event counter advances independently rather
than reusing the fixed training counter during validation.

Analyze a running or completed run with:

```sh
python analyze_dinov3_rgb_training.py \
  output/dinov3_rgb_got10k/fast_transfer_r32_e20
```

It writes `analysis/summary.json`, CSVs for training windows/epochs and validation,
and `analysis/curves.png` (loss, training IoU/confidence, LR, gradient norm,
throughput/data wait and validation AO/SR). It tolerates an unfinished final JSONL
record while the training process is writing. Use `--no-plot` for tables only.

The installed W&B SDK passes a real-video offline integration check with zero
optimizer updates. Server DNS, TLS and read-only GraphQL connectivity to
`api.wandb.ai` pass (GraphQL HTTP 200). Authenticated project access/cloud upload
must still be verified after account login.

## Architecture and transfer contract

- Initialize the official DINOv3 ViT-B/16 (12 blocks, width 768, 12 heads,
  CLS plus four register/storage tokens). Its pretrained parameters stay frozen.
- Templates `z` (initial) and `d` (online) are 128 x 128; search `x` is 256 x 256.
  Each input is a real three-channel RGB crop with ImageNet normalization.
- Blocks 1-6 process each crop independently and use an RGB-specific Q/V LoRA.
  Blocks 7-12 process the joint sequence and use shared tracking Q/V LoRA.
  The base configuration uses rank 8 / alpha 16; `fast_transfer` uses rank 32 /
  alpha 64. Both use dropout 0.05, random A and zero B.
- Retain official masked K bias, attention, LayerScale, FFN and normalization.
  There is no K update. Train role/foreground embeddings, modality embedding,
  LoRA and the MLP classification/box head: 2,665,733 trainable parameters.
- Generate RoPE from each crop's own 2D grid. Before joint attention, move all
  CLS/register tokens to a single prefix and concatenate the per-crop patch RoPE
  in exactly the same order as the patch tokens. Prefix tokens are never decoded
  as spatial patches. No DINOv2 absolute-position interpolation is used.
- `tracker.forward_features(...)` returns separately normalized `z`, `d`, `x`
  patch features. The head consumes `x` through the standard `forward` interface.
  These are joint-attention features. For independent or isolated features, use
  `extract_feature_bundle`; do not treat joint template outputs as fixed history.
  These features are available to later history, future-prediction, reconstruction
  and decision modules without changing the tracking loss or embedding logic.
- `backbone.encode_crops({name: TrackingCrop(...)})` supports named crops and
  modalities. After `backbone.add_modality('tir')`, a later tracker can pass both
  historical templates and only the available current-search crops. The new TIR
  shallow adapters are copied from learned RGB adapters, with independent storage;
  frozen DINOv3 weights, deep LoRA and role embeddings stay shared.

The RGB entry point does not implement a missing-modality tracker, future
prediction, reconstruction, decision policies or RGB-T fusion. The generic crop
interface and transfer export prepare their initialization. A later tracker still
needs availability routing, both-missing bypass, per-modality template retention
and a fusion/reconstruction readout, matching the intended research protocol.

## Official dependency and weights

Use Python >= 3.11 and PyTorch >= 2.7.1. The implementation uses the official source
directly, pinned to `6876159a11b4df116f30f667f8c9888617df0751`:

```sh
git clone https://github.com/facebookresearch/dinov3.git ../dinov3
git -C ../dinov3 checkout 6876159a11b4df116f30f667f8c9888617df0751
```

On this host it is already prepared at `/home/projects/dinov3`. It is a separate
dependency checkout, not vendored into GOLAv2. Its license remains the DINOv3
license. The source path and expected revision are configured in
`config/DINOv3RGB/dinov3/config.yaml`; relative paths resolve from the GOLAv2 root.

The default local pretrained file is:

```text
pretrained_models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

The loader initializes the official architecture and strictly loads the full
official checkpoint. It performs no implicit model downloads. `--weight_path`
instead expects a complete trained DINOv3RGB tracker checkpoint in safetensors
format, which the project saves with the filename `model.bin`.

## Data and launch

Set `GOT10k_PATH` in `consts.yaml` to the directory containing `train/` and the
optional `val/`. This host is configured for `/root/autodl-tmp/datasets/got-10k`.
Each official split must include `list.txt`; do not silently substitute a partial
directory listing. Training uses only GOT-10k's train split. Default training does
not require validation or any RGB-T dataset. The validation mixin explicitly uses
the official val split and does not affect training.

Run from the GOLAv2 root. First perform read-only setup checks:

```sh
python check_dinov3_rgb_setup.py
```

The following command starts formal training; it has **not** been submitted:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 python main.py DINOv3RGB dinov3 \
  --device cuda --disable_wandb \
  --output_dir ./output/dinov3_rgb_got10k
```

Defaults are 100 epochs, 131,072 triplets per epoch, global batch 16, AdamW with
LR 1e-4 and weight decay 0.05, cosine decay to 1e-6 with two warmup epochs, gradient
clipping at 1, and BF16 AMP. LoRA, embeddings and head use the same base LR;
one-dimensional parameters and type embeddings have zero weight decay. The loss
is IoU-aware BCE plus GIoU. These are initial settings, not tuned results.
The existing interval sampler supplies `z/d/x`; this RGB pretraining sampler does
not enforce chronological ordering of all three roles. Later causal-history or
future-prediction training must introduce an explicitly ordered sequence sampler.

For an explicitly requested short training smoke run, append `--mixin_config smoke`:
one epoch, eight triplets, global batch two, zero workers and no LR warmup. This
still runs optimizer steps; it is distinct from the forward/backward checks already
performed during implementation.

Checkpoints include the full frozen backbone, adapters, embeddings, head and an
architecture tensor that validates layer split, grids, rank, alpha and dropout.
They are intentionally self-contained and are not merged into DINOv3. Each epoch
can save model and application state, keeping the latest three epoch checkpoints.
Resume with both files from the same checkpoint directory:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main.py DINOv3RGB dinov3 --device cuda --disable_wandb \
  --weight_path /path/to/checkpoint/epoch_N/model.bin \
  --resume /path/to/checkpoint/epoch_N/state.pth \
  --output_dir ./output/dinov3_rgb_got10k_resume
```

Check the actual application-state filename in the checkpoint directory. New RNG
snapshots use primitive NumPy state values so PyTorch's default weights-only load
can read them. Older pickle-based application snapshots may require conversion.

Optional validation after obtaining trained RGB weights and the official val split:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main.py DINOv3RGB dinov3 --eval --mixin_config evaluation \
  --device cuda --disable_wandb --weight_path /path/to/model.bin \
  --output_dir ./output/dinov3_rgb_got10k_val
```

## Export for RGB-T initialization

```sh
python export_dinov3_rgb_checkpoint.py /path/to/model.bin \
  /path/to/dinov3_rgbt_init.bin --rgbt
```

This retains the entire RGB model and adds copied `tir` shallow adapters and a TIR
modality embedding. Metadata records the architecture, official source revision
and source checkpoint SHA-256. It initializes a later RGB-T model; it is not a
trained RGB-T checkpoint. The receiving backbone should add TIR before loading:

```python
tracker.backbone.add_modality('tir')
tracker.load_state_dict(safetensors.torch.load_file('dinov3_rgbt_init.bin'))
```

Omit `--rgbt` to export an annotated full RGB checkpoint instead. Neither export
overwrites an existing destination. Do not merge the task LoRA into the frozen
backbone before transfer: keeping adapters separate preserves their modality roles.

## Validation performed

On 2026-11-01 (Asia/Shanghai), using Python 3.12.3, PyTorch 2.8.0/CUDA 12.8:

- Unit tests verify official-block equivalence at zero LoRA, unchanged K projection,
  joint RoPE/prefix alignment, actual BCE/GIoU backpropagation, frozen base weights,
  full safetensors round trips, independent copied TIR adapters, foreground labels,
  single-modality online-template decoding, RNG serialization and application-state
  component restoration. All ten model/configuration tests pass; eight additional
  diagnostics tests cover persistence, W&B history, loss/gradient equivalence,
  analysis and the 20-epoch scheduler.
- Seven existing DINOv2Missing/MMLoRAT regression tests pass.
- All 9,335 GOT-10k train sequences pass manifest, required-file, image/box count
  and label-length checks. This is a file/count check, not exhaustive image decoding.
- An RTX 5090 BF16 forward/backward check uses real frames 0/20/40 from
  `GOT-10k_Train_000001`, two augmented samples, official B/16 weights and the actual
  BCE/GIoU criterion. Outputs are `[2,16,16]` scores and `[2,16,16,4]` boxes; gradients
  are finite, frozen weights have no gradients. Peak allocated memory is about
  966 MiB for that batch. **No optimizer steps or formal training were run.**
- The full training application, dataset factory, optimizer, scheduler and AMP
  construct successfully using an isolated two-sequence real-data fixture, without
  entering the training loop or changing the full-dataset cache.
- Full application state (optimizer, scheduler, sampler and RNG) passes safe
  serialization and restoration after initializing the sampler order. The export
  command also passes a save/load round trip with unchanged RGB predictions.

Run the focused tests with:

```sh
python -m unittest discover -s tests -p test_dinov3_rgb.py -v
python -m unittest discover -s tests -p 'test_*missing*.py' -v
```

These checks establish implementation compatibility, not tracking accuracy or
convergence. No trained RGB tracking scores are available yet.

## Checkpoint resume fix (v1.0.1)

The `codex/fix-rgb-training-resume` branch registers the default training runner's
`get_state` / `set_state` with its checkpoint service. Application state now contains
`objects['/runner/train/trainer']`, including AdamW parameter groups, step counters
and first/second moments, trainer iteration, LR scheduler, AMP scaler and trainable
criterion state. Optional weight-decay schedulers are saved/restored as well.
Model weights remain in the paired `model.bin` file.

Application resume requires every registered component. Old v1 state files that
contain only sampler/RNG/counter state fail with a missing trainer-state error,
instead of silently restarting the optimizer and scheduler. Lost optimizer state
cannot be reconstructed from those files. Existing model weights remain valid
for evaluation or a new fine-tuning run with `--weight_path` alone.

Verification uses the real application builder, trainer, AdamW, checkpoint dumper,
safetensors weight loader and application-state loader on a small CPU network with
dropout and a trainable criterion. It saves after two epochs/six updates, builds a
fresh application, and verifies every optimizer/scheduler/AMP state field before
continuing. The next update and all six remaining updates match uninterrupted
training exactly: losses, parameters, Adam moments, LR/weight decay, AMP scale and
growth tracker. Cases cover per-iteration and per-epoch LR schedules, each with and
without a weight-decay schedule. An enabled CPU GradScaler exercises nonempty AMP
state without using the live training GPU. These cases use accumulation 1 and no
EMA; they do not claim distributed/CUDA bitwise-equivalence coverage.

Run the focused check with:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s tests -p test_training_resume.py -v
```

The repair is developed in `/home/projects/GOLAv2-resume-fix`. It is not applied to
the active `/home/projects/GOLAv2` checkout, and does not change or restart the
ongoing `c139965` training process or its existing checkpoint files.
