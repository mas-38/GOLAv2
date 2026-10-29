# Pure DINOv2 missing-modality tracking baseline

## Version and scope

Developed on `codex/dinov2-missing-baseline`, based on upstream commit `339c737`.
The original `main` and GOLA experiment remain available for comparison. The new
`DINOv2Missing` entry point contains no LoRA/GOLA layers, rank grouping, orthogonal
loss, feature completion or diffusion. Legacy GOLA-only optimizer selection and
orthogonal loss are now explicitly enabled only by its original configuration.

## Model and protocol

- Backbone: DINOv2 ViT-B/14; full fine-tuning. Template 112 x 112, search 224 x 224.
- Keep the initial and online RGB/TIR templates, foreground/type embeddings and
  the existing MLP classification/box head. Historical templates are valid observations.
- Training samples **one state per local batch**, uniformly from `[1,0]`, `[0,1]`,
  `[1,1]`, matching SCDT's batch-level three-state sampling. Only current search
  observations are dropped. Both-missing frames are not training samples.
- Remove the missing current modality's search tokens **before patch embedding**.
  A mixed evaluation batch is grouped by availability, encoded, then restored to
  its original sample order. No missing pixels or placeholder tokens enter attention.
- RGB-only uses RGB search tokens; TIR-only uses TIR search tokens; complete frames
  use jointly encoded TIR search tokens. All cases share one prediction head.
- Both-missing evaluation frames bypass decoding and the model, preserve the last
  box/confidence and leave all templates and cropping state unchanged.
- First-frame initialization uses both real modalities, like SCDT, regardless of
  the JSON's first row. This baseline assumes both modalities exist at initialization.
- Initial templates stay fixed. Online templates update only above confidence 0.84
  and only for available modalities. **Images and foreground masks are retained
  separately per modality**. Recovery resumes normal updates.
- For transport only, a missing image slot copies the available image, matching
  SCDT. The model never embeds that copied current-search slot.
- Loss: IoU-aware BCE + GIoU. No reconstruction or orthogonal loss.
- Default learning rate: backbone/position embedding 1e-5; head/type embedding 1e-4.
  Default global training batch 8, 10 epochs. These are starting settings, not tuned results.
- Dynamic availability routing uses eager execution; torch.compile and automatic
  efficiency profiling are disabled. DDP static_graph is disabled as well.

The distinction from SCDT is intentional: this baseline predicts directly from
valid current observations and historical templates, without generating missing features.
Unlike the original SCDT template-list update, missing template slots are not overwritten.

## Run

Use the project's dependencies in `requirements.txt` (including `wandb`, even with
`--disable_wandb`, because the existing launcher imports it). Set dataset paths in
`consts.yaml`; select the evaluation dataset in `config/_dataset/test-mm.yaml`.
Use a Python/PyTorch environment with sufficient GPU memory for full fine-tuning.
Run commands from the repository root.

Training (downloads standard DINOv2 pretrained weights through the existing builder):

```sh
python main.py DINOv2Missing dinov2 --device cuda --disable_wandb --output_dir ./output/dinov2_missing
```

Do **not** pass the old GOLA/LoRA `base.bin` or grouped weights. Full tracker
checkpoints now include the backbone and are incompatible with GOLA adapters.
The `--weight_path` argument is for a trained **DINOv2Missing full checkpoint**.
The framework's ordinary safetensors loading/saving is used, with no adapter merge.

Evaluation with SCDT missing markers:

```sh
python main.py DINOv2Missing dinov2 --eval --mixin_config evaluation --device cuda --disable_wandb --weight_path /path/to/model.bin --missing_mask_file /path/to/missing_results.json --output_dir ./output/dinov2_missing_eval
```

Omit `--missing_mask_file` for complete-modality evaluation. The same option can
be provided during training to apply missing markers to the final evaluation.
It does not change random three-state training.

JSON format (sequence names must match the selected dataset):

```json
{
  "sequence_name": {
    "data": [[1, 1], [1, 0], [0, 1], [0, 0], [1, 1]]
  }
}
```

Rows include frame zero. Columns are `[RGB, TIR]`, with 1 available and 0 missing.
Missing sequence names, nonbinary flags and out-of-range frame indices fail explicitly.
Keep the same JSON for all compared methods. This implementation reads markers;
it does not generate new missing schedules. Evaluate one dataset/marker file at a time.

For ViT-L/14, append `--mixin_config large` to both training and evaluation.

## Validation

```sh
python -m unittest discover -s tests -p test_dinov2_missing.py -v
```

Five regression tests cover missing-image invariance (including NaN replacement),
mixed-batch ordering, all three training states and backbone updates, full-checkpoint
round trips, configuration routing, JSON validation, decoder skipping, selective
image/mask updates, both-missing batches and recovery.

On 2026-10-30, all five tests passed with PyTorch 2.4.0 and timm 1.0.20. A separate
full ViT-B/14 CUDA smoke check on RTX 2060 passed all three states using random
weights/inputs, actual BCE/GIoU, AMP + gradient scaling, backward and optimizer steps.
This verifies execution, not accuracy. No dataset training or benchmark results have
been produced. The local full launcher currently needs `wandb`, and `consts.yaml`
still contains server dataset paths; those pre-existing environment settings were
not changed.

## Main implementation files

- `trackit/models/methods/dinov2_missing/model.py`: pure backbone and availability routing.
- `config/DINOv2Missing/`: independent training/evaluation configuration.
- `trackit/data/methods/siamese_tracker_eval/transform/missing.py`: SCDT JSON and image loading.
- `trackit/runner/evaluation/distributed/tracker_evaluator/default/pipelines/one_stream/`:
  bypass and prediction-state handling.
- `trackit/runner/evaluation/distributed/tracker_evaluator/components/template_updater/simple.py`:
  per-modality image retention.
- `trackit/runner/evaluation/distributed/tracker_evaluator/default/pipelines/_common/online_template_foreground_indicating_mask_generation.py`:
  matching per-modality foreground-mask retention.
