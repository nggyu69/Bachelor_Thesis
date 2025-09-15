## Dataset Pipeline Overview

This project uses a 4-step pipeline driven by a shared `config.json`:

1. Generate (`Generate.py`): Renders synthetic images and COCO annotations to `paths.output_base_dir`.
2. Annotate (`Annotate.py`): Converts COCO to OBB labels, creates multiple edge/style views, and prepares datasets in white/black splits.
3. Train (`train.py`): Trains YOLO-OBB models for selected styles using dataset paths and training settings.
4. Benchmark (`Benchmark_Metrics.py`): Evaluates trained models on test sets and exports per-class metrics and LaTeX tables.

Only the parameters listed below are read from `config.json`. All other options remain in-script.

### Generate.py — Config parameters used
- `paths.scene_blend_file`: Absolute path to the Blender scene `.blend` file.
- `paths.category_map_file`: Path to the category map JSON for annotation names.
- `paths.output_base_dir`: Base directory where generated COCO dataset is written.
- `model.color`: Hex color (e.g., `#0f0f13`) applied to the imported model.
- `model.model_path`: Absolute path to the `.stl` model to import.
- `timing.start_time`: Optional epoch start time used for progress reporting. If `null`, uses current time.
- `timing.initial_count`: Baseline image count for generation progress.

Run: `blenderproc run /home/reddy/Bachelor_Thesis/Generate.py`

### Annotate.py — Config parameters used
- `paths.output_base_dir`: Base dir to read generated COCO data from.
- `paths.dataset_white_dir`: Destination root for "white" dataset variant.
- `paths.dataset_black_dir`: Destination root for "black" dataset variant.

Run: `python /home/reddy/Bachelor_Thesis/Annotate.py`

### train.py — Config parameters used
- `training.model`: Style to train (e.g., `control`, `canny`, ...).
- `training.dataset_path`: Dataset root used for training.
- `training.model_size`: YOLO size key (e.g., `n`, `s`, `m`).
- `training.epochs`: Number of epochs.
- `training.imgsz`: Image size.
- `training.patience`: Early stopping patience.
- `training.batch`: Batch size.
- `training.project_suffix`: Suffix segment for the output training directory name.
- `training.yolo_config_pattern`: Pattern for model cfg, e.g., `yolo11{size}-obb.yaml`.
- `training.yolo_weights_pattern`: Pattern for pretrained weights, e.g., `yolo11{size}.pt`.

Run: `python /home/reddy/Bachelor_Thesis/train.py`

### Benchmark_Metrics.py — Config parameters used
- `paths.test_sets_dir`: Root folder containing test sets (with `images/`, `labels/`, `data.yaml`).
- `paths.trains_base_dir`: Base directory of trained model runs (used to locate `weights/best.pt`).
- `paths.benchmarks_base_dir`: Base directory to write benchmark outputs and LaTeX.

Run: `python /home/reddy/Bachelor_Thesis/Benchmark_Metrics.py`
