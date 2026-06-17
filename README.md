# YOLOv8 Handheld Item Detection System

A real-time object detection system based on YOLOv8s, designed to identify **person, umbrella, knife, and gun** in handheld scenarios.

---

## Project Structure

```
YOLOv8_Handheld_Detection/
├── .vscode/                         # Editor settings (optional)
│
├── datasets/                        # Data preparation pipeline
│   ├── raw_images/                  # Original downloaded images
│   │   ├── human detection dataset/
│   │   ├── Knife_Dataset/
│   │   ├── Simuletic_Weapon_Umbrella_Dataset/
│   │   └── surveillance0-4/         # Web‑crawled images
│   ├── standardized_images/         # Uniform JPG format & renamed
│   ├── deduplicated_images/         # Duplicates removed (perceptual hash)
│   ├── cleaned_images/              # Manually filtered high‑quality images
│   ├── images/                      # Final labelled images (for training)
│   ├── labels/                      # YOLO format labels (class_id x y w h)
│   └── yolo_dataset/                # Split & ready‑to‑use dataset
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── data.yaml                # Dataset configuration file
│
├── runs/
│   └── detect/
│       ├── exp/weights/             # Trained model weights (best.pt, last.pt)
│       └── val/                     # Validation results & plots
│
├── test_data/                       # Sample images for quick demo
├── output/                          # Saved detection results
│
├── main.py                          # Training script with augmentations
├── Visual_System.py                 # PyQt5 interactive detection application
├── crawler.py                       # Web image crawler (Bing)
├── standardize_images.py            # Convert & rename images to JPG
├── remove_duplicates.py             # Perceptual hash de‑duplication
├── split_dataset.py                 # Stratified train/val/test split
├── count_labels.py                  # Label statistics 
├── test.py                          # Environment check script
├── requirements.txt                 # Python dependencies
├── README.md                        
└── LICENSE                          # License
```

---

## Installation

> **Prerequisites:**
> 1. [Miniconda](https://docs.anaconda.com/miniconda/) (recommended) – lightweight conda installer that keeps your system Python clean.

```bash
# 1. Clone the repository
git clone https://github.com/SquidwardinCh/YOLOv8_Handheld_Detection.git
cd YOLOv8_Handheld_Detection

# 2. Create a conda environment (Python 3.10 recommended)
conda create -n yolo_weapon python=3.10 -y
conda activate yolo_weapon

# 3. Install dependencies
pip install -r requirements.txt
```

`requirements.txt` should contain at least:
```text
torch>=2.0.0
torchvision>=0.15.0
ultralytics==8.4.60
opencv-python>=4.8.0
PyQt5>=5.15
matplotlib
numpy
pillow
imagehash
icrawler
pyyaml
tqdm
```

(Adjust versions based on your environment; the training was tested with PyTorch 2.5.1 + CUDA 12.1.)

---

## Requirements

| Package | Purpose |
|---|---|
| Python >=3.9,<3.13 | Runtime |
| PyTorch ≥ 2.0 | Deep learning framework |
| Ultralytics ==8.4.60 | YOLOv8 training & inference |
| OpenCV-Python ≥4.8 | Image & video processing |
| PyQt5 ≥5.15 | GUI for detection system |
| imagehash | Perceptual hash for de‑duplication |
| icrawler | Web image crawling |
| matplotlib | Visualization |
| pyyaml | YAML parsing |
| tqdm | Progress bars |

---

## Preprocessing

### Dataset acquisition

The dataset combines three public Kaggle sources and web‑crawled surveillance images:

- [CCTV Knife Detection Dataset](https://www.kaggle.com/datasets/simuletic/cctv-knife-detection-dataset)
- [Synthetic Weapon vs. Umbrella Detection Dataset](https://www.kaggle.com/datasets/simuletic/cctv-weapon-detection-rifles-vs-umbrellas)
- [Human Detection Dataset](https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset)
- Web images downloaded via Bing using `crawler.py`.

### Preprocessing pipeline

All preprocessing scripts reside in the repository root. Execute them in order:

1. **Crawl additional images (optional):**
   ```bash
   python crawler.py
   ```
   Downloads surveillance/pedestrian images and filters out low‑resolution or watermarked files.

2. **Standardize image formats:**
   ```bash
   python standardize_images.py
   ```
   Converts all images to RGB JPG with unified naming.

3. **Remove duplicates:**
   ```bash
   python remove_duplicates.py
   ```
   Uses perceptual hashing (pHash) to identify and remove near‑duplicate images.

4. **Manual cleaning (not scripted):**
   Manually inspect the `cleaned_images/` folder to discard low‑quality or irrelevant samples.

5. **Label the dataset:**
   Use [Make Sense](https://www.makesense.ai/) or any YOLO‑compatible tool to draw bounding boxes.
   Labels are saved in YOLO format:
   ```
   class_id x_center y_center width height
   ```
   All coordinates are normalised (0–1). Classes:
   - `0` – person
   - `1` – umbrella
   - `2` – knife
   - `3` – gun

6. **Split into train/val/test:**
   ```bash
   python split_dataset.py
   ```
   Performs a stratified random split (70% train, 15% val, 15% test) and generates `datasets/yolo_dataset/data.yaml`.

> After completing these steps, the final YOLO‑ready dataset is located at `datasets/yolo_dataset/`.

### Dataset statistics

After preprocessing, the dataset contains **254 labelled images** with the following class distribution (approximate):

| Class | Instance count |
|---|---|
| person | ~250 |
| umbrella | ~100 |
| knife | ~70 |
| gun | ~80 |

The dataset is deliberately imbalanced to reflect real‑world deployment scenarios.

---

## Training

The training script is `main.py`. It uses the Ultralytics YOLOv8 API and is configured for the optimal augmentation found in the experiments (`mixup=0.05`).

```bash
python main.py
```

**Default configuration inside `main.py`:**

| Parameter | Value | Description |
|---|---|---|
| `model` | `yolov8s.pt` | Pretrained YOLOv8s checkpoint |
| `data` | `datasets/yolo_dataset/data.yaml` | Dataset config |
| `epochs` | 50 | Training epochs (adjust for larger datasets) |
| `batch` | 16 | Batch size (GPU memory dependent) |
| `device` | `0` | GPU device (set to `cpu` if no GPU) |
| `workers` | 4 | Data loading workers |
| `mixup` | 0.05 | Mixup augmentation strength |
| `project` | `runs/detect` | Output directory |
| `name` | `exp` | Experiment name |


After training completes, the script automatically runs validation and prints:
- Precision, Recall, mAP@0.5, mAP@0.5:0.95
- Per‑class AP50
- Saves plots to `runs/detect/exp/`

The best weights are stored at `runs/detect/exp/weights/best.pt`.

---

## Evaluation & Inference

### Interactive detection application

The main detection interface is `Visual_System.py`, a PyQt5 GUI that supports three input modes.

**Launch:**
```bash
python Visual_System.py
```

**Usage:**
1. Click **“加载模型” (Load model)** – the default path points to `runs/detect/exp/weights/best.pt`.
2. Adjust confidence and IoU sliders.
3. Choose a detection mode:
   - **图片检测 (Image)**: select an image file, view original and detected results side‑by‑side.
   - **视频检测 (Video)**: select a video file; processed video is saved automatically to `output/`.
   - **摄像头检测 (Camera)**: opens webcam (ID 0) for real‑time detection.

Results are displayed in the right panel table with class name, confidence, and bounding box coordinates.

### Inference using command line

If you prefer CLI‑based inference, you can use the `yolo` command directly:

```bash
yolo detect predict \
    model=runs/detect/exp/weights/best.pt \
    source=test_data/sample.jpg \
    conf=0.25 \
    iou=0.45 \
    save=True
```

For video:
```bash
yolo detect predict model=runs/detect/exp/weights/best.pt source=your_video.mp4
```

For webcam:
```bash
yolo detect predict model=runs/detect/exp/weights/best.pt source=0
```

---

## Pretrained Checkpoints

We provide the **best model** from the mixup=0.05 experiment directly in the repository:

- **Path**: `runs/detect/exp/weights/best.pt`

Performance on the validation set:

| Metric | Value |
|---|---|
| Precision | 91.98% |
| Recall | 87.98% |
| mAP@0.5 | 93.93% |
| mAP@0.5:0.95 | 40.39% |

Per‑class AP50:

| Class | AP50 |
|---|---|
| person | 82.57% |
| umbrella | 95.47% |
| knife | 99.50% |
| gun | 98.17% |

> These metrics are from the validation split; due to the small dataset size, results on completely unseen data may vary.

---

## Model Architecture

The detection system is built on **YOLOv8s** (Ultralytics). Key components:

1. **Backbone**: Improved CSPDarknet with C2f modules – enhances gradient flow and feature reuse.
2. **Neck**: FPN (Feature Pyramid Network) + PAN (Path Aggregation Network) – fuses multi‑scale features for better small‑object detection.
3. **Head**: Decoupled head – separates classification and bounding box regression branches, combined with Anchor‑Free detection.
4. **Loss**: Combination of BCEWithLogitsLoss (classification), CIoU loss (box regression), and Distribution Focal Loss (boundary probability optimisation).

For a detailed explanation, refer to the associated course paper.

---

## Utility Modules

| Script | Function |
|---|---|
| `main.py` | End‑to‑end training with configurable augmentations |
| `Visual_System.py` | PyQt5 GUI for image/video/camera inference |
| `crawler.py` | Bing image crawler with automatic quality filtering |
| `standardize_images.py` | Format conversion & renaming |
| `remove_duplicates.py` | Perceptual hash‑based de‑duplication |
| `split_dataset.py` | Stratified train/val/test split |
| `count_labels.py` | Label distribution statistics |
| `test.py` | Quick environment check (PyTorch, CUDA) |

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation & Acknowledgements

If you find this work useful, please cite:

```bibtex
@misc{cai2026yolov8handheld,
  author = {Cai Mosheng},
  title = {Implementation of a Handheld Item Recognition System Based on YOLOv8},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/SquidwardinCh/YOLOv8_Handheld_Detection}
}
```

- YOLOv8 framework by [Ultralytics](https://github.com/ultralytics/ultralytics)
- Datasets provided by Simuletic and Konstantin Verner on Kaggle
- Reference CSDN blog: [斌擎科技. YOLOv8 dangerous weapon detection](https://blog.csdn.net/m0_68036862/article/details/148228691)
