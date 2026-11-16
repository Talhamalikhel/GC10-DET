# GC10-DET: Steel Surface Defect Detection

An object detection project using **Faster R-CNN (ResNet-50 + FPN)** to detect and localize 10 types of defects on steel surfaces.

Built with PyTorch and Torchvision, this project explores the complete object detection pipeline, from dataset preparation to model training and evaluation.

## Dataset

- **Source:** [GC10-DET – Kaggle](https://www.kaggle.com/datasets/alex000kim/gc10det)
- **Images:** 2,294 after cleaning and deduplication
- **Classes:** 10 steel surface defect categories
- **Annotations:** PASCAL VOC XML
- **Split:** 70% training, 15% validation, 15% testing

**Preprocessing included:**
- Converting grayscale images to 3 channels.
- Removing images without annotations and duplicates.
- Correcting approximately 130 annotation files with class-label typos.

## Model & Training

- **Architecture:** Faster R-CNN with ResNet-50 + FPN
- **Pretrained weights:** COCO
- **Framework:** PyTorch / Torchvision
- **Optimizer:** SGD (learning rate: 0.005, momentum: 0.9)
- **Scheduler:** StepLR
- **Training:** 7 epochs on Google Colab (T4 GPU)

The classification head was replaced to support 10 defect classes plus background.

## Results

Evaluated using `torchmetrics.MeanAveragePrecision` (COCO-style).

| Metric | Validation | Test |
|---|---:|---:|
| mAP (0.5:0.95) | 0.49 | 0.49 |
| mAP@50 | 0.84 | 0.82 |
| mAP@75 | 0.50 | — |

Validation and test results are close at the aggregate level, providing an encouraging baseline.

## Limitations & Future Work

- No data augmentation in the baseline.
- Imbalanced defect categories.
- Lower performance on visually challenging classes, particularly `inclusion` and `crease`.
- Limited training duration (7 epochs).

**Future improvements:** Data augmentation, class-imbalance handling, longer training, hyperparameter tuning, and experiments with alternative detection architectures.

## Repository Contents

- `gc10_det_faster_rcnn.ipynb` — Data preprocessing, annotation cleaning, model training, and evaluation.
- `requirements.txt` — Project dependencies.

## Acknowledgements

GC10-DET dataset: [Kaggle](https://www.kaggle.com/datasets/alex000kim/gc10det).

