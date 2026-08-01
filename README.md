# GC10-DET Steel Surface Defect Detection — Faster R-CNN

Object detection on the GC10-DET dataset (metal surface defects), using a Faster R-CNN (ResNet-50 + FPN) backbone fine-tuned with transfer learning. This was my first end-to-end computer vision detection project, built to understand the full pipeline — data auditing, annotation cleaning, dataset/dataloader design, model architecture, training, and evaluation — not just to chase the best possible score.

## Dataset

- **Source**: [GC10-DET on Kaggle](https://www.kaggle.com/datasets/alex000kim/gc10det)
- **10 defect classes**: punching_hole, welding_line, crescent_gap, water_spot, oil_spot, silk_spot, inclusion, rolled_pit, crease, waist_folding
- **2294 annotated images** (after removing 6 images with no annotation and deduplicating 12 images that appeared in multiple class folders)
- Grayscale images, converted to 3-channel for compatibility with ImageNet-pretrained weights
- Annotations in PASCAL VOC XML format
- Notable data quality issue found and fixed: ~130 annotation files had a typo in the class-10 label (`10_yaozhed` instead of `10_yaozhe`), normalized before training
- Split: 70% train / 15% validation / 15% test, stratified by each image's primary defect class

## Approach

- **Model**: `fasterrcnn_resnet50_fpn` (torchvision), pretrained on COCO, with the classification head replaced for 11 classes (10 defects + background)
- **Grayscale handling**: repeated single channel to 3 channels rather than modifying the first conv layer, to preserve the ability to use pretrained weights on a small dataset
- **Optimizer**: SGD, lr=0.005, momentum=0.9, weight_decay=0.0005, with StepLR decay
- **Training**: 7 epochs on Google Colab (T4 GPU)
- No data augmentation in this baseline (identified as a clear next step — see Limitations)

## Results

Evaluated with `torchmetrics`' `MeanAveragePrecision` (COCO-style mAP).

| Metric | Validation | Test |
|---|---|---|
| mAP (IoU 0.5:0.95) | 0.49 | 0.49 |
| mAP@50 | 0.84 | 0.82 |
| mAP@75 | 0.50 | — |

Validation and test scores are close, indicating the model generalizes consistently rather than overfitting to validation.

**Per-class mAP:**

| Class | Validation | Test |
|---|---|---|
| punching_hole | 0.51 | 0.65 |
| welding_line | 0.58 | 0.59 |
| crescent_gap | 0.71 | 0.71 |
| water_spot | 0.65 | 0.67 |
| oil_spot | 0.58 | 0.48 |
| silk_spot | 0.42 | 0.46 |
| inclusion | 0.45 | 0.23 |
| rolled_pit | 0.32 | 0.48 |
| crease | 0.30 | 0.32 |
| waist_folding | 0.40 | 0.34 |

## Limitations & next steps

- **No data augmentation** — the dataset is small (1604 training images) and imbalanced (74–884 images per class); augmentation (flips, targeted oversampling of rare classes) would likely help, especially for the weakest classes below.
- **Class imbalance** — `crease` (74 images) and `rolled_pit` (85 images) are the weakest-performing classes, consistent with having the least training data.
- **`silk_spot` underperforms despite having the most training data (884 images)** — suggests this defect type may be visually harder to localize precisely (e.g. low contrast, subtle boundaries) rather than a pure data-quantity problem. Worth further investigation.
- **`inclusion` dropped notably on test (0.45 → 0.23)** despite reasonable validation performance — worth investigating whether the test split happened to contain harder examples of this class, or whether the model is less robust on this class than validation suggested.
- **Only 7 training epochs** — a longer training run with a proper learning rate schedule would likely close some of this gap.
- Trained closer to "from scratch" on the detection head rather than just fine-tuning, as a deliberate choice to build deeper understanding of the architecture, at some cost to final performance versus a more tuned setup.

This baseline is treated as a deliberately "good enough to learn from" first project. A follow-up project will apply the lessons learned here (augmentation, better hyperparameter tuning, possibly a modern one-stage/transformer detector) aiming for stronger results.

## Files

- `gc10_det_faster_rcnn.ipynb` — full notebook: data audit, preprocessing, Dataset/DataLoader, model, training, evaluation
- `requirements.txt` — dependencies

## Acknowledgements

Dataset: Lv et al., GC10-DET (Guangdong Metal Surface Defect Dataset), sourced via Kaggle.
