# Fruit Image Classification with Transfer Learning

End-to-end fruit image classification on 30 Fruits-360 classes. The project compares a custom CNN against MobileNetV2, ResNet50 and EfficientNetB0, fine-tunes the best backbone, evaluates it on a held-out test set, analyses its errors, and serves it through a Streamlit app.

> Project 04 (SKR-INT-CV-04) | seKer AI AI & Machine Learning Internship

Items marked `TODO` are filled in from your own run of the notebook.

---

## Overview

The system accepts a fruit image, runs it through a trained deep-learning model, and returns the predicted class with its confidence. A non-technical user can operate it through a web interface without reading any code.

Pipeline: data audit, exploratory analysis, preprocessing and augmentation, model comparison, fine-tuning, evaluation, error analysis, deployment.

## Business Problem

Wherever a person is paid to look at produce and decide what it is, there is a candidate for automation: sorting lines, retail checkout, catalogue onboarding and inventory counting. A trained classifier applies the same criteria to the ten-thousandth item as to the first, which makes the result consistent and auditable.

## Objectives

- Understand and audit an image dataset before training
- Build a correct preprocessing and augmentation pipeline with clean train/validation/test splits
- Compare a custom CNN with three pre-trained backbones
- Fine-tune the best backbone and quantify the improvement
- Evaluate with precision, recall, F1 and a confusion matrix, not accuracy alone
- Inspect misclassified images and explain the failures
- Ship a working inference app

## Dataset

| Field | Value |
|---|---|
| Name | Fruits-360 (Kaggle: `moltean/fruits`), `fruits-360_100x100` version |
| Source URL | https://www.kaggle.com/datasets/moltean/fruits |
| Licence | CC BY-SA 4.0 (TODO: confirm on the dataset page) |
| Images used (after cleaning) | TODO |
| Classes used | 30, chosen with a fixed seed (42) from the classes available in both `Training` and `Test` |
| Date accessed | TODO |

Class list: TODO (printed by the notebook, also saved in `artifacts/class_names.json`).

The dataset is not committed to this repository.

## Technologies

| Layer | Tools |
|---|---|
| Language | Python |
| Data handling | NumPy, Pandas |
| Image processing | Pillow |
| Visualisation | Matplotlib |
| Deep learning | TensorFlow / Keras |
| Evaluation | Scikit-learn |
| Serving | Streamlit |
| Platform | Kaggle Notebooks (GPU) |

## Project Architecture

```
Data  ->  Preprocess  ->  Augment  ->  Model  ->  Evaluate  ->  Deploy
audit     resize          train       head       val: model      Streamlit
EDA       normalise       only        training   comparison      app
          split                       fine-tune  test: once
                  ^                                  |
                  +------ retrain after error analysis
```

Repository layout:

```
.
├── fruits_classification.ipynb   training, evaluation and error analysis
├── app.py                        Streamlit app
├── requirements.txt
├── artifacts/                    trained model, class names, config, metric CSVs
└── screenshots/
```

## Data Preprocessing

| Step | What was done |
|---|---|
| Integrity scan | Unreadable files, byte-identical duplicates, files with conflicting labels, and train/test overlaps were detected and removed. Counts: TODO |
| Resizing | All images resized to 128 x 128. The source images are 100 x 100; a slightly larger input gives the pre-trained backbones more spatial detail while keeping training fast |
| Normalisation | Each backbone uses its own `preprocess_input`; the custom CNN scales pixels to [0, 1] |
| Splits | Train/validation: stratified 85/15 split of the official `Training` folder, seed 42. Test: the official `Test` folder, evaluated once at the end |
| Augmentation | Training set only: flips, rotation, zoom, translation, brightness and contrast |
| Class imbalance | Class distribution analysed; balanced class weights are applied during training. Imbalance ratio (max/min): TODO |

Note: Fruits-360 images come from turntable captures, so near-duplicate frames can exist inside the training folder and make the validation split optimistic. The test set is the reference result.

## Model Architecture

Candidates:

| Model | Role |
|---|---|
| Custom CNN | Baseline built from scratch (4 conv blocks, batch norm, global average pooling) |
| MobileNetV2 | Fast and light |
| ResNet50 | Dependable general baseline |
| EfficientNetB0 | Accuracy-per-compute balance |

Transfer-learning models use ImageNet weights, a frozen backbone, and a head of Global Average Pooling, Dense(256, ReLU), Dropout(0.3) and a softmax layer.

**Selected architecture:** TODO

**Why:** TODO. State what was considered, what was rejected, and the evidence (validation macro-F1, training time, parameter count from the comparison table).

| Model | Val accuracy | Val macro-F1 | Params (M) | Train time (min) |
|---|---|---|---|---|
| Custom CNN | TODO | TODO | TODO | TODO |
| MobileNetV2 | TODO | TODO | TODO | TODO |
| ResNet50 | TODO | TODO | TODO | TODO |
| EfficientNetB0 | TODO | TODO | TODO | TODO |

## Training Methodology

| Setting | Value |
|---|---|
| Loss | Sparse categorical cross-entropy |
| Optimiser | Adam |
| Learning rate | 1e-3 for head training, 1e-4 for fine-tuning |
| Batch size | 64 |
| Epochs | Up to 12 for the head, 20 for the custom CNN, 10 for fine-tuning, governed by early stopping |
| Early stopping | Validation loss, patience 3, best weights restored |
| LR schedule | ReduceLROnPlateau, factor 0.3, patience 2 |
| Checkpointing | Best weights saved by validation loss |
| Precision | Mixed precision (float16) on GPU |

Fine-tuning: the last 40 layers of the backbone are unfrozen, BatchNorm layers stay frozen, and training continues at the lower learning rate.

Training curve interpretation (overfitting point, convergence, changes made): TODO

## Evaluation Results

Evaluated once on the held-out test set, before and after fine-tuning.

| Metric (test) | Before fine-tuning | After fine-tuning |
|---|---|---|
| Accuracy | TODO | TODO |
| Precision (macro) | TODO | TODO |
| Recall (macro) | TODO | TODO |
| F1 (macro) | TODO | TODO |

Full per-class reports: `artifacts/classification_report_before.csv` and `artifacts/classification_report_after.csv`.

Assessment (classes handled well, classes that struggle, and why): TODO

![Metrics before and after fine-tuning](screenshots/metrics_before_after.png)
![Confusion matrix](screenshots/confusion_matrix.png)

## Error Analysis

Misclassified test images were inspected with their true and predicted labels.

- Most frequent confusions: TODO
- Likely causes (similar-looking classes, image quality, imbalance, background): TODO
- Confidence behaviour (correct vs wrong predictions, ECE): TODO
- What was tried in response: TODO

![Misclassified examples](screenshots/misclassified.png)

## Demo

Demo video: TODO (link)

The Streamlit app lets a user upload an image, then shows the predicted class, the confidence, the top-3 predictions, and an "uncertain" result when confidence falls below an adjustable threshold.

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/<username>/fruit-classification-transfer-learning.git
cd fruit-classification-transfer-learning
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

## Usage

**Run the app**

```bash
streamlit run app.py
```

Then upload a fruit image in the browser.

**Reproduce the training**

1. Open `fruits_classification.ipynb` in a Kaggle notebook with a GPU and Internet enabled.
2. Add the `moltean/fruits` dataset as an input.
3. Run all cells. Models and reports are written to `artifacts/`.
4. Set `FAST_DEV = True` in the config cell for a quick smoke test.

## Screenshots

![App: upload and prediction](screenshots/app_prediction.png)
![App: uncertain result](screenshots/app_uncertain.png)

## Limitations

The model is trained on single fruits photographed on a white background. Cluttered real-world photos, multiple fruits in one image, or fruits outside the 30 trained classes will often be misclassified, sometimes with high confidence.

## Future Improvements

- Grad-CAM heatmaps to check that the model looks at the fruit and not the background
- Training with background and lighting augmentation, or on real-world photos
- Confidence calibration (temperature scaling) and an explicit out-of-distribution check
- All 131 classes, with a study of fine-grained confusion between varieties
- A documented REST API (FastAPI) and Docker packaging
- A model card describing intended use and limits

## Acknowledgements

Dataset: Fruits-360 by Mihai Oltean, used under its original licence.
