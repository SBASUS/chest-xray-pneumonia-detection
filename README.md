[README.md](https://github.com/user-attachments/files/34632918/README.md)
# Chest X-ray Pneumonia Detection

A Google Colab deep-learning project for chest X-ray classification. The notebook contains a primary **image-only pneumonia classification pipeline** using the Kermany/Kaggle Chest X-Ray dataset and an optional **multimodal image + clinical-text experiment** using the Indiana University (IU) Chest X-Ray dataset.

> **Research/education only:** This project is an experimental machine-learning workflow. It is not a medical device and must not be used to diagnose patients, guide treatment, or make clinical decisions.

## Project overview

| Experiment | Dataset | Task | Model |
|---|---|---|---|
| Main pipeline | Kermany / Kaggle Chest X-Ray | Binary classification: `NORMAL` vs `PNEUMONIA` | ImageNet-pretrained DenseNet121 |
| Optional Part B | IU Chest X-Ray | Weak-label abnormality classification using image and pre-diagnostic indication text | Image-only ResNet18 baseline vs ResNet18 + Bio_ClinicalBERT |

The Kermany dataset supplies labelled image folders and is used for the main image-only model. The IU experiment pairs frontal X-ray images with the report's `indication` field. The code uses report-derived `MeSH` and `Problems` fields to construct labels, but does not feed findings or impression text into the model.

## Repository contents

```text
.
├── README.md
└── final_code-3.ipynb   # Main Google Colab notebook
```

If you use a different notebook filename in your repository, update the filename above accordingly.

## Requirements

The notebook is designed for **Google Colab**. It uses:

- Python
- PyTorch and torchvision
- NumPy and pandas
- scikit-learn
- Matplotlib and seaborn
- Pillow
- `transformers` (for the IU multimodal experiment)
- `kagglehub` (for downloading the IU dataset in Part B)

The notebook installs `scikit-learn` and `seaborn` in the setup cell. It installs `transformers` and `kagglehub` in the optional IU section. PyTorch, torchvision, NumPy, pandas, Matplotlib and Pillow are generally available in Colab's runtime; if running elsewhere, install compatible versions yourself.

A GPU runtime is recommended, particularly for DenseNet121 training and Bio_ClinicalBERT. In Colab, select **Runtime → Change runtime type → T4 GPU** (or another available GPU) before running the notebook. CPU execution is possible but can be considerably slower.

## Dataset setup

### 1. Kermany / Kaggle Chest X-Ray dataset

The main pipeline reads the dataset from **Google Drive**. Download the Kermany Chest X-Ray dataset separately and place either the ZIP archive or extracted dataset folder in your Drive.

Example locations:

```text
MyDrive/chest_xray.zip
```

or

```text
MyDrive/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/                 # optional
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The notebook also accepts a flat folder structure:

```text
chest_xray/
├── NORMAL/
└── PNEUMONIA/
```

For a flat layout, the notebook creates a stratified train/validation/test split. For a split layout, the supplied test directory is held out for final evaluation. The notebook handles common nested-folder layouts and ignores `__MACOSX` folders.

**Set the dataset path:** In the Kermany data-loading cell, edit `DRIVE_DATA` to match the ZIP file or extracted folder in your Google Drive:

```python
DRIVE_DATA = Path('/content/drive/MyDrive/chest_xray')
```

For a ZIP file, use its full path, for example:

```python
DRIVE_DATA = Path('/content/drive/MyDrive/chest_xray.zip')
```

By default, `COPY_TO_LOCAL = True` copies an extracted Drive folder to Colab's local disk for faster training. Set it to `False` if you prefer to read directly from Drive.

### 2. IU Chest X-Ray dataset (optional multimodal experiment)

Part B downloads the IU dataset using KaggleHub:

```python
kagglehub.dataset_download('raddar/chest-xrays-indiana-university')
```

No manual image upload is required by this section. It reads `indiana_reports.csv` and `indiana_projections.csv`, joins them using `uid`, keeps frontal projections and resolves image paths under `images_normalized`.

The IU experiment uses the report's `indication` field as its text input. Its labels are derived from report metadata:

- `abnormal`: whether the `MeSH` field is not `normal`.
- `pneumonia`: whether the combined `MeSH` and `Problems` text contains the word `pneumonia`.

The executed notebook sets `LABEL = 'abnormal'`. The pneumonia keyword label was found in only 39 records in the notebook run, so it should not be treated as a sufficiently supported pneumonia classification target without further label validation.

## How to run in Google Colab

1. Upload `final_code-3.ipynb` to Google Drive or open it directly in Google Colab.
2. Select a GPU runtime if available.
3. Run the notebook from the beginning, in order.
4. When prompted by the notebook, mount Google Drive and make sure the Kermany dataset path in `DRIVE_DATA` is correct.
5. Check the printed image counts and class distribution. Confirm that both classes and all required splits are present.
6. Run the DenseNet121 training and evaluation cells.
7. Review the test metrics, confusion matrix, ROC curve, precision-recall curve and Grad-CAM example.
8. Use `predict_xray(path)` to test an image with the trained model, or `predict_from_drive(path)` to process a Drive image/folder.
9. To run the optional multimodal experiment, continue to **Part B** and execute its cells in order. This section installs the additional packages and downloads the IU dataset.

Do not skip the data preparation or model-training cells unless you have already loaded a compatible trained checkpoint and restored the required variables in the current runtime.

## Model details

### Main Kermany pipeline

- Input size: `224 × 224` RGB images.
- Preprocessing: ImageNet mean and standard-deviation normalisation.
- Training augmentation: random rotation, horizontal flip and mild brightness/contrast jitter.
- Encoder: ImageNet-pretrained DenseNet121.
- Fine-tuning: the final dense block and final normalisation layer are unfrozen; earlier feature layers are frozen.
- Classifier: dropout followed by a single-logit output layer.
- Loss: weighted binary cross-entropy with logits.
- Optimiser: AdamW.
- Learning rate: `1e-4`.
- Maximum epochs: `5`, with validation-based checkpoint selection and early stopping.
- Default classification threshold: `0.5`.

### Optional IU multimodal pipeline

The executed code defines two models on the same group-based IU split:

1. **Image-only baseline:** ResNet18 with a binary output layer.
2. **Multimodal model:** ResNet18 image features (512 dimensions) concatenated with the `[CLS]` representation from Bio_ClinicalBERT (768 dimensions), followed by a classification head.

The text is tokenised to a maximum sequence length of 48. The model is trained with weighted binary cross-entropy and AdamW. In the submitted code, Bio_ClinicalBERT parameters are not explicitly frozen; the text encoder is therefore trainable during the multimodal training loop.

## Results recorded in the submitted notebook

These are the outputs saved in the uploaded notebook at the time it was provided. Re-running the notebook may produce different values, especially for the IU experiment.

### Kermany DenseNet121 test results

| Metric | Result |
|---|---:|
| Accuracy | 0.8846 |
| Precision | 0.8457 |
| Sensitivity / Recall | 0.9974 |
| Specificity | 0.6966 |
| F1-score | 0.9153 |
| AUROC | 0.9515 |
| AUPRC | 0.9619 |

Confusion matrix (rows = actual class; columns = predicted class):

|  | Predicted NORMAL | Predicted PNEUMONIA |
|---|---:|---:|
| Actual NORMAL | 163 | 71 |
| Actual PNEUMONIA | 1 | 389 |

The model showed high sensitivity in this test run but a lower specificity: 71 NORMAL images were classified as PNEUMONIA. Accuracy should therefore not be considered in isolation.

### IU X-Ray results (weak-label abnormality task)

| Model | Test AUROC | Test AUPRC | Accuracy at 0.5 |
|---|---:|---:|---:|
| Image-only ResNet18 | 0.805 | 0.882 | 0.730 |
| ResNet18 + Bio_ClinicalBERT | 0.801 | 0.886 | 0.706 |

The notebook also contains an earlier run with different results (image-only AUROC 0.792; multimodal AUROC 0.816), showing run-to-run variation. In the later recorded run, shuffling the indication text left AUROC unchanged at 0.801. The results do **not** establish a consistent improvement from adding text.

## Outputs

The main pipeline saves results under:

```text
/content/pneumonia_project_outputs/
```

The notebook creates a downloadable ZIP archive:

```text
/content/pneumonia_project_outputs.zip
```

It can also copy the generated files to:

```text
MyDrive/pneumonia_project_outputs/
```

Expected outputs include:

| File | Description |
|---|---|
| `best_densenet121_pneumonia.pth` | Best validation-selected DenseNet121 weights |
| `densenet121_pneumonia_checkpoint.pth` | Model state and basic model/preprocessing metadata |
| `test_metrics.csv` / `test_metrics.json` | Held-out test metrics |
| `test_predictions.csv` | Per-image true labels, predicted labels and probabilities |
| `training_history.csv` | Epoch-wise training and validation metrics |
| `training_curves.png` | Training and validation curves |
| `confusion_matrix.png` | Test confusion matrix |
| `roc_curve.png` | ROC curve |
| `precision_recall_curve.png` | Precision-recall curve |
| `gradcam_example.png` | Example Grad-CAM visualisation |
| `project_metadata.json` | Dataset, architecture and experiment metadata |

## Important limitations

- The Kermany folder structure does not provide reliable patient identifiers. The image-level split cannot guarantee that images from the same patient are separated across train, validation and test sets.
- Dataset-specific acquisition patterns, class imbalance, duplicate images and label noise can affect performance.
- The IU abnormality and pneumonia labels are derived from report metadata rather than independently verified labels. The pneumonia keyword label is particularly sparse.
- The IU results vary between runs and the text-shuffling check did not show a measurable AUROC decrease. A multimodal performance benefit has not been demonstrated.
- Grad-CAM is a qualitative visualisation tool; it does not prove that the model uses clinically valid evidence.
- The reported results are dataset-specific and are not evidence of performance on external hospitals or patient populations.

## Reproducibility and citation

For reproducible experiments, record the notebook version, dataset version, Colab runtime, package versions, random seed, split definitions and complete metrics for each run. Do not commit patient data, private credentials, Kaggle API tokens or large datasets to a public repository.

If you use or adapt this project, cite the original datasets and the relevant model/library publications in your academic report. Check each dataset's licence and usage terms before redistribution.

## Disclaimer

This repository is intended for academic learning and research experimentation only. Predictions are model outputs, not medical diagnoses. Do not use this software for clinical decision-making.
