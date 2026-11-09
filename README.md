# Chest X-ray Pneumonia Detection

A Google Colab notebook for a chest X-ray pneumonia-classification project. The notebook contains an image-only baseline using the Kermany Chest X-Ray dataset and an optional multimodal extension using IU X-Ray image/report data.

## Repository contents

- `notebooks/final_code.ipynb` — end-to-end project notebook.

## Methods covered

### Part A — Image-only baseline (Kermany Chest X-Ray)
- Dataset discovery and class-label validation for `NORMAL` and `PNEUMONIA`
- Train/validation/test split handling
- Image preprocessing and training-time augmentation
- ImageNet-pretrained DenseNet121 transfer learning
- Binary classification with class-imbalance weighting
- Validation-based checkpoint selection and early stopping
- Held-out test evaluation, confusion matrix, and prediction export
- Grad-CAM visualizations for model interpretability

### Part B — Multimodal extension (IU X-Ray)
- Combines chest X-ray image features with clinical indication text
- Uses DenseNet121 image features and Bio_ClinicalBERT text embeddings
- Includes a comparison with an image-only model on the same study-level split
- The notebook documents limitations around data leakage and patient-level separation

## Run the notebook

1. Open `notebooks/final_code.ipynb` in Google Colab.
2. Select a GPU runtime if available.
3. Place the Kermany dataset ZIP or extracted folder in Google Drive and update `DRIVE_DATA` in the notebook.
4. Run the cells in order.
5. For the optional IU X-Ray section, place the required IU X-Ray CSV files and images in Drive and update `MM_DRIVE_DIR`.
6. Run the evaluation and export sections to generate metrics, plots, predictions, and model checkpoints.

## Results

**No experimental metrics or output artifacts are included in this initial repository upload.** The uploaded notebook contains code but has no saved cell outputs. Run it in Colab to generate actual evaluation results; do not interpret the notebook as having already achieved any particular accuracy, AUROC, precision, recall, or F1-score.

## Important limitations

- Model predictions are research outputs and are **not medical diagnoses**.
- The Kermany dataset's provided test split is kept separate when the expected folder structure is available.
- The IU X-Ray extension splits by study UID; patient-level separation is not guaranteed by that split.
- Grad-CAM is an exploratory visualization and does not establish clinically valid model reasoning.
- Dataset access, licensing, and usage conditions remain the responsibility of the user.

## Main tools

Python, PyTorch, torchvision, scikit-learn, pandas, NumPy, Matplotlib, Seaborn, and Transformers (for the multimodal extension).
