# leakage-free-breast-cancer-classifier
Leakage-free DenseNet201 for breast cancer binary classification (Benign vs Malignant) on BreakHis dataset using patient-level split. Achieves 82.28% true test accuracy on unseen patients.
# 🧬 BreakHis Breast Cancer Classification - Patient-Level Split

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.10+-red.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-82.28%25-brightgreen.svg)
![Model](https://img.shields.io/badge/Model-DenseNet201-9cf.svg)
![Dataset](https://img.shields.io/badge/Dataset-BreakHis-9cf.svg)
![Split](https://img.shields.io/badge/Split-Patient--Level-purple.svg)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Supported-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

**Leakage-Free DenseNet201 Model for Breast Cancer Binary Classification (Benign vs Malignant)**

</div>

---

## 📌 Overview

This project implements a **leakage-free DenseNet201 model** for binary classification of breast cancer histopathology images using the **BreakHis** dataset. Unlike previous approaches that used random image-level splits (which cause data leakage), this model uses **patient-level splits** where all images from a single patient are assigned to only one split. The model achieves **82.28% test accuracy**, which represents the **true, reliable performance** for unseen patients.

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| **Test Accuracy (Patient-Level)** | **82.28%** |
| **Test Accuracy (Original)** | 82.14% |
| **Best Validation Accuracy** | 87.80% |
| **Benign Precision** | 0.80 |
| **Benign Recall** | 0.76 |
| **Malignant Precision** | 0.84 |
| **Malignant Recall** | 0.86 |
| **F1-Score (Weighted)** | 0.82 |

---

## ⚠️ Why Patient-Level Split Matters

### The Data Leakage Problem

In the BreakHis dataset, each patient has multiple images (different magnifications, different slides). When we randomly split **images** (image-level split), images from the same patient can end up in both training and test sets. This causes **data leakage** — the model "memorizes" the patient rather than learning the disease, inflating accuracy artificially.

### The Solution

**Patient-level split** ensures that all images from a single patient go to only one split (train, validation, or test). This gives a **true measure** of how the model performs on **unseen patients**.

---

## 📊 Performance Comparison

| Model | Split Method | Test Accuracy | Reliability |
|-------|--------------|---------------|-------------|
| CNN Baseline | Image-Level | 88.24% | ❌ Leakage |
| DenseNet121 | Image-Level | 91.15% | ❌ Leakage |
| Ensemble (3 Models) | Image-Level | 90.99% | ❌ Leakage |
| DenseNet201 Optimized | Image-Level | **96.21%** | ❌ Leakage |
| **DenseNet201 Patient-Level** | **Patient-Level** | **82.28%** | ✅ **True** |

> **Note:** The ~14% difference between image-level (96.21%) and patient-level (82.28%) accuracy reveals the amount of data leakage in the image-level approach.

---

## 📊 Dataset Overview

The **BreakHis** (Breast Cancer Histopathological Images) dataset consists of microscopic images of breast tumor tissues.

| Feature | Description |
|---------|-------------|
| Total Images | 7,909 |
| Total Patients | 82 |
| Classes | Benign (2,480) / Malignant (5,429) |
| Magnifications | 40X, 100X, 200X, 400X |
| Image Size | 224 × 224 × 3 (RGB) |
| Format | PNG / JPG / JPEG |

### Patient-Level Split Distribution

| Split | Patients | Images | Benign | Malignant |
|-------|----------|--------|--------|-----------|
| **Train** | 56 | 5,213 | 1,378 | 3,835 |
| **Validation** | 13 | 1,319 | 530 | 789 |
| **Test** | 13 | 1,377 | 572 | 805 |
| **Total** | **82** | **7,909** | **2,480** | **5,429** |

### Class Distribution

```
Benign    ████████████████░░░░░░░░░░░░░░░░  2,480 (31.4%)
Malignant ████████████████████████████████  5,429 (68.6%)
```

> **Note:** The dataset is imbalanced (Malignant = 69%, Benign = 31%). Class weights were applied to address this.

---

## 🧠 Model Architecture

### DenseNet201 Architecture

```python
base_model = DenseNet201(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

### Layer Details

| Item | Value |
|------|-------|
| Total Parameters | 19,444,801 (74.18 MB) |
| Trainable Parameters | 1,118,977 (4.27 MB) |
| Non-trainable Parameters | 18,325,824 (69.91 MB) |

---

## 🔧 3-Phase Training Strategy

### Phase 1: Frozen Training (30 Epochs)

```python
model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
# Result: Validation Accuracy 85.52%
```

### Phase 2: Fine-Tuning (25 Epochs)

```python
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=Adam(learning_rate=5e-5), loss='binary_crossentropy', metrics=['accuracy'])
# Result: Validation Accuracy 87.80%
```

### Phase 3: Final Fine-Tuning (15 Epochs)

```python
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
# Result: Test Accuracy 82.14%
```

---

## 📈 Advanced Data Augmentation

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.6, 1.4],
    channel_shift_range=15,
    fill_mode='reflect'
)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Train/Val/Test Split | 70% / 15% / 15% (Patient-Level) |
| Optimizer | Adam |
| Class Weights | Balanced (Benign: 1.89, Malignant: 0.68) |
| Early Stopping Patience | 8 |
| ReduceLR Patience | 4 |
| Mixed Precision | float16 |

---

## 🎯 Test Time Augmentation (TTA)

```python
def predict_with_tta(img, n_aug=5):
    preds = [model.predict(img)[0][0]]
    for _ in range(n_aug):
        aug = apply_random_augmentation(img)
        preds.append(model.predict(aug)[0][0])
    return np.mean(preds)
```

**TTA Result:** Improved accuracy from 82.14% to **82.28%**

---

## 📈 Results

### Final Classification Report (TTA)

```
              precision    recall  f1-score   support
      Benign       0.80      0.76      0.78       572
   Malignant       0.84      0.86      0.85       805

    accuracy                           0.82      1377
```

### Confusion Matrix

```
[[437 135]   ← Benign: 437 correct, 135 misclassified
 [109 696]]  ← Malignant: 696 correct, 109 misclassified
```

### Training Progress

| Phase | Best Val Accuracy | Best Val Loss |
|-------|-------------------|---------------|
| Phase 1 (Frozen) | 85.52% | 0.3561 |
| Phase 2 (Fine-Tuning) | **87.80%** | **0.3320** |
| Phase 3 (Final FT) | 87.58% | 0.3418 |
| **Final Test** | **82.28%** (TTA) | - |

---

## 🚀 How to Run

### Step 1: Setup Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Dataset Preparation

Place your dataset zip file in Google Drive:

```
/content/drive/MyDrive/dataset_cancer_v1.zip
```

### Step 3: Run the Scripts in Order

```bash
# 1. Create patient-level split
python 01_create_patient_split.py

# 2. Train Phase 1 (Frozen)
python 02_phase1_frozen.py

# 3. Train Phase 2 (Fine-Tuning)
python 03_phase2_finetuning.py

# 4. Train Phase 3 + TTA + Evaluation
python 04_phase3_final_tta.py
```

### Step 4: Outputs

- Patient-level split CSV
- Trained models (`.keras`) in Drive
- Classification report and confusion matrix
- Test Accuracy: **82.28%** (Patient-Level)

---

## 🛠️ Requirements

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

### Dependencies

```
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
```

---

## 📁 Repository Structure

```
breakhis-patient-level-densenet201/
│
├── README.md
├── requirements.txt
├── 01_create_patient_split.py
├── 02_phase1_frozen.py
├── 03_phase2_finetuning.py
├── 04_phase3_final_tta.py
├── splits_patient_level.csv
└── .gitignore
```

---

## 💡 How to Use the Trained Model

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model('densenet201_patient_level_final.keras')
img = image.load_img('sample.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]
result = "Malignant" if prediction > 0.5 else "Benign"
print(f"Prediction: {result} (confidence: {prediction:.2f})")
```

---

## 🔍 Interpretation of Results

| What It Means | Details |
|---------------|---------|
| **True Accuracy (82.28%)** | Reliable performance on unseen patients |
| **Benign Recall (76%)** | 24% of Benign cases are missed |
| **Malignant Recall (86%)** | 14% of Malignant cases are missed |
| **No Data Leakage** | All patients are strictly separated |

---

## 🎓 Key Lessons Learned

### What Worked

- ✅ **Patient-level split** revealed true model performance
- ✅ **3-phase training** enabled stable learning
- ✅ **Mixed precision** sped up training by ~2-3x
- ✅ **Test Time Augmentation** added slight improvement
- ✅ **Saving checkpoints to Drive** prevented data loss

### The Data Leakage Revelation

| Approach | Accuracy | Reliability |
|----------|----------|-------------|
| Image-Level Split | 96.21% | ❌ Artificially inflated |
| **Patient-Level Split** | **82.28%** | ✅ **True performance** |

> **Critical Insight:** The ~14% gap between image-level and patient-level accuracy shows how much data leakage inflated previous results.

---

## ⚠️ Limitations

- **Limited Patients:** Only 82 patients — more data needed for robust generalization
- **Class Imbalance:** Malignant = 58 patients vs Benign = 24 patients
- **No Cross-Validation:** Single train/val/test split
- **Benign Recall:** 76% — needs improvement for clinical use

---

## 🚀 Future Improvements

| Priority | Improvement | Expected Impact |
|----------|-------------|-----------------|
| 1 | **External Data** (Camelyon, BACH) | +2-3% accuracy |
| 2 | **Deeper Fine-Tuning** (all layers) | +1-2% accuracy |
| 3 | **Ensemble with DenseNet121** | +1-2% accuracy |
| 4 | **5-Fold Cross-Validation** | More reliable evaluation |
| 5 | **Vision Transformer (ViT)** | +2-4% accuracy |

---

## 📚 Dataset Citation

```bibtex
@article{breakhis2016,
  title={BreakHis: A Dataset for Breast Cancer Histopathological Image Classification},
  author={Spanhol, F.A. and Oliveira, L.S. and Petitjean, C. and Heutte, L.},
  journal={IEEE Transactions on Biomedical Engineering},
  year={2016}
}
```

---

## 🌐 Dataset Source

[https://data.mendeley.com/datasets/jxwvdwhpc2/1](https://data.mendeley.com/datasets/jxwvdwhpc2/1)

---

## 🔑 Keywords

```
Breast Cancer, Histopathology, Binary Classification, DenseNet201, Transfer Learning,
Deep Learning, BreakHis Dataset, Medical Imaging, Benign vs Malignant,
Patient-Level Split, Data Leakage, Mixed Precision, Test Time Augmentation,
TensorFlow, Keras, Image Classification, Computational Pathology, AI in Healthcare
```

---

## 📝 License

This project is licensed under the MIT License.

---

<div align="center">

**⭐ If this project helped you, don't forget to star it! ⭐**

</div>
