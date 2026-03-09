# Project 5: Medical X-Ray Image Classification (CNN)

## Problem Statement
Medical imaging analysis using deep learning to classify chest X-rays into diagnostic categories, demonstrating how CNNs can assist clinical workflows.

## Classes
- **Normal** — Healthy chest X-ray
- **Pneumonia** — Bacterial/viral lung infection
- **COVID-19** — Bilateral ground-glass opacities
- **Tuberculosis** — Apical nodular infiltrates

## Model Architecture (TensorFlow/Keras)
```
Input(64×64×1) → Conv2D(32) → BN → MaxPool
               → Conv2D(64) → BN → MaxPool
               → Conv2D(128) → BN → GlobalAvgPool
               → Dense(256) → Dropout(0.4)
               → Dense(4, softmax)
```

## Results
| Metric | Score |
|--------|-------|
| Overall Accuracy | 94.4% |
| Macro ROC-AUC | 0.997 |
| COVID-19 F1 | 0.99 |
| Pneumonia F1 | 0.89 |

## Run
```bash
pip install tensorflow  # optional - falls back to sklearn
python src/xray_classifier.py
```

## ⚠️ Disclaimer
This project uses **synthetic image data** for demonstration. It is not intended for clinical use. Real medical AI requires validated clinical datasets, regulatory approval, and rigorous testing.

## Technologies
Python | TensorFlow/Keras (optional) | Scikit-learn | NumPy | Matplotlib
