"""
Medical X-Ray Image Classification – CNN with Synthetic Data
Demonstrates a full deep-learning pipeline: data generation, model training,
evaluation, and visualisation, without requiring external downloads.
"""

import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Optional TF import
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelBinarizer

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
VIZ_DIR  = os.path.join(os.path.dirname(__file__), '..', 'visualizations')
MDL_DIR  = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VIZ_DIR,  exist_ok=True)
os.makedirs(MDL_DIR,  exist_ok=True)

np.random.seed(42)

CLASSES    = ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis']
IMG_SIZE   = 64
N_IMAGES   = 800   # 200 per class


# ── Synthetic X-Ray Generation ───────────────────────────────────────────────────
def generate_xray(label_idx, img_size=IMG_SIZE):
    """Generate a synthetic grayscale 'X-ray' image with class-specific texture."""
    img = np.ones((img_size, img_size), dtype=np.float32) * 0.5

    # Rib-like horizontal bands
    for y in range(0, img_size, img_size // 8):
        thickness = np.random.randint(1, 3)
        img[y:y+thickness, 10:-10] *= np.random.uniform(0.6, 0.8)

    # Lung regions (left + right ellipses)
    cx1, cx2 = img_size//3, 2*img_size//3
    cy = img_size//2
    for cx in [cx1, cx2]:
        for i in range(img_size):
            for j in range(img_size):
                if ((i-cy)**2/((img_size*0.35)**2) + (j-cx)**2/((img_size*0.18)**2)) < 1:
                    img[i,j] = np.clip(img[i,j] * np.random.uniform(0.7, 0.95), 0, 1)

    # Class-specific abnormalities
    if label_idx == 1:  # Pneumonia – consolidation patches
        n = np.random.randint(2, 5)
        for _ in range(n):
            r, c = np.random.randint(15, img_size-15, 2)
            size = np.random.randint(4, 12)
            img[r:r+size, c:c+size] = np.random.uniform(0.65, 0.85)
    elif label_idx == 2:  # COVID-19 – bilateral ground-glass
        mask = np.random.rand(img_size, img_size) > 0.6
        img[mask] = np.clip(img[mask] * np.random.uniform(1.1, 1.3), 0, 1)
    elif label_idx == 3:  # Tuberculosis – apical nodules
        for _ in range(np.random.randint(2, 5)):
            r = np.random.randint(5, img_size//3)
            c = np.random.randint(10, img_size-10)
            size = np.random.randint(3, 8)
            img[r:r+size, c:c+size] = np.random.uniform(0.75, 0.95)

    # Noise
    img += np.random.normal(0, 0.04, img.shape)
    return np.clip(img, 0, 1)


def generate_dataset():
    print(f"  Generating {N_IMAGES} synthetic X-ray images ({N_IMAGES//len(CLASSES)} per class)…")
    images, labels = [], []
    per_class = N_IMAGES // len(CLASSES)
    for idx, cls in enumerate(CLASSES):
        for _ in range(per_class):
            images.append(generate_xray(idx))
            labels.append(idx)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    perm   = np.random.permutation(len(images))
    return images[perm], labels[perm]


# ── CNN Model (TF/Keras) ─────────────────────────────────────────────────────────
def build_cnn(n_classes):
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x   = layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D()(x)
    x   = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling2D()(x)
    x   = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(256, activation='relu')(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = keras.Model(inp, out)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ── Sklearn Fallback ─────────────────────────────────────────────────────────────
def train_sklearn_fallback(X_train, X_test, y_train, y_test):
    print("  (TensorFlow not available – using GradientBoosting as fallback)")
    X_tr = X_train.reshape(len(X_train), -1)
    X_te = X_test.reshape(len(X_test),   -1)
    # PCA for dimensionality reduction
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50, random_state=42)
    X_tr_pca = pca.fit_transform(X_tr)
    X_te_pca = pca.transform(X_te)
    clf  = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    clf.fit(X_tr_pca, y_train)
    proba  = clf.predict_proba(X_te_pca)
    y_pred = clf.predict(X_te_pca)
    return clf, proba, y_pred, None   # no history


# ── Visualisations ───────────────────────────────────────────────────────────────
def plot_sample_images(images, labels):
    fig, axes = plt.subplots(2, len(CLASSES), figsize=(14, 6))
    fig.suptitle('Synthetic X-Ray Samples by Class', fontsize=14, fontweight='bold')
    for col, (cls, idx) in enumerate(zip(CLASSES, range(len(CLASSES)))):
        samples = images[labels == idx][:2]
        for row in range(2):
            axes[row, col].imshow(samples[row], cmap='bone', vmin=0, vmax=1)
            axes[row, col].set_title(cls if row == 0 else '', fontsize=11)
            axes[row, col].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'xray_samples.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Sample images saved ✓")


def plot_results(y_test, y_pred, proba, history=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('X-Ray CNN – Model Evaluation', fontsize=14, fontweight='bold')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    im = axes[0].imshow(cm, cmap='Blues')
    axes[0].set_xticks(range(len(CLASSES))); axes[0].set_yticks(range(len(CLASSES)))
    axes[0].set_xticklabels(CLASSES, rotation=35, ha='right')
    axes[0].set_yticklabels(CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            axes[0].text(j, i, cm[i,j], ha='center', va='center',
                         color='white' if cm[i,j] > cm.max()/2 else 'black')
    axes[0].set_title('Confusion Matrix'); axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
    plt.colorbar(im, ax=axes[0])

    # ROC curves (one-vs-rest)
    lb = LabelBinarizer(); y_bin = lb.fit_transform(y_test)
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        auc = roc_auc_score(y_bin[:, i], proba[:, i])
        axes[1].plot(fpr, tpr, linewidth=2, label=f'{cls} (AUC={auc:.3f})')
    axes[1].plot([0,1],[0,1],'k--', linewidth=1)
    axes[1].set_xlabel('False Positive Rate'); axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curves (One-vs-Rest)'); axes[1].legend(fontsize=8)

    # Training curves (if available)
    if history is not None:
        axes[2].plot(history.history['accuracy'],     label='Train Acc', linewidth=2)
        axes[2].plot(history.history['val_accuracy'], label='Val Acc',   linewidth=2)
        axes[2].set_title('Training History')
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Accuracy')
        axes[2].legend()
        axes[2].grid(axis='y', alpha=0.3)
    else:
        # Per-class accuracy bar chart
        per_cls = [np.mean(y_pred[y_test==i] == i) for i in range(len(CLASSES))]
        bars = axes[2].bar(CLASSES, per_cls, color=['#1a73e8','#34a853','#fa7b17','#ea4335'])
        axes[2].set_ylim(0,1); axes[2].set_title('Per-Class Accuracy')
        axes[2].set_ylabel('Accuracy')
        for bar, v in zip(bars, per_cls):
            axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                         f'{v:.2f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'cnn_results.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Results plot saved ✓")


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("PROJECT 5 – Medical X-Ray Image Classification (CNN)")
    print("=" * 60)

    print("\n[1/4] Generating dataset…")
    images, labels = generate_dataset()
    print(f"  Dataset shape: {images.shape}  Classes: {CLASSES}")

    print("\n[2/4] Visualising samples…")
    plot_sample_images(images, labels)

    print("\n[3/4] Training model…")
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=42
    )

    history = None
    if HAS_TF:
        X_tr = X_train[..., np.newaxis]
        X_te = X_test[..., np.newaxis]
        model = build_cnn(len(CLASSES))
        model.summary()
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=0),
        ]
        history = model.fit(X_tr, y_train, epochs=30, batch_size=32,
                            validation_split=0.15, callbacks=callbacks, verbose=1)
        proba  = model.predict(X_te, verbose=0)
        y_pred = np.argmax(proba, axis=1)
        model.save(os.path.join(MDL_DIR, 'xray_cnn.keras'))
        print("  CNN model saved ✓")
    else:
        model, proba, y_pred, history = train_sklearn_fallback(X_train, X_test, y_train, y_test)

    print("\n[4/4] Evaluating…")
    print(classification_report(y_test, y_pred, target_names=CLASSES))
    plot_results(y_test, y_pred, proba, history)

    lb    = LabelBinarizer(); y_bin = lb.fit_transform(y_test)
    macro_auc = roc_auc_score(y_bin, proba, multi_class='ovr', average='macro')
    acc = np.mean(y_pred == y_test)

    print(f"\n{'='*60}")
    print(f"  Overall Accuracy : {acc:.4f}")
    print(f"  Macro ROC-AUC    : {macro_auc:.4f}")
    print(f"  Classes          : {CLASSES}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
