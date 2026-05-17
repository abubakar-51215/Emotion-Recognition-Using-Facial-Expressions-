"""
Comprehensive Model Evaluation — Grayscale vs RGB Inputs

Evaluates the emotion recognition model on:
  1. Grayscale images (converted to RGB)
  2. RGB images (augmented variations)
  
Reports: Accuracy, Precision, Recall, F1-Score per emotion class + overall
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH = "models/emotion_model.h5"
TEST_DIR = os.path.join("dataset", "test")
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✓ Model loaded from {MODEL_PATH}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: GRAYSCALE (converted to RGB for VGG16)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("TEST 1: GRAYSCALE IMAGES (converted to RGB)")
print("=" * 70)

test_gen_gray = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",  # Original grayscale
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

# Predict on grayscale test set (will be converted to RGB at batch level)
print("\nGenerating predictions on grayscale images...")
all_preds_gray = []
all_true_gray = []

for i, (images, labels) in enumerate(test_gen_gray):
    # Convert grayscale to RGB (repeat channels)
    if images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)
    
    preds = model.predict(images, verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(labels, axis=1)
    
    all_preds_gray.extend(pred_classes)
    all_true_gray.extend(true_classes)
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {(i + 1) * BATCH_SIZE} images...")
    
    # Stop at full dataset
    if len(all_preds_gray) >= test_gen_gray.samples:
        break

all_preds_gray = np.array(all_preds_gray[:test_gen_gray.samples])
all_true_gray = np.array(all_true_gray[:test_gen_gray.samples])

# Calculate metrics for grayscale
acc_gray = accuracy_score(all_true_gray, all_preds_gray)
prec_gray = precision_score(all_true_gray, all_preds_gray, average="weighted")
recall_gray = recall_score(all_true_gray, all_preds_gray, average="weighted")
f1_gray = f1_score(all_true_gray, all_preds_gray, average="weighted")

print(f"\n{'─' * 70}")
print(f"GRAYSCALE RESULTS (Overall):")
print(f"{'─' * 70}")
print(f"Accuracy  : {acc_gray:.4f} ({acc_gray*100:.2f}%)")
print(f"Precision : {prec_gray:.4f}")
print(f"Recall    : {recall_gray:.4f}")
print(f"F1-Score  : {f1_gray:.4f}")

# Per-class metrics for grayscale
print(f"\n{'─' * 70}")
print(f"Per-Class Metrics (Grayscale):")
print(f"{'─' * 70}")
print(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
print(f"{'─' * 70}")

prec_per_class_gray = precision_score(all_true_gray, all_preds_gray, average=None, zero_division=0)
recall_per_class_gray = recall_score(all_true_gray, all_preds_gray, average=None, zero_division=0)
f1_per_class_gray = f1_score(all_true_gray, all_preds_gray, average=None, zero_division=0)

for i, label in enumerate(EMOTION_LABELS):
    support = np.sum(all_true_gray == i)
    print(f"{label:<12} {prec_per_class_gray[i]:<12.4f} {recall_per_class_gray[i]:<12.4f} {f1_per_class_gray[i]:<12.4f} {support:<8}")

print(f"{'─' * 70}")

# ─────────────────────────────────────────────────────────────────────────────
# Test 2: RGB (with augmentation applied)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("TEST 2: RGB IMAGES (with light augmentation)")
print(f"{'=' * 70}")

test_gen_rgb = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=5,  # Light augmentation
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
).flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode="rgb",  # Native RGB
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

# Predict on RGB test set
print("\nGenerating predictions on RGB images...")
all_preds_rgb = []
all_true_rgb = []

for i, (images, labels) in enumerate(test_gen_rgb):
    preds = model.predict(images, verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(labels, axis=1)
    
    all_preds_rgb.extend(pred_classes)
    all_true_rgb.extend(true_classes)
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {(i + 1) * BATCH_SIZE} images...")
    
    # Stop at full dataset
    if len(all_preds_rgb) >= test_gen_rgb.samples:
        break

all_preds_rgb = np.array(all_preds_rgb[:test_gen_rgb.samples])
all_true_rgb = np.array(all_true_rgb[:test_gen_rgb.samples])

# Calculate metrics for RGB
acc_rgb = accuracy_score(all_true_rgb, all_preds_rgb)
prec_rgb = precision_score(all_true_rgb, all_preds_rgb, average="weighted")
recall_rgb = recall_score(all_true_rgb, all_preds_rgb, average="weighted")
f1_rgb = f1_score(all_true_rgb, all_preds_rgb, average="weighted")

print(f"\n{'─' * 70}")
print(f"RGB RESULTS (Overall):")
print(f"{'─' * 70}")
print(f"Accuracy  : {acc_rgb:.4f} ({acc_rgb*100:.2f}%)")
print(f"Precision : {prec_rgb:.4f}")
print(f"Recall    : {recall_rgb:.4f}")
print(f"F1-Score  : {f1_rgb:.4f}")

# Per-class metrics for RGB
print(f"\n{'─' * 70}")
print(f"Per-Class Metrics (RGB):")
print(f"{'─' * 70}")
print(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
print(f"{'─' * 70}")

prec_per_class_rgb = precision_score(all_true_rgb, all_preds_rgb, average=None, zero_division=0)
recall_per_class_rgb = recall_score(all_true_rgb, all_preds_rgb, average=None, zero_division=0)
f1_per_class_rgb = f1_score(all_true_rgb, all_preds_rgb, average=None, zero_division=0)

for i, label in enumerate(EMOTION_LABELS):
    support = np.sum(all_true_rgb == i)
    print(f"{label:<12} {prec_per_class_rgb[i]:<12.4f} {recall_per_class_rgb[i]:<12.4f} {f1_per_class_rgb[i]:<12.4f} {support:<8}")

print(f"{'─' * 70}")

# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("COMPARISON SUMMARY")
print(f"{'=' * 70}")
print(f"{'Metric':<15} {'Grayscale':<20} {'RGB':<20} {'Difference':<15}")
print(f"{'─' * 70}")
print(f"{'Accuracy':<15} {acc_gray:<20.4f} {acc_rgb:<20.4f} {acc_rgb - acc_gray:+.4f}")
print(f"{'Precision':<15} {prec_gray:<20.4f} {prec_rgb:<20.4f} {prec_rgb - prec_gray:+.4f}")
print(f"{'Recall':<15} {recall_gray:<20.4f} {recall_rgb:<20.4f} {recall_rgb - recall_gray:+.4f}")
print(f"{'F1-Score':<15} {f1_gray:<20.4f} {f1_rgb:<20.4f} {f1_rgb - f1_gray:+.4f}")
print(f"{'─' * 70}")

# ─────────────────────────────────────────────────────────────────────────────
# Confusion Matrices
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("CONFUSION MATRICES")
print(f"{'=' * 70}")

cm_gray = confusion_matrix(all_true_gray, all_preds_gray)
cm_rgb = confusion_matrix(all_true_rgb, all_preds_rgb)

print(f"\nGRAYSCALE Confusion Matrix:")
print(f"(rows=true, cols=predicted)")
print(f"Labels: {EMOTION_LABELS}")
print(cm_gray)

print(f"\n\nRGB Confusion Matrix:")
print(f"(rows=true, cols=predicted)")
print(f"Labels: {EMOTION_LABELS}")
print(cm_rgb)

# ─────────────────────────────────────────────────────────────────────────────
# FINAL VERDICT
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("FINAL VERDICT")
print(f"{'=' * 70}")

if acc_rgb > acc_gray:
    better = "RGB"
    diff = (acc_rgb - acc_gray) * 100
else:
    better = "GRAYSCALE"
    diff = (acc_gray - acc_rgb) * 100

print(f"\n✓ Model performs better on {better} images")
print(f"✓ Difference: {diff:.2f}%")
print(f"✓ Recommendation: Use {better} for production")
print(f"\n✓ Best Overall Accuracy achieved: {max(acc_gray, acc_rgb)*100:.2f}%")
print(f"✓ Best Model variant: {'Grayscale → RGB conversion' if better == 'RGB' else 'Direct RGB encoding'}")

print(f"\n{'=' * 70}\n")
