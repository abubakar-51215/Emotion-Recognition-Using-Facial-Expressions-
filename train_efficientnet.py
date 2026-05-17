"""
Emotion Recognition System — EfficientNetB0 Transfer Learning with Fine-tuning
Modern efficient architecture with minimal parameters but high performance.

Features:
  • EfficientNetB0 pretrained on ImageNet
  • Most efficient model in terms of size and speed
  • 2-phase training: frozen base (25 epochs) + fine-tuning (25 epochs)
  • Class weight balancing
  • Auto-resume from checkpoint
"""

import os
import json
import numpy as np
import tensorflow as tf
from typing import Optional
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0
from sklearn.utils.class_weight import compute_class_weight
import cv2

def apply_clahe(img):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image"""
    img = (img * 255).astype(np.uint8)
    if img.shape[-1] == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
    return img.astype(np.float32) / 255.0

# ─── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)  # EfficientNetB0 expects larger input
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 7
TRAIN_DIR = os.path.join("dataset", "train")
TEST_DIR = os.path.join("dataset", "test")
MODEL_DIR = "models"
LATEST_MODEL = os.path.join(MODEL_DIR, "latest_checkpoint_efficientnet.h5")
BEST_MODEL = os.path.join(MODEL_DIR, "emotion_model_efficientnet.h5")
STATE_FILE = os.path.join(MODEL_DIR, "training_state_efficientnet.json")


def save_state(epoch: int, best_val_acc: float, phase: str = ""):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({
            "last_epoch": epoch,
            "best_val_accuracy": best_val_acc,
            "phase": phase,
        }, f, indent=2)


def load_state() -> Optional[dict]:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
            if os.path.exists(LATEST_MODEL):
                state["model_path"] = LATEST_MODEL
                return state
    return None


class ResumeStateCallback(Callback):
    def __init__(self, phase: str = "frozen"):
        super().__init__()
        self.best_val_acc = 0.0
        self.phase = phase

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get("val_accuracy", 0.0)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
        try:
            self.model.save(LATEST_MODEL)
        except Exception as e:
            print(f"  ⚠️  Failed to save checkpoint: {e}")
            return
        save_state(epoch + 1, self.best_val_acc, self.phase)
        print(f"  [Checkpoint] epoch {epoch + 1}, "
              f"val_accuracy {val_acc:.4f}, best {self.best_val_acc:.4f}")


def create_data_generators():
    train_datagen = ImageDataGenerator(
        preprocessing_function=apply_clahe,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )
    test_datagen = ImageDataGenerator(preprocessing_function=apply_clahe)

    # EfficientNetB0 requires 224x224 input
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, test_generator


def build_efficientnet_model():
    """Build EfficientNetB0 transfer learning model."""
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model


def unfreeze_layers(model: Model, base_model: Model, num_layers: int = 30):
    """Unfreeze the last layers of EfficientNetB0."""
    base_model.trainable = True
    
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    print(f"\n  ✓ Unfroze last {num_layers} EfficientNetB0 layers for fine-tuning")
    print(f"  ✓ Learning rate: 0.00001 (10x lower)")


def get_class_weights(train_gen):
    labels = train_gen.classes
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    return dict(enumerate(weights))


def get_callbacks(phase: str = "frozen"):
    os.makedirs(MODEL_DIR, exist_ok=True)

    resume_cb = ResumeStateCallback(phase=phase)

    best_checkpoint = ModelCheckpoint(
        BEST_MODEL,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=15,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-8,
        verbose=1,
    )

    return [resume_cb, best_checkpoint, early_stop, reduce_lr]


def train():
    print("=" * 70)
    print("  Emotion Recognition — EfficientNetB0 Transfer Learning")
    print("=" * 70)

    train_gen, test_gen = create_data_generators()
    print(f"\nClasses: {train_gen.class_indices}")
    print(f"Training samples  : {train_gen.samples}")
    print(f"Validation samples: {test_gen.samples}")
    print(f"Input resolution  : {IMG_SIZE} (EfficientNetB0 optimized)")

    class_weights = get_class_weights(train_gen)
    print(f"Class weights     : { {k: round(v, 2) for k, v in class_weights.items()} }")

    state = load_state()
    initial_epoch = 0
    phase = "frozen"

    if state:
        initial_epoch = state["last_epoch"]
        best_so_far = state.get("best_val_accuracy", 0)
        model_path = state["model_path"]
        phase = state.get("phase", "frozen")
        
        if initial_epoch >= EPOCHS:
            print(f"\n Training already completed ({EPOCHS} epochs). Nothing to do.")
            return
        
        print(f"\n Resuming from epoch {initial_epoch + 1}/{EPOCHS}")
        print(f"   Phase: {phase.upper()}")
        print(f"   Best val_accuracy so far: {best_so_far:.4f}")
        model = load_model(model_path)
        base_model = None
        model.compile(
            optimizer=Adam(learning_rate=0.0001 if phase == "frozen" else 0.00001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
    else:
        print(f"\n Starting fresh EfficientNetB0 training for {EPOCHS} epochs")
        print(f"   Phase 1 (epochs 1-25):  Frozen base, train top layers")
        print(f"   Phase 2 (epochs 26-50): Unfrozen base, full fine-tuning")
        
        model, base_model = build_efficientnet_model()

    model.summary()

    # Phase 1: Frozen Base
    if initial_epoch < 25:
        print(f"\n{'─' * 70}")
        print("PHASE 1: Frozen EfficientNetB0 Base")
        print(f"{'─' * 70}")
        
        try:
            model.fit(
                train_gen,
                epochs=25,
                initial_epoch=initial_epoch,
                validation_data=test_gen,
                callbacks=get_callbacks("frozen"),
                class_weight=class_weights,
            )
        except KeyboardInterrupt:
            print("\n\n   Training interrupted by user (Ctrl+C)")
            print("   Run `python train_efficientnet.py` to resume.\n")
            return
        
        initial_epoch = 25
    
    # Phase 2: Unfrozen Base
    if initial_epoch >= 25 and initial_epoch < EPOCHS:
        print(f"\n{'─' * 70}")
        print("PHASE 2: Unfrozen EfficientNetB0 Layers (Fine-tuning)")
        print(f"{'─' * 70}")
        
        if base_model is None:
            base_model = model.layers[0]
        
        unfreeze_layers(model, base_model, num_layers=30)
        
        try:
            model.fit(
                train_gen,
                epochs=EPOCHS,
                initial_epoch=initial_epoch,
                validation_data=test_gen,
                callbacks=get_callbacks("unfrozen"),
                class_weight=class_weights,
            )
        except KeyboardInterrupt:
            print("\n\n   Training interrupted by user (Ctrl+C)")
            print("   Run `python train_efficientnet.py` to resume.\n")
            return

    model.save(BEST_MODEL)

    print("\n" + "=" * 70)
    print("  Training Complete (EfficientNetB0)")
    print("=" * 70)
    print(f"  Model saved to      : {BEST_MODEL}")
    print(f"  Total epochs        : {EPOCHS}")
    print(f"  Input resolution    : {IMG_SIZE}")
    print("=" * 70 + "\n")

    for f in [STATE_FILE, LATEST_MODEL]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    train()
