"""
Emotion Recognition System — Transfer Learning (VGG16)
Fine-tunes a pretrained VGG16 model for facial emotion recognition.

Features:
  • Loads VGG16 pretrained on ImageNet (frozen base)
  • Converts grayscale images to RGB for VGG16 compatibility
  • Fine-tunes only the top dense layers
  • Auto-resume from checkpoint if interrupted
  • Early stopping to prevent overfitting
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
from tensorflow.keras.applications import VGG16
from sklearn.utils.class_weight import compute_class_weight

# ─── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 20  # Transfer learning needs fewer epochs
NUM_CLASSES = 7
TRAIN_DIR = os.path.join("dataset", "train")
TEST_DIR = os.path.join("dataset", "test")
MODEL_DIR = "models"
LATEST_MODEL = os.path.join(MODEL_DIR, "latest_checkpoint_transfer.h5")
BEST_MODEL = os.path.join(MODEL_DIR, "emotion_model.h5")  # Replaces old model
STATE_FILE = os.path.join(MODEL_DIR, "training_state_transfer.json")


# ─── Training State Persistence ──────────────────────────────────────────────
def save_state(epoch: int, best_val_acc: float):
    """Write the last completed epoch and best accuracy to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({"last_epoch": epoch, "best_val_accuracy": best_val_acc}, f, indent=2)


def _find_checkpoint_model() -> Optional[str]:
    """Return the path to the transfer learning checkpoint model, or None.
    Specifically looks for the new transfer learning checkpoint, not old models."""
    if os.path.exists(LATEST_MODEL):
        return LATEST_MODEL
    # Don't try to load old models (they have incompatible input shapes)
    return None


def load_state() -> Optional[dict]:
    """Read saved training state. Returns None if no checkpoint exists."""
    checkpoint = _find_checkpoint_model()
    if checkpoint is None:
        return None
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
            state["model_path"] = checkpoint
            return state
    return {"last_epoch": 0, "best_val_accuracy": 0.0, "model_path": checkpoint}


# ─── Custom Callback: track epoch + save on Ctrl+C ──────────────────────────
class ResumeStateCallback(Callback):
    """Saves training_state_transfer.json + latest_checkpoint_transfer.h5 after every epoch."""

    def __init__(self):
        super().__init__()
        self.best_val_acc = 0.0

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
        save_state(epoch + 1, self.best_val_acc)
        print(f"  [Checkpoint] epoch {epoch + 1}, "
              f"val_accuracy {val_acc:.4f}, best {self.best_val_acc:.4f}")


# ─── Data Generators (RGB for VGG16) ──────────────────────────────────────────
def create_data_generators():
    """Create data generators that output RGB images (not grayscale)."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Load as RGB (3 channels) for VGG16
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        color_mode="rgb",  # VGG16 requires RGB
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        color_mode="rgb",  # VGG16 requires RGB
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, test_generator


# ─── Build Transfer Learning Model ────────────────────────────────────────────
def build_transfer_model():
    """Build a VGG16-based model with frozen base and fine-tuned top layers."""
    # Load pretrained VGG16 (without top classification layers)
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(48, 48, 3)
    )

    # Freeze base model weights (don't train them)
    base_model.trainable = False

    # Build new model on top
    model = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ─── Class Weights (handle imbalance) ────────────────────────────────────────
def get_class_weights(train_gen):
    labels = train_gen.classes
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    return dict(enumerate(weights))


# ─── Callbacks ────────────────────────────────────────────────────────────────
def get_callbacks():
    os.makedirs(MODEL_DIR, exist_ok=True)

    resume_cb = ResumeStateCallback()

    best_checkpoint = ModelCheckpoint(
        BEST_MODEL,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    )

    return [resume_cb, best_checkpoint, early_stop, reduce_lr]


# ─── Training ────────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  Emotion Recognition — Transfer Learning (VGG16)")
    print("=" * 60)

    # Data
    train_gen, test_gen = create_data_generators()
    print(f"\nClasses: {train_gen.class_indices}")
    print(f"Training samples  : {train_gen.samples}")
    print(f"Validation samples: {test_gen.samples}")

    # Class weights
    class_weights = get_class_weights(train_gen)
    print(f"Class weights     : { {k: round(v, 2) for k, v in class_weights.items()} }")

    # ── Check for existing checkpoint ────────────────────────────────
    state = load_state()
    initial_epoch = 0

    if state:
        initial_epoch = state["last_epoch"]
        best_so_far = state.get("best_val_accuracy", 0)
        model_path = state["model_path"]
        if initial_epoch >= EPOCHS:
            print(f"\n Training already completed ({EPOCHS} epochs). Nothing to do.")
            return
        if initial_epoch > 0:
            print(f"\n Resuming from epoch {initial_epoch + 1}/{EPOCHS}")
        else:
            print(f"\n Warm-starting from existing model (epoch unknown, restarting count)")
        print(f"   Best val_accuracy so far: {best_so_far:.4f}")
        print(f"   Loading model from: {model_path}")
        model = load_model(model_path)
        # Re-compile to get fresh optimizer
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
    else:
        print(f"\n Starting fresh transfer learning for {EPOCHS} epochs")
        model = build_transfer_model()

    model.summary()

    # ── Train with Ctrl+C safety ─────────────────────────────────────
    try:
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            initial_epoch=initial_epoch,
            validation_data=test_gen,
            callbacks=get_callbacks(),
            class_weight=class_weights,
        )
    except KeyboardInterrupt:
        print("\n\n   Training interrupted by user (Ctrl+C)")
        print("   Saving current model ...")
        try:
            model.save(LATEST_MODEL)
            print(f"   Model saved to {LATEST_MODEL}")
        except Exception as e:
            print(f"   Warning: could not save model: {e}")
        existing = load_state()
        if existing and existing.get("last_epoch", 0) > 0:
            print(f"   Progress saved - last completed epoch: {existing['last_epoch']}")
            print(f"   Best val_accuracy: {existing['best_val_accuracy']:.4f}")
        print("   Run `python train_transfer.py` again to resume.\n")
        return

    # ── Training finished ────────────────────────────────────────────
    model.save(BEST_MODEL)

    all_val_acc = history.history.get("val_accuracy", [])
    if all_val_acc:
        best_idx = int(np.argmax(all_val_acc))
        actual_epoch = initial_epoch + best_idx + 1
        print("\n" + "=" * 60)
        print("  Training Complete (Transfer Learning)")
        print("=" * 60)
        print(f"  Best epoch          : {actual_epoch}")
        print(f"  Training accuracy   : {history.history['accuracy'][best_idx]:.4f}")
        print(f"  Validation accuracy : {all_val_acc[best_idx]:.4f}")
        print(f"  Model saved to      : {BEST_MODEL}")
        print("=" * 60)
    else:
        print("\n  Training complete (no validation data recorded).")

    # Clean up checkpoint files
    for f in [STATE_FILE, LATEST_MODEL]:
        if os.path.exists(f):
            os.remove(f)
    print("  Cleaned up checkpoint files (training complete).")


if __name__ == "__main__":
    train()
