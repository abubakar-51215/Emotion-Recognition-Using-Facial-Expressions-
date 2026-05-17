"""
Emotion Recognition System — Advanced Transfer Learning (VGG16 + Fine-tuning)
Fine-tunes BOTH the frozen layers AND top layers for maximum accuracy.

Features:
  • Initial training on frozen VGG16 base (like train_transfer.py)
  • Unfroze some VGG16 layers for deeper fine-tuning
  • Extended training: 50 epochs (instead of 20)
  • Lower learning rate for fine-tuning phase
  • Auto-resume from checkpoint if interrupted
  • Early stopping with longer patience
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
BATCH_SIZE = 32  # Smaller batch for fine-tuning
EPOCHS = 50  # Extended training
NUM_CLASSES = 7
TRAIN_DIR = os.path.join("dataset", "train")
TEST_DIR = os.path.join("dataset", "test")
MODEL_DIR = "models"
LATEST_MODEL = os.path.join(MODEL_DIR, "latest_checkpoint_advanced.h5")
BEST_MODEL = os.path.join(MODEL_DIR, "emotion_model.h5")  # Final best model
STATE_FILE = os.path.join(MODEL_DIR, "training_state_advanced.json")

# ─── Training State Persistence ──────────────────────────────────────────────
def save_state(epoch: int, best_val_acc: float, phase: str = ""):
    """Write the last completed epoch and best accuracy to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({
            "last_epoch": epoch,
            "best_val_accuracy": best_val_acc,
            "phase": phase,
        }, f, indent=2)


def _find_checkpoint_model() -> Optional[str]:
    """Return the path to transfer learning checkpoint, or None."""
    if os.path.exists(LATEST_MODEL):
        return LATEST_MODEL
    # Try old transfer learning model
    if os.path.exists(os.path.join(MODEL_DIR, "latest_checkpoint_transfer.h5")):
        return os.path.join(MODEL_DIR, "latest_checkpoint_transfer.h5")
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
    return {"last_epoch": 0, "best_val_accuracy": 0.0, "model_path": checkpoint, "phase": "frozen"}


# ─── Custom Callback: track epoch + save on Ctrl+C ──────────────────────────
class ResumeStateCallback(Callback):
    """Saves training_state_advanced.json + latest_checkpoint_advanced.h5 after every epoch."""

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


# ─── Data Generators (RGB for VGG16) ──────────────────────────────────────────
def create_data_generators():
    """Create data generators that output RGB images (not grayscale)."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
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


# ─── Build Advanced Transfer Learning Model ──────────────────────────────────
def build_advanced_model():
    """Build a VGG16-based model with trainable top layers.
    
    For phase 1 (frozen): Keep VGG16 base frozen, train only top layers
    For phase 2 (unfrozen): Unfreeze some VGG16 layers and fine-tune everything
    """
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(48, 48, 3)
    )

    # Initially freeze all base layers
    base_model.trainable = False

    # Build model
    model = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
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

    return model, base_model


def unfreeze_layers(model: Model, base_model: Model, num_layers: int = 4):
    """Unfreeze the last `num_layers` layers of VGG16 for fine-tuning."""
    base_model.trainable = True
    
    # Freeze all but the last `num_layers` layers
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.00001),  # 10x lower learning rate
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    print(f"\n  ✓ Unfroze last {num_layers} VGG16 layers for fine-tuning")
    print(f"  ✓ Learning rate: 0.00001 (10x lower)")


# ─── Class Weights (handle imbalance) ────────────────────────────────────────
def get_class_weights(train_gen):
    labels = train_gen.classes
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    return dict(enumerate(weights))


# ─── Callbacks ────────────────────────────────────────────────────────────────
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
        patience=15,  # More patient in advanced training
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


# ─── Training ────────────────────────────────────────────────────────────────
def train():
    print("=" * 70)
    print("  Emotion Recognition — Advanced Transfer Learning (VGG16 + Fine-tuning)")
    print("=" * 70)

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
        print(f"   Loading model from: {model_path}")
        model = load_model(model_path)
        # Re-compile so we get a fresh optimizer state
        base_model = None
        model.compile(
            optimizer=Adam(learning_rate=0.0001 if phase == "frozen" else 0.00001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
    else:
        print(f"\n Starting fresh advanced training for {EPOCHS} epochs")
        print(f"   Phase 1 (epochs 1-25):  Frozen VGG16 base, train top layers")
        print(f"   Phase 2 (epochs 26-50): Unfrozen VGG16 layers, full fine-tuning")
        
        model, base_model = build_advanced_model()

    model.summary()

    # ── Train Phase 1: Frozen Base (Epochs 1-25) ─────────────────────
    if initial_epoch < 25:
        print(f"\n{'─' * 70}")
        print("PHASE 1: Frozen VGG16 Base (Train Top Layers Only)")
        print(f"{'─' * 70}")
        
        try:
            history_phase1 = model.fit(
                train_gen,
                epochs=25,
                initial_epoch=initial_epoch,
                validation_data=test_gen,
                callbacks=get_callbacks("frozen"),
                class_weight=class_weights,
            )
        except KeyboardInterrupt:
            print("\n\n   Training interrupted by user (Ctrl+C)")
            print("   Saving current model...")
            try:
                model.save(LATEST_MODEL)
            except Exception as e:
                print(f"   Warning: {e}")
            existing = load_state()
            if existing and existing.get("last_epoch", 0) > 0:
                print(f"   Last completed epoch: {existing['last_epoch']}")
            print("   Run `python train_advanced.py` to resume.\n")
            return
        
        initial_epoch = 25
    
    # ── Train Phase 2: Unfrozen VGG16 (Epochs 26-50) ───────────────
    if initial_epoch >= 25 and initial_epoch < EPOCHS:
        print(f"\n{'─' * 70}")
        print("PHASE 2: Unfrozen VGG16 Layers (Full Fine-tuning)")
        print(f"{'─' * 70}")
        
        if base_model is None:
            # Extract base model from sequential model
            base_model = model.layers[0]
        
        # Unfreeze last 4 VGG16 blocks
        unfreeze_layers(model, base_model, num_layers=8)
        
        try:
            history_phase2 = model.fit(
                train_gen,
                epochs=EPOCHS,
                initial_epoch=initial_epoch,
                validation_data=test_gen,
                callbacks=get_callbacks("unfrozen"),
                class_weight=class_weights,
            )
        except KeyboardInterrupt:
            print("\n\n   Training interrupted by user (Ctrl+C)")
            print("   Saving current model...")
            try:
                model.save(LATEST_MODEL)
            except Exception as e:
                print(f"   Warning: {e}")
            existing = load_state()
            if existing and existing.get("last_epoch", 0) > 0:
                print(f"   Last completed epoch: {existing['last_epoch']}")
            print("   Run `python train_advanced.py` to resume.\n")
            return

    # ── Training finished ────────────────────────────────────────────
    model.save(BEST_MODEL)

    print("\n" + "=" * 70)
    print("  Training Complete (Advanced Transfer Learning)")
    print("=" * 70)
    print(f"  Model saved to      : {BEST_MODEL}")
    print(f"  Total epochs        : {EPOCHS}")
    print(f"  Phase 1             : Epochs 1-25 (Frozen Base)")
    print(f"  Phase 2             : Epochs 26-50 (Fine-tuned Base)")
    print("=" * 70)

    # Clean up checkpoint files
    for f in [STATE_FILE, LATEST_MODEL]:
        if os.path.exists(f):
            os.remove(f)
    print("  Cleaned up checkpoint files (training complete).\n")


if __name__ == "__main__":
    train()
