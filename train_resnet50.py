"""
Emotion Recognition System — ResNet50 Transfer Learning with Fine-tuning
Competes with VGG16 for best accuracy with more modern residual architecture.

Features:
  • ResNet50 pretrained on ImageNet
  • 2-phase training: frozen base (25 epochs) + fine-tuning (25 epochs)
  • Class weight balancing
  • Auto-resume from checkpoint
  • Lower learning rate in phase 2
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
from tensorflow.keras.applications import ResNet50
from sklearn.utils.class_weight import compute_class_weight

# ─── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 7
TRAIN_DIR = os.path.join("dataset", "train")
TEST_DIR = os.path.join("dataset", "test")
MODEL_DIR = "models"
LATEST_MODEL = os.path.join(MODEL_DIR, "latest_checkpoint_resnet.h5")
BEST_MODEL = os.path.join(MODEL_DIR, "emotion_model_resnet50.h5")
STATE_FILE = os.path.join(MODEL_DIR, "training_state_resnet.json")


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


def build_resnet_model():
    """Build ResNet50 transfer learning model."""
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(48, 48, 3)
    )

    base_model.trainable = False

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
        Dropout(0.3),
        Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model


def unfreeze_layers(model: Model, base_model: Model, num_layers: int = 27):
    """Unfreeze the last layers of ResNet50."""
    base_model.trainable = True
    
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    print(f"\n  ✓ Unfroze last {num_layers} ResNet50 layers for fine-tuning")
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
    print("  Emotion Recognition — ResNet50 Transfer Learning")
    print("=" * 70)

    train_gen, test_gen = create_data_generators()
    print(f"\nClasses: {train_gen.class_indices}")
    print(f"Training samples  : {train_gen.samples}")
    print(f"Validation samples: {test_gen.samples}")

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
        print(f"\n Starting fresh ResNet50 training for {EPOCHS} epochs")
        print(f"   Phase 1 (epochs 1-25):  Frozen base, train top layers")
        print(f"   Phase 2 (epochs 26-50): Unfrozen base, full fine-tuning")
        
        model, base_model = build_resnet_model()

    model.summary()

    # Phase 1: Frozen Base
    if initial_epoch < 25:
        print(f"\n{'─' * 70}")
        print("PHASE 1: Frozen ResNet50 Base")
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
            print("   Run `python train_resnet50.py` to resume.\n")
            return
        
        initial_epoch = 25
    
    # Phase 2: Unfrozen Base
    if initial_epoch >= 25 and initial_epoch < EPOCHS:
        print(f"\n{'─' * 70}")
        print("PHASE 2: Unfrozen ResNet50 Layers (Fine-tuning)")
        print(f"{'─' * 70}")
        
        if base_model is None:
            base_model = model.layers[0]
        
        unfreeze_layers(model, base_model, num_layers=27)
        
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
            print("   Run `python train_resnet50.py` to resume.\n")
            return

    model.save(BEST_MODEL)

    print("\n" + "=" * 70)
    print("  Training Complete (ResNet50)")
    print("=" * 70)
    print(f"  Model saved to      : {BEST_MODEL}")
    print(f"  Total epochs        : {EPOCHS}")
    print("=" * 70 + "\n")

    for f in [STATE_FILE, LATEST_MODEL]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    train()
