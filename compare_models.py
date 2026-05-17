"""
Model Comparison & Benchmarking Tool
Evaluates and compares VGG16, ResNet50, EfficientNetB0, and original CNN models.
Shows accuracy, speed, and memory usage for each model.
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ─── Configuration ────────────────────────────────────────────────────────────
TEST_DIR = os.path.join("dataset", "test")
MODELS_DIR = "models"

MODELS = {
    "VGG16 (50 epochs)": os.path.join(MODELS_DIR, "latest_checkpoint_advanced.h5"),
    "ResNet50": os.path.join(MODELS_DIR, "emotion_model_resnet50.h5"),
    "EfficientNetB0": os.path.join(MODELS_DIR, "emotion_model_efficientnet.h5"),
    "VGG16 (20 epochs)": os.path.join(MODELS_DIR, "emotion_model.h5"),
    "Original CNN": os.path.join(MODELS_DIR, "latest_checkpoint.h5"),
}

EMOTION_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def load_test_data():
    """Load test dataset."""
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    
    # Try loading as RGB first (for transfer learning models)
    try:
        test_generator = test_datagen.flow_from_directory(
            TEST_DIR,
            target_size=(224, 224),  # Will handle resizing per model
            color_mode="rgb",
            batch_size=32,
            class_mode="categorical",
            shuffle=False,
        )
        return test_generator, "rgb"
    except:
        # Fall back to original size
        test_generator = test_datagen.flow_from_directory(
            TEST_DIR,
            target_size=(48, 48),
            color_mode="rgb",
            batch_size=32,
            class_mode="categorical",
            shuffle=False,
        )
        return test_generator, "rgb"


def load_test_data_custom_size(size):
    """Load test data with custom size."""
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=size,
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )
    return test_generator


def evaluate_model(model_name, model_path):
    """Evaluate a single model."""
    print(f"\n{'─' * 70}")
    print(f"  Evaluating: {model_name}")
    print(f"{'─' * 70}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"  ⚠️  Model not found: {model_path}")
        print(f"  Skipping {model_name}...\n")
        return None
    
    try:
        # Load model
        print(f"  Loading model...", end="", flush=True)
        model = tf.keras.models.load_model(model_path)
        print(" ✓")
        
        # Get input size from model
        input_shape = model.input_shape
        input_size = (input_shape[1], input_shape[2])
        print(f"  Input size: {input_size}")
        
        # Load test data with appropriate size
        print(f"  Loading test data...", end="", flush=True)
        test_gen = load_test_data_custom_size(input_size)
        print(f" ✓ ({test_gen.samples} samples)")
        
        # Model size
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  Model size: {model_size_mb:.2f} MB")
        
        # Inference speed
        print(f"  Testing inference speed...", end="", flush=True)
        num_samples = min(100, test_gen.samples)
        start_time = time.time()
        for _ in range(num_samples // 32):
            x, _ = next(test_gen)
            _ = model.predict(x, verbose=0)
        inference_time = time.time() - start_time
        avg_time_per_batch = (inference_time * 1000) / (num_samples // 32)
        print(f" ✓ ({avg_time_per_batch:.2f} ms/batch)")
        
        # Evaluation
        print(f"  Evaluating on test set...", end="", flush=True)
        test_gen.reset()
        results = model.evaluate(test_gen, verbose=0)
        test_loss = results[0]
        test_accuracy = results[1]
        print(f" ✓")
        
        # Per-class metrics
        print(f"  Computing per-class metrics...", end="", flush=True)
        test_gen.reset()
        y_true = test_gen.classes
        y_pred = np.argmax(model.predict(test_gen, verbose=0), axis=1)
        
        per_class_acc = {}
        for i, emotion in enumerate(EMOTION_CLASSES):
            mask = y_true == i
            if mask.sum() > 0:
                per_class_acc[emotion] = (y_pred[mask] == i).mean() * 100
            else:
                per_class_acc[emotion] = 0.0
        print(" ✓")
        
        results_dict = {
            "name": model_name,
            "path": model_path,
            "input_size": input_size,
            "accuracy": test_accuracy * 100,
            "loss": test_loss,
            "model_size_mb": model_size_mb,
            "inference_ms_per_batch": avg_time_per_batch,
            "per_class_accuracy": per_class_acc,
        }
        
        # Display results
        print(f"\n  🎯 Test Accuracy: {test_accuracy * 100:.2f}%")
        print(f"  📊 Test Loss: {test_loss:.4f}")
        print(f"\n  Per-Class Accuracy:")
        for emotion, acc in per_class_acc.items():
            bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
            print(f"    {emotion:12} {bar} {acc:6.2f}%")
        
        return results_dict
        
    except Exception as e:
        print(f"\n  ❌ Error evaluating model: {e}")
        return None


def compare_models():
    """Compare all trained models."""
    print("\n" + "=" * 70)
    print("  EMOTION RECOGNITION MODEL COMPARISON")
    print("=" * 70)
    
    results = {}
    
    for model_name, model_path in MODELS.items():
        result = evaluate_model(model_name, model_path)
        if result:
            results[model_name] = result
    
    if not results:
        print("\n❌ No models found to evaluate. Train models first:")
        print("  python train_advanced.py      # VGG16 (50 epochs)")
        print("  python train_resnet50.py      # ResNet50")
        print("  python train_efficientnet.py  # EfficientNetB0")
        return
    
    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY COMPARISON")
    print("=" * 70)
    
    print("\n{:<25} {:<12} {:<15} {:<18}".format(
        "Model", "Accuracy", "Loss", "Model Size (MB)"
    ))
    print("─" * 70)
    
    for model_name in sorted(results.keys(), key=lambda x: results[x]["accuracy"], reverse=True):
        r = results[model_name]
        print("{:<25} {:>10.2f}% {:<15.4f} {:<18.2f}".format(
            model_name,
            r["accuracy"],
            r["loss"],
            r["model_size_mb"],
        ))
    
    # Best model
    best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
    print("\n" + "=" * 70)
    print(f"  🏆 BEST MODEL: {best_model[0]}")
    print(f"     Accuracy: {best_model[1]['accuracy']:.2f}%")
    print("=" * 70)
    
    # Inference speed comparison
    print("\n{:<25} {:<20}".format("Model", "Inference (ms/batch)"))
    print("─" * 45)
    for model_name in sorted(results.keys(), key=lambda x: results[x]["inference_ms_per_batch"]):
        r = results[model_name]
        print("{:<25} {:<20.2f}".format(model_name, r["inference_ms_per_batch"]))
    
    # Fastest model
    fastest_model = min(results.items(), key=lambda x: x[1]["inference_ms_per_batch"])
    print(f"\n⚡ FASTEST MODEL: {fastest_model[0]} ({fastest_model[1]['inference_ms_per_batch']:.2f} ms/batch)")
    
    # Most efficient (accuracy/size ratio)
    print("\n" + "=" * 70)
    print("  EFFICIENCY RANKING (Accuracy / Model Size)")
    print("=" * 70)
    efficiency_sorted = sorted(
        results.items(),
        key=lambda x: x[1]["accuracy"] / x[1]["model_size_mb"],
        reverse=True
    )
    for i, (model_name, r) in enumerate(efficiency_sorted, 1):
        ratio = r["accuracy"] / r["model_size_mb"]
        print(f"{i}. {model_name:<25} {ratio:6.2f}%/MB")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    compare_models()
