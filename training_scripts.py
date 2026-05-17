"""
Quick Reference: Available Training Scripts & Commands
Run this to see all available training options.
"""

import os
import sys

# ANSI Colors
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'

SCRIPTS = {
    "train_transfer.py": {
        "name": "VGG16 Transfer Learning (20 Epochs)",
        "desc": "Original transfer learning model - frozen VGG16 base trained for 20 epochs",
        "time": "1-2 hours",
        "output": "emotion_model.h5 (56.77 MB)",
        "status": "ORIGINAL",
    },
    "train_advanced.py": {
        "name": "VGG16 Advanced (50 Epochs + Fine-tuning)",
        "desc": "Enhanced VGG16 with unfrozen layers - 2-phase training for maximum accuracy",
        "time": "2-4 hours",
        "output": "latest_checkpoint_advanced.h5 → emotion_model.h5",
        "status": "RECOMMENDED ⭐⭐⭐⭐⭐",
    },
    "train_resnet50.py": {
        "name": "ResNet50 Transfer Learning",
        "desc": "Modern residual architecture - skip connections for better convergence",
        "time": "3-5 hours",
        "output": "emotion_model_resnet50.h5 (102 MB)",
        "status": "ALTERNATIVE",
    },
    "train_efficientnet.py": {
        "name": "EfficientNetB0 Transfer Learning",
        "desc": "Most efficient model - smallest size, fastest inference for mobile deployment",
        "time": "4-6 hours",
        "output": "emotion_model_efficientnet.h5 (24 MB)",
        "status": "MOBILE OPTIMIZED 📱",
    },
    "compare_models.py": {
        "name": "Model Comparison & Benchmarking",
        "desc": "Compare all trained models - accuracy, speed, size, efficiency",
        "time": "~5 minutes",
        "output": "Comparison report (console)",
        "status": "ANALYSIS TOOL",
    },
}

EVALUATION_SCRIPTS = {
    "evaluate_model.py": {
        "name": "Model Evaluation (Grayscale vs RGB)",
        "desc": "Compare model performance on grayscale vs RGB inputs with per-class metrics",
        "usage": "python evaluate_model.py",
    },
}


def print_header():
    print(f"\n{BOLD}{CYAN}{'=' * 80}{RESET}")
    print(f"{BOLD}{CYAN}  EMOTION RECOGNITION SYSTEM — TRAINING SCRIPTS REFERENCE{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 80}{RESET}\n")


def print_script_info(script_name, info):
    status_color = {
        "RECOMMENDED ⭐⭐⭐⭐⭐": GREEN,
        "ORIGINAL": YELLOW,
        "ALTERNATIVE": BLUE,
        "MOBILE OPTIMIZED 📱": CYAN,
        "ANALYSIS TOOL": BLUE,
    }.get(info["status"], RESET)
    
    print(f"{BOLD}{script_name}{RESET}")
    print(f"  {BOLD}Name:{RESET} {info['name']}")
    print(f"  {BOLD}Description:{RESET} {info['desc']}")
    print(f"  {BOLD}Time Estimate:{RESET} {info['time']}")
    print(f"  {BOLD}Output:{RESET} {info['output']}")
    print(f"  {BOLD}Status:{RESET} {status_color}{info['status']}{RESET}")
    print(f"  {BOLD}Run:{RESET} {CYAN}python {script_name}{RESET}")
    print()


def print_usage_guide():
    print(f"{BOLD}{CYAN}QUICK START GUIDE{RESET}")
    print(f"{YELLOW}{'─' * 80}{RESET}")
    
    print(f"\n{BOLD}1. Train a Single Model (Recommended for Quick Results):{RESET}")
    print(f"   {CYAN}python train_advanced.py{RESET}")
    print(f"   → Best accuracy with VGG16 + fine-tuning (2-4 hours)")
    
    print(f"\n{BOLD}2. Compare All Available Models:{RESET}")
    print(f"   {CYAN}python compare_models.py{RESET}")
    print(f"   → See accuracy, speed, and size metrics (5 minutes)")
    
    print(f"\n{BOLD}3. Train Multiple Models (Full Comparison):{RESET}")
    print(f"   {CYAN}python train_advanced.py{RESET}")
    print(f"   {CYAN}python train_resnet50.py{RESET}")
    print(f"   {CYAN}python train_efficientnet.py{RESET}")
    print(f"   {CYAN}python compare_models.py{RESET}")
    print(f"   → Compare 3 architectures (12+ hours total)")
    
    print(f"\n{BOLD}4. Train for Mobile Deployment:{RESET}")
    print(f"   {CYAN}python train_efficientnet.py{RESET}")
    print(f"   → Smallest model (24 MB), fastest inference (4-6 hours)")
    
    print(f"\n{BOLD}5. Resume Interrupted Training:{RESET}")
    print(f"   {CYAN}python train_advanced.py{RESET}")
    print(f"   → Automatically resumes from last checkpoint")
    
    print()


def print_recommendations():
    print(f"{BOLD}{CYAN}MODEL SELECTION GUIDE{RESET}")
    print(f"{YELLOW}{'─' * 80}{RESET}")
    
    recommendations = [
        {
            "use_case": "Maximum Accuracy",
            "model": "train_advanced.py (VGG16 50ep)",
            "accuracy": "~65%",
            "size": "56.77 MB",
        },
        {
            "use_case": "Balanced (Speed + Accuracy)",
            "model": "train_resnet50.py",
            "accuracy": "~63%",
            "size": "102 MB",
        },
        {
            "use_case": "Mobile/Lightweight",
            "model": "train_efficientnet.py",
            "accuracy": "~61%",
            "size": "24 MB",
        },
        {
            "use_case": "Baseline (Quick Result)",
            "model": "train_transfer.py (VGG16 20ep)",
            "accuracy": "~58%",
            "size": "56.77 MB",
        },
    ]
    
    print(f"\n{BOLD}{'Use Case':<30} {'Model':<35} {'Accuracy':<12} {'Size'}{RESET}")
    print(f"{YELLOW}{'─' * 80}{RESET}")
    for rec in recommendations:
        print(f"{rec['use_case']:<30} {rec['model']:<35} {rec['accuracy']:<12} {rec['size']}")
    print()


def print_training_matrix():
    print(f"{BOLD}{CYAN}TRAINING PHASE MATRIX{RESET}")
    print(f"{YELLOW}{'─' * 80}{RESET}")
    
    print(f"\n{'Model':<25} {'Phase 1':<20} {'Phase 2':<20} {'Total Time'}")
    print(f"{YELLOW}{'─' * 80}{RESET}")
    print(f"{'VGG16 (50ep)':<25} {'1-25 (Frozen)':<20} {'26-50 (Unfrozen)':<20} {'2-4 hours'}")
    print(f"{'ResNet50':<25} {'1-25 (Frozen)':<20} {'26-50 (Unfrozen)':<20} {'3-5 hours'}")
    print(f"{'EfficientNetB0':<25} {'1-25 (Frozen)':<20} {'26-50 (Unfrozen)':<20} {'4-6 hours'}")
    print()


def print_parallelization():
    print(f"{BOLD}{CYAN}PARALLEL TRAINING (Multiple GPUs/Terminals){RESET}")
    print(f"{YELLOW}{'─' * 80}{RESET}")
    
    print(f"\n{BOLD}PowerShell (Windows):{RESET}")
    print(f"   Start-Process -NoNewWindow python train_advanced.py")
    print(f"   Start-Process -NoNewWindow python train_resnet50.py")
    print(f"   Start-Process -NoNewWindow python train_efficientnet.py")
    
    print(f"\n{BOLD}Bash (Linux/Mac):{RESET}")
    print(f"   python train_advanced.py &")
    print(f"   python train_resnet50.py &")
    print(f"   python train_efficientnet.py &")
    
    print(f"\n{YELLOW}Note: Each process uses full GPU. With single GPU, train sequentially.{RESET}\n")


def print_checkpoint_info():
    print(f"{BOLD}{CYAN}CHECKPOINT SYSTEM{RESET}")
    print(f"{YELLOW}{'─' * 80}{RESET}")
    
    print(f"\n{BOLD}Saved Automatically After Each Epoch:{RESET}")
    print(f"  • Latest checkpoint (for resuming)")
    print(f"  • Best model (only when validation accuracy improves)")
    print(f"  • Training state (epoch number, accuracy)")
    
    print(f"\n{BOLD}Checkpoint Files:{RESET}")
    print(f"  • {CYAN}latest_checkpoint_advanced.h5{RESET} (VGG16 current)")
    print(f"  • {CYAN}latest_checkpoint_resnet.h5{RESET} (ResNet50 current)")
    print(f"  • {CYAN}latest_checkpoint_efficientnet.h5{RESET} (EfficientNetB0 current)")
    print(f"  • {CYAN}training_state_*.json{RESET} (Training progress)")
    
    print(f"\n{BOLD}Resume Training:{RESET}")
    print(f"  {CYAN}python train_<model>.py  # Auto-resumes from checkpoint{RESET}\n")


def main():
    print_header()
    
    print(f"{BOLD}{CYAN}TRAINING SCRIPTS{RESET}")
    print(f"{YELLOW}{'─' * 80}{RESET}\n")
    
    for script_name, info in SCRIPTS.items():
        print_script_info(script_name, info)
    
    print_usage_guide()
    print_recommendations()
    print_training_matrix()
    print_checkpoint_info()
    print_parallelization()
    
    print(f"{BOLD}{CYAN}DOCUMENTATION{RESET}")
    print(f"{YELLOW}{'─' * 80}{RESET}")
    print(f"\nFull guide available in: {CYAN}TRAINING_GUIDE_STEP5.md{RESET}")
    print(f"\n")


if __name__ == "__main__":
    main()
