"""
Brain Tumor MRI Image Classification — Standalone Evaluation Script
===================================================================
Run evaluation on test dataset for any saved model (.keras or .h5).
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def evaluate(model_path, test_dir, output_cm=None):
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    print(f"Loading model: {model_path} ...")
    model = tf.keras.models.load_model(model_path)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=False
    )

    print(f"Evaluating on {test_gen.samples} test images...")
    probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    y_true = test_gen.classes

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "="*50)
    print("TEST EVALUATION REPORT")
    print("="*50)
    print(f"Overall Accuracy : {acc * 100:.2f}%")
    print(f"Weighted Precision: {prec * 100:.2f}%")
    print(f"Weighted Recall   : {rec * 100:.2f}%")
    print(f"Weighted F1-Score : {f1 * 100:.2f}%\n")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    if output_cm:
        os.makedirs(os.path.dirname(os.path.abspath(output_cm)), exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_title("Test Confusion Matrix", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()
        plt.savefig(output_cm, dpi=150)
        plt.close()
        print(f"\nSaved confusion matrix plot to: {output_cm}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Brain Tumor MRI Model")
    parser.add_argument("--model", type=str, default="models/mobilenet_v2.keras", help="Path to saved model")
    parser.add_argument("--test_dir", type=str, default="database/test", help="Path to test directory")
    parser.add_argument("--output_cm", type=str, default="plots/eval_confusion_matrix.png", help="Path to save confusion matrix")
    args = parser.parse_args()

    evaluate(args.model, args.test_dir, args.output_cm)
