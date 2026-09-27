"""
Brain Tumor MRI Image Classification
====================================
Training script for:
  1. Custom Convolutional Neural Network (CNN)
  2. Transfer Learning using MobileNetV2 Pretrained on ImageNet
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(__file__), "database")
TRAIN_DIR  = os.path.join(BASE_DIR, "train")
VALID_DIR  = os.path.join(BASE_DIR, "valid")
TEST_DIR   = os.path.join(BASE_DIR, "test")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
PLOTS_DIR  = os.path.join(os.path.dirname(__file__), "plots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
CLASSES     = ["glioma", "meningioma", "no_tumor", "pituitary"]
NUM_CLASSES = len(CLASSES)


# ─────────────────────────────────────────────
#  Data Generators
# ─────────────────────────────────────────────
def build_generators():
    """Build augmented train and plain valid/test generators."""
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        horizontal_flip=True,
        zoom_range=0.08,
        width_shift_range=0.05,
        height_shift_range=0.05,
        fill_mode="nearest",
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", classes=CLASSES, shuffle=True
    )
    valid_data = val_gen.flow_from_directory(
        VALID_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", classes=CLASSES, shuffle=False
    )
    test_data = val_gen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", classes=CLASSES, shuffle=False
    )
    return train_data, valid_data, test_data


# ─────────────────────────────────────────────
#  1. Custom CNN Model
# ─────────────────────────────────────────────
def build_custom_cnn():
    """Custom CNN designed for Brain Tumor MRI classification."""
    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.20),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.20),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.20),

        # Block 4
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Dense Classification Head
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.40),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="Custom_CNN")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────
#  2. Transfer Learning (MobileNetV2)
# ─────────────────────────────────────────────
def train_mobilenet_transfer(train_data, valid_data, epochs=15):
    """
    High-efficiency transfer learning using MobileNetV2.
    Precomputes frozen feature representations for CPU speed,
    trains classification head, and returns end-to-end model.
    """
    print("\n>>> Initializing MobileNetV2 base feature extractor...")
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False

    val_gen_plain = ImageDataGenerator(rescale=1.0 / 255)
    train_plain = val_gen_plain.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", classes=CLASSES, shuffle=False
    )
    
    print("Extracting training feature representations...")
    X_train_feat = base_model.predict(train_plain, verbose=1)
    y_train = tf.keras.utils.to_categorical(train_plain.classes, num_classes=NUM_CLASSES)

    print("Extracting validation feature representations...")
    valid_data.reset()
    X_val_feat = base_model.predict(valid_data, verbose=1)
    y_val = tf.keras.utils.to_categorical(valid_data.classes, num_classes=NUM_CLASSES)

    feat_dim = X_train_feat.shape[1]
    head = models.Sequential([
        layers.Input(shape=(feat_dim,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.40),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ], name="MobileNetV2_Head")

    head.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    cb = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]

    print("\n>>> Training Transfer Learning classification head...")
    history = head.fit(
        X_train_feat, y_train,
        validation_data=(X_val_feat, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        verbose=1
    )

    full_inputs = layers.Input(shape=(*IMG_SIZE, 3), name="input_image")
    features = base_model(full_inputs, training=False)
    outputs = head(features)
    full_model = models.Model(inputs=full_inputs, outputs=outputs, name="MobileNetV2_Transfer")
    full_model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return full_model, history


# ─────────────────────────────────────────────
#  Plotting Utilities
# ─────────────────────────────────────────────
def plot_history(history, model_name):
    """Plot and save training/validation accuracy and loss."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} — Training & Validation Metrics", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Training Accuracy", color="#2563EB", lw=2)
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy", color="#F97316", lw=2)
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"], label="Training Loss", color="#2563EB", lw=2)
    axes[1].plot(history.history["val_loss"], label="Validation Loss", color="#F97316", lw=2)
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_file = os.path.join(PLOTS_DIR, f"{model_name}_history.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved history plot: {out_file}")


def plot_cm(y_true, y_pred, model_name):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
        cbar=True, annot_kws={"size": 12}
    )
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    out_file = os.path.join(PLOTS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved confusion matrix: {out_file}")
    return cm


# ─────────────────────────────────────────────
#  Model Evaluation
# ─────────────────────────────────────────────
def evaluate_model(model, test_data, model_name):
    """Run comprehensive evaluation on the test set."""
    test_data.reset()
    probs = model.predict(test_data, verbose=1)
    preds = np.argmax(probs, axis=1)
    y_true = test_data.classes

    acc  = float(accuracy_score(y_true, preds))
    prec = float(precision_score(y_true, preds, average="weighted", zero_division=0))
    rec  = float(recall_score(y_true, preds, average="weighted", zero_division=0))
    f1   = float(f1_score(y_true, preds, average="weighted", zero_division=0))
    
    print(f"\n==========================================")
    print(f"  Evaluation: {model_name}")
    print(f"==========================================")
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}%")
    print(f"  Recall   : {rec * 100:.2f}%")
    print(f"  F1-Score : {f1 * 100:.2f}%\n")
    print(classification_report(y_true, preds, target_names=CLASSES, digits=4))

    cm = plot_cm(y_true, preds, model_name)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist()
    }


# ─────────────────────────────────────────────
#  Main Execution
# ─────────────────────────────────────────────
def main(force_retrain_tl=False):
    print("Preparing dataset generators...")
    train_data, valid_data, test_data = build_generators()
    print(f"Found {train_data.samples} train, {valid_data.samples} valid, {test_data.samples} test images.")

    results = {}

    # 1. Custom CNN
    print("\n" + "="*50)
    print(">>> 1. Training Custom CNN Model")
    print("="*50)
    cnn = build_custom_cnn()
    cnn_keras_path = os.path.join(MODELS_DIR, "custom_cnn.keras")
    cnn_h5_path    = os.path.join(MODELS_DIR, "custom_cnn.h5")

    callbacks_cnn = [
        EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=cnn_keras_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]

    h_cnn = cnn.fit(
        train_data,
        epochs=8,
        validation_data=valid_data,
        callbacks=callbacks_cnn,
        verbose=1
    )
    plot_history(h_cnn, "Custom_CNN")
    
    if os.path.exists(cnn_keras_path):
        cnn = models.load_model(cnn_keras_path)
    try:
        cnn.save(cnn_h5_path)
        print(f"Saved: {cnn_h5_path}")
    except Exception as e:
        print(f"Notice regarding .h5 export: {e}")

    results["Custom CNN"] = evaluate_model(cnn, test_data, "Custom_CNN")

    # 2. Transfer Learning (MobileNetV2)
    print("\n" + "="*50)
    print(">>> 2. Transfer Learning Model (MobileNetV2)")
    print("="*50)
    tl_keras_path  = os.path.join(MODELS_DIR, "mobilenet_v2.keras")
    tl_h5_path     = os.path.join(MODELS_DIR, "mobilenet_v2.h5")

    if os.path.exists(tl_keras_path) and not force_retrain_tl:
        print(f"Loading existing trained MobileNetV2 from: {tl_keras_path}")
        tl_model = models.load_model(tl_keras_path)
    else:
        tl_model, h_tl = train_mobilenet_transfer(train_data, valid_data, epochs=15)
        plot_history(h_tl, "MobileNetV2")
        tl_model.save(tl_keras_path)
        print(f"Saved: {tl_keras_path}")
        try:
            tl_model.save(tl_h5_path)
            print(f"Saved: {tl_h5_path}")
        except Exception as e:
            print(f"Notice regarding .h5 export: {e}")

    results["MobileNetV2"] = evaluate_model(tl_model, test_data, "MobileNetV2")

    # 3. Model Comparison Plot
    print("\n" + "="*50)
    print(">>> Generating Model Comparison Summary")
    print("="*50)
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#3B82F6", "#10B981"]
    for i, (name, m) in enumerate(results.items()):
        vals = [m[k] for k in metrics]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=name, color=colors[i], alpha=0.9)
        for b in bars:
            height = b.get_height()
            ax.annotate(f"{height*100:.1f}%",
                        xy=(b.get_x() + b.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Brain Tumor Classifier: Model Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    comp_plot = os.path.join(PLOTS_DIR, "model_comparison.png")
    plt.savefig(comp_plot, dpi=150)
    plt.close()
    print(f"Saved comparison plot: {comp_plot}")

    # Save results to JSON for Streamlit app
    res_path = os.path.join(os.path.dirname(__file__), "model_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results summary: {res_path}")

    print("\n==========================================")
    print("All training & evaluations completed successfully!")
    print("==========================================")


if __name__ == "__main__":
    main()
