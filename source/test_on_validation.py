import os
print("[INFO] Importing modules for validation test...")

import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from prep_dataset import prepare_datasets

model_path = '../models/asl_classifier_cnn_model_v4.keras'
train_dataset_dir = "../dataset/asl-alphabet/asl_alphabet_train/asl_alphabet_train"

output_plot_path = "../plots/test_validation_confusion_matrix_v4.png"

print("[INFO] Loading datasets...")
_, val_dataset = prepare_datasets(train_dataset_dir, validation_split=0.2, seed=42)
class_names = val_dataset.class_names
num_classes = len(class_names)

optimized_val_ds = val_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

print(f"[INFO] Loading model from {model_path}...")
model = keras.models.load_model(model_path)

print("[INFO] Predicting...")
y_true = np.concatenate([y.numpy() for _, y in optimized_val_ds], axis=0)
y_pred_probs = model.predict(optimized_val_ds, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)


print("\n--- VALIDATION CLASSIFICATION REPORT ---")
print(classification_report(
    y_true, 
    y_pred, 
    labels=np.arange(num_classes), 
    target_names=class_names, 
    zero_division=0
))
print("----------------------------------")


print("[INFO] Plotting...")

cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

plt.figure(figsize=(12, 12))
sns.heatmap(
    cm_normalized, 
    annot=False,  
    cmap='Blues',
    fmt="",
    xticklabels=class_names, 
    yticklabels=class_names,
    square=True,
    annot_kws={"size": 8}
)
plt.title("Validation Dataset", fontsize=16)
plt.ylabel('True ASL Letter')
plt.xlabel('Predicted ASL Letter')

plt.savefig(output_plot_path, bbox_inches='tight')
plt.close()

print(f"[DONE] Matrix saved to {output_plot_path}")