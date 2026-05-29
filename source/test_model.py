import os
print("[INFO] Importing modules for manual testing...")

import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

model_path = '../models/asl_classifier_cnn_model_v4.keras'
test_dataset_dir = '../dataset/asl-alphabet/asl_alphabet_test/asl_alphabet_test'
img_height, img_width = 200, 200

os.makedirs("../plots", exist_ok=True)
output_plot_path = "../plots/test_confusion_matrix_v4.png"

class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
    'nothing', 'space'
]
num_classes = len(class_names)


print(f"[INFO] Loading trained model from {model_path}...")
model = keras.models.load_model(model_path)
print("[SUCCESS] Model loaded successfully.")


print("[INFO] Processing loose test images...")

x_test_list = []
y_test_list = []

for file_name in sorted(os.listdir(test_dataset_dir)):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        img_path = os.path.join(test_dataset_dir, file_name)
        
        label_part = file_name.split('_')[0]
        
        if label_part in class_names:
           
            img = tf.keras.utils.load_img(img_path, color_mode='grayscale', target_size=(img_height, img_width))
            img_array = tf.keras.utils.img_to_array(img)
            
            x_test_list.append(img_array)
            y_test_list.append(class_names.index(label_part))


x_test = np.array(x_test_list)
y_true = np.array(y_test_list)

print(f"[SUCCESS] Loaded {len(x_test)} test images across {len(np.unique(y_true))} discovered classes.")


print("\n[INFO] Running predictions on test samples...")

y_pred_probs = model.predict(x_test, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

accuracy = np.mean(y_pred == y_true)
print(f"\n[RESULTS] Manual Test Accuracy: {accuracy * 100:.2f}%")

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(
    y_true, 
    y_pred, 
    labels=np.arange(num_classes), 
    target_names=class_names, 
    zero_division=0
))
print("-----------------------------")

print("[INFO] Plotting Test Confusion Matrix...")

cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Purples',
    xticklabels=class_names, 
    yticklabels=class_names
)
plt.title(f"Test Dataset Confusion Matrix (Accuracy: {accuracy*100:.2f}%)", fontsize=16)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.savefig(output_plot_path, bbox_inches='tight')
plt.close()

print(f"[DONE] Test completed! Matrix saved to {output_plot_path}")