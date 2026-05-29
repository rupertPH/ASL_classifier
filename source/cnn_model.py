import os
print("[INFO] Importing modules...")

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.layers import RandomRotation, RandomTranslation, RandomZoom
from keras.optimizers import Adam

import tensorflow as tf 
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from prep_dataset import prepare_datasets

#----------------------
# DATASET PREPARATION
#----------------------

print("[INFO] Preparing datasets...")
train_dataset_dir = "../dataset/asl-alphabet/asl_alphabet_train/asl_alphabet_train"

train_dataset, val_dataset = prepare_datasets(train_dataset_dir, validation_split=0.2)

print("[INFO] Plotting first batch...")
class_names = train_dataset.class_names

os.makedirs("../plots", exist_ok=True)
os.makedirs("../models", exist_ok=True)
os.makedirs("../logs/fit", exist_ok=True)

for images, labels in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"), cmap='gray' if images.shape[-1] == 1 else None)
        class_idx = int(labels[i].numpy())
        plt.title(class_names[class_idx])
        plt.axis("off")
    
    plt.savefig("../plots/train_dataset_first_batch.png", bbox_inches='tight')
    plt.close() 

print("[DONE] First batch plotted and saved.\n")

#-------------------------------------
# OPTIMALIZATION FOR APPLE SILICON M4
#-------------------------------------
print("[INFO] Setting up augmentation pipeline for datasets...")

# Dynamic augmentation pipeline outside the main model
augmentation_pipeline = Sequential([
    RandomRotation(factor=0.04, fill_mode="reflect"),
    RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="reflect"),
    RandomZoom(height_factor=0.1, width_factor=0.1, fill_mode="reflect")
])

print("[INFO] Freezing datasets directly into M4 unified RAM...")

# Apply augmentation and save directly into RAM cache
cached_train_ds = train_dataset.map(
    lambda x, y: (augmentation_pipeline(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
).cache()

cached_val_ds = val_dataset.cache()


print("[INFO] Pre-loading validation dataset into RAM...")
_ = [None for _ in cached_val_ds]
print("[DONE] Validation dataset fully cached.")

optimized_train_ds = cached_train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
optimized_val_ds = cached_val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

#---------------------
# TENSORBOARD
#---------------------

log_dir = "../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

#------------------------------
# SAVING ONLY THE BEST MODEL
#------------------------------
model_path = '../models/asl_classifier_cnn_model_v4.keras'

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path, 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

#------------------
# EARLY STOPPING
#------------------

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

#----------------------
# MODEL COMPILATION
#----------------------

num_classes = len(class_names) 

model = Sequential()

# Main CNN architecture 
model.add(tf.keras.layers.Rescaling(1.0 / 255.0, input_shape=(200, 200, 1)))

model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=(3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten()) 

model.add(Dense(256, activation="relu")) 
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))

'''
model.add(GlobalAveragePooling2D())

model.add(Dense(64, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.4))
'''
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
              metrics=['accuracy'])

#----------------------
# MODEL TRAINING
#----------------------

EPOCHS = 30

print("[INFO] Starting training...")

history = model.fit(
    optimized_train_ds,
    validation_data=optimized_val_ds,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[tensorboard_callback, checkpoint, early_stop]
)

print("[DONE]\n")

#----------------------
# SAVING MODEL
#----------------------

print("[INFO] Training finished. The best model was automatically saved in '../models/asl_classifier_cnn_model_v4.keras' during the process.")
print("[DONE]\n")

#----------------------------
# MODEL TRAINING PERFORMANCE
#----------------------------

print("[INFO] Calculating final confusion matrix...")

y_true = np.concatenate([y.numpy() for _, y in optimized_val_ds], axis=0)

y_pred_probs = model.predict(optimized_val_ds, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)


total_cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy()

print("\n--- FINAL VALIDATION CONFUSION MATRIX ---")
print(total_cm)
print("-----------------------------------------")

print("[INFO] Plotting accuracy and loss...")

pdf_path = "../plots/training_report_cnn_v4.pdf"
val_class_names = val_dataset.class_names

with PdfPages(pdf_path) as pdf:
    #---------------
    #accuracy & loss
    #---------------
    fig1 = plt.figure(figsize=(14, 5))
    #accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    #loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    pdf.savefig()
    plt.close(fig1)

    #---------------
    #confusion matrix
    #---------------
    fig2 = plt.figure(figsize=(12, 12))

    plt.imshow(total_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Final Validation Confusion Matrix", fontsize=16, pad=20)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)

    cm_normalized = total_cm.astype('float') / (total_cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    for i in range(num_classes):
        for j in range(num_classes):
            val = total_cm[i, j]
            if val > 0:
                color = "white" if cm_normalized[i, j] > 0.5 else "black"
                
                plt.text(j, i, str(int(val)), 
                         horizontalalignment="center", 
                         verticalalignment="center", 
                         color=color, 
                         fontsize=6, 
                         weight='bold')

    plt.ylabel("True ASL letter", fontsize=12)
    plt.xlabel("Predicted ASL letter", fontsize=12)
    
    plt.tight_layout()
    pdf.savefig(fig2)
    plt.close(fig2)

print("[DONE]\n")