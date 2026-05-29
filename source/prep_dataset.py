from tensorflow import keras

def prepare_datasets(train_dataset_dir, validation_split=0.2, seed=42):
    directory = train_dataset_dir
    
    print("[INFO] Preparing training dataset...")
    train_ds = keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        batch_size=128,
        image_size=(200, 200),
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset="training",
        interpolation="bilinear"
    )
    
    print("[INFO] Preparing validation dataset...")
    val_ds = keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        batch_size=128,
        image_size=(200, 200),
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset="validation",
        interpolation="bilinear"
    )
    
    print("-----------")
    print("CLASS NAMES:", train_ds.class_names)
    print("-----------")
    print("[DONE] Datasets prepared successfully.")
    
    return train_ds, val_ds


'''
print("[INFO] Plotting first batch...")
for images, labels in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        class_idx = int(labels[i].numpy())
        plt.title(class_names[class_idx])
        
        plt.axis("off")
        
    plt.show()

print("[DONE]\n")

'''