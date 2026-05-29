import os
import cv2

def process_images(root_dir):
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))

    counter = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            
            if filename.lower().endswith(('jpg')):
                full_path = os.path.join(dirpath, filename)

                #gray
                gray_img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

                if gray_img is None:
                    print(f"[WARNING] Failed to load: {full_path}!")
                    continue
                #contours
                enhanced_img = clahe.apply(gray_img)

                #saving
                cv2.imwrite(full_path, enhanced_img)

                counter += 1
                if counter % 100 == 0:
                    print(f"[INFO] {counter} images processed...")
    print("[DONE]\n")


PATH_TO_DATASET = "../dataset/asl-alphabet"
process_images(PATH_TO_DATASET)