import os
import keras

model_path = "../models/asl_classifier_dnn_model_v1b.keras"

os.makedirs("../plots", exist_ok=True)
output_image_path = "../plots/model_visualization_cnn_v4.png"

print(f"[INFO] Loading model from {model_path}...")
model = keras.models.load_model(model_path)
print("[OK] Model loaded.")

print("[INFO] Generating network plot...")
try:
    keras.utils.plot_model(
        model,
        to_file=output_image_path,
        show_shapes=True,
        show_dtype=False,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=150
    )
except Exception as e:
    print("[FAILED] Failed to generate plot. Reason: {e}")