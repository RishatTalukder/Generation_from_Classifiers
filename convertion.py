import tensorflow as tf
import os

# 1. Load your model
model_path = "models_experiment_7(best)/weights.51-0.9977.keras"
model = tf.keras.models.load_model(model_path)

# 2. Create an output directory
output_dir = "mnist_web_model"
os.makedirs(output_dir, exist_ok=True)

# 3. Save model architecture to JSON and weights to .h5
# TensorFlow.js can read this format out of the box!
with open(os.path.join(output_dir, "model.json"), "w") as f:
    f.write(model.to_json())

model.save_weights(os.path.join(output_dir, "model.weights.h5"))

print(f"Successfully exported directly to native format in '{output_dir}'!")
