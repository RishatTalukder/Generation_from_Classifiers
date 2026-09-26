import tensorflow as tf
import tensorflowjs as tfjs
import os

# 1. Path to your trained model file
model_path = "models_experiment_7(best)/weights.51-0.9977.keras"
print(f"Loading Keras model from: {model_path}")
model = tf.keras.models.load_model(model_path)

# 2. Define the output folder path
output_dir = "tfjs_model"

# 3. Perform the conversion
print("Converting model to TensorFlow.js format...")
tfjs.converters.save_keras_model(model, output_dir)

print(f"🎉 Success! Your web-ready model is saved in the '{output_dir}' folder.")
print("You should see 'model.json' and one or more '.bin' shard files inside it.")
