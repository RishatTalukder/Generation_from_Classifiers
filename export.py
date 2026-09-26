import tensorflow as tf

model_path = "models_experiment_7(best)/weights.51-0.9977.keras"
output_path = "mnist_saved_model"

print(f"Loading: {model_path}")

model = tf.keras.models.load_model(model_path)

print("Model loaded successfully.")
print(model.summary())

print(f"Exporting SavedModel to: {output_path}")

model.export(output_path)

print("Export complete!")