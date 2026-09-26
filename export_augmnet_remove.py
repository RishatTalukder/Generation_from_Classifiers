import tensorflow as tf

model_path = "models_experiment_7(best)/weights.51-0.9977.keras"

print(f"Loading: {model_path}")

model = tf.keras.models.load_model(model_path)

print("Original model:")
model.summary()


# The first layer is your data augmentation Sequential.
# The actual classifier starts after it.
classifier = tf.keras.Sequential(
    model.layers[1:],
    name="mnist_classifier"
)

# Build the inference model.
classifier.build((None, 28, 28, 1))

print("\nInference model:")
classifier.summary()


# Verify that the weights survived.
print("\nTesting inference...")

dummy_input = tf.zeros((1, 28, 28, 1))
output = classifier(dummy_input, training=False)

print("Input shape :", dummy_input.shape)
print("Output shape:", output.shape)
print("Output      :", output.numpy())


# Export
output_path = "mnist_classifier_saved_model"

print(f"\nExporting to: {output_path}")

classifier.export(output_path)

print("\n✅ Inference model exported successfully!")