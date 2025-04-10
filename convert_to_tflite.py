import tensorflow as tf

# Load your existing .keras model
model = tf.keras.models.load_model('cotton_model_v2.keras')

# Create the converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: Optimize for size/performance
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open('cotton_model_v2.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ TFLite model saved as 'cotton_model_v2.tflite'")
