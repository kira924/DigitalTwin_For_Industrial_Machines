import tensorflow as tf
import os
import sys

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def convert_to_tflite():
    # 1. Define the path to the saved_models directory
    saved_models_dir = os.path.join(parent_dir, 'saved_models')
    
    # 2. Define exact paths for the input Keras model and the output TFLite model
    keras_model_path = os.path.join(saved_models_dir, 'calibrated_model.keras')
    tflite_model_path = os.path.join(saved_models_dir, 'calibrated_model.tflite')

    print(f"Loading Keras model from: {keras_model_path}...")
    try:
        model = tf.keras.models.load_model(keras_model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Converting to TFLite format...")
    # 3. Initialize the TFLite Converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 4. Apply Optimization (Quantization) to reduce size for Mobile/Edge
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Note: Ensure complex layers like LSTM are supported during conversion
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, 
        tf.lite.OpsSet.SELECT_TF_OPS 
    ]

    # 5. Perform the conversion
    tflite_model = converter.convert()

    # 6. Save the optimized TFLite model in the saved_models folder
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    # 7. Compare sizes to verify optimization
    keras_size = os.path.getsize(keras_model_path) / 1024
    tflite_size = os.path.getsize(tflite_model_path) / 1024

    print("\nConversion Successful!")
    print(f"Original Keras Model Size: {keras_size:.2f} KB")
    print(f"Optimized TFLite Model Size: {tflite_size:.2f} KB")
    print(f"The TFLite model is saved in: {saved_models_dir}")
    print("Now you can send the TFLite file to the Mobile Developer!")

if __name__ == "__main__":
    convert_to_tflite()