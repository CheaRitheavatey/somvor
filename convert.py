# save_model_tf.py
import tensorflow as tf

try:
    # Try loading with custom objects
    model = tf.keras.models.load_model(
        'modelss/keras_model.h5',
        custom_objects={'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D},
        compile=False
    )
    
    # Save in TensorFlow format
    model.save('modelss/tf_model', save_format='tf')
    print("Model saved in TensorFlow format successfully!")
    
except Exception as e:
    print(f"Error: {e}")