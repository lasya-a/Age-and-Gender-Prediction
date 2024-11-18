import tensorflow as tf
from tensorflow.keras.models import model_from_json # type: ignore
from tensorflow.keras.utils import get_custom_objects # type: ignore

def init():
    # Register the 'Functional' class with Keras serialization system
    get_custom_objects().update({'Functional': tf.keras.Model})

    with open('lc.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json, custom_objects={'Functional': tf.keras.Model})
        loaded_model.load_weights("lc.h5")
        print("Loaded Model from disk")
        loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    return loaded_model
