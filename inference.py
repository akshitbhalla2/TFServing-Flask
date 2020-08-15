import tensorflow as tf
import numpy as np
import json
import requests
# import pillow

MODEL_URI = "http://localhost:8501/v1/models/pets:predict"
def get_prediction(image_path):
    image = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size= (128, 128)
    )
    image = tf.keras.preprocessing.image.img_to_array(image)

    # Preprocessing in the same manner as training data
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    data = json.dumps({
        "instances": image.tolist(),
    })
    response = requests.post(
        MODEL_URI,
        data=data.encode()
    )
    result = json.loads(response.text)
    pred = result["predictions"][0][0]
    if pred >= 0.5:
        return "DOG"
    else:
        return "CAT"

    #     return {
    #         "pet": "dog",
    #         "confidence": pred
    #     }
    # else:
    #     return{
    #         "pet": "not dog",
    #         "confidence": 1-pred
    #     }