"""
    Este modulo genera la predicción de la imagen cargada por el usuario
    utilizando el modelo preentrenado MobileNetV2
"""
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import cv2

modelo_mobile = None


def cargar_modelo():
    modelo_mobile = tf.keras.models.load_model(
        'image_recognition/model/models_saved/modelo_MobileNetV2.hdf5')
    print("Modelo MobileNet Cargado")
    return modelo_mobile


def Mobile_predecir_imagen(imagen: Image.Image):
    global modelo_mobile
    if modelo_mobile is None:
        modelo_mobile = cargar_modelo()

    imagen = np.asarray(imagen.resize((224, 224)))[..., :3]
    imagen = np.expand_dims(imagen, 0)
    imagen = imagen / 127.5 - 1.0

    resultado = decode_predictions(modelo_mobile.predict(imagen), 2)[0]

    respuesta = []
    respuesta.append('MobileNetV2')
    for i, res in enumerate(resultado):
        resp = {}
        resp["prediccion"] = res[1]
        resp["seguridad"] = f"{res[2]*100:0.2f} %"

        respuesta.append(resp)

    return respuesta
