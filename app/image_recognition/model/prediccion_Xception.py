"""
    Este modulo genera la predicción de la imagen cargada por el usuario
    utilizando el modelo preentrenado Xception
"""
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import cv2

modelo_xception = None


def cargar_modelo():
    modelo_xception = tf.keras.models.load_model(
        'image_recognition/model/models_saved/modelo_Xception.hdf5')
    print("Modelo Xception Cargado")
    return modelo_xception


def Xception_predecir_imagen(imagen: Image.Image):
    global modelo_xception
    if modelo_xception is None:
        modelo_xception = cargar_modelo()

    imagen = np.asarray(imagen.resize((299, 299)))[..., :3]
    imagen = np.expand_dims(imagen, 0)
    imagen = imagen / 127.5 - 1.0

    resultado = decode_predictions(modelo_xception.predict(imagen), 2)[0]

    respuesta = []
    respuesta.append('Xception')
    for i, res in enumerate(resultado):
        resp = {}
        resp["prediccion"] = res[1]
        resp["seguridad"] = f"{res[2]*100:0.2f} %"

        respuesta.append(resp)

    return respuesta
