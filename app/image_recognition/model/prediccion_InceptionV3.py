"""
    Este modulo genera la predicción de la imagen cargada por el usuario
    utilizando el modelo preentrenado InceptionV3
"""
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import cv2

modelo_inception = None


def cargar_modelo():

    modelo_inception = tf.keras.models.load_model(
        'image_recognition/model/models_saved/modelo_InceptionV3.hdf5')
    print("Modelo InceptionV3 Cargado")
    return modelo_inception


def Inception_predecir_imagen(imagen: Image.Image):
    global modelo_inception
    if modelo_inception is None:
        modelo_inception = cargar_modelo()

    imagen = np.asarray(imagen.resize((299, 299)))[..., :3]
    imagen = np.expand_dims(imagen, 0)
    imagen = imagen / 127.5 - 1.0

    resultado = decode_predictions(modelo_inception.predict(imagen), 2)[0]

    respuesta = []
    respuesta.append('InceptionV3')
    for i, res in enumerate(resultado):
        resp = {}
        resp["prediccion"] = res[1]
        resp["seguridad"] = f"{res[2]*100:0.2f} %"

        respuesta.append(resp)
    return respuesta
