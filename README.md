# Implementación de Redes Neuronales con FastAPI

En este proyecto se realizo la implementación de dos tipos de Redes Neuronales utilizando el framework FasAPI y Tensorflow/Keras, dicha implementación consiste en 3 interfaces sencillas en las cuales se pueden evidenciar las predicciones de los modelos.
La implementación se puede dividir a su vez en dos proyectos los cuales son:

  - Generador de Música con una RNN
  - Reconocimiento de Imagenes con CNN
  
  ## Generador de Música con RNN
  
  ### Descripción
  Se entrenó un modelo RNN con canciones de piano en formato MIDI, posteriormente se realizo el guardado de los pesos en cada epoca del entrenamiento.
  Posteriormente se dejo el ultimo peso generado, el cual se encuentra en la carpeta [weights_saved] para realizar la predicción del nuevo archivo MIDI.
  Para la reproducción del archivo generado en la pagina web se utilizo [MIDIjs].
  
  #### Nota
  Se debe tener en cuenta que el modelo solo genera musica de un solo instrumento.
 
  ## Reconocimiento de Imagenes con CNN
  
  ### Descripción
  Para este segundo proyecto se utilizaron 3 modelos previamente entrenados [MobileNetV2], [InceptionV3] y [Xception]. La finalidad de este proyecto fue realizar la comparación de predicciones entre los 3.
  Dicha comparación se puede evidenciar en una tabla como la siguiente:
  
  | Modelo | Predicción | % Seguridad |
  | ------ | ------ | ------ |
  | MobileNetV2 | lion | 59.28% |
  | InceptionV3 | lion | 92.27 % |
  | Xception | lion | 93.86 % |

  #### Nota
  Cabe destacar que la imagen que se predijo aparece en la parte superior de la tabla.
  
  ## Requisitos
  - [Tensorflow]
  - [Keras]
  - [Fastapi]
  - [music21]
  - [uvicorn]
  - [h5py]
  - [jinja2]
  - [aiofiles]
  
  ## Trabajo futuro
  - Agregar la función de poder entrenar nuevos modelos que generen música con datos propios y que los pesos generados se puedan descargar.
  - Implementar mas modelos pre-entrenados al proyecto de reconocimiento de imagenes.
  - Mejorar las interfaces del proyecto.
  
  [weights_saved]: <https://github.com/JorgeMejia10/implementacion_redes_neuronales_con_fastapi/tree/main/app/music_generator_model/model/weights_saved>
  [MIDIjs]: <https://www.midijs.net/>
  [MobileNetV2]: <https://keras.io/api/applications/mobilenet/#mobilenetv2-function>
  [InceptionV3]: <https://keras.io/api/applications/inceptionv3/>
  [Xception]: <https://keras.io/api/applications/xception/>
  [Tensorflow]: <https://www.tensorflow.org/install>
  [Keras]: <https://keras.io/>
  [FastAPI]: <https://fastapi.tiangolo.com/>
  [music21]: <https://pypi.org/project/music21/>
  [uvicorn]: <https://pypi.org/project/uvicorn/>
  [h5py]: <https://pypi.org/project/h5py/>
  [jinja2]: <https://pypi.org/project/Jinja2/>
  [aiofiles]: <https://pypi.org/project/aiofiles/>
  
