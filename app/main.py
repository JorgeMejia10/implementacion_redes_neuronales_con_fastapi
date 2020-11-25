import os
from enum import Enum
from io import BytesIO
from typing import List
from PIL import Image, ImageOps
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from image_recognition.model.prediccion_MobileNetV2 import Mobile_predecir_imagen
from image_recognition.model.prediccion_InceptionV3 import Inception_predecir_imagen
from image_recognition.model.prediccion_Xception import Xception_predecir_imagen
from music_generator_model.model import predecir_musica


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


def lista_musica():
    dir = './static/music_predicted/'
    content = os.listdir(dir)
    musica: list = []
    for file in content:
        if os.path.isfile(os.path.join(dir, file)) and file.endswith('.mid'):
            musica.append(file)
    return musica


def leer_imagen2file(file) -> Image.Image:
    imagen = Image.open(BytesIO(file))
    imagen.save("static/images/img.png")
    return imagen


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/reconocimiento_imagen/", response_class=HTMLResponse)
async def inicio(request: Request):
    return templates.TemplateResponse("reconocimiento_imagen.html", {"request": request})


@app.post("/reconocimiento_imagen/predecir/")
async def prediccion_imagen(request: Request, file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "¡La imagen debe ser en formato jpg o png!"
    imagen = leer_imagen2file(await file.read())
    predicciones = []
    predicciones.append(Mobile_predecir_imagen(imagen))
    predicciones.append(Inception_predecir_imagen(imagen))
    predicciones.append(Xception_predecir_imagen(imagen))
    return templates.TemplateResponse("reconocimiento_imagen.html", {"request": request, "predicciones": predicciones})


@app.get("/predecir_musica/", response_class=HTMLResponse)
async def inicio(request: Request):
    musica: List[str] = lista_musica()
    return templates.TemplateResponse("generador_musica.html", {"request": request, "lista_musica": musica})


@app.get("/predecir_musica/{action}", response_class=HTMLResponse)
async def read_item(request: Request, action: str):
    if action == "predecir":
        prediction = (predecir_musica.generar())
    musica: list = lista_musica()
    return templates.TemplateResponse("generador_musica.html", {"request": request, "lista_musica": musica, "nombre": prediction["nombre"]})
