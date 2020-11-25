"""
    Este modulo genera notas para un archivo midi usando la red neuronal entrenada
"""
import pickle
import numpy
from datetime import datetime
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Activation


def generar():
    """ Generar un archivo midi tipo Piano"""
    # Carga las notas utilizadas para entrenar el modelo
    with open('music_generator_model/model/data/notes', 'rb') as filepath:
        notas = pickle.load(filepath)

    nombre_tono = sorted(set(item for item in notas))

    n_vocab = len(set(notas))

    network_input, normalized_input = preparar_secuencia(
        notas, nombre_tono, n_vocab)
    modelo = create_network(normalized_input, n_vocab)
    prediccion = generar_notas(
        modelo, network_input, nombre_tono, n_vocab)

    return crear_midi(prediccion)


def preparar_secuencia(notas, nombre_tono, n_vocab):
    """ Preparar las secuencias utilizadas por la red neuronal """
    note_to_int = dict((note, number)
                       for number, note in enumerate(nombre_tono))

    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notas) - sequence_length, 1):
        sequence_in = notas[i:i + sequence_length]
        sequence_out = notas[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_acordes = len(network_input)

    # Remodela la entrada en un formato compatible con las capas LSTM
    normalized_input = numpy.reshape(
        network_input, (n_acordes, sequence_length, 1))
    # Normalizacion de entradas
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)


def create_network(network_input, n_vocab):
    """ Se crea la estructura de la red neuronal """
    modelo = Sequential()
    modelo.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    modelo.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    modelo.add(LSTM(512))
    modelo.add(BatchNorm())
    modelo.add(Dropout(0.3))
    modelo.add(Dense(256))
    modelo.add(Activation('relu'))
    modelo.add(BatchNorm())
    modelo.add(Dropout(0.3))
    modelo.add(Dense(n_vocab))
    modelo.add(Activation('softmax'))
    modelo.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Carga los pesos en cada nodo
    modelo.load_weights(
        'music_generator_model/model/weights_saved/weight.hdf5')
    return modelo


def generar_notas(modelo, network_input, nombre_tono, n_vocab):
    """ Genera notas para la red neuronal basada en una sequencia de notas"""
    # Se elige una secuencia aleatoria como punto de partida para la predicción
    iniciar = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note)
                       for number, note in enumerate(nombre_tono))

    acorde = network_input[iniciar]
    prediccion = []

    print('COMENZARA EL FOR DE GENERACIÓN DE NOTAS')
    for note_index in range(100):
        print(note_index)
        entrada_prediccion = numpy.reshape(acorde, (1, len(acorde), 1))
        entrada_prediccion = entrada_prediccion / float(n_vocab)

        prediction = modelo.predict(entrada_prediccion, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediccion.append(result)

        acorde.append(index)
        acorde = acorde[1:len(acorde)]
    print('FINALIZO LA GENERACION DE NOTAS')
    return prediccion


def crear_midi(prediccion):
    """
        Convierte la salida de la prediccion en notas y se crea un archivo midi a partir de las notas
    """
    offset = 0
    notas_salida = []
    # Crea objetos de notas y acordes basados en los valores generados por el modelo
    for acorde in prediccion:
        # acorde es un acorde
        if ('.' in acorde) or acorde.isdigit():
            notas_en_acorde = acorde.split('.')
            notas = []
            for nota_actual in notas_en_acorde:
                nueva_nota = note.Note(int(nota_actual))
                nueva_nota.storedInstrument = instrument.Piano()
                notas.append(nueva_nota)
            nuevo_acorde = chord.Chord(notas)
            nuevo_acorde.offset = offset
            notas_salida.append(nuevo_acorde)
        # acorde es una nota
        else:
            nueva_nota = note.Note(acorde)
            nueva_nota.offset = offset
            nueva_nota.storedInstrument = instrument.Piano()
            notas_salida.append(nueva_nota)

        # Se aumenta el desplazamiento en cada iteracióm para que las notas no se apilen
        offset += 0.5

    midi_stream = stream.Stream(notas_salida)
    fecha = datetime.now()
    fecha = f'{fecha.year}-{fecha.month}-{fecha.day}_{fecha.hour}-{fecha.minute}-{fecha.second}'
    nombre = f'music_{fecha}'
    midi_stream.write('midi', fp=f'./static/music_predicted/{nombre}.mid')
    print('Se guardo la canción')
    return {"nombre": nombre}


if __name__ == '__main__':
    generar()
