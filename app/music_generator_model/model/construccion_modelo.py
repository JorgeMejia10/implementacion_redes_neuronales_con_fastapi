"""
    Este modulo prepara datos de archivos MIDI y los prepara para el entrenamiento de la Red Neuronal
"""
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


def entrenar_red():
    """ Entrena una Red Neuronal para generar musica """
    notas = obtener_notas()

    # obtiene la cantidad de tonos
    n_vocab = len(set(notas))

    red_entrada, red_salida = preparar_secuencia(notas, n_vocab)

    modelo = crear_red_neuronal(red_entrada, n_vocab)

    entrenar(modelo, red_entrada, red_salida)


def obtener_notas():
    """ Se obtienen todas las notas y acordes de los archivos MIDI en la carpeta ./model/midi_songs """
    notas = []

    for file in glob.glob("music_generator_model/model/midi_songs/*.mid"):
        try:
            midi = converter.parse(file)

            print("Analizando %s" % file)

            notas_to_parse = None

            try:  # El archivo tiene partes de instrumento
                s2 = instrument.partitionByInstrument(midi)
                notas_to_parse = s2.parts[0].recurse()
            except:  # El archivo tiene notas en una estructura plana
                notas_to_parse = midi.flat.notas

            for element in notas_to_parse:
                if isinstance(element, note.Note):
                    notas.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notas.append('.'.join(str(n) for n in element.normalOrder))
        except:
            print("No leido %s" % file)

    with open('music_generator_model/model/data/notes', 'wb') as filepath:
        pickle.dump(notas, filepath)

    return notas


def preparar_secuencia(notas, n_vocab):
    """ Prepara la sequencia utilizada para la Red Neuronal """
    sequence_length = 100

    # Se obtienen todos los pitch names
    pitchnames = sorted(set(item for item in notas))

    # Se crea un diccionario para mapear todos los pitches a entero
    note_to_int = dict((note, number)
                       for number, note in enumerate(pitchnames))

    red_entrada = []
    red_salida = []

    # Se crea la sequencia de entrada y sus correspondientes salidas
    for i in range(0, len(notas) - sequence_length, 1):
        sequence_in = notas[i:i + sequence_length]
        sequence_out = notas[i + sequence_length]
        red_entrada.append([note_to_int[char] for char in sequence_in])
        red_salida.append(note_to_int[sequence_out])

    n_patterns = len(red_entrada)

    # Se remodela la entrada a un formato compatible con las capas LSTM
    red_entrada = numpy.reshape(
        red_entrada, (n_patterns, sequence_length, 1))
    # Se normaliza la entrada
    red_entrada = red_entrada / float(n_vocab)

    red_salida = np_utils.to_categorical(red_salida)

    return (red_entrada, red_salida)


def crear_red_neuronal(red_entrada, n_vocab):
    """ Crea la estructura de la Red Neuronal """
    modelo = Sequential()
    modelo.add(LSTM(
        512,
        input_shape=(red_entrada.shape[1], red_entrada.shape[2]),
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

    return modelo


def entrenar(modelo, red_entrada, red_salida):
    """ Entrenamiento de la Red Neuronal """
    filepath = "music_generator_model/model/weights_saved/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    modelo.fit(red_entrada, red_salida, epochs=200,
               batch_size=128, callbacks=callbacks_list)


if __name__ == '__main__':
    entrenar_red()
