import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from scipy.sparse import csr_matrix as csrm

def rnn_network(x_train,y_train,x_test,y_test):
    x_train = [x.toarray().tolist()[0] for x in x_train]
    x_test = [x.toarray().tolist()[0] for x in x_test]
    # Definir parámetros
    vocab_size = 10000     # Tamaño del vocabulario
    max_length = 100       # Longitud máxima de las secuencias de entrada
    embedding_dim = 60     # Dimensión de los vectores de embedding
    hidden_units = 128      # Número de unidades en la capa oculta
    batch_size = 64        # Tamaño del lote (batch size)
    epochs = 20            # Número de epochs (iteraciones de entrenamiento)

    # Preprocesar los datos
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_test = sequence.pad_sequences(x_test, maxlen=max_length)

    # Construir el modelo
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
    model.add(SimpleRNN(hidden_units))
    model.add(Dense(1, activation='sigmoid'))

    # Compilar el modelo
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Entrenar el modelo
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    # Evaluar el modelo
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Loss:", loss)
    print("Accuracy:", accuracy)
