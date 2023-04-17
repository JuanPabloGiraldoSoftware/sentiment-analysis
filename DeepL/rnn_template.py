import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Embedding, Dropout
from keras.optimizers import Adam

# Cargar los datos de entrenamiento y prueba
# Cargar los datos de entrenamiento y prueba
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Especificar el número máximo de palabras y la longitud máxima de la secuencia
max_words = 5000
max_len = 100

model = Sequential()

# Agregar una capa de embedding
model.add(Embedding(input_dim=max_words, output_dim=32, input_length=max_len))

# Agregar la capa recurrente
model.add(SimpleRNN(units=64, activation='tanh'))

# Agregar una capa Dropout
model.add(Dropout(0.5))

# Agregar una capa oculta
model.add(Dense(units=32, activation='relu'))

# Agregar la capa de salida
model.add(Dense(units=1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])


model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test))

