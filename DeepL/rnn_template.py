from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import accuracy_score, precision_score

def rnn_network(X_train, y_train,X_test, y_test):
    # Obtener la longitud de secuencia y número de características
    max_len = X_train.shape[1]
    num_features = X_train.shape[2]

    # Build the model
    model = Sequential()
    model.add(LSTM(128, input_shape=(max_len, num_features)))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=1)

    # Evaluate the model
    y_pred = model.predict_classes(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    # Print evaluation results
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
