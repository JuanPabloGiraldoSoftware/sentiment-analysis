from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.models import Sequential
from keras.layers import  Dense, Embedding, LSTM, Dropout, TextVectorization, Input
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer 
from keras.callbacks import EarlyStopping

def rnn_network_vect_layer(df_rev, max_features, padding_len):
  X = df_rev["review"]
  Y = df_rev["sentiment"] 

  #Vectorization Layer
  vect_layer = TextVectorization(
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=padding_len)

  #Train/Test Split
  x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.30)

  #Building Model
  model = Sequential()
  model.add(Input(shape=(1,),dtype=tf.string))
  model.add(vect_layer)
  model.add(Embedding(max_features, 128, input_length=padding_len))
  model.add(LSTM(128))
  model.add(Dropout(0.5))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


  ## Hyperparameters
  batch_size = 1024
  epochs = 20
  #val_split = 0.2
  
  #Model Checking
  vect_layer.adapt(x_train)
  print("Pass adapt vect")
  model.fit(x_train, y_train, batch_size= batch_size,epochs=epochs)
  print("Pass model fiting")
  #vect_layer.adapt(x_test)
  model.evaluate(x_test, y_test)
  print("Pass model eval")


def rnn_network_first_vect(df_rev, max_features, padding_len):
  X = df_rev["review"]
  Y = df_rev["sentiment"] 

  #Train/Test Split
  x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.30)

  rev_tokenizer = Tokenizer(num_words=max_features)
  rev_tokenizer.fit_on_texts(x_train)
  x_train_vec= rev_tokenizer.texts_to_sequences(x_train)
  x_test_vec= rev_tokenizer.texts_to_sequences(x_test)
  vocab_size = len(rev_tokenizer.word_index) + 1
  
  vx_train_pad = pad_sequences(x_train_vec, maxlen=padding_len)
  vx_test_pad = pad_sequences(x_test_vec, maxlen=padding_len)

  #EarlyStopping
  callback = EarlyStopping(monitor='val_accuracy', patience=5)

  #Building Model
  model = Sequential()
  model.add(Embedding(vocab_size, 128, input_length=padding_len))
  model.add(LSTM(128))
  model.add(Dense(64, activation='sigmoid'))
  model.add(Dropout(0.5))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  ## Hyperparameters
  batch_size = 1024
  epochs = 20
  val_split = 0.2
  
  #Model Checking
  model.fit(vx_train_pad, y_train,validation_split= val_split, epochs=epochs, batch_size=batch_size, callbacks=[callback])
  loss, accuracy = model.evaluate(vx_train_pad, y_train, verbose=False)
  print("Training Accuracy: {:.4f}".format(accuracy))
  loss, accuracy = model.evaluate(vx_test_pad, y_test, verbose=False)
  print("Testing Accuracy:  {:.4f}".format(accuracy))