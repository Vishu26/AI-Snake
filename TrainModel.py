from keras.layers import Dense, Dropout
from keras.models import Sequential
import numpy as np
import tensorflowjs as tfjs

with open('x_train.txt', 'r') as f:
	x = f.read().split(',')
x = list(map(float, x))

with open('y_train.txt', 'r') as f:
	y = f.read().split(',')
y = np.array([list(map(int, y[i:i+5])) for i in range(0, len(y), 5)])

model = Sequential()
model.add(Dense(units=256, activation='relu', input_shape=[1], kernel_regularizer='l1'))
model.add(Dense(units=512, activation='relu', kernel_regularizer='l1'))
model.add(Dropout(0.5))
model.add(Dense(units=512, activation='relu', kernel_regularizer='l1'))
model.add(Dense(units=5, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(x, y, epochs=25)

tfjs.converters.save_keras_model(model, './keras_model')