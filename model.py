from keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout
from keras.layers import Dense, Activation, Lambda
from keras.models import Sequential
from keras.utils import np_utils
import json
import argparse

def trial_model():
	channels, row, col = 3, 160, 320 #camera format

	model = Sequential()
	model.add(Lambda(lambda x: x/255 - 0.5, input_shape = (channels, row, col),
					output_shape = (channels, row, col)))
	model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode="valid", activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode="valid", activation='relu'))
	model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode="valid", activation='relu'))
	#non strided convolutional layer
	model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode="valid", activation='relu'))
	model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode="valid", activation='relu'))

	#FC layers
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(50, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))

	model.summary()
	model.compile(optimizer='adam', loss='mse')

	return model

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Steering angle model trainer')
	parser.add_argument('--batch', type=int, default=128, help='Batch size.')
	parser.add_argument('--epoch', type=int, default=10, help='Number of epochs.')
	parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
	parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')
	parser.set_defaults(skipvalidate=False)
	parser.set_defaults(loadweights=False)
	args = parser.parse_args()

	model = trial_model()

	#model.fit_generator()