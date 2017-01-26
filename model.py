from keras.layers import Dense, Conv2D, Flatten, Lambda
from keras.models import Sequential
from keras.layers import Dropout, ELU, Activation
import cv2
import math
import pandas as pd
import numpy as np
import json
import argparse
from keras.preprocessing.image import img_to_array, load_img

#data path
data_path= 'data/driving_log.csv'

rows = 66
cols = 200
channels = 3

EPOCH = 10
BATCH_SIZE = 32


def get_model():

	model = Sequential()

	model.add(Lambda(lambda x: x/255 - .5, input_shape = (rows, cols, channels),
	                                output_shape = (rows, cols, channels)))

	model.add(Conv2D(24, 5, 5, subsample=(2,2), border_mode='valid', name='hidden_1'))
	model.add(ELU())

	model.add(Conv2D(36, 5, 5, subsample=(2,2), border_mode='valid', name='hidden_2'))
	model.add(ELU())

	model.add(Conv2D(48, 5, 5, subsample=(2,2), border_mode='valid', name='hidden_3'))
	model.add(ELU())

	model.add(Conv2D(64, 3, 3, subsample=(1,1), border_mode='valid', name='hidden_4'))
	model.add(ELU())

	model.add(Conv2D(64, 3, 3, subsample=(1,1), border_mode='valid', name='hidden_5'))

	model.add(Flatten())
	model.add(Dropout(.5))#.2))
	model.add(ELU())


	model.add(Dense(100, name='FC1'))
	model.add(Dropout(.5))
	model.add(ELU())

	model.add(Dense(50, name='FC2'))
	model.add(ELU())

	model.add(Dense(10, name='FC3'))
	model.add(ELU())

	model.add(Dense(1, name='Steering_Control'))

	model.summary()

	model.compile(optimizer='adam', loss='mse')

	return model

def get_data(filenamewithpath):

	print("loading data...")
	#only first 4 columns matter
	data_frame = pd.read_csv(filenamewithpath, usecols=[0, 1, 2, 3])

	#shuffle the data
	data_frame = data_frame.sample(frac=1).reset_index(drop=True)

	print("Done loading data")

	return data_frame

def split_data(data):
	
	print("splitting data...")
	split_factor = 0.8

	num_rows_training = int(data.shape[0]*split_factor)

	tdf = data.loc[0:num_rows_training-1]
	vdf = data.loc[num_rows_training:]

	print("Done splitting data")
	return tdf, vdf

def brightness_augmentation(image):

	image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	random_brightness = .25 + np.random.uniform()

	image1[:,:,2] = image1[:,:,2]*random_brightness
	image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)

	return image1

def preprocess_image(image):

	shape = image.shape
	image = image[math.floor(shape[0]/5): shape[0]-25, 0:shape[1]]
	image = cv2.resize(image, (cols, rows), interpolation=cv2.INTER_AREA)

	return image

def get_augmented_image(row):
	
	steering = row['steering']

	camera_view = np.random.choice(['left', 'center', 'right'])

	if camera_view == 'left':
		steering += 0.25
	elif camera_view == 'right':
		steering -= 0.25
	else:
		steering = steering

	path_of_file = ("data/" + row[camera_view].strip())

	image = cv2.imread(path_of_file)

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	#brightness augmentation
	image = brightness_augmentation(image)

	#image resizing and processing
	image = preprocess_image(image)

	image = img_to_array(image)

	# decide whether to horizontally flip the image:
	# This is done to reduce the bias for turning left that is present in the training data
	flip_prob = np.random.random()
	if flip_prob > 0.5:
		# flip the image and reverse the steering angle
		steering = -steering
		image = cv2.flip(image, 1)

	return image, steering


def gen_data(dataframe, batch_size=32):

	N = dataframe.shape[0]
	batches_per_epoch = N // batch_size

	i = 0
	while(True):
		start = i*batch_size
		end = start+batch_size - 1

		X_batch = np.zeros((batch_size, rows, cols, 3), dtype=np.float32)
		y_batch = np.zeros((batch_size,), dtype=np.float32)

		j = 0

		# slice a `batch_size` sized chunk from the dataframe 
		# and generate augmented data for each row in the chunk 
		# on the fly
		for index, row in dataframe.loc[start:end].iterrows():
			X_batch[j], y_batch[j] = get_augmented_image(row)
			j += 1

		i += 1
		if i == batches_per_epoch - 1:
			# reset the index so that we can cycle over the 
			# data_frame again
			i = 0
		yield X_batch, y_batch

if __name__ == "__main__":

	temp_data = get_data(data_path)

	training_data, validation_data = split_data(temp_data)

	temp_data = None

	gen_train_data = gen_data(training_data, batch_size=BATCH_SIZE)

	gen_valid_data = gen_data(validation_data, batch_size=BATCH_SIZE)

	model = get_model()

	model.fit_generator(gen_train_data, samples_per_epoch=20000, nb_epoch=EPOCH,
						validation_data=gen_valid_data, nb_val_samples=3000)

	# serialize model to JSON
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	print("Saved model to disk")

	# serialize weights to HDF5
	model.save_weights("model.h5")
	print("Saved weights to disk")