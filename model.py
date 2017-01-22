from keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout
from keras.layers import Dense, Activation, Lambda
from keras.models import Sequential
from keras.utils import np_utils
import json
import argparse
import csv
import matplotlib.pyplot as plt

#data path
data_folder = 'data/'

def trial_model():
	row, col, channels = 66, 200, 3 #camera format

	model = Sequential()
	model.add(Lambda(lambda x: x/255 - .5, input_shape = (row, col, channels),
					output_shape = (row, col, channels)))
	model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode="valid", activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode="valid", activation='relu'))
	model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode="valid", activation='relu'))
	#non strided convolutional layer
	model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode="valid", activation='relu'))
	model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode="valid", activation='relu'))

	#FC layers
	model.add(Flatten())
	model.add(Dense(1164, activation='relu'))

	model.add(Dense(100, activation='relu'))
	
	model.add(Dense(50, activation='relu'))
	
	model.add(Dense(10, activation='relu'))
	
	model.add(Dense(1))

	model.summary()
	model.compile(optimizer='adam', loss='mse')

	return model

#to load the data from saved file
def get_data(X, y, data_folder, delta):
	log_path = data_folder + 'driving_log.csv'
	logs = []

	with open(log_path, 'rt') as f:
		reader = csv.reader(f)
		for line in reader:
			logs.append(line)
		logs_labels = logs.pop(0)

	# load center camera image
	for i in range(len(logs)):
		img_path = logs[i][0]
		img_path = data_folder+'IMG'+(img_path.split('IMG')[1]).strip()
		img = plt.imread(img_path)
		X.append(img)#image_preprocessing(img))
		y.append(float(logs[i][3]))

	# load left camera image
	for i in range(len(logs)):
		img_path = logs[i][1]
		img_path = data_folder+'IMG'+(img_path.split('IMG')[1]).strip()
		img = plt.imread(img_path)
		X.append(img)#image_preprocessing(img))
		y.append(float(logs[i][3]) + delta)

	# load right camera image
	for i in range(len(logs)):
		img_path = logs[i][2]
		img_path = data_folder+'IMG'+(img_path.split('IMG')[1]).strip()
		img = plt.imread(img_path)
		X.append(img)#image_preprocessing(img))
		y.append(float(logs[i][3]) - delta)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Steering angle model trainer')
	parser.add_argument('--batch', type=int, default=128, help='Batch size.')
	parser.add_argument('--epoch', type=int, default=10, help='Number of epochs.')
	parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
	parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')
	parser.set_defaults(skipvalidate=False)
	parser.set_defaults(loadweights=False)
	args = parser.parse_args()

	print("loading data...")

	data={}
	data['features'] = []
	data['labels'] = []

	get_data(data['features'], data['labels'],data_folder,0.3)

	model = trial_model()

	#model.fit_generator()

	# serialize model to JSON
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
    # serialize weights to HDF5
	model.save_weights("model.h5")
	print("Saved model to disk")