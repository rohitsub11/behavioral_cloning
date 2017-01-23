from keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout
from keras.layers import Dense, Activation, Lambda
from keras.models import Sequential
from keras.utils import np_utils
import json
import argparse

import pandas as pd

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
def gen_data((data_frame, batch_size=32):
    N = data_frame.shape[0]
    batches_per_epoch = N // batch_size

    i = 0
    while(True):
        start = i*batch_size
        end = start+batch_size - 1

        X_batch = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
        y_batch = np.zeros((batch_size,), dtype=np.float32)

        j = 0

        # slice a `batch_size` sized chunk from the dataframe
        # and generate augmented data for each row in the chunk on the fly
        for index, row in data_frame.loc[start:end].iterrows():
            X_batch[j], y_batch[j] = get_augmented_row(row)
            j += 1

        i += 1
        if i == batches_per_epoch - 1:
            # reset the index so that we can cycle over the data_frame again
            i = 0
        yield X_batch, y_batch

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
	data_frame = pd.read_csv('data/driving_log.csv', usecols=[0, 1, 2, 3])

    # shuffle the data
    data_frame = data_frame.sample(frac=1).reset_index(drop=True)

    # 80-20 training validation split
    training_split = 0.8

    num_rows_training = int(data_frame.shape[0]*training_split)

    training_data = data_frame.loc[0:num_rows_training-1]
    validation_data = data_frame.loc[num_rows_training:]

    # release the main data_frame from memory
    data_frame = None

    training_generator = gen_data(training_data, batch_size=args.batch_size)
    validation_data_generator = gen_data(validation_data, batch_size=args.batch_size)

	model = trial_model()

	model.fit_generator(training_generator, samples_per_epoch=10000, 
						nb_epoch=args.epoch)#, validation_data=validation_data_generator)

	# serialize model to JSON
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
    # serialize weights to HDF5
	model.save_weights("model.h5")
	print("Saved model to disk")