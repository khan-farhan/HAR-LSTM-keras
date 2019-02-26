import numpy as np
import os


Accelerometer = ["body_acc_x_", "body_acc_y_", "body_acc_z_"]

Activity = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

train_signals_path = [os.getcwd() + "/../Data/UCI HAR Dataset/train/Inertial Signals/" + Accelerometer[i] + "train.txt" for i in range(3)]
test_signals_path =  [os.getcwd() + "/../Data/UCI HAR Dataset/test/Inertial Signals/" +  Accelerometer[i] + "test.txt" for i in range(3)]

train_labels_path = os.getcwd() + "/../Data/UCI HAR Dataset/train/y_train.txt"
test_labels_path = os.getcwd() + "/../Data/UCI HAR Dataset/test/y_test.txt"

def one_hot(Y):
    Y = Y.reshape(len(Y))
    n_values = int(np.max(Y)) + 1
    return np.eye(n_values)[np.array(Y, dtype=np.int32)]  # Returns FLOATS


def load_train_acc_data():
	Acc = []

	for path in train_signals_path:
		file = open(path,'r')
		Acc.append([np.array(elem, dtype=np.float32) for elem in [ row.replace('  ', ' ').strip().split(' ') for row in file ]])
		file.close()

	return np.transpose(np.array(Acc), (1, 2, 0))



def load_train_labels():
	file = open(train_labels_path, 'r')
	labels =  np.array([elem for elem in [ row.replace('  ', ' ').strip().split(' ') for row in file ]], dtype=np.int32 )
	return one_hot(labels - 1)


def load_test_acc_data():
	Acc = []

	for path in test_signals_path:
		file = open(path,'r')
		Acc.append([np.array(elem, dtype=np.float32) for elem in [ row.replace('  ', ' ').strip().split(' ') for row in file ]])
		file.close()

	return np.transpose(np.array(Acc), (1, 2, 0))



def load_test_labels():
	file = open(test_labels_path, 'r')
	labels =  np.array([elem for elem in [ row.replace('  ', ' ').strip().split(' ') for row in file ]], dtype=np.int32 )
	return one_hot(labels - 1)




