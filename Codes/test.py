import pandas as pd
import os

data_path = os.getcwd() + "/Data/UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt"




def load_X(X_signals_paths):
	X_signals = []
    
	file = open(X_signals_paths, 'r') # Read dataset from disk, dealing with text files' syntax
	X_signals.append(
		[np.array(serie, dtype=np.float32) for serie in [ row.replace('  ', ' ').strip().split(' ') for row in file]]
		)
        
	file.close()
    
	return np.transpose(np.array(X_signals), (1, 2, 0))


a = load_X(data_path)

print(a.shape)
