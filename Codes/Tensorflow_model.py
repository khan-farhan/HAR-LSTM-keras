import numpy as np 
import os
import tensorflow as tf 
import data_load 
import time

################################################## Global Varaibles ################################
no_hidden = 32
no_input = 3
no_classes = 6
no_sampling_steps = 128
no_epoch = 100
batchsize = 100
learning_rate = 0.0025

x = tf.placeholder(tf.float32, [None, no_sampling_steps, no_input])
y = tf.placeholder(tf.float32, [None, no_classes])


weights = {
      
			'hidden': tf.Variable(tf.random_normal([no_input, no_hidden])), 
			'out': tf.Variable(tf.random_normal([no_hidden, no_classes], mean=1.0))
}

biases = {
    
			'hidden': tf.Variable(tf.random_normal([no_hidden])),
			'out': tf.Variable(tf.random_normal([no_classes]))
}


############################################# loading data #################################

Train_acc = data_load.load_train_acc_data()
Test_acc = data_load.load_test_acc_data()
Train_labels = data_load.load_train_labels()
Test_labels = data_load.load_test_labels()


print(Train_acc.shape, Test_acc.shape,Test_labels.shape,Train_labels.shape )

########################################## Creating LSTM Network ############################


def LSTM(X,weight,bias):
	X = tf.transpose(X, [1, 0, 2]) 
	X = tf.reshape(X, [-1,no_input])
	X = tf.nn.relu(tf.matmul(X, weights['hidden']) + biases['hidden'])
	X = tf.split(X, no_sampling_steps, 0)
	
	lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(no_hidden, forget_bias=1.0, state_is_tuple=True)
	lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(no_hidden, forget_bias=1.0, state_is_tuple=True)
	lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
	outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, X, dtype=tf.float32)
	lstm_last_output = outputs[-1]

	return tf.matmul(lstm_last_output, weights['out']) + biases['out']

######################################### Loading batches ##########################################

def loadBatch(X,Y,index):
	features = []
	labels = []

	for i in range(batchsize):
		features.append(X[(index*batchsize + i)%len(X)])
		labels.append(Y[(index*batchsize + i)%len(Y)])

	return np.array(features), np.array(labels)



###############################################################################################


pred = LSTM(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))  # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

################################################## Training the network #########################

def train():
	saver = tf.train.Save()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		total_loss = []
		total_acc = []
		for epoch in range(no_epoch):
			start_time = time.time()
			index = 0
			epoch_loss =  0
			epoch_accuracy =  0
			while(index*batchsize <= len(Train_acc)):
				print("loaded epoch ", epoch, "index is ", index )
				batch_x , batch_y = loadBatch(Train_acc,Train_labels,index)
				#print(batch_x.shape, batch_y.shape)
				_, loss, acc = sess.run( [optimizer, cost, accuracy], feed_dict={ x: batch_x, y: batch_y } )
				epoch_loss += loss
				epoch_accuracy += acc
				index += 1
			total_loss.append(epoch_loss)
			tota .append(epoch_accuracy)
			print("After ", epoch , "epochs the loss is ", epoch_loss ,"and accuracy is " , epoch_accuracy)
			print("Epoch Time is :" , time.time() - start_time)

		print("Training finished")


################################################## Testing ####################################################

def test():
	
	print("Testing Begin!!!")

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		index = 0
		test_loss = 0
		test_acc = 0

		while(index*batchsize <= len(Test_acc)):
			batch_x, batch_y = loadBatch(Test_acc,Test_labels,index)
			acc, loss = sess.run( [accuracy, cost], feed_dict={ x: batch_x, y: batch_y })
			print(index)
			index += 1
			test_loss += loss
			test_acc += acc

		print("Test loss " , test_loss ," and Accuracy is " ,test_acc)




if __name__ == '__main__':
	train()
	test()



