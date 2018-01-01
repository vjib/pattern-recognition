from openpyxl import load_workbook
from openpyxl import Workbook

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from statistics import mean

wb = load_workbook('C&H.xlsx')
sheet = wb.active

data={};

i=0;

yes_sample=[]
no_sample=[]

for row in sheet.iter_rows():
	if i>=1:
		size=len(row)
		name=row[0].value
		state=0
		col=12
		#print(name)
		p=[]
		while col<size and row[col].value is not None:
			if col>=12:
				price=row[col].value
				#print(price)
				p.append(price)
			col=col+1;
		data[name]=p
		if row[1].value==1:
			state=1
			yes_sample.append(p)
		else:
			no_sample.append(p)
		#print(state)

	i=i+1

def initSample( yes,no ):
	in_group=[]
	out_group=[]
	k=10
	for sample in yes:
		total=len(sample)
		p=[]
		for i in range(0,k):
			index=total*(i)/(k)
			index=math.floor(index)
			price=sample[index]
			min2=min(sample)
			max2=max(sample)
			#if len(sample[index:math.floor(index+(i/k))])>0:
				#price=sum(sample[index:math.floor(index+(i/k))])/len(sample[index:math.floor(index+(i/k))])
			price=(price-min2)/(max2-min2)
			p.append(price)
		in_group.append(p)
		out_group.append([1.0,0.0])
	for sample in no:
		total=len(sample)
		p=[]
		for i in range(0,k):
			index=total*(i)/(k)
			index=math.floor(index)
			price=sample[index]
			min2=min(sample)
			max2=max(sample)
			#if len(sample[index:math.floor(index+(i/k))])>0:
				#price=sum(sample[index:math.floor(index+(i/k))])/len(sample[index:math.floor(index+(i/k))])
			price=(price-min2)/(max2-min2)
			p.append(price)
		in_group.append(p)
		out_group.append([0.0,1.0])
	
	combined = list(zip(in_group, out_group))
	random.shuffle(combined)

	in_group[:], out_group[:] = zip(*combined)

	return [in_group,out_group]

samples=initSample(yes_sample,no_sample)

pivot=math.floor(4*len(samples[0])/5)

x = np.array(samples[0][0:pivot], dtype=np.float32)
y = np.array(samples[1][0:pivot], dtype=np.float32)

#print(samples[0])

# Parameters
learning_rate = 0.1
num_steps = 500
display_step = 100

# Network Parameters
n_hidden_1 = 16 # 1st layer number of neurons
n_hidden_2 = 16 # 2nd layer number of neurons


num_input = len(x[0]) 
num_classes = len(y[0])

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Create model
def neural_net(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

#pivot=math.floor(len(samples)/2)

# Start training
with tf.Session() as sess:

	#Run the initializer
	sess.run(init)

	for step in range(1, num_steps+1):

		# Run optimization op (backprop)
		output=sess.run(train_op, feed_dict={X: x, Y: y})

		if step % display_step == 0 or step == 1:
		# Calculate batch loss and accuracy
			loss, acc = sess.run([loss_op, accuracy], feed_dict={X: x, Y: y})
			print("Step " + str(step) + ", Minibatch Loss= {:.4f}".format(loss) + ", Training Accuracy= {:.3f}".format(acc))

	print("Optimization Finished!")

	output=sess.run(accuracy, feed_dict={X: x, Y: y})
	print("Training Accuracy:", output)


	x=[[1,0.7,0.5,0.3,0.2,0.1,0.4,0.9,0.7,1],
		[0.8,0.4,0.6,0.2,0.05,0.3,0.7,0.6,0.75,0.9],
		[1.0,0.5,0.0,0.0,0.0,0.0,0.5,1.0,0.7,1.0],
		[1.0,0.7,0.4,0.1,0.0,0.3,0.8,0.5,0.8,1.0],
		[1.0,0.0,0.5,0.1,0.3,0.6,0.2,0.7,0.9,0.7],
		[0.8,0.2,0.0,0.2,0.5,0.4,0.6,0.4,0.7,1.0],
		[1.0,0.5,0.2,0.0,0.0,0.0,0.0,0.2,0.5,1.0],
		[0.9,0.2,0.3,0.0,0.2,0.0,0.4,0.7,0.5,1.0],
		[1.0,0.0,0.4,0.3,0.7,0.5,0.7,0.8,0.7,1.0],
		[1.0,0.8,0.7,0.6,0.5,0.0,0.4,0.7,0.6,0.9],
		[1.0,0.0,0.0,0.1,0.0,0.0,0.7,0.6,0.6,1.0],
		[0.9,0.4,0.1,0.2,0.0,0.0,0.2,0.7,0.5,0.6],
		[0.7,0.4,0.1,0.6,0.6,0.1,0.9,0.7,0.9,0.4]]

	y=[[1,0],[1,0],[1,0],[0,1],[0,1],[1,0],[0,1],[0,1],[1,0],[1,0],[1,0],[0,1],[0,1]]

	#pivot=math.floor(len(samples[0])/2)
	x2 = np.array(samples[0][pivot+1:len(samples[0])-1], dtype=np.float32)
	y2 = np.array(samples[1][pivot+1:len(samples[0])-1], dtype=np.float32)


	def CH_Test(x,y):
		output=sess.run(prediction, feed_dict={X: np.array(x, dtype=np.float32)})

		i=0
		acc=0

		for y_p in output:
			state=y[i][0]==1
			state_p=(y_p[0]>=y_p[1])
			if state==state_p:
				acc=acc+1
			i=i+1

		return acc/len(y)
	

	acc=CH_Test(x2,y2)
	print("Testing Accuracy:",acc)

#https://medium.com/@nonthakon/%E0%B8%A5%E0%B8%AD%E0%B8%87%E0%B9%83%E0%B8%8A%E0%B9%89-tensorflow-%E0%B8%97%E0%B8%B3-linear-regression-f9e05d734441
