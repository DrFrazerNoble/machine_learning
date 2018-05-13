
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# The central unit of data in TensorFlow is a tensor.
# A tensor's rank is its number of dimensions.
# A tensor's shape is its length along each dimension.

A = tf.constant(3, name='A') # A rank 0 tensor.
B = tf.constant([3], name='B') # A rank 1 tensor.
C = tf.constant([3, 2], name='C') # A rank 2 tensor.
D = tf.constant([[3, 2], [1, 0]], name='D')

print(A)
print(B)
print(C)
print(D)
print('\n')

# A computational graph is a series of operations arranged into a graph.
# Graphs consist of operations and tensors.

A = tf.constant(1.0, dtype=tf.float32, shape=None, name='A')
B = tf.constant(2.0, dtype=tf.float32, shape=None, name='B')

SUM = tf.add(A,B,name='SUM');

print(A)
print(B)
print(SUM)
print('\n')

# To visualise a computation graph, we use TensorBoard. First, we need to create an event file.

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

# To view the graph, type the following command into the command prompt: tensorboard --logdir="."

# To evaluate tensors, we use a tf.Session() object.

with tf.Session() as sess:

	sum = sess.run(SUM)

	print(sum)
	print('\n')

# We can pass data to a graph using a placeholder.

A = tf.placeholder(tf.float32, name='A')
B = tf.placeholder(tf.float32, name='B')

SUM = tf.add(A,B,name='SUM')

with tf.Session() as sess:

	sum = sess.run(SUM, {A: 1.0, B: 2.0})

	print(sum)
	print('\n')

# Datasets are the preferred method of streaming data into a model.
# To get a tensor from a dataset, it needs to be converted to an iterator, and then call the iterator's get_next method.

data = [
	[0, 0],
	[1, 1],
	[2, 2],
	[3, 3],
	[4, 4]
]

slices = tf.data.Dataset.from_tensor_slices(data)
next_item = slices.make_one_shot_iterator().get_next()

with tf.Session() as sess:

	while True:
		try:
			print(sess.run(next_item))
		except tf.errors.OutOfRangeError:
			print('\n')
			break

# A trainable model must modify values in the graph. Layers are the preferred way to add trainable parameters.
# Layers package together variables and operations.

A = tf.placeholder(tf.float32, shape=[None, 3], name='A')
linear_model = tf.layers.Dense(units=1)
B = linear_model(A)

init = [tf.global_variables_initializer()]

with tf.Session() as sess:

	sess.run(init)

	print(sess.run(B, {A: [[1, 2, 3]]}))
	print('\n')