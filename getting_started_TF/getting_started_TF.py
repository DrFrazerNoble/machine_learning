'''A Python TensorFlow Program'''

#Import statements for TensorFlow.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

#Create two constants, A and B, with values, 1.0 and 2.0, respectively.
A = tf.constant(1.0, dtype=tf.float32, shape=None, name='A')
B = tf.constant(2.0, dtype=tf.float32, shape=None, name='B') 
#Here, we've defined the constants in a long-hand way; we could define, e.g. A, in a short-hand way by: # A = tf.constant(1.0).

#Define a graph, which adds A and B together.
C = tf.add(A,B, name='C')

#Create a session
with tf.Session() as s:

	#We can use TensorBoard to visualise the graph.
	tb_writer = tf.summary.FileWriter('./graphs', s.graph);

	#Execute the graph.
	c = s.run(C)

	#Display the results on the console.
	print(c)

	tb_writer.close();

	#To display the graph, type the following into the console:
	#	tensorboard --logdir="./graphs" --port 5000
	#follow the on-screen prompt to view the graph.