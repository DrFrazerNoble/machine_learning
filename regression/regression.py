
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

x = tf.constant([[1],[2],[3],[4]], dtype=tf.float32, name='x')
y = tf.constant([[2],[4],[6],[8]], dtype=tf.float32, name='y')

linear_model = tf.layers.Dense(units=1)

y_ = linear_model(x)

loss = tf.losses.mean_squared_error(labels=y, predictions=y_);

optimiser = tf.train.GradientDescentOptimizer(0.01);
train = optimiser.minimize(loss);

init = tf.global_variables_initializer()

with tf.Session() as sess:

	sess.run(init)

	for i in range(100):
		_, loss_value = sess.run((train, loss))
		print(loss_value)

	print(sess.run(y_))
	print('\n')