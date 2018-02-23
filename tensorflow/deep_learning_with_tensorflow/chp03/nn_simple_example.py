# As a linear problem, use relu will have a poor performance.

import tensorflow as tf
from numpy.random import RandomState

batch_size = 7 # change 8 to 7 for module 128

w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# biases1 = tf.Variable(0, dtype=tf.float32)
# biases2 = tf.Variable(0, dtype=tf.float32)
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_= tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

a = tf.matmul(x, w1)
# a = tf.nn.relu(tf.matmul(x, w1) + biases1)
y = tf.matmul(a, w2)
# y = tf.nn.relu(tf.matmul(a, w2) + biases2)
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
                                + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm = RandomState(1)
X = rdm.rand(128, 2)
Y = [[int(x1+x2<1)] for (x1, x2) in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(sess.run(w1))
    print(sess.run(w2))
    print('\n')

    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size) % 128
        end = (i*batch_size) % 128 + batch_size
        sess.run([train_step, y, y_], feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, y_:Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

    print("\n")
    print(sess.run(w1))
    print(sess.run(w2))

