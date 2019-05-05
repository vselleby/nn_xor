import tensorflow as tf

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

n_input = tf.placeholder(tf.float32, shape=[None, 2])
n_output = tf.placeholder(tf.float32, shape=[None, 1])

hidden_nodes = 2

b_hidden = tf.Variable(tf.zeros([2]))
W_hidden = tf.Variable(tf.random_uniform([2, hidden_nodes]))
a_hidden = tf.sigmoid(tf.matmul(n_input, W_hidden) + b_hidden)


b_output = tf.Variable(tf.zeros([1]))
W_output = tf.Variable(tf.random_uniform([hidden_nodes, 1]))
a_output = tf.sigmoid(tf.matmul(a_hidden, W_output) + b_output)
squared_error = tf.square(n_output - a_output)

loss = tf.reduce_mean(squared_error)
optimizer = tf.train.GradientDescentOptimizer(1)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for epoch in xrange(0, 10001):
    sess.run(train, feed_dict={n_input: X, n_output: y})

print(sess.run(a_output, feed_dict={n_input: X}))
