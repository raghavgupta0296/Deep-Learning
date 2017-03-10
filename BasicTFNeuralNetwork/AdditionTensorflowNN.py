import tensorflow as tf
import numpy as np

seed = 14

# Generate addition data
dataX = np.array(np.random.randint(1,10,[300,2]))
dataY = []
for i in dataX:
    j = i[0]+i[1]
    dataY.append(j)
dataY = np.array(dataY)
dataY = np.reshape(dataY,[dataY.shape[0],1])

# Create tensorflow placeholders
X = tf.placeholder(tf.float32,[None,2])
Y = tf.placeholder(tf.float32,[None,1])

# Hidden units. For addition, keeping them low (but >=2) turned out best
hidden_neurons = 2

# Parameters
weights = {
    'hidden_layer': tf.Variable(tf.random_normal([2,hidden_neurons],seed=seed)),
    'output_layer': tf.Variable(tf.random_normal([hidden_neurons,1],seed=seed))
}

biases = {
    'hidden_layer': tf.Variable(tf.random_normal([hidden_neurons],seed=seed)),
    'output_layer': tf.Variable(tf.random_normal([1],seed=seed))
}

# Hidden Layer
h1 = tf.add(tf.matmul(X,weights['hidden_layer']),biases['hidden_layer'])
# Output Regression Layer
o1 = tf.add(tf.matmul(h1,weights['output_layer']),biases['output_layer'])

# MSE Error loss
cost = tf.reduce_sum(tf.squared_difference(o1,Y))
# RMSProp optimizer
optimizer = tf.train.RMSPropOptimizer(0.001).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run([optimizer],feed_dict={X:dataX,Y:dataY})
    print('Prediction-------------------------------')
    print(" Enter numbers to add : ")
    a1 = input()
    a2 = input()
    o = sess.run([o1],feed_dict={X:[[a1,a2]]})
    ou = np.reshape(np.array(o),[1])
    print('predicted : ',o,'; approx ',np.round(ou))
