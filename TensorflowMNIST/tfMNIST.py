from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

# print (mnist)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32,shape=[None,784])
y = tf.placeholder(tf.float32,shape=[None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

pred = tf.matmul(x,W) + b

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

correctPred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
acc = tf.reduce_mean(tf.cast(correctPred,tf.float32))

for i in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x:batch[0],y:batch[1]})

print (acc.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels}))

