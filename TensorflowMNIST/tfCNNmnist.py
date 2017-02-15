from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

x = tf.placeholder(tf.float32,shape=[None,784])
y = tf.placeholder(tf.float32,shape=[None,10])

def weight_ini(sh):
    r = tf.truncated_normal(shape=sh,stddev=0.1)
    return tf.Variable(r)
def bias_ini(sh):
    r =  tf.constant(0.1,shape=sh)
    return tf.Variable(r)
def conv2d_ini(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def maxPool2_ini(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x_im = tf.reshape(x,shape=[-1,28,28,1])

W1 = weight_ini([5,5,1,32])
b1 = bias_ini([32])
o1 = conv2d_ini(x_im,W1)+b1
o1 = tf.nn.relu(o1)
o1 = maxPool2_ini(o1)
W2 = weight_ini([5,5,32,64])
b2 = bias_ini([64])
o2 = conv2d_ini(o1,W2)+b2
o2 = tf.nn.relu(o2)
o2 = maxPool2_ini(o2)
W3 = weight_ini([7*7*64,1024])
b3 = bias_ini([1024])
o2_flat = tf.reshape(o2,[-1,7*7*64])
o3 = tf.matmul(o2_flat,W3)+b3
o3 = tf.nn.relu(o3)

dropout_prob = tf.placeholder(tf.float32)
o3_drop = tf.nn.dropout(o3,dropout_prob)

W4 = weight_ini([1024,10])
b4 = bias_ini([10])
o4 = tf.matmul(o3_drop,W4)+b4

loss = tf.nn.softmax_cross_entropy_with_logits(o4,y)
train_step = tf.train.AdamOptimizer(epsilon=1e-4).minimize(loss)
correct_preds = tf.equal(tf.argmax(o4,1),tf.argmax(y,1))
acc = tf.reduce_mean(tf.cast(correct_preds,tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        accuracy = acc.eval(feed_dict={x:batch[0],y:batch[1],dropout_prob:1.0})
        print ("Iter %d, accuracy %g"%(i,accuracy))
    train_step.run(feed_dict={x:batch[0],y:batch[1],dropout_prob:0.5})

test_acc = acc.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels,dropout_prob:1.0})
print ("Test Accuracy : %g"%test_acc)
