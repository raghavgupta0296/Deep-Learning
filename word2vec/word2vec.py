import numpy as np
import tensorflow as  tf
import zipfile
import collections
import math

tf.set_random_seed(1001)

"""
While using Rap.txt

a = "burn a lot \n i'm".split('\n')
b=[]
for i in range(len(a)):
	b.append(a[i].split())
	b.append('\n')
c=[]
for i in b:
	for j in i:
		c.append(j)

"""


def takeOutData(s="word2vecData.zip"):
    with zipfile.ZipFile(s) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


f = open("cleanedRapLyrics.txt")
data = tf.compat.as_str(f.read()).split()

# data = takeOutData()

# print data[:30]

vocab_size = 5000


def integerizeData(data):
    most_common = [['UNK', 0]]
    unk_count = 0
    most_common.extend(collections.Counter(data).most_common(vocab_size - 1))
    word2int = dict()
    data2 = list()
    for common, _ in most_common:
        word2int[common] = len(word2int)

    for word in data:

        if word in word2int:
            data2.append(word2int[word])
        else:
            data2.append(0)
            unk_count += 1

    most_common[0][1] = unk_count
    int2word = dict(zip(word2int.values(), word2int.keys()))
    return data2, most_common, word2int, int2word


data2, most_common, word2int, int2word = integerizeData(data)

del data

print("no of words : ", len(set(data2)))
print("no. of most common : ", len(most_common))
print("size of dict : ", len(int2word))

# print data2[:30]

data_index = 0

def generateBatch(batch_size, num_skips, skip_window):
    global data_index
    span = 2 * skip_window + 1
    buf = collections.deque(maxlen=span)
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    for _ in range(span):
        buf.append(data2[data_index])
        data_index = (data_index + 1) % len(data2)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = np.random.randint(0, span)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buf[skip_window]
            labels[i * num_skips + j, 0] = buf[target]
        buf.append(data2[data_index])
        data_index = (data_index + 1) % len(data2)
    return batch, labels


batch_size = 128
num_skips = 2
skip_window = 1

batch, labels = generateBatch(batch_size, num_skips, skip_window)

# print batch[:10],labels[:10]

neg_samples = 64
embedding_size = 128

validation_set = np.random.choice(50, 3, replace=False)

graph = tf.Graph()

with graph.as_default():
    train_x = tf.placeholder(tf.int32, shape=[batch_size])
    train_y = tf.placeholder(tf.int32, shape=[batch_size, 1])

    valid_set = tf.constant(validation_set, dtype=tf.int32)

    embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_x)

    w = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    b = tf.Variable(tf.zeros([vocab_size]))

    loss = tf.reduce_mean(tf.nn.nce_loss(weights=w, biases=b, labels=train_y, inputs=embed, num_sampled=neg_samples,
                                         num_classes=vocab_size))

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalised_embeddings = embeddings / norm

    valid_embed = tf.nn.embedding_lookup(normalised_embeddings, valid_set)

    similarity = tf.matmul(valid_embed, normalised_embeddings, transpose_b=True)

    init = tf.global_variables_initializer()

iters = 10001

with tf.Session(graph=graph) as session:
    init.run()
    print("Initialised")
    avg_loss = 0

    for i in range(iters):
        batch_inputs, batch_labels = generateBatch(batch_size, num_skips, skip_window)
        _, loss_val = session.run([optimizer, loss], feed_dict={train_x: batch_inputs, train_y: batch_labels})
        avg_loss += loss_val

        if i % 500 == 0:
            if i > 0:
                avg_loss /= 2000
        print("Avg Loss at ", i, "/", iters, " : ", avg_loss)
        avg_loss = 0

        if i % 1000 == 0:
            similar = similarity.eval()
            for j in range(3):
                print("v_set", validation_set[j])
                v_word = int2word[validation_set[j]]
                top_k = 5
                nearest = (-similar[j, :]).argsort()[1:top_k + 1]
                print("Nearest to word ", v_word, " : ")
                for k in range(top_k):
                    # print("near k :", nearest[k])
                    close_word = int2word[nearest[k]]
                    print(close_word, ", ", )
    final_embeddings = normalised_embeddings.eval()

# PLOTTING

def plot_with_labels(low_dim_embs, labels, filename='tsneRap.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 500
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labelsr = [int2word[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labelsr)

