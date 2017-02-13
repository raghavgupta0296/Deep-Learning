import tensorflow as tf
import numpy as np

def create_dataset(n_clusters,n_samples_in_cluster,n_features,seed):
	np.random.seed(seed)
	all_data=[]
	for i in range(n_clusters):
		datapoints = tf.random_normal((n_samples_in_cluster,n_features),mean=0.0,stddev=5.0,seed=seed,dtype=tf.float32,name='cluster_{}'.format(i))
		all_data.append(datapoints)
	all_datapoints = tf.concat(0,all_data,name='all_datapoints')
	return all_datapoints

def randomize_centroids(all_datapoints,n_clusters,n_features,seed):
	centroids = []
	"""
	#c = np.random.randint(size=(1,n_features),low=(np.amin([all_datapoints],axis=0),np.amin([all_datapoints],axis=1)),high=(np.amax([all_datapoints],axis=0),np.amax([all_datapoints],axis=1)))
	
	random_indices = tf.random_shuffle(tf.range(0,tf.shape(all_datapoints)[0]))
	i = tf.slice(random_indices,[0,],[n_clusters,])
	centroids = tf.gather(all_datapoints,i)
	# OR
	"""
	random_datapoints = tf.random_shuffle(all_datapoints,seed=seed)
	centroids = tf.slice(random_datapoints,[0,0],[n_clusters,-1])
	
	return centroids
	
def assign_cluster(all_datapoints,centroids):
	expanded_ad = tf.expand_dims(all_datapoints,0)
	expanded_c = tf.expand_dims(centroids,1)
	distances = tf.reduce_sum(tf.square(tf.sub(expanded_ad,expanded_c)),2)
	mins = tf.argmin(distances,0)	
	return mins		

def update_centroids(all_datapoints,mins,n_clusters):
	mins = tf.to_int32(mins)
	partitions = tf.dynamic_partition(all_datapoints,mins,n_clusters)
	centroids = tf.concat(0,[tf.expand_dims(tf.reduce_mean(partition,0),0) for partition in partitions])
	return centroids

seed = 507
n_clusters = 3
n_features = 2
n_samples_in_cluster = 4
n_epochs = 10
convergence = False

all_datapoints = create_dataset(n_clusters,n_samples_in_cluster,n_features,seed)
centroids = randomize_centroids(all_datapoints,n_clusters,n_features,seed) 
mins = assign_cluster(all_datapoints,centroids)
new_centroids = update_centroids(all_datapoints,mins,n_clusters)

model = tf.global_variables_initializer()

epochs = 1
with tf.Session() as session:
	session.run(model)
	print session.run(all_datapoints)	
	print session.run(centroids)
	print " Epoch 1 "
	print session.run(mins)
	a = session.run(new_centroids)
	print a
	epochs +=1
	while convergence is False and epochs<n_epochs:
		print "\n Epoch : ",epochs
		print session.run(mins)
		epochs += 1
		b = session.run(new_centroids)
		if (a!=b).all():
			a = b
			print a
		else:
			convergence = True		
