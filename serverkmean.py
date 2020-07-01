# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 09:01:38 2020

@author: KARTH
"""



import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


cluster=[]
num_points = 100
dimensions = 1
rng = np.random.RandomState(0)
colors = rng.rand(100)
points1 = np.array([1,2,1,2,1,2,1,2,2,1,2,2,1,2,2,1,2,1,2],dtype=float)
points2 = np.array([3,3,3,9,3,3,3,3,3,3,3,3,3,3,3,9,3,3,3],dtype=float)
points3 = np.array([1,1,1,20,1,4,4,7,4,3,34,2,34,34,34,50,34,34,34],dtype=float)
points4 = np.array([1,1,1,52,1,4,4,7,4,3,34,2,34,34,34,90,34,34,34],dtype=float)
points5 = np.array([10,10,10,10,10,500,500,500,500,500,1000,1000,1000,1000,1000,1000,1000,1000,1000],dtype=float)
points6= np.array([1,1,1,1,1,52,52,52,52,52,100,100,100,100,100,100,100,100,100],dtype=float)
grades_range = np.random.uniform(0, 1000, [num_points])
#points = np.random.uniform(0, 1000, [num_points, 2])
plt.scatter(points1, points6)
plt.scatter(points2, points6)
plt.scatter(points3, points6)
plt.scatter(points4, points6)
plt.scatter(points5, points6)

#plt.set_xlabel('Grades Range')
#plt.set_ylabel('Grades Scored')
#plt.set_title('scatter plot')


#print(points1)
#print(points2)

points=np.column_stack((points1,points2,points3,points4,points5,points6))
print(points)
#print(points)
# plt.show()
# plt.scatter(points)
# plt.ylabel('some numbers')
# plt.show()

def input_fn():
  return tf.compat.v1.train.limit_epochs(
      tf.convert_to_tensor(points, dtype=tf.float32), num_epochs=1)

num_clusters = 3
kmeans = tf.compat.v1.estimator.experimental.KMeans(
    num_clusters=num_clusters, use_mini_batch=False)

# train
num_iterations = 10
previous_centers = None
for _ in range(num_iterations):
  kmeans.train(input_fn)
  cluster_centers = kmeans.cluster_centers()
  plt.show()
  plt.plot(cluster_centers)
  plt.ylabel('Iteration:' + str(num_iterations))
  plt.show()
  if previous_centers is not None:
    print('delta:', cluster_centers - previous_centers)
  previous_centers = cluster_centers
  print('score:', kmeans.score(input_fn))
print('cluster centers:', cluster_centers)


# map the input points to their clusters
cluster_indices = list(kmeans.predict_cluster_index(input_fn))
for i, point in enumerate(points):
  cluster_index = cluster_indices[i]
  center = cluster_centers[cluster_index]
  cluster.append(cluster_index)
  print('point:', point, 'is in cluster', cluster_index, 'centered at', center)
  

outputpoints=np.column_stack((points,np.array(cluster)))
print(cluster)
print(outputpoints)