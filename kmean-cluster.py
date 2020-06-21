# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:29:30 2020

@author: KARTH
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



num_points = 100
dimensions = 1
points1 = np.random.uniform(0, 25, [num_points, dimensions])
points2 = np.random.uniform(75, 100, [num_points, dimensions])
points3 = np.random.uniform(150, 200, [num_points, dimensions])
grades_range = np.random.uniform(0, 200, [num_points, dimensions])
#points = np.random.uniform(0, 1000, [num_points, 2])
plt.scatter(grades_range, points1, color='r')
plt.scatter(grades_range, points2, color='b')
plt.scatter(grades_range, points3, color='g')
#plt.set_xlabel('Grades Range')
#plt.set_ylabel('Grades Scored')
#plt.set_title('scatter plot')

#print(points)
#print(points1)
#print(points2)

points=np.concatenate((points1,points2,points3), axis=1)
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
  print('point:', point, 'is in cluster', cluster_index, 'centered at', center)