import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from minisom import MiniSom
import glob

files = glob.glob('D:/Victor/scaleogram_img/*.png')

images = []
i = 0
for i in range(len(files)):
    img = np.array(plt.imread(files[i]).flatten())
    images = np.append(images, img)

images = np.reshape(images,(1708, 497664)) 
images_norm = images.astype(float) / 255

som_shape = (32, 32)
input_len = images_norm.shape[1]
som = MiniSom(som_shape[0], som_shape[1], input_len, sigma=1.0, learning_rate=0.5)
som.train_random(images_norm, 64)

neuron_clusters = np.array([som.winner(x) for x in images_norm])

# # Elbow Method

wcss = []

for i in range(20):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(neuron_clusters)
    wcss.append(kmeans.inertia_)
    
# K-Means with Optimized no. of clusters

kmeans = KMeans(n_clusters = 6)
kmeans.fit(neuron_clusters)

image_clusters = kmeans.predict(neuron_clusters)

value_counts = np.bincount(image_clusters)

x = np.arange(len(value_counts))

plt.bar(x, value_counts)

plt.xticks(x)

plt.xlabel("No. of Samples")
plt.ylabel("Severity")

plt.title("Severity using Clustering")

plt.show()