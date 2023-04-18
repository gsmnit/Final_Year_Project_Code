import json
import random
from math import sqrt
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

def rgb_to_hex(rgb):
    # Convert the RGB values to hexadecimal strings
    r, g, b = rgb
    r_hex = hex(r)[2:].zfill(2)
    g_hex = hex(g)[2:].zfill(2)
    b_hex = hex(b)[2:].zfill(2)

    # Concatenate the hexadecimal strings
    hex_code = "#" + r_hex + g_hex + b_hex

    return hex_code

def my_matric(p1,p2):
    return sqrt( (p1[0] - p2[0]) * (p1[0] - p2[0]) +  (p1[1] - p2[1]) * (p1[1] - p2[1]) )

def find_cluster(centres,feature):
    cluster = 0
    min_dist = 100000000
    for i in range(len(centres)):
        curr_dist = my_matric(centres[i],feature)
        if curr_dist < min_dist:
            min_dist = curr_dist
            cluster = i

    return cluster


with open('Mon.json') as f:
    data = json.load(f)

# print(data)

features = np.array(data)
# print(len(features))
# print(features)
epsilon = 2
min_samples = 4
cluster_method = 'xi'
matric = my_matric

scaler = StandardScaler()
# features = scaler.fit_transform(features)
clustering = OPTICS(max_eps=epsilon, min_samples=min_samples, cluster_method=cluster_method, metric=my_matric).fit(features)

labels = clustering.labels_

no_clusters = len(np.unique(labels))
no_noise = np.sum(np.array(labels) == -1, axis=0)

print('Estimated no. of clusters: %d' % no_clusters)
print('Estimated no. of noise points: %d' % no_noise)

# Define the number of colors and their dimensionality
num_colors = 500
color_dim = 3  # Red, green, and blue values

# # Generate random values for the colors
colors = np.random.randint(0, 256, size=(num_colors, color_dim))

# Convert the array to unsigned 8-bit integers
colors = colors.astype(np.uint8)

# Convert the RGB values to hex codes
hex_colors = np.apply_along_axis(rgb_to_hex, 1, colors)

# Print the resulting array
# print(hex_colors)
# print(type(hex_colors))
# print(type(labels))

# Generate scatter plot for training data
colors = np.empty(len(features), dtype='U10')
# print(labels)
# print(labels[1])

i = 0
for label in labels:
    colors[i] = hex_colors[label]
    i += 1
# print(colors)

# Generate plot of clusters
# plt.scatter(features[:,0], features[:,1], c=colors, marker="o", picker=True)
# plt.title(f'OPTICS clustering')
# plt.xlabel('Arrival time')
# plt.ylabel('charging duration')
# plt.show()

# Generate reachability plot
# reachability = clustering.reachability_[clustering.ordering_]
# plt.plot(reachability)
# plt.title('Reachability plot')
# plt.show()


# get the number of clusters
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# print(n_clusters)
# print(labels)
# print(type(labels))

# find centre of all clusters
centres = np.empty(n_clusters, dtype=np.ndarray)
num_of_points_in_cluster = np.empty(n_clusters, dtype=int)
kde_arr = []


for i in range(n_clusters):
    # select data points for the current cluster
    cluster_points = features[labels == i]
    # pdd for arrival time and for power-duration
    kde_1 = KernelDensity(kernel='gaussian', bandwidth=0.5)
    kde_2 = KernelDensity(kernel='gaussian',bandwidth=0.5)
    kde_1.fit(cluster_points[:,0][:,None])
    X = cluster_points[:,-2:]
    kde_2.fit(X)
    kde_arr.append([kde_1,kde_2])

    x = np.linspace(-4, 24, 500).reshape(-1, 1)
    # evaluate the PDF at the given range of values
    log_pdf = kde_1.score_samples(x)
    # convert the log probabilities to probabilities
    pdf = np.exp(log_pdf)
    # plot the resulting PDF
    plt.plot(x, pdf)
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.show()

    # Create grid of points to evaluate the PDF
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Evaluate the PDF at the grid points
    pdf = np.exp(kde_2.score_samples(grid_points))
    pdf = pdf.reshape(xx.shape)

    # Plot the PDF
    plt.imshow(pdf, cmap=plt.cm.Blues, extent=[x_min, x_max, y_min, y_max], origin='lower')
    plt.colorbar()
    plt.title('2D Probability Density Function')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    # calculate the mean or median of the data points
    center = np.mean(cluster_points, axis=0)  # or np.median(cluster_points, axis=0)
    centres[i] = center
    num_of_points_in_cluster[i] = len(cluster_points)

    # print("Cluster", i + 1, "center:", center, "number of points in cluster", len(cluster_points))

for i in range(len(features)):
    if labels[i] == -1:
        cluster = find_cluster(centres, features[i])
        num_of_points_in_cluster[cluster] += 1

# # check
# print(len(features))
sum = np.sum(num_of_points_in_cluster)
# print(sum)

prob_dist = {}
for i in range(len(num_of_points_in_cluster)):
    prob_dist[i] = num_of_points_in_cluster[i]/sum

# print(prob_dist)

# 100 monte carlo simulation
MC = 1
n_trials = 14
simulations = []
for s in range(MC):
    outcomes = random.choices(list(prob_dist.keys()), list(prob_dist.values()), k=n_trials)
    num_cluster = np.zeros(n_clusters, dtype=np.int32)
    simulation = []
    for j in outcomes:
        num_cluster[j] += 1
    print(outcomes)
    print(num_cluster)
    for i in range(n_clusters):
        sessions = np.empty(0, dtype=np.ndarray)
        if(num_cluster[i]):
            arrival_times = kde_arr[i][0].sample(num_cluster[i])
            power_durations = kde_arr[i][1].sample(num_cluster[i])
            # print(arrival_times)
            # print(power_durations)
            sessions = np.concatenate((arrival_times,power_durations),axis=1)
            print(sessions)
        simulation.append(sessions)
    simulations.append(simulation)

# print(simulations)


