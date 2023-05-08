import json
import random
from math import sqrt
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

def rgb_to_hex(rgb):
    """Convert the RGB values to hexadecimal strings"""
    r, g, b = rgb
    r_hex = hex(r)[2:].zfill(2)
    g_hex = hex(g)[2:].zfill(2)
    b_hex = hex(b)[2:].zfill(2)

    # Concatenate the hexadecimal strings
    hex_code = "#" + r_hex + g_hex + b_hex

    return hex_code

def my_matric(p1,p2):
    """Euclidian distance will be used as matrics for optics clustering"""
    return sqrt( (p1[0] - p2[0]) * (p1[0] - p2[0]) +  (p1[1] - p2[1]) * (p1[1] - p2[1]) )

def find_cluster(centres,feature):
    """find closest cluster centre for given feature using all available centres"""
    cluster = 0
    min_dist = 100000000
    for i in range(len(centres)):
        curr_dist = my_matric(centres[i],feature)
        if curr_dist < min_dist:
            min_dist = curr_dist
            cluster = i

    return cluster


def ev_forcast(day, num_sessions,week_forcast,week_num):
    file_name = day + '.json'
    print(file_name)
    with open(file_name) as f:
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
    # features = scaler.fit_transform(features)  ## scaling is not necessary because both features for clustering arrival time and duration are in comperative ranges
    clustering = OPTICS(max_eps=epsilon, min_samples=min_samples, cluster_method=cluster_method, metric=my_matric).fit(features)

    labels = clustering.labels_

    num_clusters = len(np.unique(labels))
    num_noise = np.sum(np.array(labels) == -1, axis=0)

    print('Estimated no. of clusters: %d' % num_clusters)
    print('Estimated no. of noise points: %d' % num_noise)

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
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # print(num_clusters)
    # print(labels)
    # print(type(labels))

    # find centre of all clusters
    centres = np.empty(num_clusters, dtype=np.ndarray)
    num_of_points_in_cluster = np.empty(num_clusters, dtype=int)
    kde_arr = []

    for i in range(num_clusters):
        # select data points for the current cluster
        cluster_points = features[labels == i]
        # pdf for arrival time
        kde_1 = KernelDensity(kernel='gaussian', bandwidth=0.5)
        # pdf for power-duration
        kde_2 = KernelDensity(kernel='gaussian', bandwidth=0.5)
        kde_1.fit(cluster_points[:, 0][:, None])
        X = cluster_points[:, -2:]
        kde_2.fit(X)
        kde_arr.append([kde_1, kde_2])

        # x = np.linspace(-4, 24, 500).reshape(-1, 1)
        # # evaluate the PDF at the given range of values
        # log_pdf = kde_1.score_samples(x)
        # # convert the log probabilities to probabilities
        # pdf = np.exp(log_pdf)
        # # plot the resulting PDF
        # plt.plot(x, pdf)
        # plt.xlabel('Value')
        # plt.ylabel('Probability Density')
        # plt.show()

        # # Create grid of points to evaluate the PDF
        # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        # xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        # grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        #
        # # Evaluate the PDF at the grid points
        # pdf = np.exp(kde_2.score_samples(grid_points))
        # pdf = pdf.reshape(xx.shape)
        #
        # # Plot the PDF
        # plt.imshow(pdf, cmap=plt.cm.Blues, extent=[x_min, x_max, y_min, y_max], origin='lower')
        # plt.colorbar()
        # plt.title('2D Probability Density Function')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.show()

        # calculate the mean or median of the data points
        center = np.mean(cluster_points, axis=0)  # or np.median(cluster_points, axis=0)
        centres[i] = center
        num_of_points_in_cluster[i] = len(cluster_points)

        # print("Cluster", i + 1, "center:", center, "number of points in cluster", len(cluster_points))

    for i in range(len(features)):
        if labels[i] == -1:
            cluster = find_cluster(centres, features[i])
            num_of_points_in_cluster[cluster] += 1

    # # check len(features) is equal to sum of num of points in clusters
    # print(len(features))
    sum = np.sum(num_of_points_in_cluster)
    # print(sum)

    prob_dist = {}
    for i in range(len(num_of_points_in_cluster)):
        prob_dist[i] = num_of_points_in_cluster[i] / sum

    # print(prob_dist)

    # 100 monte carlo simulation
    MC = 100
    simulations = []
    for s in range(MC):
        outcomes = random.choices(list(prob_dist.keys()), list(prob_dist.values()), k=num_sessions)
        num_cluster_sim = np.zeros(num_clusters, dtype=np.int32)
        simulation = []
        power_sum = 0
        for j in outcomes:
            num_cluster_sim[j] += 1
        # print(outcomes)
        # print(num_cluster)
        for i in range(num_clusters):
            # sessions = np.empty(0, dtype=np.ndarray)
            if (num_cluster_sim[i]):
                arrival_times = kde_arr[i][0].sample(num_cluster_sim[i])
                power_durations = kde_arr[i][1].sample(num_cluster_sim[i])
                # print(arrival_times)
                # print(power_durations)
                # print(type(power_durations))
                for j in range(num_cluster_sim[i]):
                    simulation.append((week_num[day]+arrival_times[j]/24, power_durations[j][1]))
                # sessions = np.concatenate((arrival_times, power_durations), axis=1)
                power_sum += np.sum(power_durations[:, 1])
                # print(sessions)
            # simulation.append(sessions)
        simulations.append([simulation, power_sum])

    # print(simulations)
    total_power_sum = 0
    for simulation in simulations:
        total_power_sum += simulation[1]
        # print(simulation[1])

    avg_pow = total_power_sum / 100
    print('average power sum', avg_pow)

    min_power = 1000000
    forcast_sessions = None
    curr_min_diff = 1000000
    for i in range(100):
        if abs(avg_pow - simulations[i][1]) < curr_min_diff:
            min_power = simulations[i][1]
            curr_min_diff = abs(avg_pow - simulations[i][1])
            forcast_sessions = simulations[i][0]

    for session in forcast_sessions:
        arrival_times = session[0][0];
        power = session[1]
        week_forcast.append((arrival_times,power))
    print(min_power)
    # print(week_forcast)

    # total = 0
    # for tup in week_forcast:
    #     total += tup[1]
    # print(total)

week_forcast = []

week_num = {
    'Mon': 0,
    'Tue': 1,
    'Wed': 2,
    'Thu': 3,
    'Fri': 4,
    'Sat': 5,
    'Sun': 6
}

sessions ={
    'Mon':49,
    'Tue':57,
    'Wed':47,
    'Thu':55,
    'Fri':17,
    'Sat':2,
    'Sun':5
}

# ev_forcast('Mon', sessions['Mon'],week_forcast,week_num)
for day in ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']:
    ev_forcast(day, sessions[day],week_forcast,week_num)


# Plotting
week_forcast = sorted(week_forcast, key=lambda x: x[0])
x = [i[0] for i in week_forcast]
y = [i[1] for i in week_forcast]
plt.plot(x,y)

# Customize the plot
plt.title("estimated Power consumption for week 7 sept 2021 to 14 sept 2021" )
plt.xlabel("time")
plt.ylabel("power(kWh)")
plt.grid(True)

# Show the plot
plt.show()

