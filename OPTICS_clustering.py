from datetime import datetime
import json
from math import sqrt
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import random

def rgb_to_hex(rgb):
    # Convert the RGB values to hexadecimal strings
    r, g, b = rgb
    r_hex = hex(r)[2:].zfill(2)
    g_hex = hex(g)[2:].zfill(2)
    b_hex = hex(b)[2:].zfill(2)

    # Concatenate the hexadecimal strings
    hex_code = "#" + r_hex + g_hex + b_hex

    return hex_code


def convert_time_format(time_str):
    time_str = time_str[-12:-4]
    time_obj = datetime.strptime(time_str, '%H:%M:%S')
    hour_val = time_obj.hour + time_obj.minute/60.0
    return hour_val

def time_duration(time1,time2):
    if time2 >= time1:
        return time2-time1
    else:
        return 24-time1 + time2

def my_matric(p1,p2):
    return sqrt( (p1[0] - p2[0]) * (p1[0] - p2[0]) +  (p1[1] - p2[1]) * (p1[1] - p2[1]) )


with open('acndata_session_kwh=1.json') as f:
    data = json.load(f)

features = []
for item in data['_items']:
    arrival_hour = convert_time_format(item['connectionTime'])
    unplug_hour = convert_time_format(item['disconnectTime'])
    connection_time = time_duration(arrival_hour,unplug_hour)
    kwh = item['kWhDelivered']
    feature = [arrival_hour, connection_time, kwh]
    features.append(feature)

features = np.array(features)
print(len(features))
# print(features)
epsilon = 5
min_samples = 10
cluster_method = 'xi'
matric = my_matric

scaler = StandardScaler()
features = scaler.fit_transform(features)
clustering = OPTICS(max_eps=epsilon, min_samples=min_samples, cluster_method=cluster_method, metric=my_matric).fit(features)

labels = clustering.labels_

no_clusters = len(np.unique(labels))
no_noise = np.sum(np.array(labels) == -1, axis=0)

print('Estimated no. of clusters: %d' % no_clusters)
print('Estimated no. of noise points: %d' % no_noise)

# Define the number of colors and their dimensionality
num_colors = 100
color_dim = 3  # Red, green, and blue values

# Generate random values for the colors
colors = np.random.randint(0, 256, size=(num_colors, color_dim))

# Convert the array to unsigned 8-bit integers
colors = colors.astype(np.uint8)

# Convert the RGB values to hex codes
hex_colors = np.apply_along_axis(rgb_to_hex, 1, colors)

# Print the resulting array
print(hex_colors)
print(type(hex_colors))
print(type(labels))

# Generate scatter plot for training data
colors = np.empty(len(features), dtype='U10')
# print(labels)
# print(labels[1])

i = 0
for label in labels:
    colors[i] = hex_colors[label]
    i += 1

print(colors)

plt.scatter(features[:,0], features[:,1], c=colors, marker="o", picker=True)
plt.title(f'OPTICS clustering')
plt.xlabel('Axis X[0]')
plt.ylabel('Axis X[1]')
plt.show()