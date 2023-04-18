import numpy as np
from sklearn.neighbors import KernelDensity

# Load your dataset
data = np.random.normal(size=1000)

# Create a KernelDensity instance with a bandwidth of 0.5
kde = KernelDensity(bandwidth=0.5)

# Fit the KDE model to your dataset
kde.fit(data[:, None])

# Generate a set of values to estimate the PDF
x = np.linspace(-5, 5, 1000)

# Estimate the PDF at each generated value
log_prob = kde.score_samples(x[:, None])

# Normalize the PDF
prob = np.exp(log_prob)
prob /= prob.sum()

# Sample from the estimated PDF
samples = kde.sample(100)

# Print the results
print("Estimated PDF:")
print(prob)
print("Samples:")
print(samples)