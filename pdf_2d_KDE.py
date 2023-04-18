import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Generate some random data
np.random.seed(42)
X = np.random.normal(size=(100, 2))
print(X)
# Create KDE model
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)

# Create grid of points to evaluate the PDF
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid_points = np.column_stack([xx.ravel(), yy.ravel()])

# Evaluate the PDF at the grid points
pdf = np.exp(kde.score_samples(grid_points))
pdf = pdf.reshape(xx.shape)

# Plot the PDF
plt.imshow(pdf, cmap=plt.cm.Blues, extent=[x_min, x_max, y_min, y_max], origin='lower')
plt.colorbar()
plt.title('2D Probability Density Function')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Sample from the distribution
samples = kde.sample(10000)
plt.scatter(samples[:, 0], samples[:, 1], s=5)
plt.title('Samples from 2D Probability Density Function')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

