import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_classes = 10
n_samples_per_class = 50
n_dimensions = 512
centers_std = 200.0  # Standard deviation for generating class centers
sample_std = 20.0  # Standard deviation for generating samples around centers

# Generate class centers (10 different centers in 512 dimensions)
centers = np.random.normal(0, centers_std, size=(n_classes, n_dimensions))

# Generate samples for each class
X = []  # Features
y = []  # Labels

for i in range(n_classes):
    # Generate samples following Gaussian distribution around each center
    samples = np.random.normal(
        centers[i], sample_std, size=(n_samples_per_class, n_dimensions)
    )
    X.append(samples)
    y.extend([i] * n_samples_per_class)

X = np.vstack(X)
y = np.array(y)

# Standardize the features

# Apply UMAPX
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply UMAP
reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
X_embedded = reducer.fit_transform(X_scaled)

# Create visualization
plt.figure(figsize=(10, 8))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap="Spectral", s=5)
plt.gca().set_aspect("equal", "datalim")
plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
# plt.title('UMAP visualization of 10 Gaussian clusters in 512D')
# plt.xlabel('UMAP 1')
# plt.ylabel('UMAP 2')

# Save the plot
plt.savefig("umap_visualization.png", dpi=300, bbox_inches="tight")
plt.close()
