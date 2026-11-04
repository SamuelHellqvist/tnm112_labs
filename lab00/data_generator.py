import numpy as np
import matplotlib.pyplot as plt


class DataGenerator:
    def __init__(self):
        self.x = []
        self.y = []

    def generate(self, K, N, sigma, seed=None):
        """Generate a 2D dataset with K clusters of N points each.

        K     - number of classes/clusters
        N     - number of data points in each class
        sigma - std of the clusters
        seed  - optional seed for reproducible results
        """
        rng = np.random.default_rng(seed)
        means = rng.random((K, 2))
        self.x = np.vstack(
            [rng.normal(means[i], sigma, (N, 2)) for i in range(K)]
        )
        self.y = np.repeat(np.arange(K), N)

    def plot_data(self):
        """Plot the generated dataset, using a distinct marker for each class."""
        markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
        for k in np.unique(self.y):
            plt.plot(
                self.x[self.y == k, 0],
                self.x[self.y == k, 1],
                markers[k % len(markers)],
                label=f'Class {k}',
            )
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()

    def rotate(self, ang):
        """Rotate the generated points by ang radians."""
        rotation = np.array([
            [np.cos(ang), -np.sin(ang)],
            [np.sin(ang), np.cos(ang)],
        ])
        self.x = self.x @ rotation.T


    def export_data(self, filename):
        """Save the dataset and labels to a NumPy .npz file."""
        np.savez(filename, x=self.x, y=self.y)

    def import_data(self, filename):
        """Load a dataset and labels from a NumPy .npz file."""
        with np.load(filename) as data:
            self.x = data['x']
            self.y = data['y']


if __name__ == "__main__":
    dg = DataGenerator()
    dg.generate(3, 200, 0.1)
    dg.plot_data()