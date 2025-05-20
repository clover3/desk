import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# Sample data generation
def generate_data(num_series=3, points=100):
    x = np.linspace(0, 10, points)
    data_series = []

    for i in range(num_series):
        # Create different patterns for each series
        y = np.sin(x + i * np.pi / 4) * (i + 1) / 2 + np.random.normal(0, 0.1, points)
        data_series.append((x, y, f"Series {i + 1}"))

    return data_series


# Create and plot the 3D-style visualization
def plot_3d_style_series(data_series, elev=30, azim=45, save_path=None):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Color map for different series
    colors = cm.viridis(np.linspace(0, 1, len(data_series)))

    # Plot each series with 3D effect
    for idx, (x, y, label) in enumerate(data_series):
        # Add z-dimension as index to create the 3D effect
        z = np.zeros_like(x) + idx

        # Plot the main line
        ax.plot(x, y, z, color=colors[idx], label=label, linewidth=2)

        # Add vertical lines to ground for 3D effect
        for i in range(0, len(x), len(x) // 10):
            ax.plot([x[i], x[i]], [y[i], y[i]], [0, z[i]],
                    color=colors[idx], alpha=0.3, linestyle='--')

        # Optional: Plot shadow/projection on the xz-plane
        ax.plot(x, np.ones_like(x) * np.min(y) - 0.5, z,
                color=colors[idx], alpha=0.3)

    # Customize the plot
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.view_init(elev=elev, azim=azim)  # Set the viewing angle

    # Remove z-ticks since they're just for the 3D effect
    ax.set_zticks([])

    # Add a legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1))

    # Tight layout to ensure everything fits
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


# Generate data and create the plot
if __name__ == "__main__":
    # Generate sample data (3 series by default)
    data = generate_data(num_series=4, points=200)

    # Create the 3D-style plot
    plot_3d_style_series(data, elev=20, azim=30, save_path="3d_style_plot.png")

    plt.show()