# code for plotting figure showing seperability in 2d and 3d. Chat GPT was used for implementation of this


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_sphere(ax, center, radius, color, alpha=0.6):
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=alpha)

def plot_circle_projection(ax, center, radius, color, alpha=0.3):
    theta = np.linspace(0, 2*np.pi, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = np.zeros_like(x)  # Projected onto z=0 plane
    ax.plot(x, y, z, color=color, alpha=alpha, linewidth=2)

def plot_filled_circle_projection(ax, center, radius, color, alpha=0.3, z_plane=0):
    theta = np.linspace(0, 2*np.pi, 100)
    r = np.linspace(0, radius, 50)
    r, theta = np.meshgrid(r, theta)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    z = np.full_like(x, z_plane)  # Flat on z=0 plane
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)


def plot_plane(ax, z_height=3, size=2, color='gray', alpha=0.2):
    x = np.linspace(0, 5, 10)
    y = np.linspace(0, 5, 10)
    x, y = np.meshgrid(x, y)
    z = np.full_like(x, z_height)
    ax.plot_surface(x, y, z, color=color, alpha=alpha, rstride=1, cstride=1, edgecolor='none')

def plot_sphere_plane_intersection(ax, center, radius, z_plane, color='black', linewidth=2):
    x0, y0, z0 = center
    dz = z_plane - z0
    if abs(dz) <= radius:
        r = np.sqrt(radius**2 - dz**2)
        theta = np.linspace(0, 2*np.pi, 100)
        x = x0 + r * np.cos(theta)
        y = y0 + r * np.sin(theta)
        z = np.full_like(x, z_plane)
        ax.plot(x, y, z, color=color, linewidth=linewidth, linestyle='--')

# Create a single 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')


# Plot 3D spheres
plot_sphere(ax, center=(3, 2, 2.5), radius=1, color='blue')
plot_sphere(ax, center=(2, 3, 4), radius=1, color='red')

# Project their outlines as circles onto the z=0 plane
plot_filled_circle_projection(ax, center=(3, 2), radius=1, color='blue')
plot_filled_circle_projection(ax, center=(2, 3), radius=1, color='red')

plot_plane(ax, z_height=3.25, size=2, color='gray', alpha=0.2)

plot_sphere_plane_intersection(ax, center=(3, 2, 2.5), radius=1, z_plane=3.25, color='black')
plot_sphere_plane_intersection(ax, center=(2, 3, 4), radius=1, z_plane=3.25, color='black')

# Formatting
ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_zlim([0, 5])
ax.view_init(elev=13, azim=30)

plt.tight_layout()
# plt.show()
plt.savefig("linearSep.pdf")
