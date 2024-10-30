import numpy as np

# Example random coordinates on the sphere with radius 0.5
x, y, z = round(0.5 * np.sin(np.random.uniform(0, np.pi)) * np.cos(np.random.uniform(0, 2 * np.pi)), 5), \
          round(0.5 * np.sin(np.random.uniform(0, np.pi)) * np.sin(np.random.uniform(0, 2 * np.pi)), 5), \
          round(0.5 * np.cos(np.random.uniform(0, np.pi)), 5)

print(x, y, z)
# Calculate yaw (psi)
yaw = round(np.arctan2(y, x), 5)

# Calculate pitch (theta)
pitch = round(np.arctan2(z, np.sqrt(x**2 + y**2)), 5)

# Roll (phi) is zero for looking at the origin
roll = 0.0

print(yaw, pitch, roll)