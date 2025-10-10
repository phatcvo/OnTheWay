import numpy as np
import control as ctrl
import matplotlib.pyplot as plt

# System parameters
m = 1000.0  # Mass of the vehicle (kg)
b = 50.0    # Damping coefficient (Ns/m)
k = 3000.0  # Spring constant (N/m)

# State-space representation
A = np.array([[0, 1], [-k/m, -b/m]])
B = np.array([[0], [1/m]])
C = np.array([[1, 0]])
D = np.array([[0]])

# Define the Q and R matrices for the cost function
Q = np.diag([1, 1])  # State cost weights
R = np.array([[0.1]])  # Control input cost weight

# Compute the LQR controller gain
K, _, _ = ctrl.lqr(A, B, Q, R)

# Simulation parameters
dt = 0.01  # Time step
num_steps = 1000  # Number of simulation steps

# Initial conditions
x0 = np.array([[0], [0]])  # Initial state [position, velocity]

# Lists to store simulation results
states = []
controls = []

# Simulate the closed-loop system
x = x0
for _ in range(num_steps):
    u = -np.dot(K, x)  # Calculate control input using LQR gain
    x = np.dot(A, x) + np.dot(B, u)  # Update state
    states.append(x)
    controls.append(u)

# Extract position and velocity for plotting
positions = [state[0] for state in states]
velocities = [state[1] for state in states]

# Plot results
plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.plot(np.arange(0, num_steps*dt, dt), positions)
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.subplot(212)
plt.plot(np.arange(0, num_steps*dt, dt), velocities)
plt.xlabel('Time (s)')
plt.ylabel('Velocity')
plt.tight_layout()
plt.show()