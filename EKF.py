# import numpy as np


# def  state_trasition(x, u):
#     A = np.array([[0, 5],[0.01, 0]])
#     B = np.arrap([0, 1])
#     x = np.dot(A, x) + np.dot(B, u)
    
#     return x

# def measurement_function(x):
#     H = np.array([1, 0])
#     z = np.dot(H, x)
#     return z

# xhat = np.array([0, 0])
# P = np.identity(2)

# Q = np.identity(2) * 0.01
# R = np.identity(1) * 0.1

# true_state = np.array()    

import numpy as np
import matplotlib.pyplot as plt

# # Define the system dynamics (discrete-time)
# def system_dynamics(x, u):
# # Assuming a simple 2D system: x(k+1) = A * x(k) + B * u(k)
#     A = np.array([[1, 0.5],[0, 1]])
#     B = np.array([[0], [0.5]])
#     x = np.dot(A, x) + np.dot(B, u)
    
#     return x

# # Define the Jacobian matrix for the system dynamics
# def jacobian_f(x, u):
#     # For the 2D system, the Jacobian is:
#     A = np.array([[1, 0], [0, 1]])
#     return A

# # EKF prediction step
# def predict(x_est, P_est, u):
#     # Predicted state estimate
#     x_pred = system_dynamics(x_est, u)
    
#     # Predicted error covariance
#     P_pred = np.dot(np.dot(jacobian_f(x_est, u), P_est), jacobian_f(x_est, u).T) + Q
    
    
#     return x_pred, P_pred

# # EKF update step
# def update(x_pred, P_pred, z):
#     # Measurement model (example: z(k) = C * x(k) + noise)
#     C = np.array([[1, 0]])  # Jacobian of the measurement model
    
#     # Measurement residual
#     y = z - np.dot(C, x_pred)
    
#     # Kalman gain
#     K = np.dot(np.dot(P_pred, C.T), np.linalg.inv(np.dot(np.dot(C, P_pred), C.T) + R))
    
#     # Updated state estimate
#     x_est = x_pred + np.dot(K, y)
    
#     # Updated error covariance
#     P_est = np.dot((np.eye(x_est.shape[0]) - np.dot(K, C)), P_pred)
    
#     return x_est, P_est

# # Define the process and measurement noise covariance matrices
# Q = np.array([[0.1, 0], [0, 0.0]])  # Process noise covariance
# R = np.array([[0.1]])  # Measurement noise covariance

# # Initial state estimate and covariance
# x_est = np.array([[0], [0]])
# P_est = np.array([[1, 0], [0, 1]])

# # Simulate measurements
# num_steps = 100
# true_states = []
# measurements = []

# for t in range(num_steps):
#     # True state evolution (for simulation)
#     true_states.append(x_est)
    
#     # Simulate a noisy measurement
#     z = np.dot(np.array([[1]]), x_est) + np.random.normal(0, np.sqrt(R[0, 0]))
#     measurements.append(z)
    
#     # EKF prediction and update
#     x_pred, P_pred = predict(x_est, P_est, np.array([[-2]])) 
#     print(x_pred, P_pred)
#     x_est, P_est = update(x_pred, P_pred, z)

# # Extract x and y coordinates for plotting
# true_x = [state[0, 0] for state in true_states]
# true_y = [state[1, 0] for state in true_states]
# estimated_x = [state[0, 0] for state in measurements]
# estimated_y = [state[1, 0] for state in measurements]

# # Plot the true and estimated states
# plt.figure()
# plt.plot(true_x, true_y, label="True States")
# plt.plot(estimated_x, estimated_y, 'ro', label="Estimated States")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.legend()
# plt.title("Extended Kalman Filter Example (2D)")
# plt.grid(True)
# plt.show()


# F = np.array([[1, 0.5], [0, 1]])
# G = np.array([[0], [0.5]])
# u = -2
# x0 = np.array([[0], [5]])


# P0 = np.array([[0.01, 0], [0, 1]])
# # Q = np.array([[1, 0], [0, 1]])
# Q0 = np.identity(2) * 0.1
# R = np.identity(1) * 0.01
# L = np.identity(2)
# H = np.array([0.011, 0])
# M = np.identity(1) 

# x = np.dot(F, x0) + np.dot(G, u)
# P = np.dot(np.dot(F, P0), F.T) + np.dot(np.dot(L, Q0), L.T)
# # K = np.dot(np.dot(P, H.T), np.linalg.inv(np.dot(np.dot(H, P), H.T) + np.dot(np.dot(M, R), M.T)))
# print(np.linalg.inv(np.dot(np.dot(H, P), H.T) + np.dot(np.dot(M, R), M.T)))
# print(H)
# print(np.transpose(H))

import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic noisy data
x = np.linspace(0, 10, 100)
true_signal = np.sin(x)
noisy_data = true_signal + np.random.normal(0, 0.2, len(x))

# Smoothing parameters
alpha = 0.01  # Learning rate
num_iterations = 1000  # Number of iterations

# Initialize the smoothed signal
smoothed_signal = noisy_data.copy()

# Gradient descent to minimize the smoothing objective
for _ in range(num_iterations):
    gradient = np.zeros_like(smoothed_signal)
    
    # Compute the gradient (dL/dx) of the smoothing objective
    for i in range(1, len(smoothed_signal) - 1):
        gradient[i] = 2 * (smoothed_signal[i - 1] - 2 * smoothed_signal[i] + smoothed_signal[i + 1])
    
    # Update the smoothed signal using gradient descent
    smoothed_signal += alpha * gradient

# Plot the original, noisy, and smoothed signals
plt.figure(figsize=(10, 6))
plt.plot(x, true_signal, label="True Signal")
plt.plot(x, noisy_data, label="Noisy Data", alpha=0.5)
plt.plot(x, smoothed_signal, label="Smoothed Signal", linewidth=2)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Signal Smoothing using Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()