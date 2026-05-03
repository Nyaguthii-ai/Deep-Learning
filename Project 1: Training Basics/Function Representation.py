#Function Representation

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#EXERCISE 1

#Creating a 1D linear function and plotting it where beta represents the y-intercept
#and  omega the gradient/ slope of the function

# Define a 1D linear function
def linear_function_1D(x, beta, omega):
    return beta + omega * x

# Create an array of x values
x = np.arange(0, 10, 0.01)

# Set parameters
beta = 10 #3.0
omega = -2 #1.5

# Compute y values
y = linear_function_1D(x, beta, omega)
# Compute y1 values
y1 = y

# Plot
plt.figure(figsize=(7, 4))
plt.plot(x, y1, label=f"beta={beta}, omega={omega}")
# y label
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.xlabel('x')
plt.ylabel('y1')
plt.title('1D Linear Function')
plt.legend()
plt.grid(True)
plt.show()


#-------------------------------------------------------------------------------------
# 🔍 Sample 3D Linear Plane: y = 3 + 1*x1 + 2*x2 using a 2D Linear Function
#EXERCISE 2

# Set sample values
beta2 = -3.0 #3.0
omega1 = 0.0 #1.0
omega2 = 0 #2.0

# Create a grid of x1 and x2 values
x1_vals = np.linspace(-5, 5, 50)
x2_vals = np.linspace(-5, 5, 50)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Compute y values based on the linear plane formula
Y = beta2 + omega1 * X1 + omega2 * X2

# Plotting the 3D surface
fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Y, cmap='plasma', edgecolor='k', alpha=0.85)

# Labels and title
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
ax.set_title('3D Linear Plane: $y = 3 + 1x_1 + 2x_2$')

plt.tight_layout()
plt.show()


#-------------------------------------------------------------------------------------
#EXERCISE 3
#
#

# Individual equation approach
def linear_function_3D(x3, x4, beta, omega1, omega2):
    return beta + omega1 * x3 + omega2 * x4
# Sample input
x3 = 2  #4
x4 = -3 #-1
x5 = 2

# Parameters
beta1 = 1.0 #0.5
beta2 = 0 #0.2
beta3 = -2.0 
omega11, omega12 = 1.0, -2.0 #omega13 = -0.3
omega21, omega22 = 0.5, 0.1 #omega23 = 1.2
omega31, omega32 = -1.0, 3.0

# Compute outputs individually
y1 = linear_function_3D(x3, x4, beta1, omega11, omega12)
y2 = linear_function_3D(x3, x4, beta2, omega21, omega22)
y3 = linear_function_3D(x3, x4, beta3, omega31, omega32)

# Matrix-vector form
import numpy as np

# Inputs as vector
x_vec = np.array([[x3], [x4]])

# Weight matrix
omega_mat = np.array([
    [omega11, omega12],
    [omega21, omega22],
    [omega31, omega32]
])

# Bias vector
beta_vec = np.array([[beta1], [beta2], [beta3]])

# Compute matrix-vector output
y_vec = beta_vec + np.matmul(omega_mat, x_vec)
y_fixed = y_vec

# Display result
print("y_fixed =")
print(y_fixed)

#-------------------------------------------------------------------------------------
#EXERCISE 4

# Create x values
x_exp = np.linspace(-3, 3, 400)
x_log = np.linspace(0.01, 10, 400)  # avoid log(0)

# Compute y values
y_exp = np.exp(x_exp)
y_log = np.log(x_log)

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot exponential
axs[0].plot(x_exp, y_exp, color='green')
axs[0].set_title('Exponential Function: $y = \exp(x)$')
axs[0].set_xlabel('$x$')
axs[0].set_ylabel('$\exp(x)$')
axs[0].grid(True)

# Plot logarithm
axs[1].plot(x_log, y_log, color='purple')
axs[1].set_title('Logarithm Function: $y = \log(x)$')
axs[1].set_xlabel('$x$')
axs[1].set_ylabel('$\log(x)$')
axs[1].grid(True)

plt.tight_layout()
plt.show()

#-------------------------------------------------------------------------------------
#EXERCISE 5
# Create x values
x_exp = np.linspace(-3, 3, 400)
x_log = np.linspace(0.01, 10, 400)  # avoid log(0)

# Compute y values
y_exp = np.exp(x_exp)
y_exp_neg = np.exp(-x_exp)
y_log = np.log(x_log)
y_log_plus1 = np.log(x_log + 1)

# Create subplots: 2 rows, 2 columns
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot exponential
axs[0, 0].plot(x_exp, y_exp, color='green', label='$\exp(x)$')
axs[0, 0].set_title('Exponential Function: $y = \\exp(x)$')
axs[0, 0].set_xlabel('$x$')
axs[0, 0].set_ylabel('$y$')
axs[0, 0].grid(True)
axs[0, 0].legend()

# Plot exponential of negative x
axs[0, 1].plot(x_exp, y_exp_neg, color='blue', label='$\exp(-x)$')
axs[0, 1].set_title('Bonus: $y = \\exp(-x)$')
axs[0, 1].set_xlabel('$x$')
axs[0, 1].set_ylabel('$y$')
axs[0, 1].grid(True)
axs[0, 1].legend()

# Plot logarithm
axs[1, 0].plot(x_log, y_log, color='purple', label='$\log(x)$')
axs[1, 0].set_title('Logarithm Function: $y = \\log(x)$')
axs[1, 0].set_xlabel('$x$')
axs[1, 0].set_ylabel('$y$')
axs[1, 0].grid(True)
axs[1, 0].legend()

# Plot log(x + 1)
axs[1, 1].plot(x_log, y_log_plus1, color='orange', label='$\log(x + 1)$')
axs[1, 1].set_title('Bonus: $y = \\log(x + 1)$')
axs[1, 1].set_xlabel('$x$')
axs[1, 1].set_ylabel('$y$')
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.tight_layout()
plt.show()
