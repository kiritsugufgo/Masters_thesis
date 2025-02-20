import numpy as np 
import matplotlib.pyplot as plt

def z_function(x, y):
    return np.sin(5*x) * np.cos(5*y) / 5

def calc_gradient(x, y):
    return np.cos(5*x) * np.cos(5*y), -np.sin(5*x) * np.sin(5*y)

x = np.arange(-1, 1, 0.05)
y = np.arange(-1, 1, 0.05)

X, Y = np.meshgrid(x, y)

Z = z_function(X, Y)

current_pos = (0.0, 0.5, z_function(0.0, 0.5))
current_pos2 = (0.3, 0.9, z_function(0.3, 0.9))
current_pos3 = (-0.2, 0.5, z_function(-0.2, 0.5))


learning_rate = 0.01

ax = plt.subplot(projection='3d', computed_zorder=False)

for _ in range(1000):
    
    #stocastic version 
    noise_x, noise_y = np.random.normal(0, 0.1, 2)
    noisy_pos = (current_pos[0] + noise_x, current_pos[1] + noise_y)
    X_deriv, Y_deriv = calc_gradient(noisy_pos[0], noisy_pos[1])
    X_new, Y_new = current_pos[0] - learning_rate * X_deriv, current_pos[1] - learning_rate * Y_deriv
    current_pos = (X_new, Y_new, z_function(X_new, Y_new)) 
    
    X_deriv, Y_deriv = calc_gradient(current_pos2[0], current_pos2[1])
    X_new, Y_new = current_pos2[0] - learning_rate * X_deriv, current_pos2[1] - learning_rate * Y_deriv
    current_pos2 = (X_new, Y_new, z_function(X_new, Y_new)) 
    
    X_deriv, Y_deriv = calc_gradient(current_pos3[0], current_pos3[1])
    X_new, Y_new = current_pos3[0] - learning_rate * X_deriv, current_pos3[1] - learning_rate * Y_deriv
    current_pos3 = (X_new, Y_new, z_function(X_new, Y_new)) 
    
    
    ax.plot_surface(X, Y, Z, cmap='viridis', zorder=0)
    ax.scatter(current_pos[0], current_pos[1], current_pos[2], color='cyan', zorder=1)
    ax.scatter(current_pos2[0], current_pos2[1], current_pos2[2], color='magenta', zorder=1)
    ax.scatter(current_pos3[0], current_pos3[1], current_pos3[2], color='magenta', zorder=1)
    plt.pause(0.001)
    ax.clear()