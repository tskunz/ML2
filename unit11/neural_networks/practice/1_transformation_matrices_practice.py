"""
Transformation Matrices Practice Exercises
---------------------------------------

This script contains hands-on exercises for understanding and implementing
transformation matrices in neural networks.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_points(original, transformed, title1="Original Points", title2="Transformed Points"):
    """Helper function to visualize original and transformed points."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.scatter(original[:, 0], original[:, 1], alpha=0.5)
    plt.title(title1)
    plt.grid(True)
    plt.axis('equal')
    
    plt.subplot(122)
    plt.scatter(transformed[:, 0], transformed[:, 1], alpha=0.5)
    plt.title(title2)
    plt.grid(True)
    plt.axis('equal')
    
    plt.show()

# Exercise 1: Basic Linear Transformation
# -------------------------------------
def exercise1():
    """Implement a simple scaling transformation."""
    print("Exercise 1: Scaling Transformation")
    
    # Generate random 2D points
    points = np.random.randn(100, 2)
    
    # TODO: Create a scaling matrix that doubles the x coordinates
    # and halves the y coordinates
    scaling_matrix = np.array([
        [2.0, 0.0],
        [0.0, 0.5]
    ])
    
    # Apply transformation
    transformed_points = points @ scaling_matrix
    
    # Visualize results
    plot_points(points, transformed_points, "Original Points", "Scaled Points")
    
    return scaling_matrix

# Exercise 2: Rotation Matrix
# -------------------------
def exercise2():
    """Implement a rotation transformation."""
    print("Exercise 2: Rotation Transformation")
    
    # Generate random 2D points
    points = np.random.randn(100, 2)
    
    # TODO: Create a rotation matrix for 45 degrees counterclockwise
    theta = np.pi/4  # 45 degrees in radians
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Apply transformation
    transformed_points = points @ rotation_matrix
    
    # Visualize results
    plot_points(points, transformed_points, "Original Points", "Rotated Points")
    
    return rotation_matrix

# Exercise 3: Composite Transformations
# ----------------------------------
def exercise3():
    """Implement a composite transformation (scaling + rotation)."""
    print("Exercise 3: Composite Transformation")
    
    # Generate random 2D points
    points = np.random.randn(100, 2)
    
    # TODO: Create scaling and rotation matrices
    scaling_matrix = np.array([
        [1.5, 0.0],
        [0.0, 0.5]
    ])
    
    theta = np.pi/6  # 30 degrees
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # TODO: Apply composite transformation (first scale, then rotate)
    transformed_points = points @ scaling_matrix @ rotation_matrix
    
    # Visualize results
    plot_points(points, transformed_points, "Original Points", "Scaled and Rotated Points")
    
    return scaling_matrix, rotation_matrix

# Exercise 4: Neural Network Layer Simulation
# ----------------------------------------
def exercise4():
    """Simulate a simple neural network layer transformation."""
    print("Exercise 4: Neural Network Layer Simulation")
    
    # Generate random 2D points
    points = np.random.randn(100, 2)
    
    # TODO: Create a weight matrix and bias vector for a simple neural network layer
    W = np.array([
        [1.5, -0.5],
        [0.5, 1.0]
    ])
    b = np.array([1.0, -1.0])
    
    # Apply affine transformation: y = xW + b
    transformed_points = points @ W + b
    
    # Apply non-linear activation (ReLU)
    activated_points = np.maximum(0, transformed_points)
    
    # Visualize results
    plot_points(points, activated_points, "Input Layer", "After ReLU Activation")
    
    return W, b

if __name__ == "__main__":
    print("Transformation Matrices Practice Exercises")
    print("----------------------------------------")
    
    # Run exercises
    scaling_matrix = exercise1()
    rotation_matrix = exercise2()
    scaling_rotation = exercise3()
    W, b = exercise4()
    
    print("\nExercise completed! Try modifying the transformation matrices to see different effects.") 