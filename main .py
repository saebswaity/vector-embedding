import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_reduction_matrix_C(matrix):
    """
    Calculate the reduction matrix C using Singular Value Decomposition (SVD).
    
    Parameters:
    - matrix: The input matrix
    
    Returns:
    - C: The reduction matrix
    """
    U, S, Vt = np.linalg.svd(matrix)
    C = np.dot(np.diag(np.sqrt(S)), Vt)
    return C

def visualize_3d_vectors(matrix, labels):
    """
    Visualize 3D vectors using the reduction matrix C and labels.
    
    Parameters:
    - matrix: The input matrix
    - labels: Labels for each vector
    """
    # Calculate the reduction matrix C
    C = calculate_reduction_matrix_C(matrix)

    # Selecting the first three rows of C for 3D visualization
    C_3d = C[:3, :]
    C_3d0 = np.c_[C_3d, np.zeros(3)]

    # Increase the size of the figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Adding arrows
    origin = [0, 0, 0]
    ax.quiver(*origin, C_3d[0, :], C_3d[1, :], C_3d[2, :])

    # Scatter plot for points
    ax.scatter(C_3d0[0, :], C_3d0[1, :], C_3d0[2, :], c=['red', 'green', 'blue', 'cyan', 'magenta', 'black', 'brown', 'gray'], marker='o')

    # Adding labels for each point
    for i, word in enumerate(labels + ['origin']):
        ax.text(C_3d0[0, i], C_3d0[1, i], C_3d0[2, i], word)

    # Adding axis labels
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')

    # Remove x-axis and y-axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Displaying the plot
    plt.show()

# Given W' matrix and labels
W_prime = np.array([
    # love buck like water life hate depression
    [ 1.0, 0.0,  0.7, 0.3,  0.5, -0.9, -0.3], # love
    [ 0.0, 1.0,  0.0, 0.6,  0.0,  0.0,  0.0], # bucket
    [ 0.7, 0.0,  1.0, 0.2,  0.2, -0.7, -0.1], # like
    [ 0.3, 0.6,  0.2, 1.0,  0.6,  0.0,  0.0], # water
    [ 0.5, 0.0,  0.2, 0.6,  1.0, -0.5,  0.4], # life
    [-0.9, 0.0, -0.7, 0.0, -0.5,  1.0,  0.6], # hate
    [-0.3, 0.0, -0.1, 0.0,  0.4,  0.6,  1.0]  # depression
])
 
word_labels = ['love', 'bucket', 'like', 'water', 'life', 'hate', 'depression']

# Visualize 3D vectors using the reduction matrix C
visualize_3d_vectors(W_prime, word_labels)
