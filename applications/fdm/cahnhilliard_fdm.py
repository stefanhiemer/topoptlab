import numpy as np
import matplotlib.pyplot as plt

def cahn_hilliard_fd(dim=2, grid_size=128, 
                     dx=1.0, dt=0.01, 
                     epsilon=1.0, M=1.0, 
                     n_steps=500,
                     display=True):
    """
    Solves the Cahn-Hilliard equation using finite differences.
    
    Parameters:
        dim (int): Dimensionality of the problem (2 or 3).
        grid_size (int): Size of the grid in each dimension.
        dx (float): Spatial step size.
        dt (float): Time step size.
        epsilon (float): Interfacial width parameter.
        M (float): Mobility.
        n_steps (int): Number of time steps.
    """
    # Grid and initial condition
    if dim == 2:
        c = np.random.rand(grid_size, grid_size) * 0.1 - 0.05  # Small random initial perturbation
    elif dim == 3:
        c = np.random.rand(grid_size, grid_size, grid_size) * 0.1 - 0.05
    else:
        raise ValueError("Only 2D and 3D cases are supported.")
    
    # Laplacian operator (finite difference)
    def laplacian(f):
        if dim == 2:
            return (
                np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) +
                np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1) - 4 * f
            ) / dx**2
        elif dim == 3:
            return (
                np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) +
                np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1) +
                np.roll(f, 1, axis=2) + np.roll(f, -1, axis=2) - 6 * f
            ) / dx**2
    
    # Time-stepping loop
    for step in np.arange(n_steps):
        # Compute the chemical potential
        mu = -epsilon**2 * laplacian(c) + c**3 - c
        
        # Update the concentration field
        c += dt * M * laplacian(mu)
        
        # Optional: Visualization for 2D
        if dim == 2 and step % (n_steps // 100) == 0 and display:
            plt.imshow(c, cmap='RdBu', origin='lower')
            plt.colorbar(label='Concentration')
            plt.title(f"Step {step}")
            plt.pause(0.001)
            plt.clf()
    
    # Final visualization
    if dim == 2 and display:
        plt.imshow(c, cmap='RdBu', origin='lower')
        plt.colorbar(label='Concentration')
        plt.title("Final Step")
        plt.show()
    elif dim == 3:
        print("Simulation complete. Use 3D visualization tools to analyze results.")
    
    return c

if __name__ == "__main__":
    #
    ndim = 2
    n = 128
    display=True
    #
    import sys
    if len(sys.argv)>1: 
        ndim = int(sys.argv[1])
    if len(sys.argv)>2: 
        n = int(sys.argv[2])
    if len(sys.argv)>3: 
        display = bool(int(sys.argv[3]))
    # Run the simulation 
    cahn_hilliard_fd(dim=ndim, grid_size=n, 
                     dx=1.0, dt=0.01, 
                     epsilon=1.0, M=1.0, 
                     n_steps=int(1e5),
                     display=display)
