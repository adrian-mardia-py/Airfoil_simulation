"""
Author: Adrián Martín Díaz

04/05/2021 ----> v1
14/04/2025 ----> v2
---
v1:
---
Code that, given a two-dimensional thin airfoil defined by a polynomial up to degree 3,
computes the circulation density along the chord, the complex potential function,
and plots the streamlines around the airfoil.
Parallelization with CUDA is used for handling the matrices.

v2:
---
The singularity of the leading-edge vortex has been corrected.

TODO
----
- Handle angle of attack
- Compute lift coefficient, moment coefficient, aerodynamic center, and moment center
- Improve memory management visualization
- Generate the NACA profile equation by just providing its digits
- Create airfoil animation
- Generate streamlines by clicking on the plot
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import PruebasCurvasNivel as pcn

from numpy import pi
from numba import cuda, njit
from numba.cuda import libdevice as ld
from timeit import default_timer as timer
from inputimeout import inputimeout, TimeoutOccurred


@cuda.jit
def funcionCorriente(
    Tr: np.float32,
    Ti: np.float32, 
    Ci: np.float32, 
    D: np.float32,
    rango: np.float32, 
    gamma: np.float32) -> None:
    """
    Parameters
    ----------
    Tr -> Real coordinate mesh\n
    Ti -> Imaginary coordinate mesh\n
    Ci -> Imaginary solution mesh\n
    D -> Solution mesh\n
    rango -> Vortex positions\n
    gamma -> Circulation at each point\n

    Returns
    -------
    None

    Considerations
    ---------------
    All units and arrays must be of type np.float32, works best with Numba
    
    Function
    -------
    Using the GPU in parallel, creates a matrix with the values of the stream function at each point
    """
    
    # Create matrix indices in parallel
    idx, jdx, kdx = cuda.grid(3)
    
    # Ensure evaluation only within memory boundaries
    if idx < Tr.shape[0] and jdx < Tr.shape[1] and kdx < Tr.shape[2]:
        # Define some variables
        Trx = Tr[idx, jdx, 0]
        Tix = Ti[idx, jdx, 0]

        # Optional exclusion zone (not evaluated)
        if False: #abs(Tr[idx, jdx, 0] - 3.5/2) < 3.5/2 and abs(Ti[idx, jdx, 0]) < 0.1:
            Ci[idx, jdx, kdx] = 0
        else:
            modulo = ld.sqrt(Trx**2 + Tix**2)
            argumento = ld.atan2(Tix, Trx)
            rgx = rango[kdx]

            # For the stream function, imaginary part of the complex potential
            Ci[idx, jdx, kdx] = (gamma[kdx] / (2 * pi)) * ld.log(modulo**2 + rgx**2 - 2*modulo*rgx*ld.cos(argumento))

            # Add each vortex
            cuda.atomic.add(D, (idx, jdx), Ci[idx, jdx, kdx])

            # Add free stream
            cuda.atomic.add(D, (idx, jdx), 10 * ld.cos(10) * Tix)

@njit
def circulacion(dPerfil: np.float32, z: np.float32, alfa: np.float32) -> np.float32:
    """
    Parameters
    ----------
    dPerfil -> Airfoil derivative polynomial coefficients\n
    z -> Position of each vortex\n

    Returns
    --------
    Array with circulation at each vortex ---> np.float32

    Considerations
    ---------------
    - Fourier coefficients computed for a cubic airfoil polynomial (before differentiation)
    - Alpha in radians
    - At the leading edge, a singularity (infinite vorticity) occurs. It has been approximated, but as a result some streamlines cross the airfoil.
      v2 -> Circulation at the leading edge is approximated as half of the previous value, according to *Potential Aerodynamics*. Being an integrable
            singularity with zero integral, it has no effect on resulting forces and/or moments.

    Function
    -------
    Creates a discrete circulation distribution for each vortex along the airfoil
    """

    # Fourier coefficients of the derivative polynomial ax^2 + bx + c 
    A0 = np.float32(0.25 * (3 * dPerfil[0] + 2 * dPerfil[1] * dPerfil[2]))
    A1 = np.float32(-0.5 * (dPerfil[0] + dPerfil[1]))
    A2 = np.float32(0.5 * dPerfil[0])

    # Angle of attack, velocity and x differential
    dx = z[1] - z[0]
    uInf = 10

    # Normalize chord range from 0 to 1 (avoids division by zero) and compute circulation
    x = z / (z[z.shape[0] - 1]) 
    gamma = np.zeros_like(x).astype(np.float32)

    for i in range(x.shape[0] - 1):
        if x[i] == 0:
            #gamma[i] = np.float32(-2 * uInf * dx * ((alfa - A0) * 2 / np.sin(0.01) + A2 * np.sin(2)))
            gamma[i] = 0
        else:
            gamma[i] = np.float32(-2 * uInf * dx * ((alfa - A0) * (2 - 2*x[i]) / (np.sin(np.arccos(1 - 2*x[i]))) + A1 * np.sin(np.arccos(1 - 2*x[i])) + A2 * np.sin(2 * np.arccos(1 - 2*x[i]))))

    gamma[0] = 0.5 * gamma[1] # Smooth circulation at leading edge

    # Return computed circulation
    return gamma

def titulo(cadena: str) -> None:
    print("\n", "-" * 15, cadena.upper(), "-" * 15)

def main(perfil: np.float32, alfa: np.float32 = 0) -> None:
    """
    Parameters
    ----------
    perfil -> Polynomial coefficients of the airfoil to analyze

    Returns
    --------
    None

    Function
    ---------------
    Plots the contour of the streamlines around the input airfoil
    """

    titulo("Running program")

    # Differentiate airfoil equation
    derivadaPerfil = np.array([(perfil.shape[0] - 1 - i) * perfil[i] for i in range(perfil.shape[0])])
    dPerfil = derivadaPerfil[:-1].astype(np.float32)

    # Create meshes and matrices
    #numVortices = 60
    cuerda = 3

    numVortices = int(input("\nEnter the number of vortices to place along the airfoil chord --> "))
    
    x = np.linspace(-4, 4, 2000).astype(np.float32)
    y = np.linspace(-4, 4, 2000).astype(np.float32)
    z = np.linspace(0, cuerda, numVortices).astype(np.float32)

    [Tr, Ti, Z] = np.meshgrid(x, y, z, indexing="ij") #T = Tr + j*Ti

    Ci = np.zeros_like(Tr).astype(np.float32)   # Imaginary solution mesh for each vortex (empty)
    D = np.zeros_like(Tr[:, :, 0]).astype(np.float32)   # Final solution mesh (empty)

    print('\nMeshes created successfully')

    # Compute circulation at each vortex
    titulo("Computing circulation")
    gamma = circulacion(dPerfil, z, alfa)

    print(f'\nCirculation distribution along the chord:\n')
    print(gamma)

    # Transfer matrices to GPU
    titulo("Computing stream function")

    Tr_gpu = cuda.to_device(Tr)
    Ti_gpu = cuda.to_device(Ti)
    Ci_gpu = cuda.to_device(Ci)
    D_gpu  = cuda.to_device(D)
    z_gpu  = cuda.to_device(z)
    gamma_gpu = cuda.to_device(gamma)
    
    # Define blocks and threads per block
    threads_per_block = (16, 16, 4)  #16*16*4 = 1024 threads, maximum
    blocks_per_grid_x = (Tr.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (Tr.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid_z = (Tr.shape[2] + threads_per_block[2] - 1) // threads_per_block[2]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

    print(f'\nGPU memory usage:\n\n-Threads per block {threads_per_block}\n--Total {threads_per_block[0] * threads_per_block[1] * threads_per_block[2]} threads')
    print(f'\n-Blocks per grid {blocks_per_grid}\n--Total: {blocks_per_grid_x * blocks_per_grid_y * blocks_per_grid_z} blocks')
    print(f'\nMatrix size ({Tr.shape[0]}, {Tr.shape[1]}, {Tr.shape[2]})\n--Total: {Tr.shape[0] * Tr.shape[1] * Tr.shape[2]} elements per matrix')
    
    # Execute GPU function
    a = timer()

    funcionCorriente[blocks_per_grid, threads_per_block](
        Tr_gpu,
        Ti_gpu,
        Ci_gpu,
        D_gpu, 
        z_gpu,
        gamma_gpu
    )
    
    cuda.synchronize()  

    D_gpu.copy_to_host(D)

    b = timer()
    
    print(f'\nOperations completed in {b-a:.2f} seconds')
    titulo("Creating contour plot")
    
    c = timer()

    # Parameters to optimize contour creation by reducing matrix density
    Tr_plot = Tr[::4, ::4, 0]
    Ti_plot = Ti[::4, ::4, 0]
    D_plot = D[::4, ::4]
    
    # Create contour plot and line simulating vortex line
    pcn.contornoPerfil(D, Tr[:, :, 0], Ti[:, :, 0])
    
    plt.title(f'Streamlines around airfoil with alpha = {alfa}, equation -> {perfil[0]:.2f}x^3 + {perfil[1]:.2f}x^2 + {perfil[2]:.2f}x + {perfil[3]:.2f}')
    plt.plot(z, np.zeros_like(z), 'k-')
    plt.plot(z, np.zeros_like(z), 'rd', markersize = 1)

    d = timer()
    
    print(f'\nContour plot created in {d-c:.2f} seconds')
    
    # Show flow around airfoil
    plt.show(block=True)

    # Ask user whether to save data matrix
    e = timer()
    while True:
        # If no response in 15 seconds, data will not be saved
        try:
            guardarMatriz: str = inputimeout(prompt = "\n\nDo you want to save the data matrix? (yes/no): ", timeout=15)
        except TimeoutOccurred:
            print("\n\nNo selection detected, data will not be saved")
            break

        if guardarMatriz.upper() == None:
            print("No selection, data will not be saved")
            break

        if guardarMatriz.upper() == "YES":
            titulo("Saving data")

            # For storing the data
            os.makedirs("data", exist_ok=True)

            # Guardar el archivo en data/
            ruta: str = os.path.join("data", "flujo_data.npz")

            np.savez(ruta, mallaX = Tr[:, :, 0], mallaY = Ti[:, :, 0], flujo = D)

            break

        elif guardarMatriz.upper() == "NO":
            break

        print("\n\nInvalid selection, please try again (Do not use accents)")
    
    # Free GPU memory
    del Tr_gpu,Ti_gpu, Ci_gpu, D_gpu, z_gpu, gamma_gpu
    cuda.current_context().deallocations.clear()

    titulo("Program finished successfully")

# Run program
if __name__ == "__main__":
    # Example airfoil equation: 5x^3 + 5x^2 + 3x + 4
    epsilon = 0.5
    perfil: np.float32 = epsilon * np.array([5, 5, 3, 4])

    # To view documentation of each function set doc=True, to run program set doc=False
    doc = True

    # Clear terminal
    os.system('cls')

    if not doc:
        main(perfil)
    else:
        titulo("funcionCorriente")
        print(funcionCorriente.__doc__)

        titulo("Circulacion")
        print(circulacion.__doc__)

        titulo("main")
        print(main.__doc__)

        titulo("PruebasCurvasNivel")
        print(pcn.__doc__)

        titulo("contornoPerfil")
        print(pcn.contornoPerfil.__doc__)
