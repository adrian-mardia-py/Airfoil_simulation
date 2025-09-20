"""
Module:
------
Allows creating and determining the number of evaluations needed for the matrix
created in the program FlujoAlrededorPerfil.py

Functions:
----------
contornoPerfil()

Considerations:
---------------
It is necessary to save the matrix when running FlujoAlrededorPerfil.py
"""


import numpy as np
import matplotlib.pyplot as plt
import os


def contornoPerfil(D: np.float64, X: np.float64, Y: np.float64, niveles: np.int32 = 23) -> None:
    """
    Parameters:
    ----------
    D -> Matrix containing the stream function\n
    X, Y -> Coordinate matrices\n
    niveles -> number of evaluations to be performed in each of the three contour zones\n

    Returns:
    --------
    None

    Function:
    ---------
    Creates the contour plot of the airfoil with a higher density of lines
    in the region near the plate
    """

    # The contour lines will start from the very front, the first column of the stream function matrix
    mitad = int(D.shape[1] / 2)
    D1 = D[10, :]

    alto = int(D.shape[0] / 3)

    # Create the levels, can be modified manually
    niveles1 = np.linspace(1, alto, niveles).astype(np.int32)
    niveles2 = np.linspace(alto + 1, 2*alto, niveles).astype(np.int32)
    niveles3 = np.linspace(2*alto + 1, 3*alto, niveles).astype(np.int32)

    # Evaluate the stream function matrix at the selected levels
    D2_1 = np.array([D1[i-1] for i in niveles1])
    D2_2 = np.array([D1[i-1] for i in niveles2])
    D2_3 = np.array([D1[i-1] for i in niveles3])

    # Sort the evaluations and remove duplicates for the contour plot
    nivelesExternos = np.sort(np.unique(np.concatenate((D2_1, D2_3))))
    nivelesPerfil = np.sort(D2_2)

    # Create the contour plot
    plt.contour(X, Y, D, nivelesExternos, colors='lightblue', linestyles='solid')
    plt.contour(X, Y, D, nivelesPerfil, linestyles='solid')


if __name__ == '__main__':

    ruta: str = os.path.join("data", "flujo_data.npz")
    
    data = np.load(ruta)

    X = data['mallaX']
    Y = data['mallaY']
    D = data['flujo']

    niveles = np.int32(input('Enter the desired number of evaluations: '))

    contornoPerfil(D, X, Y, niveles)

    plt.show()
