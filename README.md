# Thin Airfoil Flow Simulation with GPU Acceleration

This project simulates the aerodynamic interference of a thin airfoil using **linear potential theory** and **GPU parallelization**.

## Overview
The analysis of thin airfoils can be simplified by replacing the camber line with a distribution of vortices that "bend" the airflow.  
Since the governing equations (Laplace’s equation) are linear, the total flow solution can be expressed as the sum of independent elemental solutions.  

This property makes the problem highly suitable for **parallel computing on GPUs**.

## Methodology
The implementation follows these steps:
1. **Vortex contribution**:  
   - For each vortex, its induced velocity field is calculated.  
   - The contribution is stored as a layer of a 3D matrix.  

2. **Field superposition**:  
   - All layers are summed using **atomic operations**.  
   - The free-stream velocity is added to obtain the total flow field.  

3. **Vortex strength calculation**:  
   - The intensity of each vortex is computed using optimized routines decorated with `@njit` from the **Numba** library.  

4. **Streamline visualization**:  
   - The module `PruebasCurvasNivel.py` evaluates streamlines in the flow field.  
   - Only the central streamlines are highlighted, clearly showing how most streamlines are deflected below the airfoil, producing the **high-pressure region responsible for lift**.  

## Technical Details
- All GPU operations are executed using **`np.float32` precision**, which provides the best performance with CUDA.  
- Parallelization is achieved with **Numba CUDA** kernels.  
- Visualization is handled with Python plotting tools.  

## Requirements
- Python 3.9+  
- Numba  
- NumPy  
- Matplotlib (for visualization)  

Install dependencies with:
```bash
pip install -r requirements.txt
