# SPH
### **SPH**(Smoothed-particle hydrodynamics) implementation with built-in adjustable parameter using Taichi Lang.

## Objective
- **Smoothed-particle hydrodynamics** (SPH) is a computational method used for simulating the mechanics of continuum media, such as solid mechanics and fluid flows. It is a meshfree Lagrangian method (where the co-ordinates move with the fluid), and the resolution of the method can easily be adjusted with respect to variables such as density.[wiki] 

- SPH was originally developed for astrophysical problems. To glance over the basic of SPH, you may find useful to cosult the folowing toy project [Simulating star with SPH](https://github.com/sillsill777/vpython-projects).

- In this project with the SPH formalism, we will numerically solve fluid equations which governs the movement of fluid flow. Further we will consider several effects governing fluid motion such as viscosity and surface tension. Also handle the issue of Fluid-Rigid coupling.

## Result
<img src="./image/default.gif">
[Fig. 1 Default setting]

## Program Description
- Our program consists of two phases. First phase is set-up phase for simulating SPH and second phase is a simulation phase.  
### First Phase (set-up phase)
- If you run the `run_simulation.py`, you will see the following window.

