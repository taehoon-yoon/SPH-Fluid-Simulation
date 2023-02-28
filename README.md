# Smoothed Particle Hydrodynamics
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
<img src="./image/png/phase1.png">
[Fig.2 Phase1]

This is the phase where you can edit the SPH particle system. For example, you can change the width, height and depth of fluid blocks. Further you can add more fluid blocks and decide whether to include static rigid body(dragon sculpture) for simulation.

- Axis description and definition of `(x0, y0, z0)` and `(x1, y1, z1)`

<img src="./image/png/front-view.png" height="300">
[Fig3. Axis description]

<img src="./image/png/side-view.png" height="300">
[Fig4. `(x0, y0, z0)` and `(x1, y1, z1)`]


- Widget Option Description

`Start` : Start the SPH simulation. (Go to phase 2)

`Add Fluid Block` : Add aditional fluid block. 

`Delete Recent Fluid Block` : Delete most recent fluid block.

`Include Rigid Object` : Whether to include static rigid body(dragon sculpture) for simulation. If it is not checked, then rigid body will not be presented in simulation.

`x0_{}, y0_{}, z0_{}` : Fluid block { } start point. Since by default there is only one fluid block, there exists only `x0_1, y0_1, z0_1` . If you add more fluid block by `Add Fluid Block` , then there will be more slider bar to adjust the position of fluid block. Eacn slider range from minimal possible point (`origin` + some margin) to maximal possible point ( `(x1, y1, z1)` - some margin).

`x1_{}, y1_{}, z1_{}` : Fluid block { } end point. You can find the definition of `(x0, y0, z0)` and `(x1, y1, z1)` in the [Fig. 3]. (marked as green) Eacn slider range from minimal possible point ( `(x0, y0, z0)` + some margin) to maximal possible point ( `maxium boundary point` (black line vertex) - some margin).

### :warning: WARNING :warning: : Due to GPU memory problem, when you want to change the value of `(x0, y0, z0)` and `(x1, y1, z1)` you must **click** the point you want that value to be located not sliding the bar. Please consult the following gif.

<img src="./image/phase1-manual.gif">
