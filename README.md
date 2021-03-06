## [Untethered Soft Robots - Design and Closed Loop Motion Planning with Discrete Elastic Rods]

Numerical simulation for untethered soft robots using DER method.

***

***

## How to Use

### Dependencies
Install the following C++ dependencies:
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
  - Eigen is used for various linear algebra operations.

- [Pardiso](https://pardiso-project.org/)
  - Pardiso is used for the solving of linear systems.

- [OpenGL / GLUT](https://www.opengl.org/)
  - OpenGL / GLUT is used for rendering the knot through a simple graphic.
  - Simply install through apt package manager:
      ```bash
    sudo apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev
    ```
- Lapack (*usually preinstalled on your computer*)

***
### Compiling

g++ -I 'Eigen link' main.cpp world.cpp elasticRod.cpp setInput.cpp timeStepper.cpp inertialForce.cpp externalGravityForce.cpp elasticStretchingForce.cpp geometry.cpp elasticBendingForce.cpp dampingForce.cpp -lGL -lglut -lGLU -lpardiso600-GNU720-X86-64 -llapack -lgfortran -fopenmp -lpthread -lm -Ofast -o simDER

***

***
### Geometry
The input file for geometry can be generated by "geometry.py" or "geometry.cpp".

- ```nodes.txt``` - Position of input nodes, N*2, [node x position, node y position].
- ```stretching.txt``` - Stretching element, N*2, [nodeIndex1, nodeIndex2].
- ```bending.txt``` - Bending element, N*4, [nodeIndex1, nodeIndex2, nodeIndex3, relative stiffness].
- ```actuationSequence.txt``` - Actuation sequence of four limbs. "1" represents activation and "0" represents no activation 
- ```actuation.txt``` - Nodes of four limbs. Each column corresponds to a single limb. 


***

### Setting Parameters

All simulation parameters are set through a parameter file ```option.txt```.

Specifiable parameters are as follows (we use SI units):
- ```RodLength``` - Contour length of the rod.
- ```rodCurvature ``` - Undeformed curvature of rod.
- ```rodDensity ``` - Density of rod on a limb.
- ```frameDensity ``` - Density of rod on the frame.
- ```deltaLength ``` - Length of each rod.
- ```rodRadius ``` - Radius of the rod.
- ```frameWidth ``` - Width of the frame.
- ```frameLength ``` - Length of the frame.
- ```limbNumber ``` - Number of limbs.
- ```youngM ``` - Young's modulus of rod.
- ```Poisson ``` - Poisson ratio of rod.
- ```deltaTime ``` - Time of each step in the simulation.
- ```totalTime ``` - Total time in the simulation.
- ```tol ``` - Tolerance for convergence.
- ```maxIteration ``` - Max iteration for convergence.
- ```gVector ``` - Gravity vector.
- ```render ``` - Render option. "1" represents "on" and "0" represents "off".
- ```saveData ``` - Save data option. "1" represents "on" and "0" represents "off".
- ```cf1 ``` - Drag coefficient of the frame in orthogonal direction.
- ```cf2 ``` - Drag coefficient of the frame in tangent direction.
- ```cr1 ``` - Drag coefficient of the limb in orthogonal direction.
- ```cr2 ``` - Drag coefficient of the limb in tangent direction.
- ```phaseDelay ``` - Phase offset between front limbs and rear limbs.
- ```frequency ``` - Actuation frequency of limbs.
- ```tauLogisitic ``` - Coefficient in the equation characterizing Young's modulus change of limbs during actuation and deactivation.
- ```actuationTime ``` - Actuation time of each limb.
- ```actA - actD ``` - Coefficient in the equation characterizing curvature change of limbs during actuation.
- ```deactA - deactD ``` - Coefficient in the equation characterizing curvature change of limbs during deactuation.
- ```addMass ``` - Coefficient of added mass.
- ```EGaIn ``` - EGaIn volume ratio in "%".





***
### Fitting data for actuation

- ```fitF00_act - fitF50_act ``` - Coefficient in the equation characterizing curvature change of the limb during actuation at each frequency for a corresponding volume ratio of EGaIn.
- ```fitF00_deact - fitF50_deact ``` - Coefficient in the equation characterizing curvature change of the limb during deactuation at each frequency for a corresponding volume ratio of EGaIn.

***

***
### Running the Simulation
Once parameters are set to your liking, the simulation can be run from the terminal by running the provided script:

./simDER option.txt

***

***
### Saving the Data
Make sure you set the option to save the data from your simulations. You can easily run it in a loop in a python script:

```python
#!/usr/bin/python3

import os

num_tests = 2000

for i in range(num_tests):
    print('Starting test: ' + str(i))
    os.system("./simDER option.txt")
```
Once this data is collected, it can be processed into a .npy file for direct implementation in numpy.

***

***
## Planning/Control Software

***

### Dependencies
To setup the environment, make the 'conda_setup.bash' file executable, then run it. Say yes to its prompts, then follow the text instructions it spits out.

To make conda_setup executable, run

$chmod+x conda_setup.bash

$./conda_setup.bash

in the folder that conda_setup.bash is found in.

***

### Running the Planner

Once a data library is collected, the controller can be run from the terminal by running:

```./frog_planner```.

This script simply calls:

```python3 src/main.py 'False' 'brute' 'si' 1 'data/simfrog_numpy_data_2022-01-26.npy' '/data/results'```.

Options can be adjusted as follows:

- Option 1: T/F if there is a calibration grid. The code works well without it, so I use 'False'.
- Option 2: Planner uses nearest neighbor ('brute') or locally weighted regression ('lwr') to determine the next state.
- Option 3: Type of path to follow. Options are line 'li', sinusoid 'si', and ellipsoid 'el'.
- Option 4: Depth of tree of the planner (integer). Works well with smaller values (e.g. 2).
- Option 5: is the numpy file containing the transition models from the DER library (e.g. 'data/simfrog_numpy_data_2022-01-26.npy').
- Option 6: directory of saved results.

***