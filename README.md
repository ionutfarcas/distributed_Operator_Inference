# Distributed Operator Inference (dOpInf)

Distributed Operator Inference (dOpInf) [1][## References] is an algorithm for fast and scalable construction of predictive physics-based reduced-order models (ROMs) trained from data sets of extremely large state dimension.
The algorithm learns structured physics-based ROMs [3] that approximate the dynamical systems underlying those data sets
This repository provides a step-by-step tutorial using a 2D Navier-Stokes flow over a step scenario as a case study based on Ref. [2].
The goal of this tutorial is to guide users through the implementation process and make dOpInf accessible for integration into complex application scenarios.

## Contents

- [**sequential_OpInf.py**](./sequential_OpInf.py) script containing a reference, serial implementation of OpInf
- [**distributed_OpInf.py**](./distributed_OpInf.py) script containing the distributed memory implementation of dOpInf based on MPI
- [**config/**](/config/) contains the config file [**config.py**](/config/config.py) for setting up
- [**navier_stokes_benchmark/**](/navier_stokes_benchmark/) containing the training data in HDF5 format for the considered 2D Navier-Stokes example
- [**utils/**](/utils/) contains a script [**utils.py**](/utils/utils.py) with several auxiliary functions used in the sequential and distributed OpInf implementations
- [**postprocessing/**](/postprocessing/) folder containing several utilities for postprocessing the reduced model solution
- [**runtimes/**](/runtimes/) option to save total CPU times of sequential and distributed OpInf implementations for, e.g., scaling studies (see the first few lines in the [**sequential_OpInf.py**](./sequential_OpInf.py) and [**distributed_OpInf.py**](./distributed_OpInf.py) scripts)
- [**high_fidelity_code/**](./high_fidelity_code/) folder containing the script to run the high-fidelity 2D Navier-Stokes model and a few extra utilities:
    - [**generate_high_fidelity_data.py**](./high_fidelity_code/generate_high_fidelity_data.py) script used to generate the high-fidelity data by solving the 2D Navier-Stokes equations using [FEniCS](https://fenicsproject.org/). This implementation closely follows this [tutorial](https://fenicsproject.org/pub/tutorial/html/._ftut1009.html)
    - [**extract_probe_indices.py**](./high_fidelity_code/extract_reference_probe_data.py) script used to extract the snapshot indices for user-specified probe locations for comparing the reference with the reduced solutions
    - [**extract_reference_probe_data.py**](./high_fidelity_code/extract_reference_probe_data.py) script used to extract the high-fidelity data at the indices found by [**extract_probe_indices.py**](./high_fidelity_code/extract_reference_probe_data.py) script. Note that the full dataset was not uploaded here due to its size, but it can be regenerated by running the high-fidelity code ([**generate_high_fidelity_data.py**](./high_fidelity_code/generate_high_fidelity_data.py))
- [**distributed_OpInf_tutorial_2D_Navier_Stokes.ipynb**](./distributed_OpInf_tutorial_2D_Navier_Stokes.ipynb) Jupyter Notebook containing a detailed, step-by-step tutorial that implements dOpInf in the considered 2D Navier-Stokes example

## Installation

The code is written in Python.
I recommend creating a new `conda` environment and installing the prerequisites listed in [requirements.txt](./requirements.txt).

```shell
$ conda deactivate                                      # Deactivate any current environments.
$ conda create -n distributed-OpInf python=3.12         # Create a new environment.
$ conda activate distributed-OpInf                      # Activate the new environment.
(distributed-OpInf) $ pip install -r requirements.txt   # Install required packages.
```

Alternatively, create a new virtual environment with `venv`.

You will also need mpi4py to run the code.
To install mpi4py within the <em>distributed-OpInf</em> virtual environment, you first need to make sure you have an MPI implementation an your system.
On a Debian-based OS, for example, you can install OpenMPI as
```shell
$ sudo apt-get install libopenmpi-dev
```
followed by
```shell
$ env MPICC=/yourpath/mpicc pip install mpi4py
```

Alternatively,
``` shell
$ conda install mpi4py
```
should install mpi4py + all required dependencies.
If there are any problems with dependencies,
``` shell
$ conda install package_name
```
should do the trick.

## Running the code

#### Running the Jupyter Notebook in parallel
To run the Jupyter Notebook in parallel, you will also need [IPython](https://ipyparallel.readthedocs.io/en/latest/) for parallel computing.
This can be installed as
``` shell
$ conda install ipyparallel
```
You then need to start a cluster (collection of IPython engines to use in parallel) to run the Notebook in parallel via
``` shell
$ mpiexec -n <number_of_processes> ipcluster start -n <number_of_engines>
```
Finally, open the Jupyter Notebook as
``` shell
$ jupyter-notebook distributed_OpInf_tutorial_2D_Navier_Stokes.ipynb
```

#### Running the script that implements dOpInf in parallel

The script that implements dOpInf ([**distributed_OpInf.py**](./distributed_OpInf.py)) can be run as
``` shell
$ mpiexec -n <number_of_processes> python3 distributed_OpInf.py
```
or
``` shell
$ mpirun -n <number_of_processes> python3 distributed_OpInf.py
```

#### Running the high-fidelity 2D Navier-Stokes code
For running the high fidelity code via the script [**generate_high_fidelity_data.py**](./high_fidelity_code/generate_high_fidelity_data.py), you will need to install [FEniCS](https://fenicsproject.org/) on your system. To this end, you can follow the steps summarized [here](https://fenicsproject.org/download/archive/).

## References
[1] Farcas, I.-G., Gundevia, R. P., Munipalli, R., and Willcox, K. E., "Distributed computing for physics-based data-driven reduced
modeling at scale: Application to a rotating detonation rocket engine," 2024. arXiv:2407.09994 (https://arxiv.org/abs/2407.09994)

[2] Farcas, I.-G., Gundevia, R. P., Munipalli, R., and Willcox, K. E., "A Parallel Implementation of Reduced-Order Modeling of Large-Scale Systems," <em>In Proceedings of AIAA SciTech Forum & Exhibition, Orlando, FL, January 2025, Session: PC-15, High Performance Computing</em>

[3] Peherstorfer, B., and Willcox, K., "Data-driven operator inference for nonintrusive projection-based model reduction," <em>Computer
Methods in Applied Mechanics and Engineering</em>, Vol. 306, 2016, pp. 196–215. https://doi.org/https://doi.org/10.1016/j.cma.2016.03.025.
