# Distributed Operator Inference (dOpInf)

<!--Minitutorial [MT6](https://meetings.siam.org/sess/dsp_programsess.cfm?SESSIONCODE=82504)/[MT7](https://meetings.siam.org/sess/dsp_programsess.cfm?SESSIONCODE=83058), SIAM Conference on Computational Science and Engineering ([CSE25](https://www.siam.org/conferences-events/siam-conferences/cse25/))\
[Ionut-Gabriel Farcas](https://scholar.google.com/citations?user=Cts5ePIAAAAJ), Virginia Tech\
[Shane A. McQuarrie](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ), Sandia National Laboratories\
[Steffen Werner](https://scholar.google.com/citations?user=F2v1uKAAAAAJ), Virginia Tech\
March 2025, Fort Worth, TX -->

## Contents

- [**sequential_OpInf.py**](./sequential_OpInf.py) script that contains a reference, serial implementation of OpInf
- [**distributed_OpInf.py**](./distributed_OpInf.py) script that contains the distributed memory implementation of dOpInf based on MPI
- [**config/**](/config/) contains the config file [**config.py**](/config/config.py) for setting up
- [**navier_stokes_benchmark/**](/navier_stokes_benchmark/) contains the training data in HDF5 format for the considered 2D Navier-Stokes example
- [**utils/**](/utils/) contains a script [**utils.py**](/utils/utils.py) with several auxiliary functions used in the sequential and distributed OpInf implementations
- [**postprocessing/**](/postprocessing/) folder that contains several utilities for postprocessing the reduced model solution
- [**runtimes/**](/runtimes/) option to save total CPU times of sequential and distributed OpInf implementations for, e.g., scaling studies (see the first few lines in the [**sequential_OpInf.py**](./sequential_OpInf.py) and [**distributed_OpInf.py**](./distributed_OpInf.py) scripts)


## Installation

The code is written in Python.
I recommend creating a new `conda` environment and installing the prerequisites listed in [requirements.txt](./requirements.txt).

```shell
$ conda deactivate                                      # Deactivate any current environments.
$ conda create -n distributed-OpInf python=3.11         # Create a new environment.
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
should install all required dependencies for mpi4py.

<!--If you wish to run the -->

## References
- Farcas, I.-G., Gundevia, R. P., Munipalli, R., and Willcox, K. E., <em>Distributed computing for physics-based data-driven reduced
modeling at scale: Application to a rotating detonation rocket engine,</em> 2024. arXiv:2407.09994 (https://arxiv.org/abs/2407.09994)
