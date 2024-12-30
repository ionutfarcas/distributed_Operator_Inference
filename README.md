# Distributed Operator Inference (dOpInf)

<!--Minitutorial [MT6](https://meetings.siam.org/sess/dsp_programsess.cfm?SESSIONCODE=82504)/[MT7](https://meetings.siam.org/sess/dsp_programsess.cfm?SESSIONCODE=83058), SIAM Conference on Computational Science and Engineering ([CSE25](https://www.siam.org/conferences-events/siam-conferences/cse25/))\
[Ionut-Gabriel Farcas](https://scholar.google.com/citations?user=Cts5ePIAAAAJ), Virginia Tech\
[Shane A. McQuarrie](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ), Sandia National Laboratories\
[Steffen Werner](https://scholar.google.com/citations?user=F2v1uKAAAAAJ), Virginia Tech\
March 2025, Fort Worth, TX -->

## Contents

- [**sequential_OpInf.py/**](./sequential_OpInf.py) script that contains a reference, serial implementation of OpInf
- [**distributed_OpInf.py/**](./distributed_OpInf.py) script that contains the distributed memory implementation of dOpInf based on MPI
- [**config/**](/config/) contains the config file


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
To install mpi4py within the distributed-OpInf virtual environment, you first need to make sure you have an MPI implementation an your system.
On a Debian-based OS, for example, you can install OpenMPI as
```shell
$ sudo apt-get install libopenmpi-dev
```
followed by
```shell
$ env MPICC=/yourpath/mpicc pip3 install mpi4py
```

Alternatively,
``` shell
$ conda install mpi4py
```
should install all required dependencies for mpi4py.

<!--If you wish to run the -->

## References
Farcas, I.-G., Gundevia, R. P., Munipalli, R., and Willcox, K. E., "Distributed computing for physics-based data-driven reduced
modeling at scale: Application to a rotating detonation rocket engine" arXiv:2407.09994 (https://arxiv.org/abs/2407.09994)
