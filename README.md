# Distributed Operator Inference

<!--Minitutorial [MT6](https://meetings.siam.org/sess/dsp_programsess.cfm?SESSIONCODE=82504)/[MT7](https://meetings.siam.org/sess/dsp_programsess.cfm?SESSIONCODE=83058), SIAM Conference on Computational Science and Engineering ([CSE25](https://www.siam.org/conferences-events/siam-conferences/cse25/))\
[Ionut-Gabriel Farcas](https://scholar.google.com/citations?user=Cts5ePIAAAAJ), Virginia Tech\
[Shane A. McQuarrie](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ), Sandia National Laboratories\
[Steffen Werner](https://scholar.google.com/citations?user=F2v1uKAAAAAJ), Virginia Tech\
March 2025, Fort Worth, TX

## Contents

This minitutorial is presented in two parts.

- [**TimeDomain/**](/TimeDomain/) contains data and examples of data-driven model reduction when observations of the system state are available.
- [**FrequencyDomain/**](./FrequencyDomain/) contains data and examples of data-driven model reduction when frequency input-output observations are available.

See [slides.pdf (**TODO**)](./slides.pdf) for the presentation slides-->.

## Installation

The code is written in Python.
We recommend creating a new `conda` environment and installing the prerequisites listed in [requirements.txt](./requirements.txt).

```shell
$ conda deactivate                                      # Deactivate any current environments.
$ conda create -n distributed-OpInf python=3.11          # Create a new environment.
$ conda activate distributed-OpInf                      # Activate the new environment.
(cse-minitutorial) $ pip install -r requirements.txt    # Install required packages.
```

Alternatively, create a new virtual environment with `venv`.

<!--If you wish to run the -->

## References
[Farcas, I.-G., Gundevia, R. P., Munipalli, R., and Willcox, K. E., ``Distributed computing for physics-based data-driven reduced
modeling at scale: Application to a rotating detonation rocket engine" arXiv:2407.09994] (https://arxiv.org/abs/2407.09994)
