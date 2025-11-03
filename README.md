# Inference-Methods-for-Stochastic-Compartmental-Models

This repository contains the code to reproduce and results used in the work "Comparison of Simulation-based Inference Methods for Stochastic Compartmental Models".


## Usage
The code sould be usable on any local machine where Python 3.12 and Julia 1.10 are installed. The methods have the following specific requirements.

### PF
- Manifest.toml
- Project.toml

Additionally, the Particles.jl and ParticlesDE.jl packages must be installed from the local files.

### CNF
- requirements.txt

## Structure

Following folders and content are available in this repository:
- **code:** contains all the code
- **data:** contains the synthetically generated data for the different models.
- **figures:** contains figures presented in the paper.
- **output:** contains result from running both methods and ues for the creation of the figures.
- **petab:** contains the PEtab problem for the two variant model from (Gudina et al. 2021)



**References:**

- Gudina EK, Ali S, Girma E, et al (2021) Seroepidemiology and model-based prediction of SARS CoV 2 in Ethiopia: longitudinal cohort study among front-line hospital workers and communities. The Lancet Global Health 9(11):e1517–e1527. https://doi.org/10.1016/S2214-109X(21)00386-7

