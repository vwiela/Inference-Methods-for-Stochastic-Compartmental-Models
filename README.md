# Inference-Methods-for-Stochastic-Compartmental-Models

This repository contains the code to reproduce and results used in the work "Assessment of Simulation-based Inference Methods for Stochastic Compartmental Models".


## Usage
The code sould be usable on any local machine where Python 3.10 and Julia 1.10 are installed. The methods have the following specific requirements.

**Note:** The use of Julia and Python interactively, does need some caution when setting up the environments. It is recommended to first build the Python virtual environemnt, then configure the Julia callable, before installing _pyjulia_(https://github.com/JuliaPy/pyjulia) and _PyCall.jl_ (https://github.com/JuliaPy/PyCall.jl). Guidelines on how to use the local python environment in PyCall can be found [here](https://github.com/JuliaPy/PyCall.jl).

Alternatively, using [PythonCall.jl](https://github.com/JuliaPy/pyjulia?tab=readme-ov-file) provides a harmonic interface between Python and Julia.

Alternatively, it is also possible to use a different Julia callable for the call from Python than for the native Julia calculations. User should only make sure, that the same version is used.

Independent of the chosen Julia and Python integration, the requirements for both are listed in the following files.

### Julia
- Manifest.toml
- Project.toml

Additionally, the Particles and ParticlesDE packages must be installed from ParticleFilters.jl (https://github.com/lcontento/ParticleFilters.jl).


### Python
- requirements.txt

## Structure

Following folders and content are available in this repository:
- **code:** contains all the code
  - **cnfcomp**  
    Python package with BayesFlow wrappers for CNF training, prior definitions, and routines for generating comparison figures.
  - **epmodels**  
    Contains the stochastic compartmental models used in experiments:
    - `sir` — classic Susceptible–Infected–Recovered (SIR) model
    - `seir2v` — Susceptible–Exposed–Infected–Recovered (SEIR) model with two variants
  - **notebooks**  
    Jupyter notebooks demonstrating experiments for both models:
    - `SIR_model` — notebooks for the SIR model
    - `SEIR_model` — notebooks for the SEIR‑2V model
  - **pf_utils**  
    Julia utilities for particle filtering experiments.
- **data:** contains the synthetically generated data for the different models.
- **figures:** contains figures presented in the paper.
- **output:** contains result from running both methods and ues for the creation of the figures.
- **petab:** contains the PEtab problem for the two variant model from (Gudina et al. 2021)

---

## CNF Training and Inference

For retraining the neural network (full pipeline), the central entry point is:

```python
def train_validate_infer(
    config,
    prior_dict,
    folder,
    id,
    model="seir2v",
    observationtype="normal",
    simsize=100000,
    epochs=100,
    batch_size=32,
    validation_sims=400,
    summary_dim=14,
):
    """Train CNF model, validate performance, and run inference."""
```

This function handles simulation, training, validation, and inference in one pipeline.

For inference using pre‑saved checkpoints, use:

```python
def load_validate_infer(
    config,
    prior_dict,
    folder,
    id,
    model="seir2v",
    observationtype="normal",
    summary_dim=14,
):
    """Load trained CNF model from checkpoint and run inference."""
```

## PF Inference
For inference using the pseudo-marginal MH algorithm, the scripts
`ModelFilterRun.jl` need to be run. The scripts support parallelization across independent runs, if several workers are available. Hyperparameters must be changed within the scripts before running.

Evaluation of the results in the jupyter notebooks, directly uses the result files, saved by the scripts.

**References:**

- Gudina EK, Ali S, Girma E, et al (2021) Seroepidemiology and model-based prediction of SARS CoV 2 in Ethiopia: longitudinal cohort study among front-line hospital workers and communities. The Lancet Global Health 9(11):e1517–e1527. https://doi.org/10.1016/S2214-109X(21)00386-7

