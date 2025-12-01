# Inference-Methods-for-Stochastic-Compartmental-Models

This repository contains the code to reproduce and results used in the work "Assessment of Simulation-based Inference Methods for Stochastic Compartmental Models".


## Usage
The code sould be usable on any local machine where Python 3.10 and Julia 1.10 are installed. The methods have the following specific requirements.

### PF
- Manifest.toml
- Project.toml

Additionally, the Particles.jl and ParticlesDE.jl packages must be installed from ParticleFilters.jl (https://github.com/lcontento/ParticleFilters.jl).

### CNF
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

**References:**

- Gudina EK, Ali S, Girma E, et al (2021) Seroepidemiology and model-based prediction of SARS CoV 2 in Ethiopia: longitudinal cohort study among front-line hospital workers and communities. The Lancet Global Health 9(11):e1517–e1527. https://doi.org/10.1016/S2214-109X(21)00386-7

