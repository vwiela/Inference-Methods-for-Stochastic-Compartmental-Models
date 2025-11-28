from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import bayesflow as bf
import bayesflow.diagnostics as diag
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.networks import InvertibleNetwork, SequenceNetwork
from bayesflow.trainers import Trainer

from keras.backend import clear_session

from pathlib import Path

import logging

from .utils import time_counter, setup_logger
from epmodels.variant_model import generative_model_seir2v
from epmodels.sir_model import generative_model_sir


def configurator(input_dict, prior_dict):
    data = input_dict["sim_data"]

    # Extract prior draws and z-standardize with previously computed means and transform
    params = input_dict["prior_draws"].astype(np.float32)
    params = prior_dict["transform"](params)
    params = (params - prior_dict["prior_means"]) / prior_dict["prior_stds"]

    out_dict = {
        "summary_conditions": data,
        "parameters": params,
    }

    return out_dict


def train_cnf(
    config,
    prior_dict,
    savepath,
    id,
    model="seir2v",
    observationtype="normal",
    simsize=100000,
    epochs=100,
    batch_size=32,
    validation_sims=400,
    summary_dim=14,
):

    summary_net = SequenceNetwork(summary_dim=summary_dim, lstm_units=64)

    inference_net = InvertibleNetwork(
        num_params=len(prior_dict["prior"].param_names),
        num_coupling_layers=8,
        coupling_design="spline",
    )
    amortizer = AmortizedPosterior(inference_net, summary_net, name="Amortizer_" + id)

    wrapped_configurator = partial(configurator, prior_dict=prior_dict)
    if model == "seir2v":
        wrapped_generative_model = partial(
            generative_model_seir2v,
            config=config,
            prior_dict=prior_dict,
            observationtype=observationtype,
        )
    elif model == "sir":
        wrapped_generative_model = partial(
            generative_model_sir,
            config=config,
            prior_dict=prior_dict,
            observationtype=observationtype,
        )

    trainer = Trainer(
        amortizer,
        wrapped_generative_model,
        configurator=wrapped_configurator,
        checkpoint_path=savepath + "checkpoints/",
        max_to_keep=1,
    )
    amortizer.summary()

    if model == "seir2v":
        offline_data = generative_model_seir2v(
            simsize, config, prior_dict, observationtype=observationtype
        )
    elif model == "sir":
        offline_data = generative_model_sir(
            simsize, config, prior_dict, observationtype=observationtype
        )
    history = trainer.train_offline(
        offline_data, epochs=epochs, batch_size=batch_size, validation_sims=validation_sims
    )

    f = diag.plot_losses(history["train_losses"], history["val_losses"], moving_average=True)
    outfile = savepath + "figures/loss"
    Path(outfile).parent.mkdir(exist_ok=True, parents=True)
    f.savefig(outfile + ".pdf")

    return trainer


def load_trained_cnf(
    config, prior_dict, savepath, id, model="seir2v", observationtype="binomial", summary_dim=14
):

    summary_net = SequenceNetwork(summary_dim=summary_dim, lstm_units=64)

    inference_net = InvertibleNetwork(
        num_params=len(prior_dict["prior"].param_names),
        num_coupling_layers=8,
        coupling_design="spline",
    )
    amortizer = AmortizedPosterior(inference_net, summary_net, name="Amortizer_" + id)

    wrapped_configurator = partial(configurator, prior_dict=prior_dict)
    if model == "seir2v":
        wrapped_generative_model = partial(
            generative_model_seir2v,
            config=config,
            prior_dict=prior_dict,
            observationtype=observationtype,
        )
    elif model == "sir":
        wrapped_generative_model = partial(
            generative_model_sir,
            config=config,
            prior_dict=prior_dict,
            observationtype=observationtype,
        )

    trainer = Trainer(
        amortizer,
        wrapped_generative_model,
        configurator=wrapped_configurator,
        checkpoint_path=savepath + "checkpoints/",
    )
    amortizer.summary()

    trainer.load_pretrained_network()

    return trainer


def validate_cnf(
    trainer, config, prior_dict, savepath, id, model="seir2v", observationtype="binomial"
):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info("Validating " + trainer.amortizer.name)
    # Generate validation data
    if model == "seir2v":
        validation_sims = trainer.configurator(
            generative_model_seir2v(2000, config, prior_dict, observationtype=observationtype)
        )
    elif model == "sir":
        validation_sims = trainer.configurator(
            generative_model_sir(2000, config, prior_dict, observationtype=observationtype)
        )
    # Extract unstandardized prior draws and transform to original scale
    prior_samples = (
        validation_sims["parameters"] * prior_dict["prior_stds"] + prior_dict["prior_means"]
    )
    prior_samples = prior_dict["transform_inv"](prior_samples)

    # Generate 100 posterior draws for each of the 2000 simulated data sets
    post_samples = trainer.amortizer.sample(validation_sims, n_samples=100)

    # Unstandardize posterior draws into original scale
    post_samples = post_samples * prior_dict["prior_stds"] + prior_dict["prior_means"]
    post_samples = prior_dict["transform_inv"](post_samples)

    # Create ECDF plot
    f = diag.plot_sbc_ecdf(post_samples, prior_samples, param_names=prior_dict["prior"].param_names)
    outfile = savepath + "figures/sbc_ecdf_1"
    Path(outfile).parent.mkdir(exist_ok=True, parents=True)
    f.savefig(outfile + ".pdf")

    f = diag.plot_sbc_ecdf(
        post_samples,
        prior_samples,
        stacked=True,
        difference=True,
        legend_fontsize=12,
        fig_size=(6, 5),
    )
    outfile = savepath + "figures/sbc_ecdf_2"
    f.savefig(outfile + ".pdf")

    f = diag.plot_sbc_histograms(
        post_samples, prior_samples, param_names=prior_dict["prior"].param_names
    )
    outfile = savepath + "figures/sbc_histogram"
    f.savefig(outfile + ".pdf")

    post_samples = trainer.amortizer.sample(validation_sims, n_samples=1000)
    post_samples = post_samples * prior_dict["prior_stds"] + prior_dict["prior_means"]
    post_samples = prior_dict["transform_inv"](post_samples)

    f = diag.plot_recovery(post_samples, prior_samples, param_names=prior_dict["prior"].param_names)
    outfile = savepath + "figures/recovery"
    f.savefig(outfile + ".pdf")

    f = diag.plot_z_score_contraction(
        post_samples, prior_samples, param_names=prior_dict["prior"].param_names
    )
    outfile = savepath + "figures/z_score_contraction"
    f.savefig(outfile + ".pdf")
    plt.close("all")


def infer_cnf(amortizer, data, prior_dict, savepath, id):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info("Infering dataset " + id + " via " + amortizer.name)
    # Obtain posterior draws given real data
    post_samples_obs = amortizer.sample({"summary_conditions": data}, 10000)

    # Undo standardization to get parameters on their original (unstandardized) scales
    post_samples_obs = post_samples_obs * prior_dict["prior_stds"] + prior_dict["prior_means"]
    post_samples_obs = prior_dict["transform_inv"](post_samples_obs)

    # Save post_samples to csv
    outfile = savepath + "posterior_samples.csv"
    Path(outfile).parent.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame(post_samples_obs)
    df.to_csv(outfile, index=False)

    # 2d plot of sample posterior draws vs prior
    f = diag.plot_posterior_2d(post_samples_obs, prior=prior_dict["prior"])
    outfile = savepath + "figures/posterior"
    Path(outfile).parent.mkdir(exist_ok=True, parents=True)
    f.savefig(outfile + ".pdf")
    plt.close("all")


def infer_cnf_nofig(amortizer, data, prior_dict, savepath, id):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info("Infering dataset " + id + " via " + amortizer.name)
    # Obtain posterior draws given real data
    post_samples_obs = amortizer.sample({"summary_conditions": data}, 10000)
    # Undo standardization to get parameters on their original (unstandardized) scales
    post_samples_obs = post_samples_obs * prior_dict["prior_stds"] + prior_dict["prior_means"]
    post_samples_obs = prior_dict["transform_inv"](post_samples_obs)

    # Save post_samples to csv
    outfile = savepath + "posterior_" + id + ".csv"
    Path(outfile).parent.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame(post_samples_obs)
    df.to_csv(outfile, index=False)


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
    Path(folder + "logfile.log").parent.mkdir(exist_ok=True, parents=True)
    logger = setup_logger(id, folder + "logfile.log")
    t1 = time_counter()
    trainer = train_cnf(
        config,
        prior_dict,
        folder,
        id,
        model=model,
        observationtype=observationtype,
        simsize=simsize,
        epochs=epochs,
        batch_size=batch_size,
        validation_sims=validation_sims,
        summary_dim=summary_dim,
    )
    t2 = time_counter()
    logger.info("Time for training " + trainer.amortizer.name + ": " + str(t2 - t1))
    t1 = time_counter()
    validate_cnf(
        trainer, config, prior_dict, folder, id, model=model, observationtype=observationtype
    )
    t2 = time_counter()
    logger.info("Time for validating " + trainer.amortizer.name + ": " + str(t2 - t1))
    t1 = time_counter()
    infer_cnf(trainer.amortizer, config["obs_data"], prior_dict, folder, id)
    t2 = time_counter()
    logger.info(
        "Time for infering "
        + config["dataname"]
        + " via "
        + trainer.amortizer.name
        + ": "
        + str(t2 - t1)
    )
    plt.close("all")
    clear_session()


def load_validate_infer(
    config,
    prior_dict,
    folder,
    id,
    model="seir2v",
    observationtype="normal",
    summary_dim=14,
):
    Path(folder + "logfile2.log").parent.mkdir(exist_ok=True, parents=True)
    logger = setup_logger(id, folder + "logfile2.log")
    t1 = time_counter()
    trainer = load_trained_cnf(
        config,
        prior_dict,
        folder,
        id,
        model=model,
        observationtype=observationtype,
        summary_dim=summary_dim,
    )
    t2 = time_counter()
    logger.info("Time for loading pre-trained net" + trainer.amortizer.name + ": " + str(t2 - t1))
    t1 = time_counter()
    validate_cnf(
        trainer, config, prior_dict, folder, id, model=model, observationtype=observationtype
    )
    t2 = time_counter()
    logger.info("Time for validating " + trainer.amortizer.name + ": " + str(t2 - t1))
    t1 = time_counter()
    infer_cnf(trainer.amortizer, config["obs_data"], prior_dict, folder, id)
    t2 = time_counter()
    logger.info(
        "Time for infering "
        + config["dataname"]
        + " via "
        + trainer.amortizer.name
        + ": "
        + str(t2 - t1)
    )
    plt.close("all")
    clear_session()


def train_validate_infer_amort(
    list_config,
    prior_dict,
    folder,
    id,
    model="seir2v",
    observationtype="binnormalomial",
    simsize=100000,
    epochs=100,
    batch_size=32,
    validation_sims=400,
    summary_dim=14,
):
    Path(folder + "logfile.log").parent.mkdir(exist_ok=True, parents=True)
    logger = setup_logger(id, folder + "logfile.log")
    t1 = time_counter()
    trainer = train_cnf(
        list_config[0],
        prior_dict,
        folder,
        id,
        model=model,
        observationtype=observationtype,
        simsize=simsize,
        epochs=epochs,
        batch_size=batch_size,
        validation_sims=validation_sims,
        summary_dim=summary_dim,
    )
    t2 = time_counter()
    logger.info("Time for training " + trainer.amortizer.name + ": " + str(t2 - t1))
    t1 = time_counter()
    validate_cnf(
        trainer, list_config[0], prior_dict, folder, model=model, observationtype=observationtype
    )
    t2 = time_counter()
    logger.info("Time for validating " + trainer.amortizer.name + ": " + str(t2 - t1))
    t1 = time_counter()
    infer_cnf(
        trainer.amortizer,
        list_config[0]["obs_data"],
        prior_dict,
        folder,
        list_config[0]["dataname"],
    )
    t2 = time_counter()
    logger.info(
        "Time for infering "
        + list_config[0]["dataname"]
        + " via "
        + trainer.amortizer.name
        + ": "
        + str(t2 - t1)
    )
    for i in range(1, len(list_config)):
        t1 = time_counter()
        infer_cnf(
            trainer.amortizer,
            list_config[i]["obs_data"],
            prior_dict,
            folder,
            list_config[i]["dataname"],
        )
        t2 = time_counter()
        logger.info(
            "Time for infering "
            + list_config[i]["dataname"]
            + " via "
            + trainer.amortizer.name
            + ": "
            + str(t2 - t1)
        )
    clear_session()
