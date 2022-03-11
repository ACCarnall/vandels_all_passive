import numpy as np
import bagpipes as pipes
import glob
import sys
import os
import matplotlib.pyplot as plt
from copy import deepcopy


def analysis_func(fit):
    import matplotlib.pyplot as plt
    fit.posterior.get_advanced_quantities()

    fig = plt.figure(figsize=(12, 5.))
    ax = plt.subplot()

    y_scale = pipes.plotting.add_spectrum(fit.galaxy.spectrum, ax)
    pipes.plotting.add_spectrum_posterior(fit, ax, y_scale=y_scale)

    noise_post = fit.posterior.samples["noise"]*10**-y_scale
    noise_perc = np.percentile(noise_post, (16, 50, 84), axis=0).T
    noise_max = np.max(np.abs(noise_perc))
    noise_perc -= 1.05*noise_max

    ax.plot(fit.galaxy.spectrum[:,0], noise_perc[:, 1], color="darkorange")

    ax.fill_between(fit.galaxy.spectrum[:,0], noise_perc[:, 0],
                    noise_perc[:, 2], color="navajowhite", alpha=0.7)

    ax.plot(fit.galaxy.spectrum[:,0],
            np.median(fit.posterior.samples["spectrum"], axis=0)*10**-y_scale,
            color="black", lw=0.5)

    pipes.plotting.add_observed_photometry_linear(fit.galaxy, ax, y_scale=y_scale)

    fit.posterior.samples["const_spectrum"] = np.zeros((fit.n_posterior, fit.galaxy.spectrum.shape[0]))

    for i in range(fit.n_posterior):
        fit.fitted_model._update_model_components(fit.posterior.samples2d[i, :])
        const_model_comp = deepcopy(fit.fitted_model.model_components)
        del const_model_comp["dblplaw"]
        #const_model_comp["dust"]["eta"] = 5.

        if i == 0:
            model_galaxy = pipes.model_galaxy(const_model_comp,
                                              filt_list=fit.galaxy.filt_list,
                                              spec_wavs=fit.galaxy.spec_wavs,
                                              index_list=fit.galaxy.index_list)

        else:
            model_galaxy.update(const_model_comp)

        fit.posterior.samples["const_spectrum"][i, :] = model_galaxy.spectrum[:, 1]

        #ax.plot(fit.galaxy.spectrum[:, 0], fit.posterior.samples["const_spectrum"][i, :]*10**-y_scale, lw=0.5, color="green", alpha=0.4)

    const_spec_post = fit.posterior.samples["const_spectrum"]*10**-y_scale#*10**2
    const_spec_perc = np.percentile(const_spec_post, (16, 50, 84), axis=0).T

    ax.fill_between(fit.galaxy.spectrum[:,0], const_spec_perc[:, 0],
                    const_spec_perc[:, 2], color="green", alpha=0.7)

    ymax = ax.get_ylim()[1]
    ax.set_ylim(-2.1*noise_max, ymax)
    ax.axhline(0., color="gray", zorder=1, lw=1.)
    ax.axhline(-1.05*noise_max, color="gray", zorder=1, lw=1., ls="--")

    plt.savefig("pipes/plots/" + fit.run + "/" + fit.galaxy.ID + "_gp.pdf",
                bbox_inches="tight")

    plt.close()
