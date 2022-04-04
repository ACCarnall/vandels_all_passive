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

    ymax = ax.get_ylim()[1]
    ax.set_ylim(-2.1*noise_max, ymax)
    ax.axhline(0., color="gray", zorder=1, lw=1.)
    ax.axhline(-1.05*noise_max, color="gray", zorder=1, lw=1., ls="--")

    plt.savefig("pipes/plots/" + fit.run + "/" + fit.galaxy.ID + "_gp.pdf",
                bbox_inches="tight")

    plt.close()
