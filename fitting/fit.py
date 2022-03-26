import numpy as np
import bagpipes as pipes
import sys

from astropy.table import Table

sys.path.append("../utils")

from load_vandels import *
from get_cat_filt_list import *
from plot_gp_noise import *


def get_fit_instructions_v14():
    """ Set up the desired fit_instructions dictionary. """

    dust = {}
    dust["type"] = "Salim"
    dust["eta"] = 2.
    dust["Av"] = (0., 4.)
    dust["delta"] = (-0.3, 0.3)
    dust["delta_prior"] = "Gaussian"
    dust["delta_prior_mu"] = 0.
    dust["delta_prior_sigma"] = 0.1
    dust["B"] = (0., 5.)

    nebular = {}
    nebular["logU"] = -3.

    zmet_factor = (0.02/0.014)

    dblplaw = {}
    dblplaw["massformed"] = (0., 13.)
    dblplaw["metallicity"] = (0.2/zmet_factor, 2.5/zmet_factor)
    dblplaw["metallicity_prior"] = "log_10"
    dblplaw["alpha"] = (0.01, 1000.)
    dblplaw["alpha_prior"] = "log_10"
    dblplaw["beta"] = (0.01, 1000.)
    dblplaw["beta_prior"] = "log_10"
    dblplaw["tau"] = (0.1, 15.)

    calib = {}
    calib["type"] = "polynomial_bayesian"

    calib["0"] = (0.75, 1.25)
    calib["0_prior"] = "Gaussian"
    calib["0_prior_mu"] = 1.
    calib["0_prior_sigma"] = 0.1

    calib["1"] = (-0.25, 0.25)
    calib["1_prior"] = "Gaussian"
    calib["1_prior_mu"] = 0.
    calib["1_prior_sigma"] = 0.1

    calib["2"] = (-0.25, 0.25)
    calib["2_prior"] = "Gaussian"
    calib["2_prior_mu"] = 0.
    calib["2_prior_sigma"] = 0.1

    noise = {}
    noise["type"] = "GP_exp_squared"
    noise["scaling"] = (0.1, 10.)
    noise["scaling_prior"] = "log_10"
    noise["norm"] = (0.0001, 0.1)
    noise["norm_prior"] = "log_10"
    noise["length"] = (0.01, 1.)
    noise["length_prior"] = "log_10"

    fit_instructions = {}
    fit_instructions["dust"] = dust
    fit_instructions["dblplaw"] = dblplaw
    fit_instructions["nebular"] = nebular
    fit_instructions["t_bc"] = 0.01
    fit_instructions["redshift"] = (0., 10.)

    fit_instructions["veldisp"] = (100., 500.)
    fit_instructions["veldisp_prior"] = "log_10"

    fit_instructions["calib"] = calib
    fit_instructions["noise"] = noise

    return fit_instructions


cat = Table.read("../catalogues/vandels_passive_master_dr4_z.fits").to_pandas()
cat.index = cat["ID"].str.decode("utf8").str.strip().values

cat = cat.groupby(cat["Z_VANDELS_DR4_FLAG"] >= 3).get_group(True)
cat = cat.groupby(cat["Z_VANDELS_DR4"] >= 0.95).get_group(True)

IDs = cat.index.values
redshifts = cat["Z_VANDELS_DR4"].values

cat_filt_list = get_cat_filt_list(IDs)
fit_instructions = get_fit_instructions_v14()

fit_cat = pipes.fit_catalogue(IDs, fit_instructions, load_vandels, run="v14",
                              cat_filt_list=cat_filt_list, vary_filt_list=True,
                              redshifts=redshifts, redshift_sigma=0.005,
                              make_plots=True, full_catalogue=True,
                              n_posterior=1000, analysis_function=analysis_func)

fit_cat.fit(n_live=1000, verbose=True, mpi_serial=False, track_backlog=False)
