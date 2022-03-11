import numpy as np
import bagpipes as pipes
import glob
from astropy.io import fits
import sys

# Point these to the locations of the photometry catalogues (v1.0) and
# a directory containing the best spectra files on your computer.
phot_path = "/Users/adam/work/data/vandels/catalogues"
spec_path = "/Users/adam/work/data/vandels/dr4"


def load_vandels(ID):
    """ Top-level function for loading vandels spectra + photometry. """
    spectrum = load_vandels_spec(ID)
    photometry = load_vandels_phot(ID)

    return spectrum, photometry


def load_vandels_spec(ID):
    """ Load DR2 public release versions of vandels spectrum files. """

    # if the ID is short, assume they want objects from the SELECT cats
    if len(ID) < 12:
        ID = old_id_to_new(ID)

    field, num, catalogue = parse_ids(ID)

    if "UDS" in ID:
        field = "UDS"

    if "CDFS" in ID:
        field = "CDFS"

    fname = glob.glob(spec_path + "/sc_" + field + num + "*.fits")[0]
    hdulist = fits.open(fname)
    print(ID, hdulist[0].header["PND Z"], hdulist[0].header["PND ZFLAGS"])


    maxwav = hdulist[0].header["CRVAL1"] + hdulist[0].header["CDELT1"]*hdulist[0].data.shape[0]
    wavs = np.arange(hdulist[0].header["CRVAL1"], maxwav, hdulist[0].header["CDELT1"])

    spectrum = np.c_[wavs,
                     hdulist[0].data,
                     hdulist[3].data]

    mask = (spectrum[:,0] < 9250.) & (spectrum[:,0] > 5200.)

    spectrum = spectrum[mask,:]

    spectrum = bin(spectrum, 2)

    # Get rid of any points with zeros in the error spectrum (hack)
    spectrum[(spectrum[:,2] == 0.), 1] = 0.
    spectrum[(spectrum[:,2] == 0.), 2] = 9.9999*10**99

    return spectrum


def load_vandels_spec_nobin(ID):
    """ Load DR2 public release versions of vandels spectrum files. """

    # if the ID is short, assume they want objects from the SELECT cats
    if len(ID) < 12:
        ID = old_id_to_new(ID)

    field, num, catalogue = parse_ids(ID)

    if "UDS" in ID:
        field = "UDS"

    if "CDFS" in ID:
        field = "CDFS"

    fname = glob.glob(spec_path + "/sc_" + field + num + "*.fits")[0]
    hdulist = fits.open(fname)

    maxwav = hdulist[0].header["CRVAL1"] + hdulist[0].header["CDELT1"]*hdulist[0].data.shape[0]
    wavs = np.arange(hdulist[0].header["CRVAL1"], maxwav, hdulist[0].header["CDELT1"])

    spectrum = np.c_[wavs,
                     hdulist[0].data,
                     hdulist[3].data]

    mask = (spectrum[:,0] < 9250.) & (spectrum[:,0] > 5200.)

    spectrum = spectrum[mask,:]

    # Get rid of any points with zeros in the error spectrum (hack)
    spectrum[(spectrum[:,2] == 0.), 1] = 0.
    spectrum[(spectrum[:,2] == 0.), 2] = 9.9999*10**99

    return spectrum


def load_vandels_spec_full(ID):
    """ Load DR2 public release versions of vandels spectrum files. """

    # if the ID is short, assume they want objects from the SELECT cats
    if len(ID) < 12:
        ID = old_id_to_new(ID)

    field, num, catalogue = parse_ids(ID)

    if "UDS" in ID:
        field = "UDS"

    if "CDFS" in ID:
        field = "CDFS"

    fname = glob.glob(spec_path + "/sc_" + field + num + "*.fits")[0]
    hdulist = fits.open(fname)

    maxwav = hdulist[0].header["CRVAL1"] + hdulist[0].header["CDELT1"]*hdulist[0].data.shape[0]
    wavs = np.arange(hdulist[0].header["CRVAL1"], maxwav, hdulist[0].header["CDELT1"])

    spectrum = np.c_[wavs,
                     hdulist[0].data,
                     hdulist[3].data]

    # Get rid of any points with zeros in the error spectrum (hack)
    spectrum[(spectrum[:,2] == 0.), 1] = 0.
    spectrum[(spectrum[:,2] == 0.), 2] = 9.9999*10**99
    #spectrum = bin(spectrum, 2)
    return spectrum


def load_vandels_phot(ID):
    """ Load vandels photometry from the v1.0 catalogues. """

    # if the ID is short, assume they want objects from the SELECT cats
    if len(ID) < 12:
        ID = old_id_to_new(ID)

    if "HST" in ID:
        photometry = load_vandels_hst_phot(ID)

    if "GROUND" in ID:
        photometry = load_vandels_ground_phot(ID)

    # Limit SNR to 20 sigma in each band
    for i in range(len(photometry)):
        if np.abs(photometry[i,0]/photometry[i,1]) > 20.:
            photometry[i,1] = np.abs(photometry[i,0]/20.)

    # Limit SNR of IRAC1 and IRAC2 channels to 10 sigma.
    for i in range(1,3):
        if np.abs(photometry[-i,0]/photometry[-i,1]) > 10.:
            photometry[-i,1] = np.abs(photometry[-i,0]/10.)

    # blow up the errors associated with any N/A points in the photometry
    for i in range(len(photometry)):
        if photometry[i,0] == 0. or photometry[i,1] <= 0:
            photometry[i,0] = 0.
            photometry[i,1] = 9.9*10**99.

    return photometry


def load_vandels_ground_phot(ID):
    """ Load vandels photometry from the v1.0 GROUND catalogues. """

    field, num, catalogue = parse_ids(ID)

    if "UDS" in ID:
        field = "UDS"
        offsets = np.loadtxt("../filters/offsets_uds_ground.txt")

    if "CDFS" in ID:
        field = "CDFS"
        offsets = np.loadtxt("../filters/offsets_cdfs_ground.txt")


    hdulist = fits.open(phot_path + "/VANDELS_" + field
                    + "_GROUND_PHOT_v1.0.fits")

    num_mask = np.isin(hdulist[1].data["ID"], np.array([num]).astype(int))

    if num_mask.sum() == 0:
        sys.exit("Object not found in catalogue")

    elif num_mask.sum() == 1:
        tablerow = num_mask.argmax()

    elif num_mask.sum() == 2:
        cat_mask = (hdulist[1].data["CAT"] == catalogue)
        combined_mask = cat_mask & num_mask
        tablerow = combined_mask.argmax()

    fluxes = []
    fluxerrs = []

    for name in hdulist[1].columns.names:
        if "2as" in name or "tphot" in name:
            if not "err" in name:
                fluxes.append(hdulist[1].data[name][tablerow])

                if "2as" in name:
                    fluxes[-1] *= hdulist[1].data["isoFactor"][tablerow]

            else:
                fluxerrs.append(hdulist[1].data[name][tablerow])

                if "2as" in name:
                    fluxerrs[-1] *= hdulist[1].data["isoFactor"][tablerow]

    photometry = np.zeros((len(fluxes),2))
    photometry[:,0] = fluxes
    photometry[:,1] = fluxerrs

    photometry[:,0] *= offsets

    return photometry


def load_vandels_hst_phot(ID):
    """ Load vandels photometry from the v1.0 HST catalogues. """

    field, num, catalogue = parse_ids(ID)

    if "UDS" in ID:
        field = "UDS"
        offsets = np.loadtxt("../filters/offsets_uds_hst.txt")

    if "CDFS" in ID:
        field = "CDFS"
        offsets = np.loadtxt("../filters/offsets_cdfs_hst.txt")

    hdulist = fits.open(phot_path + "/VANDELS_" + field
                    + "_HST_PHOT_v1.0.fits")

    num_mask = np.isin(hdulist[1].data["ID"], np.array([num]).astype(int))

    if num_mask.sum() == 0:
        sys.exit("Object not found in catalogue")

    elif num_mask.sum() == 1:
        tablerow = num_mask.argmax()

    elif num_mask.sum() == 2:
        cat_mask = (hdulist[1].data["CAT"] == catalogue)
        combined_mask = cat_mask & num_mask
        tablerow = combined_mask.argmax()

    row = np.array(hdulist[1].data[tablerow][5:])

    photometry = np.zeros((int(row.shape[0]/2), 2))
    photometry[:,0] = [row[i] for i in np.arange(0, row.shape[0], 2)]
    photometry[:,1] = [row[i] for i in np.arange(1, row.shape[0], 2)]

    photometry[:,0] *= offsets

    return photometry


def bin(spectrum, binn):
    """ Bins up a two or three column spectrum by a given factor. """

    binn = int(binn)
    nbins = int(len(spectrum)/binn)
    binspec = np.zeros((nbins, spectrum.shape[1]))
    for i in range(binspec.shape[0]):
        binspec[i, 0] = np.mean(spectrum[i*binn:(i+1)*binn, 0])
        binspec[i, 1] = np.mean(spectrum[i*binn:(i+1)*binn, 1])
        if spectrum.shape[1] == 3:
            sq_sum = np.sum(spectrum[i*binn:(i+1)*binn, 2]**2)
            binspec[i,2] = (1./float(binn))*np.sqrt(sq_sum)

    return binspec


def make_ids(field, num, catalogue):
    """ Turns a list of fields, id nums and catalogues into unique
    vandels IDs for use with the v1.0 photometry catalogues. """

    if np.array(field).ndim == 0:
        num = str(num)
        while len(num) < 6:
            num = "0" + num

        return field + num + catalogue

    else:
        IDs = []

        for i in range(len(field)):

            while len(str(num[i])) < 6:
                num[i] = "0" + str(num[i])

            IDs.append(field[i] + num[i] + catalogue[i])

        return IDs


def parse_ids(IDs):
    """ Turns unique vandels IDs for use with the v1.0 photometry
    catalogues into a list of fields, id nums and catalogues. """

    field = []
    num = []
    catalogue = []

    if not isinstance(IDs, (list, np.ndarray)):
        IDs = [IDs]

    for ID in IDs:
        catalogue.append(ID[-6:])
        ID = ID[:-6]
        num.append(ID[-6:])
        ID = ID[:-6]
        field.append(ID)

    if len(IDs) == 1:
        return field[0], num[0], catalogue[0]

    else:
        return field, num, catalogue


def old_id_to_new(ID):
    """ Turns an ID for a vandels spectrum of the form CDFS_000001 into
    a unique vandels ID for use with the v1.0 photometry catalogues. """

    if "UDS_0" in ID:
        field = "UDS-HST"

    elif "UDS_" in ID:
        field = "UDS-GROUND"

    elif "CDFS_0" in ID:
        field = "CDFS-HST"

    elif "CDFS_" in ID:
        field = "CDFS-GROUND"

    return field + ID[-6:] + "SELECT"


IDs = ["UDS-GROUND198829SELECT"]

for ID in IDs:
    galaxy = pipes.galaxy(ID, load_vandels_spec_full, photometry_exists=False)

    galaxy.spectrum[:, 0] /= (1. + 1.1474)

    galaxy.plot()
