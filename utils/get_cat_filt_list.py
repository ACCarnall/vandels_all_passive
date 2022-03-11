import numpy as np


def get_cat_filt_list(IDs):
    # Set up the filt_lists for the four sets of photometry.
    fields = ["UDS_HST", "UDS_GROUND", "CDFS_HST", "CDFS_GROUND"]
    filt_list_paths = ["../filters/" + f for f in fields]
    filt_list_paths = [f + "_filt_list.txt" for f in filt_list_paths]
    filt_lists = [np.loadtxt(f, dtype=str) for f in filt_list_paths]
    filt_list_dict = dict(zip(fields, filt_lists))

    # Create list of filt_lists for the catalogue of objects to be fitted.
    cat_filt_list = []

    for i in range(len(IDs)):

        if IDs[i].startswith("UDS-GROUND"):
                cat_filt_list.append(filt_list_dict["UDS_GROUND"])

        elif IDs[i].startswith("UDS-HST"):
                cat_filt_list.append(filt_list_dict["UDS_HST"])

        elif IDs[i].startswith("CDFS-GROUND"):
                cat_filt_list.append(filt_list_dict["CDFS_GROUND"])

        elif IDs[i].startswith("CDFS-HST"):
                cat_filt_list.append(filt_list_dict["CDFS_HST"])

    return cat_filt_list
