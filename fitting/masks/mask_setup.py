import numpy as np
from astropy.table import Table

cat = Table.read("../../catalogues/vandels_passive_master_dr4_z.fits").to_pandas()
cat.index = cat["ID"].str.decode("utf8").str.strip().values

cat = cat.groupby(cat["Z_VANDELS_DR4_FLAG"] >= 3).get_group(True)
cat = cat.groupby(cat["Z_VANDELS_DR4"] >= 0.95).get_group(True)

IDs = cat.index.values
redshifts = cat["Z_VANDELS_DR4"]

for i in range(IDs.shape[0]):

    obj_mask = np.zeros((4,2))

    obj_mask[0,:] = [6860., 6920.]
    obj_mask[1,:] = [7150., 7340.]
    obj_mask[2,:] = [7575., 7725.]
    obj_mask[3,:] = [-25., 25.] + 3727.*(redshifts[i]+1)

    np.savetxt(IDs[i] + "_mask", obj_mask)
