"""
Quick measurents.

Script to do some quick measurements of glucest. Edit of original file by
Alfredo for his "quick_measurments.py" file of CHOP 10 in that folder.
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import analysis.constants as const
import copy
# %% Get data

cest = nib.load(
    const.PATH_IMAGES + "CHOP_10/b0b1correctedcestmap_10162019_S10.nii"
).get_fdata()

seg = nib.load(const.PATH_IMAGES + "CHOP_10/manual_segmentation_3.nii.gz").get_fdata()

# %%

cest_nan = copy.deepcopy(cest)

cest_nan[cest < const.CUTOFF[0]] = np.nan
cest_nan[cest > const.CUTOFF[1]] = np.nan

sns.heatmap(cest_nan, square=True)
plt.show()

cest[cest < const.CUTOFF[0]] = const.CUTOFF[0]
cest[cest > const.CUTOFF[1]] = const.CUTOFF[1]
sns.heatmap(cest, square=True)

# %%
# left sided CEST
seg = seg[:, :, 0]
seg_left = (seg == 1).astype(int)
vol_left = np.sum(seg_left)
left = seg_left * cest
left_nan = seg_left * cest_nan

sns.heatmap(left, square=True)

# %%
# left[left < 1] = np.nan
cest_left = np.nanmean(left)
cest_left_nan = np.nanmean(cest_nan)

print(f"{cest_left}, {cest_left_nan}")
# %%
# right sided CEST
seg_right = (seg == 2).astype(int)
vol_right = np.sum(seg_right)

right = seg_right * cest
right_nan = seg_right * cest_nan

sns.heatmap(right, square=True)

# right[right < 1] = np.nan
cest_right = np.nanmean(right)
cest_right_nan = np.nanmean(right_nan)
print(f"{cest_right}, {cest_right_nan}")
# %%
# AI
ai_vol = np.abs(vol_left - vol_right) / (vol_left + vol_right)
ai_cest = np.abs(cest_left - cest_right) / (cest_left + cest_right)

# Measure the volume
vol_seg = nib.load(
    "/Users/allucas/Documents/research/CNT/hiha_glucest/CHOP_10/manual_segmentation_2.nii.gz"
).get_fdata()
vol_left = np.sum(vol_seg == 1)
vol_right = np.sum(vol_seg == 2)
ai_vol = np.abs(vol_left - vol_right) / (vol_left + vol_right)

print("CEST Left: ", cest_left)
print("\n")
print("CEST Right: ", cest_right)
print("\n")
print("CEST AI: ", ai_cest)
print("\n")
print("Vol Left: ", vol_left)
print("\n")
print("Vol right: ", vol_right)
print("\n")
print("Vol AI: ", ai_vol)
print("\n")
