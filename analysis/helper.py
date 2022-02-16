"""
Helper utilities.

The code is to help with some anlyses.
"""
# pylint: disable=import-error, unused-import
# %% Imports
import sys
import os
import copy
import matplotlib
import numpy as np
import seaborn as sns
import nibabel as nib
import nibabel.processing
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
import analysis.constants as const

# %%
# =============================================================================
# Analysis functions
# =============================================================================


def reslice_glucest_project(path_c3d, reference_image, moving_image, output_name, print_bool=True):
    """
    Summary: Resclice MRI images.

    Purpose: to resclice T2 and segmentation images to be in the same plane
    and dimnsions as the glucest image for analysis

    Parameters
    ----------
    reference_image : TYPE
        DESCRIPTION.
    moving_image : TYPE
        DESCRIPTION.
    output_name : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    cmd = f"{path_c3d} {reference_image} {moving_image} -interpolation NearestNeighbor \
        -reslice-identity -o {output_name}"
    execute_command(cmd, print_bool=print_bool)


def reslice_wrapper(files, subject, reslice=True):
    """
    Purpose: wrapper function for reslice_glucest_project function above.

    Parameters
    ----------
    files : TYPE
        DESCRIPTION.
    subject : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    print(files["sub"][subject])

    # getting file names
    cest_path = files["cest"][subject]
    t2_path = files["T2"][subject]
    hipp_path = files["hipp"][subject]

    # making new folder to store the rescliced images
    reslice_path = os.path.dirname(cest_path) + "/resclice"
    check_path_and_make(reslice_path, reslice_path)

    # making new file names
    files["T2_reslice"][subject] = reslice_path + "/T2_to_cest.nii"
    files["hipp_reslice"][subject] = reslice_path + "/hipp_to_cest.nii"
    t2_resclice_path = files["T2_reslice"][subject]
    hipp_resclice_path = files["hipp_reslice"][subject]

    # Checking if cest imagee is 2D or 3D. If 2D, must add thir dimension
    img_cest = nib.load(cest_path)
    cest = img_cest.get_fdata()
    cest_3D = cest.reshape((cest.shape[0], cest.shape[1], 1))
    img_cest_３D = nib.Nifti1Image(cest_3D, img_cest.affine, img_cest.header)
    cest_path_3D_fname = os.path.splitext(os.path.basename(cest_path))[0] + "_3D.nii"
    cest_path_3D = os.path.join(reslice_path, cest_path_3D_fname)
    nib.save(img_cest_３D, cest_path_3D)

    if reslice:
        # resclice T2
        reslice_glucest_project(const.PATH_C3D, cest_path_3D, t2_path, t2_resclice_path)
        # resclice hipp
        reslice_glucest_project(const.PATH_C3D, cest_path_3D, hipp_path, hipp_resclice_path)
    return cest_path, t2_resclice_path, hipp_resclice_path, hipp_path, cest_path_3D


# %%
# =============================================================================
# Plotting functions
# =============================================================================


def plot_cest(imaging_paths, cutoff_lower, cutoff_upper, title=""):
    """
    Plot glucest with t2 and hippocampal segmentation.

    Returns
    -------
    None.

    """
    cest_path = imaging_paths[0]
    t2_resclice_path = imaging_paths[1]
    hipp_resclice_path = imaging_paths[2]
    hipp_path = imaging_paths[3]

    # loading mri data
    cest_img = nib.load(cest_path)
    t2_img = nib.load(t2_resclice_path)
    hipp_img = nib.load(hipp_resclice_path)
    hipp_img_3d = nib.load(hipp_path)

    # re-orienting images to RAS orientation
    # https://nipy.org/nibabel/image_orientation.html
    if len(cest_img.shape) > 2:
        cest_img = nib.as_closest_canonical(cest_img)
        t2_img = nib.as_closest_canonical(t2_img)
        hipp_img = nib.as_closest_canonical(hipp_img)
        hipp_img_3d = nib.as_closest_canonical(hipp_img_3d)

    # getting array data
    cest = cest_img.get_fdata()
    t2 = t2_img.get_fdata()
    hipp = hipp_img.get_fdata()
    hipp_3D = hipp_img_3d.get_fdata()

    # reshape to 2D for plotting
    cest = cest.reshape((cest.shape[0], cest.shape[1]))
    hipp = hipp.reshape((hipp.shape[0], hipp.shape[1]))
    t2 = t2.reshape((t2.shape[0], t2.shape[1]))

    # removing data that falls out of range (artifacts) and converting them to Nans
    cest_nan = copy.deepcopy(cest)
    cest_nan[cest < cutoff_lower] = np.nan
    cest_nan[cest > cutoff_upper] = np.nan

    # cest for plotting, so that nans don't mess up
    cest_plot = copy.deepcopy(cest)
    cest_plot[cest < cutoff_lower] = 0
    cest_plot[cest > cutoff_upper] = 0

    sx, sy, sz = t2_img.header.get_zooms()
    cest_voxel_volume = sx * sy * sz
    # calculating summary stats
    cest_total_mean = np.nanmean(cest_nan)
    cest_total_median = np.nanmedian(cest_nan)
    cest_total_std = np.nanstd(cest_nan)#/cest_total_mean
    cest_total_pixels = len(cest_nan[~np.isnan(cest_nan)])
    cest_total_volume = cest_total_pixels * cest_voxel_volume / 1000

    cest_hipp = cest_nan[np.logical_or(hipp == 1, hipp == 2)]
    cest_hipp_mean = np.nanmean(cest_hipp)
    cest_hipp_median = np.nanmedian(cest_hipp)
    cest_hipp_std = np.nanstd(cest_hipp)#/cest_hipp_mean
    cest_hipp_pixels = len(cest_hipp)
    cest_hipp_volume = cest_hipp_pixels * cest_voxel_volume / 1000

    cest_left = cest_nan[hipp == 1]
    cest_left_mean = np.nanmean(cest_left)
    cest_left_median = np.nanmedian(cest_left)
    cest_left_std = np.nanstd(cest_left)#/cest_left_mean
    cest_left_pixels = len(cest_left)
    cest_left_volume = cest_left_pixels * cest_voxel_volume / 1000

    cest_right = cest_nan[hipp == 2]
    cest_right_mean = np.nanmean(cest_right)
    cest_right_median = np.nanmedian(cest_right)
    cest_right_std = np.nanstd(cest_right)#/cest_right_mean
    cest_right_pixels = len(cest_right)
    cest_right_volume = cest_right_pixels * cest_voxel_volume / 1000


    #Calculate assymmetry
    # AI = [  [|L -R|] / [L+R]   ] * 100
    AI = np.abs(cest_left_mean - cest_right_mean) / (cest_left_mean + cest_right_mean) *100


    # Making 3d hippocampus image conform to 256x256x256 and 1x1x1mm so that we can
    # compare across patients for volume in mm rather than pixels
    sx, sy, sz = hipp_img_3d.header.get_zooms()
    voxel_volume = sx * sy * sz
    hipp_total_volume = np.sum(np.logical_or(hipp_3D == 1, hipp_3D == 2)
                               ) * voxel_volume / 1000  # in cm3
    hipp_left_volume = np.sum((hipp_3D == 1)) * voxel_volume / 1000  # in cm3
    hipp_right_volume = np.sum((hipp_3D == 2)) * voxel_volume / 1000  # in cm3

    #####
    # %%
    fig, axes = plot_make(r=2, c=3, size_length=8, size_height=4)
    axes = axes.flatten()
    sns.heatmap(t2, square=True, ax=axes[0], cbar=False,
                xticklabels=False, yticklabels=False, cmap="Greys_r")
    sns.heatmap(t2, square=True, ax=axes[1], cbar=False,
                xticklabels=False, yticklabels=False, cmap="Greys_r")
    sns.heatmap(t2, square=True, ax=axes[2], cbar=False,
                xticklabels=False, yticklabels=False, cmap="Greys_r")
    sns.heatmap(cest_nan, square=True, ax=axes[1], cbar=False, cmap="magma", center=8,
                xticklabels=False, yticklabels=False, vmin=cutoff_lower, vmax=cutoff_upper)
    sns.heatmap(cest_nan, square=True, ax=axes[2], cbar=False, cmap="magma", center=8,
                xticklabels=False, yticklabels=False,
                mask=~(np.logical_or(hipp == 1, hipp == 2)),
                vmin=cutoff_lower, vmax=cutoff_upper)

    # adding colorbar
    cb_ax = fig.add_axes([0.90, 0.54, 0.01, 0.34])
    sns.heatmap(cest_nan, ax=axes[1], cmap="magma", center=8,
                cbar=True, vmin=cutoff_lower, vmax=cutoff_upper,
                cbar_ax=cb_ax, square=True, xticklabels=False, yticklabels=False)

    cb_ax.set_ylabel('CEST value', rotation=90)

    # labeling Left and right
    left_side = np.where(hipp == 1)[0]
    np.where(hipp == 2)
    half = hipp.shape[0] / 2
    if left_side[0] > half:
        axes[2].axes.text(hipp.shape[1] * 0.02, hipp.shape[0] * 0.97, ha='left', fontsize=7,
                          s="Left", color="white")
        axes[2].axes.text(hipp.shape[1] * 0.02, hipp.shape[0] * 0.08, ha='left', fontsize=7,
                          s="Right", color="white")
    else:
        axes[2].axes.text(hipp.shape[1] * 0.02, hipp.shape[0] * 0.97, ha='left', fontsize=7,
                          s="Right", color="white")
        axes[2].axes.text(hipp.shape[1] * 0.02, hipp.shape[0] * 0.08, ha='left', fontsize=7,
                          s="Left", color="white")

    fig.suptitle(title)
    binwidth = 1
    color_all = "#777777"
    color_left = "#5252d6"
    color_right = "#d69452"
    sns.histplot(cest_nan.flatten(), ax=axes[3], binrange=[
                 0, cutoff_upper], binwidth=binwidth, color=color_all, kde=True)
    sns.histplot(cest_nan[hipp == 1], ax=axes[3], binrange=[
                 0, cutoff_upper], binwidth=binwidth, color=color_left, kde=True)
    sns.histplot(cest_nan[hipp == 2], ax=axes[3], binrange=[
                 0, cutoff_upper], binwidth=binwidth, color=color_right, kde=True)

    axes[3].set_yscale('log')
    sns.ecdfplot(cest_nan.flatten(), ax=axes[4], color=color_all)
    sns.ecdfplot(cest_nan[hipp == 1], ax=axes[4], color=color_left)
    sns.ecdfplot(cest_nan[hipp == 2], ax=axes[4], color=color_right)

    axes[3].tick_params(axis='both', which='major', labelsize=7)
    axes[4].tick_params(axis='both', which='major', labelsize=7)

    axes[3].set_xlabel('CEST Value', fontsize=7)
    axes[3].set_ylabel('Number of Pixels', fontsize=7)
    axes[4].set_xlabel('CEST Value', fontsize=7)
    axes[4].set_ylabel('Proportion of Pixels', fontsize=7, labelpad=2)
    axes[4].set_title('ECDF Plot', fontsize=7)
    axes[3].set_title('Distribution of CEST values', fontsize=7)
    axes[0].set_title('T2 image', fontsize=7)
    axes[1].set_title('CEST and T2 image', fontsize=7)
    axes[2].set_title('Hippocampus CEST and T2 image', fontsize=7)

    axes[3].spines['top'].set_visible(False)
    axes[3].spines['right'].set_visible(False)
    axes[4].spines['top'].set_visible(False)
    axes[4].spines['right'].set_visible(False)

    axes[5].set_xlim([0, 100])
    axes[5].set_ylim([0, 100])
    axes[5].axis('off')

    patch_all = mpatches.Patch(color='#333333', label='All pixels')
    patch_left = mpatches.Patch(color='#26269f', label='Left')
    patch_right = mpatches.Patch(color='#9f6326', label='Right')
    axes[3].legend(handles=[patch_all, patch_left, patch_right],
                   frameon=False, loc=1, prop={'size': 6})

    fs = 7
    axes[5].axes.text(-5, 90, ha='left', fontsize=fs,
                      s=f"CEST total \n      mean: {np.round(cest_total_mean,1)}, \
    median: {np.round(cest_total_median,1)}, \
    sd: {np.round(cest_total_std,1)}, \
    \n      pixels: {cest_total_pixels}")
    # volume: {np.round(cest_total_volume,1)} cm^3")
    axes[5].axes.text(-5, 65, ha='left', fontsize=fs,
                      s=f"CEST hipp \n      mean: {np.round(cest_hipp_mean,1)}, \
    median: {np.round(cest_hipp_median,1)}, \
    sd: {np.round(cest_hipp_std,1)}, \
    \n      pixels: {cest_hipp_pixels}")
    # volume: {np.round(cest_hipp_volume,1)} cm^3")
    axes[5].axes.text(-5, 40, ha='left', fontsize=fs,
                      s=f"CEST left \n      mean: {np.round(cest_left_mean,1)}, \
    median: {np.round(cest_left_median,1)}, \
    sd: {np.round(cest_left_std,1)}, \
    \n      pixels: {cest_left_pixels}")
    # volume: {np.round(cest_left_volume,1)} cm^3")
    axes[5].axes.text(-5, 15, ha='left', fontsize=fs,
                      s=f"CEST right \n      mean: {np.round(cest_right_mean,1)}, \
    median: {np.round(cest_right_median,1)}, \
    sd: {np.round(cest_right_std,1)}, \
    \n      pixels: {cest_right_pixels}")
    # volume: {np.round(cest_right_volume,1)} cm^3")
    axes[5].axes.text(-5, 0, ha='left', fontsize=fs,
                      s=f"Hippocampus 3D volume \n\
    total: {np.round(hipp_total_volume,1)}, \
    left: {np.round(hipp_left_volume,1)}, \
    right: {np.round(hipp_right_volume,1)} cm^3")

    return [cest_total_mean, cest_total_median, cest_total_std, cest_total_pixels, cest_total_volume,
            cest_hipp_mean, cest_hipp_median, cest_hipp_std, cest_hipp_pixels, cest_hipp_volume,
            cest_left_mean, cest_left_median, cest_left_std, cest_left_pixels, cest_left_volume,
            cest_right_mean, cest_right_median, cest_right_std, cest_right_pixels, cest_right_volume,
            hipp_total_volume, hipp_left_volume, hipp_right_volume, AI]

# %%


def plot_make(r=1, c=1, size_length=None, size_height=None, dpi=300,
              sharex=False, sharey=False, squeeze=True):
    """
    Purpose: To make a generlized plotting function.

    Parameters
    ----------
    r : TYPE, optional
        DESCRIPTION. The default is 1.
    c : TYPE, optional
        DESCRIPTION. The default is 1.
    size_length : TYPE, optional
        DESCRIPTION. The default is None.
    size_height : TYPE, optional
        DESCRIPTION. The default is None.
    dpi : TYPE, optional
        DESCRIPTION. The default is 300.
    sharex : TYPE, optional
        DESCRIPTION. The default is False.
    sharey : TYPE, optional
        DESCRIPTION. The default is False.
    squeeze : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    axes : TYPE
        DESCRIPTION.

    """
    if size_length is None:
        size_length = 4 * c
    if size_height is None:
        size_height = 4 * r
    fig, axes = plt.subplots(r, c, figsize=(size_length, size_height), dpi=dpi,
                             sharex=sharex, sharey=sharey, squeeze=squeeze)
    return fig, axes

# %%
# =============================================================================
# Basic utility functions
# =============================================================================


def execute_command(cmd, print_bool=True):
    """
    Summary: Execute a given command.

    Parameters
    ----------
    cmd : string
        terminal syntax command line.
    print_bool : Boolean, optional
        whetheere or not to print out the command given. The default is True.

    Returns
    -------
    None.

    """
    if print_bool:
        print(f"\n\nExecuting Command Line: \n{cmd}\n\n")
    os.system(cmd)


def check_path_and_make(path_to_check, path_to_make, make=True, print_bool=True, exist_ok=True):
    """
    Check if path_to_check exists.

    If so, then option to make a second directory path_to_make (may be same as path_to_check)

    Parameters
    ----------
    path_to_check : TYPE
        DESCRIPTION.
    path_to_make : TYPE
        DESCRIPTION.
    make : TYPE, optional
        DESCRIPTION. The default is True.
    print_bool : TYPE, optional
        DESCRIPTION. The default is True.
    exist_ok : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """
    if not os.path.exists(path_to_check):
        if print_bool:
            print(f"\nFile or Path does not exist:\n{path_to_check}")
    if make:
        if os.path.exists(path_to_make):
            if print_bool:
                print(f"Path already exists\n{path_to_make}")
        else:
            os.makedirs(path_to_make, exist_ok=exist_ok)
            if print_bool:
                print("Making Path")
