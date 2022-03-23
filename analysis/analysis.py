"""
HI/HA images.

The purpose of this code is to analyze HI/HA images.
We compare controls and patients with HIHA
We do this by looking at glucest images as a proxy for glutamate levels and brain
activation/funtion
We are primarily interested in the hippocampal areas, but also look at distribution
of glucest values in other areas
Primary metrics: glucest means and medians in hippocampus. Assymetry index,
hippocamal volumes, and distributions of glucest values across the brain.
"""
# pylint: disable=import-error, unused-import
# %% Imports
import copy
import os
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
import nibabel.processing
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import datasets, linear_model
import analysis.helper as hp
import analysis.constants as const
import scipy.ndimage


# %% CEST spreadsheet analysis
# =============================================================================
# Recreation of plots given by data from rosenfeld (the HIHA_cest.csv file)
# We look at the peak cest values (the greater cest value between the left or
# right hippocampus)
# =============================================================================
df = pd.read_csv(const.PATH_HIHA_CEST + "HIHA_cest.csv")

fig, axes = hp.plot_make(r=1, c=1)
sns.violinplot(data=df, x="HIHA", y="peak_cest", ax=axes)
sns.swarmplot(
    data=df, x="HIHA", y="peak_cest", color="white", edgecolor="gray", ax=axes
)


# %% Other metrics (like assymmetry index)

# Difference in left and right cest
df["cest_diff"] = abs((df["hippo_avg_cest_left"] - df["hippo_avg_cest_right"]) /
                      (df["hippo_avg_cest_left"] + df["hippo_avg_cest_right"]))

fig, axes = hp.plot_make(r=1, c=1)
sns.violinplot(data=df, x="HIHA", y="cest_diff", ax=axes)
sns.swarmplot(data=df, x="HIHA", y="cest_diff", color="white", edgecolor="gray", ax=axes)


# %% Imaging Analsysis
# =============================================================================
# Analysis of images
# =============================================================================
FILES = copy.deepcopy(const.FILES)

FILES["T2_reslice"] = "none"
FILES["hipp_reslice"] = "none"

data = pd.DataFrame(columns=["sub", "group", "cest_total_mean", "cest_total_median",
                             "cest_total_std", "cest_total_pixels", "cest_total_volume",
                             "cest_hipp_mean", "cest_hipp_median", "cest_hipp_std",
                             "cest_hipp_pixels", "cest_hipp_volume", "cest_left_mean",
                             "cest_left_median", "cest_left_std", "cest_left_pixels",
                             "cest_left_volume", "cest_right_mean", "cest_right_median",
                             "cest_right_std", "cest_right_pixels", "cest_right_volume",
                             "hipp_total_volume", "hipp_left_volume", "hipp_right_volume", "asymmetry_index"])
# %% Rescliec and plot
# =============================================================================
# Resclice images and plot
# Purpose: T2 and hippocampal images are not in same plane/dimensions as the
# 2D glucest, so we must re-sclice them using c3D
# c3D command: https://sourceforge.net/p/c3d/git/ci/master/tree/doc/c3d.md#-
# reslice-itk-resample-image-using-affine-transform
# =============================================================================


for i in range(len(FILES)):
    SUBJECT = i
    imaging_paths = hp.reslice_wrapper(
        FILES, SUBJECT, reslice=False)
    values = hp.plot_cest(imaging_paths, cutoff_lower=const.CUTOFF[0],
                          cutoff_upper=const.CUTOFF[1], title=FILES["sub"][SUBJECT])

    data = data.append(dict(sub=FILES["sub"][SUBJECT],
                       group=FILES["group"][SUBJECT]), ignore_index=True)
    data.iloc[i, 2:] = values

    # plt.savefig(f"plots/summary_plots/{FILES['sub'][SUBJECT]}.png", dpi=600)

for h in range(2, len(data.columns)):
    col = data.columns[h]
    data[col] = pd.to_numeric(data[col])

# %% Get peak cest
data["peak_cest"] = np.nan
for i in range(len(data)):
    peak_left = data.loc[i, "cest_left_mean"]
    peak_right = data.loc[i, "cest_right_mean"]
    if peak_left > peak_right:
        data.loc[i, "peak_cest"] = peak_left
    else:
        data.loc[i, "peak_cest"] = peak_right



# %% Detection of outliers
peak_cest = data.loc[:, "peak_cest"]
data["peak_cest_zscore"] = np.nan

peak_cest_control = data.loc[data['group'] == "control", "peak_cest"]
peak_cest_control_mean = np.mean(peak_cest_control)
peak_cest_control_sd = np.std(peak_cest_control)
data["peak_cest_zscore"] = (peak_cest-peak_cest_control_mean)/peak_cest_control_sd
# %% Save data

data_drop = data.drop(["cest_total_volume",  "cest_hipp_volume",  "cest_hipp_volume", "cest_left_volume", "cest_right_volume"], axis = 1)
data_drop.to_csv("analysis/CEST_measurements.csv")

# %% Statistics

statistics =  pd.DataFrame(columns=["measurement", "control_mean", "control_sd", "HIHA_mean", "HIHA_sd", "pvalue"])

for s in range(2, 23):
    MEASURE = data_drop.columns[s]
    print(f"{s}, {MEASURE}")
    v1 = data.loc[data['group'] == "control", MEASURE]
    v2 = data.loc[data['group'] == "patient", MEASURE]
    control_mean = np.nanmean(v1)
    control_sd = np.nanstd(v1)
    HIHA_mean = np.nanmean(v2)
    HIHA_sd = np.nanstd(v2)
    pvalue = st.mannwhitneyu(v1, v2)[1]
    pvalue = st.ttest_ind(v1, v2)[1]


    statistics = statistics.append(dict(measurement=MEASURE, control_mean = control_mean,
                       control_sd = control_sd, HIHA_mean = HIHA_mean, HIHA_sd = HIHA_sd,
                       pvalue= pvalue), ignore_index=True)

statistics.to_csv("analysis/CEST_statistics_table.csv")
# %%Plot comparisons of subjects and controls
# =============================================================================
# Plot comparisons of subjects and controls
# =============================================================================

MEASURE = "asymmetry_index"
TITLE = MEASURE

fig, axes = hp.plot_make(r=1, c=1)
sns.violinplot(data=data, x="group", y=MEASURE, ax=axes)
sns.swarmplot(data=data, x="group", y=MEASURE, color="black", edgecolor="gray", ax=axes)

v1 = data.loc[data['group'] == "control", MEASURE]
v2 = data.loc[data['group'] == "patient", MEASURE]

pvalue = st.mannwhitneyu(v1, v2)[1]
print(pvalue)

axes.set_title(f"{TITLE}, p-val: {np.round(pvalue, 4)}", fontsize=7)

# plt.savefig(f"plots/group_plots/{MEASURE}.png", dpi=600)


# %% Figure 3 plots of comparisons

def plot_group_comparison(measure, data, ylim=None, size_length=10.5, size_height=6,
                          line_width_axes=6, ms=12, capthick=5, elinewidth=6,
                          capsize=17, stripplot_size=10, line_width_boxplot=5,
                          boxplot_width=0.4, err_bar_position_1=0.35, err_bar_position_2=1.35):
    """
    Single plots of comparing subjects and controls.

    Parameters
    ----------
    measure : string
        the measurement you want to plot.
    title : string
        plot title.
    data : dataframe
        data from ode blocks above which amass all the data together into a df.

    Returns
    -------
    plot.

    """
    fig, axes = hp.plot_make(r=1, c=1, size_length=size_length, size_height=size_height)
    sns.boxplot(data=data, x="group", y=measure, whis=np.inf, width=boxplot_width,
                linewidth=line_width_boxplot,
                palette=[const.COLOR_CONTROLS, const.COLOR_SUBJCTS],
                boxprops=dict(edgecolor="black"), medianprops=dict(color="black"),
                whiskerprops=dict(color="black"), capprops=dict(color="black"))
    sns.stripplot(data=data, x="group", y=measure, color="black", linewidth=0, ax=axes,
                  size=stripplot_size)
    if ylim is not None:
        axes.set_ylim(ylim)

    v1 = data.loc[data['group'] == "control", measure]
    v2 = data.loc[data['group'] == "patient", measure]
    mean_v1, st_error_v1 = np.mean(v1), st.sem(v1)
    mean_v2, st_error_v2 = np.mean(v2), st.sem(v2)

    h_1 = st_error_v1 * st.t.ppf((1 + 0.95) / 2.0, len(v1)-1)
    h_2 = st_error_v2 * st.t.ppf((1 + 0.95) / 2.0, len(v2)-1)

    axes.errorbar(err_bar_position_1, mean_v1, yerr=h_1, fmt='s', ms=ms,
                  color=const.COLOR_CONTROLS_DARK, ecolor=const.COLOR_CONTROLS,
                  capthick=capthick, elinewidth=elinewidth, capsize=capsize)
    axes.errorbar(err_bar_position_2, mean_v2, yerr=h_2, fmt='s', ms=ms,
                  color=const.COLOR_SUBJCTS_DARK, ecolor=const.COLOR_SUBJCTS,
                  capthick=capthick, elinewidth=elinewidth, capsize=capsize)

    control_95ci = st.t.interval(alpha=0.95, df=len(v1)-1, loc=np.mean(v1), scale=st.sem(v1))
    patient_95ci = st.t.interval(alpha=0.95, df=len(v1)-1, loc=np.mean(v2), scale=st.sem(v2))
    pvalue = st.mannwhitneyu(v1, v2)[1]
    pvalue = st.ttest_ind(v1, v2)[1]
    print(pvalue)

    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_linewidth(line_width_axes)
    axes.spines['left'].set_linewidth(line_width_axes)
    axes.tick_params(width=line_width_axes, length=8)

    axes.set_title(f"p-val: {np.round(pvalue, 8)}", fontsize=7)


# %%
data.columns
MEASURE = "asymmetry_index"
plot_group_comparison("asymmetry_index", data,  size_length=7, size_height=8.5,
                      line_width_axes=6, ms=12, capthick=5, elinewidth=6,
                      capsize=17, stripplot_size=10, line_width_boxplot=5,
                      boxplot_width=0.4, err_bar_position_1=0.35, err_bar_position_2=1.35)
plt.savefig("plots/figure_03/asymmetry_index.pdf", dpi=600)


plot_group_comparison("cest_hipp_pixels", data,  size_length=3, size_height=3,
                      line_width_axes=6, ms=5, capthick=2, elinewidth=2,
                      capsize=8, stripplot_size=4, line_width_boxplot=1,
                      boxplot_width=0.4, err_bar_position_1=0.35, err_bar_position_2=1.35)
plt.savefig("plots/figure_03/cest_hipp_pixels.pdf", dpi=600)


plot_group_comparison("peak_cest", data, ylim=[7.5, 13], size_length=7, size_height=8.5,
                      line_width_axes=6, ms=12, capthick=5, elinewidth=6,
                      capsize=17, stripplot_size=10, line_width_boxplot=5,
                      boxplot_width=0.4, err_bar_position_1=0.35, err_bar_position_2=1.35)

plt.savefig("plots/figure_03/peak_cest.pdf", dpi=600)
# plot_group_comparison("hipp_total_volume", data,  size_length=3.5, size_height=2)
# %% plot relationships between variables (confounding variables and associations?)

fig, axes = hp.plot_make(r=1, c=1,  size_length=2, size_height=2.5)
sns.regplot(x="cest_hipp_pixels", y="cest_hipp_std", data=data,
            ax=axes, color="#444444", line_kws=dict(lw=5), scatter_kws=dict(edgecolor="none"))
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.spines['bottom'].set_linewidth(6)
axes.spines['left'].set_linewidth(6)
axes.tick_params(width=6, length=8)
plt.savefig("plots/figure_03/hipp_pixels_vs_std.pdf", dpi=600)


fig, axes = hp.plot_make(r=1, c=1,  size_length=2, size_height=2.5)
sns.regplot(x="cest_hipp_pixels", y="peak_cest", data=data,
            ax=axes, color="#444444", line_kws=dict(lw=5), scatter_kws=dict(edgecolor="none"))
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.spines['bottom'].set_linewidth(6)
axes.spines['left'].set_linewidth(6)
axes.tick_params(width=6, length=8)
plt.savefig("plots/figure_03/hipp_pixels_vs_peak.pdf", dpi=600)


fig, axes = hp.plot_make(r=1, c=1,  size_length=1.4, size_height=2.5)
sns.residplot(x="cest_hipp_pixels", y="cest_hipp_std", data=data, ax=axes,
              color="#444444", line_kws=dict(lw=5), scatter_kws=dict(edgecolor="none"))
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.spines['bottom'].set_linewidth(6)
axes.spines['left'].set_linewidth(6)
axes.tick_params(width=6, length=8)
plt.savefig("plots/figure_03/hipp_pixels_vs_std_resid.pdf", dpi=600)


fig, axes = hp.plot_make(r=1, c=1,  size_length=1.4, size_height=2.5)
sns.residplot(x="cest_hipp_pixels", y="peak_cest", data=data, ax=axes,
              color="#444444", line_kws=dict(lw=5), scatter_kws=dict(edgecolor="none"))
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.spines['bottom'].set_linewidth(6)
axes.spines['left'].set_linewidth(6)
axes.tick_params(width=6, length=8)
plt.savefig("plots/figure_03/hipp_pixels_vs_peak_resid.pdf", dpi=600)

# =============================================================================
# Plots of L/R hipp
# =============================================================================


fig, axes = hp.plot_make(r=1, c=1)
sns.residplot(x="cest_left_pixels", y="cest_left_std", data=data, ax=axes,
              scatter_kws={"s": 80})
sns.residplot(x="cest_right_pixels", y="cest_right_std", data=data, ax=axes,
              scatter_kws={"s": 80})
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)

fig, axes = hp.plot_make(r=1, c=1)
sns.residplot(x="cest_hipp_pixels", y="peak_cest", data=data, ax=axes,
              scatter_kws={"s": 80})
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)


fig, axes = hp.plot_make(r=1, c=1)
sns.regplot(x="cest_left_pixels", y="cest_left_std", data=data, ax=axes)
sns.regplot(x="cest_right_pixels", y="cest_right_std", data=data, ax=axes)
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)


# =============================================================================
# Trying out linear regression models
# =============================================================================


X = np.array(data.loc[:, ["cest_left_pixels"]])
y = np.array(data.loc[:, ["cest_left_std"]]).reshape((len(data)))


model = sm.OLS(y, sm.add_constant(X))
model_fit = model.fit()
params = model_fit.params
print(model_fit.summary())

########  Trying random stuff out with linear regression
pred_ols = model_fit.get_prediction()
iv_l = pred_ols.summary_frame()["obs_ci_lower"]
iv_u = pred_ols.summary_frame()["obs_ci_upper"]

sns.scatterplot(x="cest_hipp_volume", y="cest_hipp_std", data=data)
sns.lineplot(x=X.flatten(), y=model_fit.fittedvalues)
sns.lineplot(x=X.flatten(), y=iv_u, ls="--")
sns.lineplot(x=X.flatten(), y=iv_l, ls="--")

model_fit

sns.scatterplot(x=X.flatten(), y=model_fit.resid)

model_fit.get_prediction()
model_fit.fittedvalues
model_fit.resid + model_fit.fittedvalues
y


pred_ols

tmp_df = pd.DataFrame(columns=["group", "resid"])
tmp_df["group"] = data.loc[:, "group"]
tmp_df["resid"] = model_fit.resid

fig, axes = hp.plot_make(r=1, c=1)
sns.violinplot(data=tmp_df, x="group", y="resid", ax=axes)
sns.swarmplot(data=tmp_df, x="group", y="resid", ax=axes, color="black", edgecolor="gray")
v1 = tmp_df.loc[tmp_df['group'] == "control", "resid"]
v2 = tmp_df.loc[tmp_df['group'] == "patient", "resid"]

st.ttest_ind(v1, v2)

pvalue = st.mannwhitneyu(v1, v2)[1]

# %% Make C006 plot for manuscript

# =============================================================================
# Making example plot for figure of manuscript using C006's data
# =============================================================================
i = 2
SUBJECT = i
imaging_paths = hp.reslice_wrapper(FILES, SUBJECT, reslice=False)

values = hp.plot_cest(imaging_paths, cutoff_lower=const.CUTOFF[0],
                      cutoff_upper=const.CUTOFF[1], title=FILES["sub"][SUBJECT])

cutoff_lower = const.CUTOFF[0]
cutoff_upper = const.CUTOFF[1]

cest_path = imaging_paths[0]
t2_resclice_path = imaging_paths[1]
hipp_resclice_path = imaging_paths[2]
hipp_path = imaging_paths[3]

cest_img = nib.load(cest_path)
t2_img = nib.load(t2_resclice_path)
hipp_img = nib.load(hipp_resclice_path)
hipp_img_3d = nib.load(hipp_path)

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

fig, axes = hp.plot_make(r=1, c=1, size_length=8, size_height=8)
sns.heatmap(t2, square=True, ax=axes, cbar=False,
            xticklabels=False, yticklabels=False, cmap="Greys_r")
sns.heatmap(cest_nan, square=True, ax=axes, cbar=False, cmap="magma", center=8,
            xticklabels=False, yticklabels=False, vmin=cutoff_lower, vmax=cutoff_upper)

# plt.savefig("plots/C006/C006.png", dpi=600)


fig, axes = hp.plot_make(r=1, c=1, size_length=2, size_height=2)
sns.heatmap(t2, square=True, ax=axes, cbar=False,
            xticklabels=False, yticklabels=False, cmap="Greys_r")
sns.heatmap(cest_nan, square=True, ax=axes, cbar=False, cmap="magma", center=8,
            xticklabels=False, yticklabels=False, mask=~(np.logical_or(hipp == 1, hipp == 2)),
            vmin=cutoff_lower, vmax=cutoff_upper)

cb_ax = fig.add_axes([0.91, 0.2, 0.02, 0.6])
sns.heatmap(cest_nan, ax=axes, cmap="magma", center=8,
            cbar=True, vmin=cutoff_lower, vmax=cutoff_upper,
            cbar_ax=cb_ax, square=True, xticklabels=False, yticklabels=False)

cb_ax.set_ylabel('CEST value', rotation=90)
# plt.savefig("plots/C006/C006_cest_hipp.pdf", dpi=600)


fig, axes = hp.plot_make(r=2, c=1, size_length=5, size_height=5)
BINWIDTH = 0.25
COLOR_ALL = "#777777"
COLOR_LEFT = "#5252d6"
COLOR_RIGHT = "#d69452"
LINE_WIDTH = 5
mean = np.nanmean(cest_nan[np.logical_or(hipp == 1, hipp == 2)])
sd = np.nanstd(cest_nan[np.logical_or(hipp == 1, hipp == 2)])
axes[0].axvline(mean, lw=LINE_WIDTH, color="#000000")
axes[0].axvline(mean + sd, lw=LINE_WIDTH, color="#000000", ls=":")
axes[0].axvline(mean - sd, lw=LINE_WIDTH, color="#000000", ls=":")
sns.histplot(cest_nan[hipp == 1], ax=axes[0], linewidth=0, binrange=[
             0, cutoff_upper], binwidth=BINWIDTH, color=COLOR_LEFT, kde=True)
sns.histplot(cest_nan[hipp == 2], ax=axes[0], linewidth=0, binrange=[
             0, cutoff_upper], binwidth=BINWIDTH, color=COLOR_RIGHT, kde=True)

axes[0].set_xlim([4, 14])
axes[1].set_xlim([4, 14])
# axes[3].set_yscale('log')
sns.ecdfplot(cest_nan[hipp == 1], ax=axes[1], color=COLOR_LEFT)
sns.ecdfplot(cest_nan[hipp == 2], ax=axes[1], color=COLOR_RIGHT)

axes[0].tick_params(axis='both', which='major', labelsize=7)
axes[1].tick_params(axis='both', which='major', labelsize=7)

axes[1].set_xlabel('CEST Value', fontsize=7)
axes[1].set_ylabel('Number of Pixels', fontsize=7)
# axes[4].set_xlabel('CEST Value', fontsize=7)
axes[1].set_ylabel('Proportion of Pixels', fontsize=7, labelpad=2)
axes[1].set_title('ECDF Plot', fontsize=7)
axes[0].set_title('Distribution of CEST values', fontsize=7)

axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

patch_left = mpatches.Patch(color='#26269f', label='Left')
patch_right = mpatches.Patch(color='#9f6326', label='Right')
axes[0].legend(handles=[patch_left, patch_right],
               frameon=False, loc=1, prop={'size': 6})

# plt.savefig("plots/C006/C006_cest_dist.pdf", dpi=600)
# %% individual plots for manuscript
# =============================================================================
# Make individual plots for manuscript
# =============================================================================

i = 18
SUBJECT = i
name = FILES["sub"][SUBJECT]
imaging_paths = hp.reslice_wrapper(FILES, SUBJECT, reslice=False)

values = hp.plot_cest(imaging_paths, cutoff_lower=const.CUTOFF[0],
                      cutoff_upper=const.CUTOFF[1], title=FILES["sub"][SUBJECT])

cest_path = imaging_paths[0]
t2_resclice_path = imaging_paths[1]
hipp_resclice_path = imaging_paths[2]
hipp_path = imaging_paths[3]

cest_img = nib.load(cest_path)
t2_img = nib.load(t2_resclice_path)
hipp_img = nib.load(hipp_resclice_path)

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

fig, axes = hp.plot_make(r=1, c=1, size_length=8, size_height=8)
sns.heatmap(t2, square=True, ax=axes, cbar=False,
            xticklabels=False, yticklabels=False, cmap="Greys_r")
sns.heatmap(cest_nan, square=True, ax=axes, cbar=False, cmap="magma", center=8,
            xticklabels=False, yticklabels=False, vmin=cutoff_lower, vmax=cutoff_upper)

# plt.savefig(f"plots/single_plots/{name}_whole.png", dpi=600, bbox_inches='tight')


fig, axes = hp.plot_make(r=1, c=1, size_length=8, size_height=8)
sns.heatmap(t2, square=True, ax=axes, cbar=False,
            xticklabels=False, yticklabels=False, cmap="Greys_r")
sns.heatmap(cest_nan, square=True, ax=axes, cbar=False, cmap="magma", center=8,
            xticklabels=False, yticklabels=False, mask=~(np.logical_or(hipp == 1, hipp == 2)),
            vmin=cutoff_lower, vmax=cutoff_upper)

# plt.savefig(f"plots/single_plots/{name}_hipp.png", dpi=600, bbox_inches='tight')


fig, axes = hp.plot_make(r=1, c=2, size_length=10, size_height=7.3)
LINE_WIDTH = 6
BINWIDTH = 0.25
COLOR_ALL = "#777777"
COLOR_LEFT = "#5252d6"
COLOR_RIGHT = "#d69452"

mean = np.nanmean(cest_nan[np.logical_or(hipp == 1, hipp == 2)])
sd = np.nanstd(cest_nan[np.logical_or(hipp == 1, hipp == 2)])
axes[0].axvline(mean, lw=LINE_WIDTH, color="#000000")
axes[0].axvline(mean + sd, lw=LINE_WIDTH, color="#000000", ls=":")
axes[0].axvline(mean - sd, lw=LINE_WIDTH, color="#000000", ls=":")

sns.histplot(cest_nan[hipp == 1], ax=axes[0], linewidth=0, binrange=[0, cutoff_upper],
             binwidth=BINWIDTH, color=COLOR_LEFT, kde=True, line_kws=dict(lw=LINE_WIDTH))
sns.histplot(cest_nan[hipp == 2], ax=axes[0], linewidth=0, binrange=[0, cutoff_upper],
             binwidth=BINWIDTH, color=COLOR_RIGHT, kde=True, line_kws=dict(lw=LINE_WIDTH))

axes[0].set_xlim([0, 20])
axes[1].set_xlim([0, 20])
axes[0].set_ylim([0, 60])
# axes[3].set_yscale('log')
sns.ecdfplot(cest_nan[hipp == 1], ax=axes[1], color=COLOR_LEFT, lw=LINE_WIDTH)
sns.ecdfplot(cest_nan[hipp == 2], ax=axes[1], color=COLOR_RIGHT, lw=LINE_WIDTH)

axes[0].tick_params(axis='both', which='major', labelsize=7)
axes[1].tick_params(axis='both', which='major', labelsize=7)

axes[0].axes.xaxis.set_ticklabels([])
axes[0].axes.yaxis.set_ticklabels([])
axes[1].axes.xaxis.set_ticklabels([])
axes[1].axes.yaxis.set_ticklabels([])

# axes[1].set_xlabel('CEST Value', fontsize=7)
axes[1].set_ylabel('', fontsize=7)
# axes[4].set_xlabel('CEST Value', fontsize=7)
axes[0].set_ylabel('', fontsize=7, labelpad=2)
# axes[1].set_title('ECDF Plot', fontsize=7)
# axes[0].set_title('Distribution of CEST values', fontsize=7)

axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# patch_left = mpatches.Patch(color='#26269f', label='Left')
# patch_right = mpatches.Patch(color='#9f6326', label='Right')
# axes[0].legend(handles=[patch_left, patch_right],
#               frameon=False, loc=1, prop={'size': 6})
for a in range(2):
    axes[a].tick_params(width=4)
    for axis in ['top', 'bottom', 'left', 'right']:
        axes[a].spines[axis].set_linewidth(LINE_WIDTH)
# plt.savefig(f"plots/single_plots/{name}_plots.pdf", dpi=600)


# %% Plot comparisons of controls and subjects

# Detection of outliers
peak_cest = data.loc[:, "peak_cest"]
peak_cest_mean = np.mean(peak_cest)
peak_cest_sd = np.std(peak_cest)
(peak_cest-peak_cest_mean)/peak_cest_sd

peak_cest_control = data.loc[data['group'] == "control", "peak_cest"]
peak_cest_control_mean = np.mean(peak_cest_control)
peak_cest_control_sd = np.std(peak_cest_control)
(peak_cest-peak_cest_control_mean)/peak_cest_control_sd


# %%Plots for summary figure
# i = 16 for CHOP 10, and i = 8 for control 13, i =7 for C 11, i = 12 for CHOP 5
i = 12
SUBJECT = i
imaging_paths = hp.reslice_wrapper(FILES, SUBJECT, reslice=False)

# values = hp.plot_cest(imaging_paths, cutoff_lower=const.CUTOFF[0],
#                      cutoff_upper=const.CUTOFF[1], title=FILES["sub"][SUBJECT])

cutoff_lower = const.CUTOFF[0]
cutoff_upper = const.CUTOFF[1]

cest_path = imaging_paths[0]
t2_resclice_path = imaging_paths[1]
hipp_resclice_path = imaging_paths[2]
hipp_path = imaging_paths[3]

cest_img = nib.load(cest_path)
t2_img = nib.load(t2_resclice_path)
hipp_img = nib.load(hipp_resclice_path)
hipp_img_3d = nib.load(hipp_path)

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


cest = scipy.ndimage.gaussian_filter(cest, sigma=1)
# removing data that falls out of range (artifacts) and converting them to Nans
cest_nan = copy.deepcopy(cest)
cest_nan[cest < cutoff_lower] = np.nan
cest_nan[cest > cutoff_upper] = np.nan

#fig, axes = hp.plot_make(r=1, c=1, size_length=8, size_height=8)
# sns.heatmap(t2, square=True, ax=axes, cbar=False,
#            xticklabels=False, yticklabels=False, cmap="Greys_r")
# sns.heatmap(cest_nan, square=True, ax=axes, cbar=False, cmap="magma", center=8,
#            xticklabels=False, yticklabels=False, vmin=cutoff_lower, vmax=cutoff_upper)


if i == 8:
    x_1, x_2, y_1, y_2, t2_cent = 59, 150, 86, 137, 200  # for control 13
if i == 16:
    x_1, x_2, y_1, y_2, t2_cent = 59, 149, 98, 154, 550  # for CHOP 10
if i == 7:
    x_1, x_2, y_1, y_2, t2_cent = 64, 158, 115, 159, 150  # for t2_cent 11
if i == 16:
    x_1, x_2, y_1, y_2, t2_cent = 57, 151, 98, 154, 550  # for CHOP 10
if i == 7:
    x_1, x_2, y_1, y_2, t2_cent = 63, 157, 108, 164, 150  # for Control 11
if i == 12:
    x_1, x_2, y_1, y_2, t2_cent = 56, 150, 92, 148, 550  # for CHOP 5

VMIN = 2
VMAX = 15
CENTER = 8
fig, axes = hp.plot_make(r=1, c=1, size_length=2, size_height=2)
sns.heatmap(t2[x_1:x_2, y_1:y_2], square=True, ax=axes, cbar=False,
            xticklabels=False, yticklabels=False, cmap="Greys_r", center=t2_cent)
sns.heatmap(cest_nan[x_1:x_2, y_1:y_2], square=True, ax=axes, cbar=False, cmap="magma",
            center=CENTER, xticklabels=False, yticklabels=False,
            mask=~(np.logical_or(hipp[x_1:x_2, y_1:y_2] == 1, hipp[x_1:x_2, y_1:y_2] == 2)),
            vmin=VMIN, vmax=VMAX)

# plt.savefig(f"plots/summary_figure/{FILES['sub'][SUBJECT]}_cest.png", dpi=600, bbox_inches='tight')

cb_ax = fig.add_axes([0.91, 0.2, 0.02, 0.6])
sns.heatmap(cest_nan[x_1:x_2, y_1:y_2], ax=axes, cmap="magma", center=CENTER,
            cbar=True, vmin=VMIN, vmax=VMAX,
            cbar_ax=cb_ax, square=True, xticklabels=False, yticklabels=False,
            mask=~(np.logical_or(hipp[x_1:x_2, y_1:y_2] == 1, hipp[x_1:x_2, y_1:y_2] == 2)))



plt.savefig(
    f"plots/summary_figure/{FILES['sub'][SUBJECT]}_cest.pdf", dpi=600, bbox_inches='tight')

# plt.savefig("plots/C006/C006_cest_hipp.pdf", dpi=600)
np.nanmin(cest_nan)
np.nanmax(cest_nan)
# %%
if i == 16:
    COLOR = const.COLOR_SUBJCTS
if i == 7:
    COLOR = const.COLOR_CONTROLS
cest_nan_hipp = cest_nan[np.logical_or(hipp == 1, hipp == 2)]

fig, axes = hp.plot_make(r=1, c=1, size_length=3, size_height=1.5)
BINWIDTH = 1
COLOR_ALL = "#777777"
COLOR_LEFT = "#5252d6"
COLOR_RIGHT = "#d69452"
LINE_WIDTH = 5
mean = np.nanmean(cest_nan_hipp)
sd = np.nanstd(cest_nan_hipp)
axes.axvline(mean, lw=LINE_WIDTH, color="#000000")
axes.axvline(mean + sd, lw=LINE_WIDTH, color="#000000", ls=":")
axes.axvline(mean - sd, lw=LINE_WIDTH, color="#000000", ls=":")
sns.histplot(cest_nan_hipp, ax=axes, linewidth=0, binrange=[
             0, cutoff_upper], binwidth=BINWIDTH, color=COLOR, kde=True,line_kws = dict(lw=5))


axes.set_xlim([0, 20])
# axes[3].set_yscale('log')

axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.spines['bottom'].set_linewidth(6)
axes.spines['left'].set_linewidth(6)
axes.tick_params(width=6, length=8)
plt.savefig(
    f"plots/summary_figure/{FILES['sub'][SUBJECT]}_cest_distribution.pdf", dpi=600, bbox_inches='tight')


