#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is based on the MNE tutorials:
https://mne.tools/stable/auto_tutorials/index.html (accessed March 31, 2026)
"""

import gc
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt#from mne import concatenate_stcs
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.multitest import multipletests
import mne
from mne.minimum_norm import apply_inverse, make_inverse_operator
from mne.time_frequency import psd_array_welch
import pingouin as pg
mne.set_log_level('WARNING')
from nibabel.freesurfer import io as fsio
from scipy.stats import ttest_ind
import sys


########## need to specify ##########
# participant ID
subject = 11

# frequency range
f_min = 1
f_max = 4
#####################################

duration = 20
invmethod='dSPM'
lambda2=1./9. 

# data
subjectname = f"Dream{subject}"
epochfile = f"D11_Notch_BP_500Hz_autoICA_{duration}s_epo.fif"
noisecovfile1 = f"D{subject}_day1_noise_cov.fif"
noisecovfile2 = f"D{subject}_day2_noise_cov.fif"
noisecovfile3 = f"D{subject}_day3_noise_cov.fif"

# out names
figname = f"D{subject}_{f_min}-{f_max}Hz_{duration}sec_{invmethod}.png"
morph_diff = f"D{subject}_PSDdiff_{f_min}-{f_max}Hz_{duration}sec_{invmethod}"
morph_full = f"D{subject}_PSDfull_{f_min}-{f_max}Hz_{duration}sec_{invmethod}"
morph_less = f"D{subject}_PSDless__{f_min}-{f_max}Hz_{duration}sec_{invmethod}"


# make path for files
morph_savepath_diff = Path.home() / "Data" / "Dream" / "MNE" / "Morph" / morph_diff
morph_savepath_full = Path.home() / "Data" / "Dream" / "MNE" / "Morph" / morph_full
morph_savepath_less = Path.home() / "Data" / "Dream" / "MNE" / "Morph" / morph_less


subjects_dir = "/Applications/freesurfer/subjects/"
ffwd = f"Dream{subject}-fwd.fif" 
path_fwd = Path(subjects_dir) / f"Dream{subject}" / "bem" / ffwd


epochpath = Path.home() / "Data" / "Dream" / "MNE" / "Epoch" / epochfile
path_noise_cov1 = Path.home() / "Data" / "Dream" / "MNE" / "NoiseCov" / noisecovfile1
path_noise_cov2 = Path.home() / "Data" / "Dream" / "MNE" / "NoiseCov" / noisecovfile2
path_noise_cov3 = Path.home() / "Data" / "Dream" / "MNE" / "NoiseCov" / noisecovfile3
 
 
# read epochs and fwd files
epochs = mne.read_epochs(epochpath, preload=True)
epochs.pick_types(meg=True, eeg=False)

fwd = mne.read_forward_solution(path_fwd)
noise_cov_day1 = mne.read_cov(path_noise_cov1)
noise_cov_day2 = mne.read_cov(path_noise_cov2)
noise_cov_day3 = mne.read_cov(path_noise_cov3)
 

##### create inverse operator
if 'noisecovfile1' in locals():  
    inverse_operator_day1 = mne.minimum_norm.make_inverse_operator(
        epochs.info, fwd, noise_cov_day1, loose=0.2, depth=0.8, rank='info'
    )

if 'noisecovfile2' in locals():  
    inverse_operator_day2 = mne.minimum_norm.make_inverse_operator(
        epochs.info, fwd, noise_cov_day2, loose=0.2, depth=0.8, rank='info'
    )

if 'noisecovfile3' in locals():  
    inverse_operator_day3 = mne.minimum_norm.make_inverse_operator(
        epochs.info, fwd, noise_cov_day3, loose=0.2, depth=0.8, rank='info'
    )


##### calculate spectral power 
# filter epochs
ix1 = (
    (epochs.metadata["dream"] == 1) &
    (~epochs.metadata["whitedream"].isin([1, 0.5])) &
    (~epochs.metadata["unsure_dream"].isin([1, 0.5])) &
    (~epochs.metadata["wake_thinking"].isin([1, 0.5])) &
    (~epochs.metadata["wake_nothinking"].isin([1, 0.5])) &
    (~epochs.metadata["asleep_nodream"].isin([1, 0.5])))

ix2 = (
    (~epochs.metadata["dream"].isin([1, 0.5])) &  
    (~epochs.metadata["whitedream"].isin([1, 0.5])) &
    (~epochs.metadata["unsure_dream"].isin([1, 0.5])) &
    (~epochs.metadata["wake_thinking"].isin([1, 0.5])) &
    (~epochs.metadata["wake_nothinking"].isin([1, 0.5])) &
    (epochs.metadata["asleep_nodream"]==1))

epochs_full = epochs[ix1]
epochs_less = epochs[ix2]

##### frequency power analysis
sfreq = epochs.info['sfreq']

# Welch parameters
n_per_seg = int(2 * sfreq) # 2sec segments
n_overlap = int(n_per_seg / 2) # 50% overlap
    
# calculate stc_band for each epoch
psds_full = []
for i in range(len(epochs_full)):
    ep_data = epochs_full.get_data()[i]  # shape: (n_channels, n_times)
    epoch = mne.EvokedArray(ep_data, epochs_full.info, tmin=epochs_full.tmin)
    
    if   epochs_full.metadata.iloc[i]['ExpDay'] == 1:
        stc = apply_inverse(epoch, inverse_operator_day1, lambda2=lambda2, method=invmethod)
    elif epochs_full.metadata.iloc[i]['ExpDay'] == 2:
        stc = apply_inverse(epoch, inverse_operator_day2, lambda2=lambda2, method=invmethod)
    elif epochs_full.metadata.iloc[i]['ExpDay'] == 3:
        stc = apply_inverse(epoch, inverse_operator_day3, lambda2=lambda2, method=invmethod)
    else:
        raise ValueError(f"Unexpected ExpDay value: {epochs_full.metadata.ExpDay[i]}") 
                
    data = stc.data
    psd, freqs = psd_array_welch(data, sfreq=sfreq, fmin=f_min, fmax=f_max, 
                               n_fft=n_per_seg, n_overlap=n_overlap,
                               window='hamming', average='mean')  # psd shape: (n_sources, n_freqs)
    band_power = psd.mean(axis=1)  # shape: (n_sources,)
    psds_full.append(band_power)
    
X_full = np.array(psds_full)  # shape: (n_epochs_full, n_sources)


psds_less = []
for i in range(len(epochs_less)):
    ep_data = epochs_less.get_data()[i]  # shape: (n_channels, n_times)
    epoch = mne.EvokedArray(ep_data, epochs_less.info, tmin=epochs_less.tmin)
    
    if   epochs_less.metadata.iloc[i]['ExpDay'] == 1:
        stc = apply_inverse(epoch, inverse_operator_day1, lambda2=lambda2, method=invmethod)
    elif epochs_less.metadata.iloc[i]['ExpDay'] == 2:
        stc = apply_inverse(epoch, inverse_operator_day2, lambda2=lambda2, method=invmethod)
    elif epochs_less.metadata.iloc[i]['ExpDay'] == 3:
        stc = apply_inverse(epoch, inverse_operator_day3, lambda2=lambda2, method=invmethod)
    else:
        raise ValueError(f"Unexpected ExpDay value: {epochs_less.metadata.ExpDay[i]}") 
                
    data = stc.data
    psd, freqs = psd_array_welch(data, sfreq=sfreq, fmin=f_min, fmax=f_max, 
                              n_fft=n_per_seg, n_overlap=n_overlap,
                              window='hamming', average='mean')  # psd shape: (n_sources, n_freqs)  
    
    band_power = psd.mean(axis=1)  # shape: (n_sources,)
    psds_less.append(band_power)
    
X_less = np.array(psds_less)  # shape: (n_epochs_less, n_sources)

del band_power, psd, freqs
gc.collect()
 

##### ROI
# read labels of Desikan-Killiany atlas
labels = mne.read_labels_from_annot(f"Dream{subject}", parc='aparc', subjects_dir=subjects_dir, hemi='both')
    
label_dict = {lab.name: lab for lab in labels}

frontal_labels = ['superiorfrontal-lh','rostralmiddlefrontal-lh','caudalmiddlefrontal-lh','parsopercularis-lh','parstriangularis-lh','parsorbitalis-lh','lateralorbitofrontal-lh','medialorbitofrontal-lh','frontalpole-lh','precentral-lh','paracentral-lh',
                     'superiorfrontal-rh','rostralmiddlefrontal-rh','caudalmiddlefrontal-rh','parsopercularis-rh','parstriangularis-rh','parsorbitalis-rh','lateralorbitofrontal-rh','medialorbitofrontal-rh','frontalpole-rh','precentral-rh','paracentral-rh']


posterior_labels = ['postcentral-lh','supramarginal-lh','superiorparietal-lh','inferiorparietal-lh','precuneus-lh','lingual-lh','pericalcarine-lh','cuneus-lh','lateraloccipital-lh',
                 'postcentral-rh','supramarginal-rh','superiorparietal-rh','inferiorparietal-rh','precuneus-rh','lingual-rh','pericalcarine-rh','cuneus-rh','lateraloccipital-rh']


roi_dict = {"Frontal": frontal_labels, "Posterior": posterior_labels}


lh_vertno, rh_vertno = stc.vertices
stc_vertices = np.concatenate([lh_vertno, rh_vertno])

frontal_indices = []
posterior_indices = []

def label_to_stc_indices_from_stc_vertices(label, stc_vertices):
    label_verts_in_stc = np.intersect1d(label.vertices, stc_vertices)
    return np.where(np.isin(stc_vertices, label_verts_in_stc))[0]

for label in labels:
    if label.name in frontal_labels:
        idx = label_to_stc_indices_from_stc_vertices(label, stc_vertices)
        frontal_indices.extend(idx.tolist())
    elif label.name in posterior_labels:
        idx = label_to_stc_indices_from_stc_vertices(label, stc_vertices)
        posterior_indices.extend(idx.tolist())
        

# take unique indices
frontal_indices = np.unique(frontal_indices).astype(int)
posterior_indices = np.unique(posterior_indices).astype(int)


## mean power per ROI per epoch
# frontal
dreamful_frontal = X_full[:, frontal_indices].mean(axis=1)
dreamless_frontal = X_less[:, frontal_indices].mean(axis=1)

# posterior
dreamful_posterior = X_full[:, posterior_indices].mean(axis=1)
dreamless_posterior = X_less[:, posterior_indices].mean(axis=1)


# ttest with Bayse factor
alternative = "less" if f_min < 20 else "greater"
tt_frontal = pg.ttest(dreamful_frontal, dreamless_frontal, paired=False, alternative=alternative)
tt_posterior = pg.ttest(dreamful_posterior, dreamless_posterior, paired=False, alternative=alternative)


# correction of p-values based on the Holm's method
pvals = [tt_frontal["p-val"].iloc[0],tt_posterior["p-val"].iloc[0]]
reject, pvals_holm, _, _ = multipletests(pvals, alpha=0.05, method="holm")


#### Plot
def get_asterisk(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''
    
fig, ax = plt.subplots(figsize=(6, 5))

# jittered x position
x1 = np.random.normal(loc=0.9, scale=0.03, size=len(dreamful_frontal))
x2 = np.random.normal(loc=1.1, scale=0.03, size=len(dreamless_frontal))
x3 = np.random.normal(loc=1.5, scale=0.03, size=len(dreamful_posterior))
x4 = np.random.normal(loc=1.7, scale=0.03, size=len(dreamless_posterior))

ax.scatter(x1, dreamful_frontal, color='red', alpha=0.7, label='Experience', marker='o', facecolors='none')
ax.scatter(x2, dreamless_frontal, color='blue', alpha=0.7, label='No Experience', marker='o', facecolors='none')
ax.scatter(x3, dreamful_posterior, color='red', alpha=0.7, marker='o', facecolors='none')
ax.scatter(x4, dreamless_posterior, color='blue', alpha=0.7, marker='o', facecolors='none')


# add horizontal bars for mean values
ax.hlines(np.mean(dreamful_frontal), xmin=0.8, xmax=1.0, color='red', linewidth=2)
ax.hlines(np.mean(dreamless_frontal), xmin=1.0, xmax=1.2, color='blue', linewidth=2)
ax.hlines(np.mean(dreamful_posterior), xmin=1.4, xmax=1.6, color='red', linewidth=2)
ax.hlines(np.mean(dreamless_posterior), xmin=1.6, xmax=1.8, color='blue', linewidth=2)


y_min_frontal = min(np.min(dreamful_frontal), np.min(dreamless_frontal))
y_min_posterior = min(np.min(dreamful_posterior), np.min(dreamless_posterior))

y_min_all = min(y_min_frontal, y_min_posterior)

y_max_frontal = max(np.max(dreamful_frontal), np.max(dreamless_frontal))
y_max_posterior = max(np.max(dreamful_posterior), np.max(dreamless_posterior))


# to adjust position of asterisks
y_max_frontal = y_max_frontal+0.0001
y_max_posterior = y_max_posterior+0.0001
y_max_all = max(y_max_frontal, y_max_posterior)

yrange = y_max_all - y_min_all

pos_asterisk = yrange/100
pos_bracket = yrange/100

if pvals_holm[0] < 0.05:  # frontal
    ax.plot([0.9, 1.1], [y_max_frontal, y_max_frontal], color='black', linewidth=1.5)
    ax.plot([0.9, 0.9], [y_max_frontal, y_max_frontal - pos_bracket], color='black', linewidth=1.5)# left bracket
    ax.plot([1.1, 1.1], [y_max_frontal, y_max_frontal - pos_bracket], color='black', linewidth=1.5)# right bracket
    ax.text(1.0, y_max_frontal + pos_asterisk/10, get_asterisk(pvals_holm[0]), ha='center', va='bottom', fontsize=16)

if pvals_holm[1] < 0.05:  # parieto-occipital
    ax.plot([1.5, 1.7], [y_max_posterior, y_max_posterior], color='black', linewidth=1.5)
    ax.plot([1.5, 1.5], [y_max_posterior, y_max_posterior - pos_bracket], color='black', linewidth=1.5)# left bracket
    ax.plot([1.7, 1.7], [y_max_posterior, y_max_posterior - pos_bracket], color='black', linewidth=1.5)# right bracket
    ax.text(1.6, y_max_posterior + pos_asterisk/10, get_asterisk(pvals_holm[1]), ha='center', va='bottom', fontsize=16)
        
# set axis
ax.set_xticks([1.0, 1.6])
ax.set_xticklabels(['Frontal', 'Posterior'], fontsize=12)
ax.set_ylabel(f"Mean Source Power ({duration}sec) {invmethod}", fontsize=12)
ax.set_xlim(0.7, 1.9)

ax.set_ylim(y_min_all-yrange/20, y_max_all+yrange/10) 
ax.legend(['Experience', 'No experience'], loc='upper left')

plt.tight_layout()
plt.title(f"{f_min}-{f_max}Hz")
plt.show()
path_save = Path.home() / "Data" / "Dream" / "MNE" / "ROI" / figname
plt.savefig(path_save, dpi=300)


print(f"{figname}")
print()
print("Frontal")
print(tt_frontal.to_string())
print()
print("Posterior")
print(tt_posterior.to_string())
print()
print(pvals_holm)


#####　visualize on the brain
# vertex indeces for left and right hemispheres
X_vert_lh, X_vert_rh = stc.vertices
used_vertices = [X_vert_lh, X_vert_rh]

src = mne.read_source_spaces(f"{subjects_dir}/Dream{subject}/bem/Dream{subject}-oct-6-src.fif")
    
n_lh = len(src[0]['vertno'])
lh_vertno = src[0]['vertno']
rh_vertno = src[1]['vertno']
used_vertices = [lh_vertno, rh_vertno]
n_total = n_lh + len(rh_vertno)

# check if numbers of X_full_final are consistent
print(f"X_full.shape[1] = {X_full.shape[1]}, total vertices = {n_total}")
if X_full.shape[1] != n_total:
    used_vertices = [X_vert_lh, X_vert_rh]

# make source estimate of mean power for X_full and X_less
mean_full = X_full.mean(axis=0)
mean_less = X_less.mean(axis=0)

# make source estimate for power difference
diff_data = X_full.mean(axis=0) - X_less.mean(axis=0)  # shape: (n_sources,)

stc_full = mne.SourceEstimate(mean_full[:, np.newaxis], vertices=used_vertices,tmin=0, tstep=1, subject=f"Dream{subject}")
stc_less = mne.SourceEstimate(mean_less[:, np.newaxis], vertices=used_vertices,tmin=0, tstep=1, subject=f"Dream{subject}")

## visualize PSD differences
stc_diff = mne.SourceEstimate(diff_data[:, np.newaxis], vertices=used_vertices,tmin=0, tstep=1, subject=f"Dream{subject}")


# visualize PDS difference
brain = stc_diff.plot(
    subject=f"Dream{subject}",
    subjects_dir=subjects_dir,
    hemi='both',
    surface='inflated',
    smoothing_steps=15,
)

# brain = stc_diff.plot(
#     subject=f"Dream{subject}",
#     subjects_dir=subjects_dir,
#     hemi='rh',
#     surface='inflated',
#     smoothing_steps=15,
# )


# save data after morphing to the template brain
src_from = inverse_operator_day1["src"]
morph = mne.compute_source_morph(src=src_from, subject_from=stc_diff.subject, subject_to="fsaverage", subjects_dir=subjects_dir)


stc_diff_fsavg = morph.apply(stc_diff)
stc_full_fsavg = morph.apply(stc_full)
stc_less_fsavg = morph.apply(stc_less)

stc_diff_fsavg.save(morph_savepath_diff, overwrite=True)
stc_full_fsavg.save(morph_savepath_full, overwrite=True)
stc_less_fsavg.save(morph_savepath_less, overwrite=True)
