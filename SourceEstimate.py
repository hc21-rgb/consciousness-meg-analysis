"""
Source reconstruction based on the 
based on the MNE-python tutorial <https://mne.tools/stable/auto_tutorials/inverse/30_mne_dspm_loreta.html> (last access: 2025/12/28)

"""


import gc
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt#from mne import concatenate_stcs
from statsmodels.stats.multitest import fdrcorrection
import mne
from mne.minimum_norm import apply_inverse, make_inverse_operator
from mne.time_frequency import psd_array_welch
import pingouin as pg
mne.set_log_level('WARNING')
from nibabel.freesurfer import io as fsio
from scipy.stats import ttest_ind
import sys


############# needs to be specified ##########

# participant/participant ID
subject = 11

# frequency range
f_min = 1
f_max = 4
##############################################

# analysis duration prior to awakening
duration = 20

# inverse method
invmethod='dSPM'

# signal to noise ratio
snr = 3.0
lambda2 = 1.0 / snr**2

# false discovery rate
fdr_alpha = 0.05

# threshold for log BF
threshold = np.log10(3) 


stage = 'N1N2' 
subjectname = f"Dream{subject}"
epochfile = f"D{subject}_Notch_BP_500Hz_autoICA_{duration}s_epo.fif"
noisecovfile1 = f"D{subject}_day1_noise_cov.fif"
noisecovfile2 = f"D{subject}_day2_noise_cov.fif"
noisecovfile3 = f"D{subject}_day3_noise_cov.fif"

# out names
figname = f"D{subject}_dream_{stage}_{f_min}-{f_max}Hz_{duration}sec_{invmethod}.png"
figname_bfhisto = f"D{subject}_dream_{stage}_{f_min}-{f_max}Hz_{duration}sec_{invmethod}_bfhisto_bf3.png"
morphname = f"D{subject}_dream_PSDdiff_{stage}_{f_min}-{f_max}Hz_{duration}sec_{invmethod}"
################

# make paths for files
morph_savepath = Path.home() / "Data" / "Dream" / "MNE" / "Morph" / morphname
subjects_dir = "/Applications/freesurfer/subjects/"
ffwd = f"Dream{subject}-fwd.fif" 
path_fwd = Path(subjects_dir) / f"Dream{subject}" / "bem" / ffwd
epochpath = Path.home() / "Data" / "Dream" / "MNE" / "Epoch" / epochfile
path_noise_cov1 = Path.home() / "Data" / "Dream" / "MNE" / "NoiseCov" / noisecovfile1
path_noise_cov2 = Path.home() / "Data" / "Dream" / "MNE" / "NoiseCov" / noisecovfile2
path_noise_cov3 = Path.home() / "Data" / "Dream" / "MNE" / "NoiseCov" / noisecovfile3
if 'noisecovfile4' in locals():    
 path_noise_cov4 = Path.home() / "Data" / "Dream" / "MNE" / "NoiseCov" / noisecovfile4
 
 
# read epochs and fwd files
epochs = mne.read_epochs(epochpath, preload=True)
epochs.pick_types(meg=True, eeg=False)

fwd = mne.read_forward_solution(path_fwd)
noise_cov_day1 = mne.read_cov(path_noise_cov1)
noise_cov_day2 = mne.read_cov(path_noise_cov2)
noise_cov_day3 = mne.read_cov(path_noise_cov3)
if 'noisecovfile4' in locals():    
 noise_cov_day4 = mne.read_cov(path_noise_cov4)
 
# create inverse operators
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

if 'noisecovfile4' in locals():  
    inverse_operator_day4 = mne.minimum_norm.make_inverse_operator(
        epochs.info, fwd, noise_cov_day4, loose=0.2, depth=0.8, rank='info'
    )


# filter epochs
ix1 = (
    (epochs.metadata["dream"] == 1) &
    (epochs.metadata["sleepstage"].isin(["N1", "N2"])) &
    (~epochs.metadata["whitedream"].isin([1, 0.5])) &
    (~epochs.metadata["unsure_dream"].isin([1, 0.5])) &
    (~epochs.metadata["wake_thinking"].isin([1, 0.5])) &
    (~epochs.metadata["wake_nothinking"].isin([1, 0.5])) &
    (~epochs.metadata["asleep_nodream"].isin([1, 0.5])))

ix2 = (
    (~epochs.metadata["dream"].isin([1, 0.5])) &  
    (epochs.metadata["sleepstage"].isin(["N1", "N2"])) &
    (~epochs.metadata["whitedream"].isin([1, 0.5])) &
    (~epochs.metadata["unsure_dream"].isin([1, 0.5])) &
    (~epochs.metadata["wake_thinking"].isin([1, 0.5])) &
    (~epochs.metadata["wake_nothinking"].isin([1, 0.5])) &
    (epochs.metadata["asleep_nodream"]==1))

epochs_full = epochs[ix1] # experience
epochs_less = epochs[ix2] # no-experience


### compute power spectral density

# sampling frequency
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
    elif epochs_full.metadata.iloc[i]['ExpDay'] == 4:
            stc = apply_inverse(epoch, inverse_operator_day4, lambda2=lambda2, method=invmethod)
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
    elif epochs_less.metadata.iloc[i]['ExpDay'] == 4:
            stc = apply_inverse(epoch, inverse_operator_day4, lambda2=lambda2, method=invmethod)
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
 

# exclude exp day which does not includes both full and less epochs. 
day_full = epochs_full.metadata.ExpDay
day_less = epochs_less.metadata.ExpDay
common_days = np.intersect1d(day_full.unique(), day_less.unique())

X_full_list = []
X_less_list = []
day_full_list = []
day_less_list = []

for day in common_days:
    idx_full = (day_full == day).values
    idx_less = (day_less == day).values

    Xf = X_full[idx_full]   # shape: (n_full_day, n_sources)
    Xl = X_less[idx_less]   # shape: (n_less_day, n_sources)

    n_full = Xf.shape[0]
    n_less = Xl.shape[0]

    full_day = [day] * n_full
    less_day = [day] * n_less

    X_full_list.append(Xf)
    X_less_list.append(Xl)
    day_full_list.extend(full_day)
    day_less_list.extend(less_day)

X_full_final = np.vstack(X_full_list)
X_less_final = np.vstack(X_less_list)


### ROI
# read labels of Desikan-Killiany atlas
labels = mne.read_labels_from_annot(f"Dream{subject}", parc='aparc', subjects_dir=subjects_dir, hemi='both')

label_dict = {lab.name: lab for lab in labels}

frontal_labels = ['superiorfrontal-lh','rostralmiddlefrontal-lh','caudalmiddlefrontal-lh','parsopercularis-lh','parstriangularis-lh','parsorbitalis-lh','lateralorbitofrontal-lh','medialorbitofrontal-lh','frontalpole-lh','precentral-lh','paracentral-lh',
                     'superiorfrontal-rh','rostralmiddlefrontal-rh','caudalmiddlefrontal-rh','parsopercularis-rh','parstriangularis-rh','parsorbitalis-rh','lateralorbitofrontal-rh','medialorbitofrontal-rh','frontalpole-rh','precentral-rh','paracentral-rh']


parocc_labels = ['postcentral-lh','supramarginal-lh','superiorparietal-lh','inferiorparietal-lh','precuneus-lh','lingual-lh','pericalcarine-lh','cuneus-lh','lateraloccipital-lh',
                 'postcentral-rh','supramarginal-rh','superiorparietal-rh','inferiorparietal-rh','precuneus-rh','lingual-rh','pericalcarine-rh','cuneus-rh','lateraloccipital-rh']


temporal_labels = ['entorhinal-lh','parahippocampal-lh','temporalpole-lh','fusiform-lh','superiortemporal-lh','middletemporal-lh','inferiortemporal-lh','transversetemporal-lh','bankssts-lh',
                   'entorhinal-rh','parahippocampal-rh','temporalpole-rh','fusiform-rh','superiortemporal-rh','middletemporal-rh','inferiortemporal-rh','transversetemporal-rh','bankssts-rh']


roi_dict = {"Frontal": frontal_labels, "ParOcc": parocc_labels, "Temporal": temporal_labels}


lh_vertno, rh_vertno = stc.vertices
stc_vertices = np.concatenate([lh_vertno, rh_vertno])

frontal_indices = []
parocc_indices = []
temporal_indices = []

def label_to_stc_indices_from_stc_vertices(label, stc_vertices):
    label_verts_in_stc = np.intersect1d(label.vertices, stc_vertices)
    return np.where(np.isin(stc_vertices, label_verts_in_stc))[0]

for label in labels:
    if label.name in frontal_labels:
        idx = label_to_stc_indices_from_stc_vertices(label, stc_vertices)
        frontal_indices.extend(idx.tolist())
    elif label.name in parocc_labels:
        idx = label_to_stc_indices_from_stc_vertices(label, stc_vertices)
        parocc_indices.extend(idx.tolist())
    elif label.name in temporal_labels:
        idx = label_to_stc_indices_from_stc_vertices(label, stc_vertices)
        temporal_indices.extend(idx.tolist())


# take unique indices
frontal_indices = np.unique(frontal_indices).astype(int)
parocc_indices = np.unique(parocc_indices).astype(int)
temporal_indices = np.unique(temporal_indices).astype(int)


### mean power per ROI per epoch
# frontal
dreamful_frontal = X_full_final[:, frontal_indices].mean(axis=1)
dreamless_frontal = X_less_final[:, frontal_indices].mean(axis=1)

# parieto-occipital/posterior
dreamful_parocc = X_full_final[:, parocc_indices].mean(axis=1)
dreamless_parocc = X_less_final[:, parocc_indices].mean(axis=1)

# temporal
dreamful_temporal = X_full_final[:, temporal_indices].mean(axis=1)
dreamless_temporal = X_less_final[:, temporal_indices].mean(axis=1)

# t-test with Bayes factor
alternative = "less" if f_min < 20 else "greater"
tt_frontal = pg.ttest(dreamful_frontal, dreamless_frontal, paired=False, alternative=alternative)
tt_parocc = pg.ttest(dreamful_parocc, dreamless_parocc, paired=False, alternative=alternative)
tt_temporal = pg.ttest(dreamful_temporal, dreamless_temporal, paired=False, alternative=alternative)


#### plot
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

markers = ['o', 'o', 'o', 'o', 'o']
unique_days = sorted(np.unique(day_full_list))

# jittered x position
x1 = np.random.normal(loc=0.9, scale=0.03, size=len(dreamful_frontal))
x2 = np.random.normal(loc=1.1, scale=0.03, size=len(dreamless_frontal))
x3 = np.random.normal(loc=1.5, scale=0.03, size=len(dreamful_parocc))
x4 = np.random.normal(loc=1.7, scale=0.03, size=len(dreamless_parocc))
x5 = np.random.normal(loc=2.1, scale=0.03, size=len(dreamful_temporal))
x6 = np.random.normal(loc=2.3, scale=0.03, size=len(dreamless_temporal))


for i, day in enumerate(unique_days):
    idx_full = np.array(day_full_list) == day
    idx_less = np.array(day_less_list) == day

    marker = markers[i % len(markers)]

    ax.scatter(x1[idx_full], dreamful_frontal[idx_full], color='red', alpha=0.7, label='Experience', marker=marker, facecolors='none')
    ax.scatter(x2[idx_less], dreamless_frontal[idx_less], color='blue', alpha=0.7, label='No Experience', marker=marker, facecolors='none')
    ax.scatter(x3[idx_full], dreamful_parocc[idx_full], color='red', alpha=0.7, marker=marker, facecolors='none')
    ax.scatter(x4[idx_less], dreamless_parocc[idx_less], color='blue', alpha=0.7, marker=marker, facecolors='none')
    ax.scatter(x5[idx_full], dreamful_temporal[idx_full], color='red', alpha=0.7, marker=marker, facecolors='none')
    ax.scatter(x6[idx_less], dreamless_temporal[idx_less], color='blue', alpha=0.7, marker=marker, facecolors='none')


# add horizontal bars for mean values
ax.hlines(np.mean(dreamful_frontal), xmin=0.8, xmax=1.0, color='red', linewidth=2)
ax.hlines(np.mean(dreamless_frontal), xmin=1.0, xmax=1.2, color='blue', linewidth=2)
ax.hlines(np.mean(dreamful_parocc), xmin=1.4, xmax=1.6, color='red', linewidth=2)
ax.hlines(np.mean(dreamless_parocc), xmin=1.6, xmax=1.8, color='blue', linewidth=2)
ax.hlines(np.mean(dreamful_temporal), xmin=2.0, xmax=2.2, color='red', linewidth=2)
ax.hlines(np.mean(dreamless_temporal), xmin=2.2, xmax=2.4, color='blue', linewidth=2)

# in case values are too low
if tt_frontal["p-val"].iloc[0] < 0.05:  # frontal
    y_max_frontal = max(np.max(dreamful_frontal), np.max(dreamless_frontal)) + 0.002
    ax.plot([0.9, 1.1], [y_max_frontal, y_max_frontal], color='black', linewidth=1.5)
    ax.plot([0.9, 0.9], [y_max_frontal, y_max_frontal - 0.0005], color='black', linewidth=1.5)# left bracket
    ax.plot([1.1, 1.1], [y_max_frontal, y_max_frontal - 0.0005], color='black', linewidth=1.5)# right bracket
    ax.text(1.0, y_max_frontal + 0.0002, get_asterisk(tt_frontal["p-val"].iloc[0]), ha='center', va='bottom', fontsize=16)

if tt_parocc["p-val"].iloc[0] < 0.05:  # parieto-occipital
    y_max_parocc = max(np.max(dreamful_parocc), np.max(dreamless_parocc)) + 0.002
    ax.plot([1.5, 1.7], [y_max_parocc, y_max_parocc], color='black', linewidth=1.5)
    ax.plot([1.5, 1.5], [y_max_parocc, y_max_parocc - 0.0005], color='black', linewidth=1.5)# left bracket
    ax.plot([1.7, 1.7], [y_max_parocc, y_max_parocc - 0.0005], color='black', linewidth=1.5)# right bracket
    ax.text(1.6, y_max_parocc + 0.0002, get_asterisk(tt_parocc["p-val"].iloc[0]), ha='center', va='bottom', fontsize=16)

if tt_temporal["p-val"].iloc[0] < 0.05:  # parieto-occipital
    y_max_temporal = max(np.max(dreamful_temporal), np.max(dreamless_temporal)) + 0.002 
    ax.plot([2.1, 2.3], [y_max_temporal, y_max_temporal], color='black', linewidth=1.5)
    ax.plot([2.1, 2.1], [y_max_temporal, y_max_temporal - 0.0005], color='black', linewidth=1.5)# left bracket
    ax.plot([2.3, 2.3], [y_max_temporal, y_max_temporal - 0.0005], color='black', linewidth=1.5)# right bracket
    ax.text(2.2, y_max_temporal + 0.0002, get_asterisk(tt_temporal["p-val"].iloc[0]), ha='center', va='bottom', fontsize=16)
    
    
# set axis
ax.set_xticks([1.0, 1.6, 2.2])
ax.set_xticklabels(['Frontal', 'Parietal_Occipital', 'Temporal'], fontsize=12)
ax.set_ylabel(f"Mean Source Power ({duration}sec) {invmethod}_raw", fontsize=12)
ax.set_xlim(0.7, 2.5)
ax.set_ylim(0.01, 0.037)
#ax.set_ylim(0, 0.2)
ax.legend(['Experience', 'No experience'], loc='upper left')

plt.tight_layout()
plt.title(f"{f_min}-{f_max}Hz")
plt.show()
path_save = Path.home() / "Data" / "Dream" / "MNE" / "ROI" / figname
plt.savefig(path_save, dpi=300)


# Bayes factor
nx = X_full_final.shape[0]   
ny = X_less_final.shape[0]   
n_vert = X_full_final.shape[1]

# obtain t-val
tvals = np.empty(n_vert)

for v in range(n_vert):
    t, _ = ttest_ind(X_full_final[:, v], X_less_final[:, v], equal_var=True, nan_policy='omit')
    tvals[v] = t

bf10 = np.array([pg.bayesfactor_ttest(t=tval, nx=nx, ny=ny, paired=False, alternative=alternative) for tval in tvals])

# log10 (compress dynamic range)
logBF = np.log10(bf10)

###　visualize on the brain
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

# power difference
diff_data = X_full_final.mean(axis=0) - X_less_final.mean(axis=0)  # shape: (n_sources,)

# visualize PSD differences
diff_stc = mne.SourceEstimate(
    diff_data[:, np.newaxis],  # shape: (n_sources, 1)
    vertices=used_vertices,
    tmin=0, tstep=1,
    subject=f"Dream{subject}"
)

# visualize t-vals
stc_tvals = mne.SourceEstimate(
    tvals[:, np.newaxis],  # shape: (n_sources, 1)
    vertices=used_vertices,
    tmin=0, tstep=1,
    subject=f"Dream{subject}"
)


brain = stc_tvals.plot(
    subject=f"Dream{subject}",
    subjects_dir=subjects_dir,
    hemi='both',
    surface='inflated',
    smoothing_steps=15,
    views=['dorsal']
)

# for label in labels:
#     if label.name in frontal_labels:
#         brain.add_label(label, color='red', alpha=0.7, borders=True)
#     elif label.name in parocc_labels:
#         brain.add_label(label, color='blue', alpha=0.7, borders=True)
#     elif label.name in temporal_labels:
#         brain.add_label(label, color='green', alpha=0.7, borders=True)

# visualize Bayes factor on the brain
stc_bf = mne.SourceEstimate(
    data=logBF[:, np.newaxis],
    vertices=used_vertices,
    tmin=0, tstep=1,
    subject=f"Dream{subject}"
)
    
brain = stc_bf.plot(
    subject=f"Dream{subject}",
    subjects_dir=subjects_dir,
    #hemi='split',
    hemi='both',
    surface='inflated',
    smoothing_steps=15,
    #views=["lat", "med"]
    views=['dorsal']
)
# for label in labels:
#     if label.name in frontal_labels:
#         brain.add_label(label, color='red', alpha=0.7, borders=True)
#     elif label.name in parocc_labels:
#         brain.add_label(label, color='blue', alpha=0.7, borders=True)
#     elif label.name in temporal_labels:
#         brain.add_label(label, color='green', alpha=0.7, borders=True)


# function to plot histogram per ROI
def extract_roi_values(stc, roi_label_names, threshold=threshold):
    values = []

    for lname in roi_label_names:
        lab = label_dict.get(lname)
        if lab is None:
            print(f"{lname} not found, skip.")
            continue

        stc_roi = stc.in_label(lab)
        if stc_roi.data.size == 0:
            continue

        vals = stc_roi.data.ravel()
        values.append(vals)

    if len(values) == 0:
        return np.array([]), np.nan
    
    vals_all = np.concatenate(values)

    # number of vertices log10BF >= threshold 
    num_over = np.sum(vals_all >= threshold)

    # number of vertices within a ROI
    num_total = vals_all.size

    # percentage of sources above threshold
    percentage = 100 * num_over / num_total

    return vals_all, percentage


# min and max across ROIs
all_vals = []

for roi_labels in roi_dict.values():
    vals, percentage    = extract_roi_values(stc_bf, roi_labels, threshold)
    if vals.size > 0:
        all_vals.append(vals)

if len(all_vals) == 0:
    raise RuntimeError("no values were obtained from the ROI")

all_vals = np.concatenate(all_vals)
xmin = all_vals.min()
xmax = all_vals.max()

# histogram of log10BF in each ROI
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

#less
for col, (roi_name, roi_labels) in enumerate(roi_dict.items()):
    ax = axes[col]
    vals, per = extract_roi_values(stc_bf, roi_labels, threshold)
    
    if vals.size == 0:
        ax.set_title(f"{roi_name} (no data)")
        ax.set_xlim(xmin, xmax)
        continue

    ax.hist(vals, bins=40, alpha=0.8, range=(xmin, xmax), color='gray')
    ax.set_title(f"{roi_name}, BF ≥ 3: {int(np.round(per,1))}%,  (n={vals.size})")
    ax.set_xlabel("log10 BF")
    ax.set_ylabel("Number of vertices")
    ax.set_xlim(-5, 5)
    ax.axvline(-np.log10(3), linestyle='--', linewidth=1, color='black')
    ax.axvline(np.log10(3), linestyle='--', linewidth=1, color='black')

plt.tight_layout()

path_save_bfhisto = Path.home() / "Data" / "Dream" / "MNE" / "ROI" / figname_bfhisto
fig.savefig(path_save_bfhisto, dpi=300)


print(f"{figname}")
print()
print("Frontal")
print(tt_frontal.to_string())
print()
print("Parietal-Occipital")
print(tt_parocc.to_string())
print()
print("Temporal")
print(tt_temporal.to_string())


### convert diff_data to morph and save
src = inverse_operator_day1['src']
morph = mne.compute_source_morph(src=src, subject_from=f"Dream{subject}",subject_to='fsaverage',subjects_dir=subjects_dir)
stc_fsavg = morph.apply(diff_stc)
stc_fsavg.save(morph_savepath)
