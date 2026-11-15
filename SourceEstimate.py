#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run-wise source reconstruction and Experience vs No-experience comparison.
1. Read run-wise epoch files from Epoch/D{participant}/.
2. Create an inverse operator separately for each run.
3. Reconstruct each epoch in each run in source space.
4. Confirm that all runs use the same source-space vertices.
5. Concatenate source-power arrays only after source reconstruction.
6. Preserve the original day-level rule: retain only ExpDay values containing
   both Experience and No-experience epochs.
7. Plot mean source power for each epoch per ROI and save the plot
"""

import gc
from pathlib import Path

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
import pingouin as pg
from statsmodels.stats.multitest import multipletests
from mne.minimum_norm import apply_inverse
from mne.time_frequency import psd_array_welch

mne.set_log_level("WARNING")


# =============================================================================
# Settings
# =============================================================================
participant = 11

f_min = 20
f_max = 50

duration = 20
invmethod = "dSPM"
lambda2 = 1.0 / 9.0
stage = "allstages"

# Pattern used to find run-wise epoch files.
# Example directory:
# ~/Data/Dream/MNE/Epoch/D5/
#
# If all files in the directory are the intended epoch files, use "*-epo.fif".
epoch_pattern = f"*{duration}s_epo.fif"

# Noise covariance files are still selected by experimental day.
# Remove/comment entries for days that do not exist for a participant.
noise_cov_files = {
    1: f"D{participant}_day1_noise_cov.fif",
    2: f"D{participant}_day2_noise_cov.fif",
    3: f"D{participant}_day3_noise_cov.fif",
    4: f"D{participant}_day4_noise_cov.fif",
}


# =============================================================================
# Paths
# =============================================================================
subjects_dir = "/Applications/freesurfer/subjects/"
subject_name = f"Dream{participant}"

epoch_dir = Path.home() / "Data" / "Dream" / "MNE" / "Epoch" / f"D{participant}"
noise_cov_dir = Path.home() / "Data" / "Dream" / "MNE" / "NoiseCov"
fig_save_dir = Path.home() / "Data" / "Dream" / "MNE" / "Results" 

fwd_file = Path(subjects_dir) / subject_name / "bem" / f"{subject_name}-fwd.fif"

figname = (
    f"D{participant}_{stage}_{f_min}-{f_max}Hz_{duration}sec_"
    f"{invmethod}_2ROIs_ICAzsc.png"
)

csv_name = (
    f"D{participant}_{stage}_{f_min}-{f_max}Hz_{duration}sec_"
    f"{invmethod}_ROIpower_by_epoch.csv"
)
csv_path = fig_save_dir / csv_name


# =============================================================================
# Helper functions
# =============================================================================
def get_condition_masks(epochs):
    """Return boolean masks for Experience and No-experience epochs."""
    if epochs.metadata is None:
        raise RuntimeError("Epoch metadata are required, but metadata=None.")

    md = epochs.metadata

    required_columns = {
        "dream",
        "whitedream",
        "unsure_dream",
        "wake_thinking",
        "wake_nothinking",
        "asleep_nodream",
        "ExpDay",
    }
    missing = required_columns.difference(md.columns)
    if missing:
        raise RuntimeError(f"Missing metadata columns: {sorted(missing)}")

    ix_exp = (
        (md["dream"] == 1)
        & (~md["whitedream"].isin([1, 0.5]))
        & (~md["unsure_dream"].isin([1, 0.5]))
        & (~md["wake_thinking"].isin([1, 0.5]))
        & (~md["wake_nothinking"].isin([1, 0.5]))
        & (~md["asleep_nodream"].isin([1, 0.5]))
    )

    ix_noexp = (
        (~md["dream"].isin([1, 0.5]))
        & (~md["whitedream"].isin([1, 0.5]))
        & (~md["unsure_dream"].isin([1, 0.5]))
        & (~md["wake_thinking"].isin([1, 0.5]))
        & (~md["wake_nothinking"].isin([1, 0.5]))
        & (md["asleep_nodream"] == 1)
    )

    return ix_exp.to_numpy(), ix_noexp.to_numpy()


def get_single_expday(epochs, epoch_file):
    """Return the unique ExpDay for a run; require one day per run file."""
    days = epochs.metadata["ExpDay"].dropna().unique()
    if len(days) != 1:
        raise RuntimeError(
            f"{epoch_file.name}: expected exactly one ExpDay per run, "
            f"but found {days.tolist()}"
        )
    return int(days[0])


def source_band_power_for_epochs(
    epochs_subset,
    inverse_operator,
    f_min,
    f_max,
    lambda2,
    invmethod,
    expected_vertices=None,
):
    """Source reconstruct each epoch and return band power per source vertex."""
    if len(epochs_subset) == 0:
        return np.empty((0, 0)), expected_vertices

    sfreq = epochs_subset.info["sfreq"]
    n_per_seg = int(2 * sfreq)      # 2-s segments
    n_overlap = int(n_per_seg / 2)  # 50% overlap

    band_powers = []

    for ep_data in epochs_subset.get_data():
        epoch = mne.EvokedArray(
            ep_data,
            epochs_subset.info,
            tmin=epochs_subset.tmin,
        )

        stc = apply_inverse(
            epoch,
            inverse_operator,
            lambda2=lambda2,
            method=invmethod,
        )

        current_vertices = [stc.vertices[0].copy(), stc.vertices[1].copy()]

        if expected_vertices is None:
            expected_vertices = current_vertices
        else:
            same_lh = np.array_equal(current_vertices[0], expected_vertices[0])
            same_rh = np.array_equal(current_vertices[1], expected_vertices[1])
            if not (same_lh and same_rh):
                raise RuntimeError(
                    "Source-space vertices differ between runs/epochs. "
                    "All runs must use the same source space before concatenation."
                )

        psd, _ = psd_array_welch(
            stc.data,
            sfreq=sfreq,
            fmin=f_min,
            fmax=f_max,
            n_fft=n_per_seg,
            n_overlap=n_overlap,
            window="hamming",
            average="mean",
        )

        band_power = psd.mean(axis=1)
        band_powers.append(band_power)

    return np.vstack(band_powers), expected_vertices


def label_to_stc_indices(label, used_vertices):
    """Map a FreeSurfer label to STC row indices, respecting hemisphere."""
    lh_vertices, rh_vertices = used_vertices

    if label.hemi == "lh":
        return np.where(np.isin(lh_vertices, label.vertices))[0]
    if label.hemi == "rh":
        return len(lh_vertices) + np.where(np.isin(rh_vertices, label.vertices))[0]

    raise ValueError(f"Unexpected label hemisphere: {label.hemi}")


# =============================================================================
# Read forward solution and noise covariance matrices
# =============================================================================
fwd = mne.read_forward_solution(fwd_file)

noise_cov_by_day = {}
for day, filename in noise_cov_files.items():
    cov_path = noise_cov_dir / filename
    if cov_path.exists():
        noise_cov_by_day[day] = mne.read_cov(cov_path)
    else:
        print(f"Noise covariance not found; day {day} will be unavailable: {cov_path}")

if not noise_cov_by_day:
    raise FileNotFoundError("No noise covariance files were found.")


# =============================================================================
# Find run-wise epoch files
# =============================================================================
epoch_files = sorted(epoch_dir.glob(epoch_pattern))

# Fallback is useful when filenames do not contain the duration string.
if len(epoch_files) == 0:
    epoch_files = sorted(epoch_dir.glob("*zsc-epo.fif"))

if len(epoch_files) == 0:
    raise FileNotFoundError(f"No epoch files found in {epoch_dir}")

print("Run-wise epoch files:")
for p in epoch_files:
    print(f"  {p.name}")
print()


# =============================================================================
# Run-wise source reconstruction
# =============================================================================
exp_arrays = []
noexp_arrays = []
exp_days = []
noexp_days = []
exp_runs = []
noexp_runs = []

# Vertex list shared by every reconstructed epoch/run.
used_vertices = None

for run_index, epoch_file in enumerate(epoch_files, start=1):
    print(f"Processing run {run_index}: {epoch_file.name}")

    epochs_run = mne.read_epochs(epoch_file, preload=True)
    epochs_run.pick_types(meg=True, eeg=False)

    if epochs_run.metadata is None:
        raise RuntimeError(f"{epoch_file.name}: metadata are required.")

    day = get_single_expday(epochs_run, epoch_file)

    if day not in noise_cov_by_day:
        raise RuntimeError(
            f"{epoch_file.name}: ExpDay={day}, but no noise covariance "
            f"was loaded for day {day}."
        )

    # IMPORTANT: inverse operator is made separately for each run using
    # that run's Info, while the same participant forward solution is used.
    inverse_operator_run = mne.minimum_norm.make_inverse_operator(
        epochs_run.info,
        fwd,
        noise_cov_by_day[day],
        loose=0.2,
        depth=0.8,
        rank="info",
    )

    ix_exp, ix_noexp = get_condition_masks(epochs_run)
    epochs_exp_run = epochs_run[ix_exp]
    epochs_noexp_run = epochs_run[ix_noexp]

    print(
        f"  ExpDay={day}; Experience={len(epochs_exp_run)}, "
        f"No-experience={len(epochs_noexp_run)}"
    )

    # Experience
    if len(epochs_exp_run) > 0:
        X_exp_run, used_vertices = source_band_power_for_epochs(
            epochs_exp_run,
            inverse_operator_run,
            f_min,
            f_max,
            lambda2,
            invmethod,
            expected_vertices=used_vertices,
        )
        exp_arrays.append(X_exp_run)
        exp_days.extend([day] * len(X_exp_run))
        exp_runs.extend([run_index] * len(X_exp_run))

    # No experience
    if len(epochs_noexp_run) > 0:
        X_noexp_run, used_vertices = source_band_power_for_epochs(
            epochs_noexp_run,
            inverse_operator_run,
            f_min,
            f_max,
            lambda2,
            invmethod,
            expected_vertices=used_vertices,
        )
        noexp_arrays.append(X_noexp_run)
        noexp_days.extend([day] * len(X_noexp_run))
        noexp_runs.extend([run_index] * len(X_noexp_run))

    del epochs_run, inverse_operator_run
    gc.collect()


if not exp_arrays:
    raise RuntimeError("No Experience epochs remained after filtering.")
if not noexp_arrays:
    raise RuntimeError("No No-experience epochs remained after filtering.")
if used_vertices is None:
    raise RuntimeError("No source estimates were generated.")

# Concatenate only AFTER run-wise source reconstruction.
X_exp_all = np.vstack(exp_arrays)
X_noexp_all = np.vstack(noexp_arrays)
exp_days = np.asarray(exp_days, dtype=int)
noexp_days = np.asarray(noexp_days, dtype=int)
exp_runs = np.asarray(exp_runs, dtype=int)
noexp_runs = np.asarray(noexp_runs, dtype=int)

print()
print(f"All reconstructed Experience epochs: {X_exp_all.shape[0]}")
print(f"All reconstructed No-experience epochs: {X_noexp_all.shape[0]}")
print(f"Number of source vertices: {X_exp_all.shape[1]}")


# =============================================================================
# Use only experimental days containing BOTH conditions
# =============================================================================
common_days = np.intersect1d(np.unique(exp_days), np.unique(noexp_days))

if len(common_days) == 0:
    raise RuntimeError(
        "No ExpDay contains both Experience and No-experience epochs."
    )

keep_exp = np.isin(exp_days, common_days)
keep_noexp = np.isin(noexp_days, common_days)

X_full_final = X_exp_all[keep_exp]
X_less_final = X_noexp_all[keep_noexp]

day_full_final = exp_days[keep_exp]
day_less_final = noexp_days[keep_noexp]
run_full_final = exp_runs[keep_exp]
run_less_final = noexp_runs[keep_noexp]

print(f"Common ExpDay values retained: {common_days.tolist()}")
print(f"Final Experience epochs: {len(X_full_final)}")
print(f"Final No-experience epochs: {len(X_less_final)}")
print()


# =============================================================================
# ROI definitions
# =============================================================================
labels = mne.read_labels_from_annot(
    subject_name,
    parc="aparc",
    subjects_dir=subjects_dir,
    hemi="both",
)

frontal_labels = [
    "superiorfrontal-lh", "rostralmiddlefrontal-lh", "caudalmiddlefrontal-lh",
    "parsopercularis-lh", "parstriangularis-lh", "parsorbitalis-lh",
    "lateralorbitofrontal-lh", "medialorbitofrontal-lh", "frontalpole-lh",
    "superiorfrontal-rh", "rostralmiddlefrontal-rh", "caudalmiddlefrontal-rh",
    "parsopercularis-rh", "parstriangularis-rh", "parsorbitalis-rh",
    "lateralorbitofrontal-rh", "medialorbitofrontal-rh", "frontalpole-rh",
]

parocc_labels = [
    "supramarginal-lh", "superiorparietal-lh", "inferiorparietal-lh",
    "precuneus-lh", "lingual-lh", "pericalcarine-lh", "cuneus-lh",
    "lateraloccipital-lh",
    "supramarginal-rh", "superiorparietal-rh", "inferiorparietal-rh",
    "precuneus-rh", "lingual-rh", "pericalcarine-rh", "cuneus-rh",
    "lateraloccipital-rh",
]

lh_vertno, rh_vertno = used_vertices

frontal_indices = []
parocc_indices = []

for label in labels:
    if label.name in frontal_labels:
        frontal_indices.extend(label_to_stc_indices(label, used_vertices).tolist())
    elif label.name in parocc_labels:
        parocc_indices.extend(label_to_stc_indices(label, used_vertices).tolist())

frontal_indices = np.unique(frontal_indices).astype(int)
parocc_indices = np.unique(parocc_indices).astype(int)

if len(frontal_indices) == 0 or len(parocc_indices) == 0:
    raise RuntimeError("One or more ROI definitions matched no source vertices.")


# =============================================================================
# Mean source power per ROI per epoch
# =============================================================================
dreamful_frontal = X_full_final[:, frontal_indices].mean(axis=1)
dreamless_frontal = X_less_final[:, frontal_indices].mean(axis=1)

dreamful_parocc = X_full_final[:, parocc_indices].mean(axis=1)
dreamless_parocc = X_less_final[:, parocc_indices].mean(axis=1)


# =============================================================================
# Plot ROI power
# =============================================================================
fig, ax = plt.subplots(figsize=(6, 5))

x1 = np.random.normal(loc=0.9, scale=0.03, size=len(dreamful_frontal))
x2 = np.random.normal(loc=1.1, scale=0.03, size=len(dreamless_frontal))
x3 = np.random.normal(loc=1.5, scale=0.03, size=len(dreamful_parocc))
x4 = np.random.normal(loc=1.7, scale=0.03, size=len(dreamless_parocc))

ax.hlines(np.mean(dreamful_frontal), xmin=0.8, xmax=1.0, color='black',linewidth=2)
ax.hlines(np.mean(dreamless_frontal), xmin=1.0, xmax=1.2, color='black', linewidth=2)
ax.hlines(np.mean(dreamful_parocc), xmin=1.4, xmax=1.6, color='black', linewidth=2)
ax.hlines(np.mean(dreamless_parocc), xmin=1.6, xmax=1.8, color='black', linewidth=2)

ax.scatter(x1, dreamful_frontal,color='red',  alpha=0.7, label="Experience", marker="o", facecolors="none")
ax.scatter(x2, dreamless_frontal, color='blue', alpha=0.7, label="No Experience", marker="o", facecolors="none")
ax.scatter(x3, dreamful_parocc, color='red', alpha=0.7, marker="o", facecolors="none")
ax.scatter(x4, dreamless_parocc, color='blue', alpha=0.7, marker="o", facecolors="none")


y_min_all = min(
    np.min(dreamful_frontal),
    np.min(dreamless_frontal),
    np.min(dreamful_parocc),
    np.min(dreamless_parocc),
)

y_max_all = max(
    np.max(dreamful_frontal),
    np.max(dreamless_frontal),
    np.max(dreamful_parocc),
    np.max(dreamless_parocc),
)

yrange = y_max_all - y_min_all
if yrange == 0:
    yrange = max(abs(y_max_all), 1.0) * 0.1

ax.set_xticks([1.0, 1.6])
ax.set_xticklabels(["Frontal", "Posterior"], fontsize=12)
ax.set_ylabel(f"Mean Source Power ({duration}sec) {invmethod}", fontsize=12)
ax.set_xlim(0.7, 1.9)
ax.set_ylim(y_min_all - yrange / 20, y_max_all + yrange / 10)
ax.legend(loc="upper left")
ax.set_title(f"{f_min}-{f_max} Hz")

plt.tight_layout()
path_save = fig_save_dir / figname
plt.savefig(path_save, dpi=300)
plt.show()


# =============================================================================
# Whole-brain mean source-power maps after run-wise reconstruction
# =============================================================================
mean_full = X_full_final.mean(axis=0)
mean_less = X_less_final.mean(axis=0)
diff_data = mean_full - mean_less

stc_full = mne.SourceEstimate(
    mean_full[:, np.newaxis],
    vertices=used_vertices,
    tmin=0,
    tstep=1,
    subject=subject_name,
)

stc_less = mne.SourceEstimate(
    mean_less[:, np.newaxis],
    vertices=used_vertices,
    tmin=0,
    tstep=1,
    subject=subject_name,
)

stc_diff = mne.SourceEstimate(
    diff_data[:, np.newaxis],
    vertices=used_vertices,
    tmin=0,
    tstep=1,
    subject=subject_name,
)

brain = stc_diff.plot(
    subject=subject_name,
    subjects_dir=subjects_dir,
    hemi="both",
    surface="inflated",
    smoothing_steps=15,
)

# brain = stc_diff.plot(
#     subject=subject_name,
#     subjects_dir=subjects_dir,
#     hemi="lh",
#     surface="inflated",
#     smoothing_steps=15,
# )

# brain = stc_diff.plot(
#     subject=subject_name,
#     subjects_dir=subjects_dir,
#     hemi="rh",
#     surface="inflated",
#     smoothing_steps=15,
# )


# =============================================================================
# Save mean ROI power for each epoch as CSV
# =============================================================================
df_exp_roi = pd.DataFrame({
    "Participant": participant,
    "ExpDay": day_full_final,
    "Run": run_full_final,
    "Condition": "experience",
    "Frontal": dreamful_frontal,
    "Posterior": dreamful_parocc,
})

df_noexp_roi = pd.DataFrame({
    "Participant": participant,
    "ExpDay": day_less_final,
    "Run": run_less_final,
    "Condition": "no-experience",
    "Frontal": dreamless_frontal,
    "Posterior": dreamless_parocc,
})

roi_epoch_df = pd.concat(
    [df_exp_roi, df_noexp_roi],
    ignore_index=True
)

roi_epoch_df = roi_epoch_df.sort_values(
    ["ExpDay", "Run", "Condition"]
).reset_index(drop=True)

roi_epoch_df.insert(
    0, "CSV_Epoch_ID",
    np.arange(1, len(roi_epoch_df) + 1)
)

roi_epoch_df.to_csv(csv_path, index=False)

print()
print(f"ROI power CSV saved: {csv_path}")
print(roi_epoch_df.head())
