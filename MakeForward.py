"""
Head model and forward computation based on the NNE-python tutorial
based on the MNE-python tutorial <https://mne.tools/stable/auto_tutorials/forward/30_forward.html#sphx-glr-auto-tutorials-forward-30-forward-py> (last access: 2025/12/28)
"""

import mne
from pathlib import Path


############# needs to be specified ##########

# participant/participant ID
subject = "Dream11"

# raw file containing channel locations and types
raw_fname = "/Users/Data/Dream/MEG/240527/run1_raw_tsss.fif" # Dream11 day3

#############################################

# paths to FreeSurfer reconstructions
FREESURFER_HOME = "/Applications/freesurfer"
subjects_dir = f"{FREESURFER_HOME}/subjects"


plot_bem_kwargs = dict(
    subject=subject,
    subjects_dir=subjects_dir,
    brain_surfaces="white",
    orientation="coronal",
    slices=[50, 100, 150, 200],
)
mne.viz.plot_bem(**plot_bem_kwargs)


# path to the transformation file obtained through coregistration
trans = f"/Applications/freesurfer/subjects/{subject}/mri/transforms/{subject}-trans.fif"


info = mne.io.read_info(raw_fname)
mne.viz.plot_alignment(
    info,
    trans,
    subject=subject,
    dig=True,
    meg=["helmet", "sensors"],
    subjects_dir=subjects_dir,
    surfaces="head",
)

src = mne.setup_source_space(
    subject, spacing="oct6", add_dist="patch", subjects_dir=subjects_dir
)
print(src)

mne.viz.plot_bem(src=src, **plot_bem_kwargs)


sphere = (0.0, 0.0, 0.04, 0.09)
vol_src = mne.setup_volume_source_space(
    subject,
    subjects_dir=subjects_dir,
    sphere=sphere,
    sphere_units="m",
    add_interpolator=False,
)
print(vol_src)

mne.viz.plot_bem(src=vol_src, **plot_bem_kwargs)

surface = "/Applications/freesurfer/subjects/{subject}/bem/inner_skull.surf"

vol_src = mne.setup_volume_source_space(
    subject, subjects_dir=subjects_dir, surface=surface, add_interpolator=False
)
print(vol_src)
mne.viz.plot_bem(src=vol_src, **plot_bem_kwargs)

fig = mne.viz.plot_alignment(
    subject=subject,
    subjects_dir=subjects_dir,
    surfaces="white",
    coord_frame="mri",
    src=src,
)
mne.viz.set_3d_view(
    fig,
    azimuth=173.78,
    elevation=101.75,
    distance=0.30,
    focalpoint=(-0.03, -0.01, 0.03),
)


conductivity = (0.3,)  # for single layer
model = mne.make_bem_model(
    subject=subject, ico=4, conductivity=conductivity, subjects_dir=subjects_dir
)
bem = mne.make_bem_solution(model)


fwd = mne.make_forward_solution(
    raw_fname,
    trans=trans,
    src=src,
    bem=bem,
    meg=True,
    eeg=False,
    mindist=5.0,
    n_jobs=None,
    verbose=True,
)
print(fwd)

# save fwd
fwd_path = Path(subjects_dir) / subject / "bem" / f"{subject}-fwd.fif"
mne.write_forward_solution(str(fwd_path), fwd, overwrite=True)


print(f"Before: {src}")
print(f'After:  {fwd["src"]}')


leadfield = fwd["sol"]["data"]
print(f"Leadfield size : {leadfield.shape[0]} sensors x {leadfield.shape[1]} dipoles")


fwd_fixed = mne.convert_forward_solution(
    fwd, surf_ori=True, force_fixed=True, use_cps=True
)
leadfield = fwd_fixed["sol"]["data"]
print(f"Leadfield size : {leadfield.shape[0]} sensors x {leadfield.shape[1]} dipoles")