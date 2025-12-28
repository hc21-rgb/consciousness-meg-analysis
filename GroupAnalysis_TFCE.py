"""
Group analysis with Threshold-Free Cluster Enhancement (TFCE)
"""

import mne
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Qt5Agg') 
mne.set_log_level('WARNING')
from mne.stats import spatio_temporal_cluster_1samp_test 
import glob


############# needs to be specified ##########
frange = "delta"
#frange = "gamma"
#########################################

# path to data
stc_dir = Path("/Users/Data/Dream/MNE/Morph") / frange

# alpha level
alpha = 0.05

# read fsaverage info
subjects_dir = "/Applications/freesurfer/subjects"
fs_src = mne.setup_source_space("fsaverage", spacing="ico5", add_dist=False, subjects_dir=subjects_dir, verbose=False)
adjacency = mne.spatial_src_adjacency(fs_src)
fs_vertices = [fs_src[0]["vertno"], fs_src[1]["vertno"]]
n_vertices = len(fs_vertices[0]) + len(fs_vertices[1])

# read PSD data
stc_files = sorted(glob.glob(str(stc_dir / "*-lh.stc")))
stcs = [mne.read_source_estimate(f.replace("-lh.stc", ""), subject='fsaverage')
        for f in stc_files]


X_diff = np.stack([stc.data[:, 0] for stc in stcs])

if frange == "delta":
    T_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_1samp_test(
        X_diff,
        n_permutations=2000, # try 5 for debugging
        tail= -1, # -1: full-less < 0, 1:full-less > 0, 0:two-sided,  
        threshold=dict(start=0, step= -0.1), #TFCE #if tail = -1, step < 0   
        adjacency=adjacency,
        n_jobs=1,
        out_type='mask'
    )
    sig_idx = np.where(cluster_pv > 1-alpha)[0] # cluster_pv values should be inversed in the case of one tail (less)

else: # i.e., frange == "gamma"
   T_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_1samp_test(
       X_diff,
       n_permutations=2000,
       tail= 1,   
       threshold=dict(start=0, step= 0.1),
       adjacency=adjacency,
       n_jobs=1,
       out_type='mask'
    )
   sig_idx = np.where(cluster_pv < alpha)[0]

print(f"detected clusters: {len(sig_idx)} / {len(clusters)}")

# visualize
T_source = T_obs.squeeze()
sig_mask = np.zeros(T_source.shape[0], dtype=bool)

for k in np.where(sig_idx)[0]:
   sig_mask |= clusters[sig_idx[k]]

T_vis = T_source.copy()
T_vis[~sig_mask] = 0.0

stc_T = mne.SourceEstimate(T_vis[:, np.newaxis], fs_vertices, 0, 1, subject='fsaverage')
brain = stc_T.plot(hemi='split', views=['lat', 'med'], subjects_dir=subjects_dir)#,

# check T value distribution
stc_T_all = mne.SourceEstimate(T_source[:, np.newaxis], fs_vertices, 0, 1, subject='fsaverage')
brain = stc_T_all.plot(hemi='split', views=['lat', 'med'], subjects_dir=subjects_dir)#,