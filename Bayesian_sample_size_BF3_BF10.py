"""
Model-aligned Bayesian Sample-Size Design for an MEG Study of the Neural Correlates of Consciousness

FINAL CONFIRMATORY MODEL
------------------------
For each outcome separately:

    y_scaled ~ State
               + (1 + State || Participant)
               + (1 | Participant:Day)

State coding:
    no-experience = -0.5
    experience    = +0.5

Thus beta_state is exactly:
    Experience - No-experience

"||" is implemented by independent participant random intercepts and
participant random slopes (no intercept-slope correlation).

IMPORTANT: FIXED EXTERNAL SCALING
---------------------------------
For Bayes-factor analysis, the confirmatory outcomes should NOT be standardized
using the confirmatory sample's own SD.  This script determines, from the pilot
P5/P11 data only:

    pilot_center
    pilot_residual_sd

and defines:

    y_scaled = (y_raw - pilot_center) / pilot_residual_sd

The SAME fixed center/scale should be used in the final confirmatory analysis.
This keeps the prior beta_state ~ Normal(0, 0.5) fixed before looking at
confirmatory neural outcomes.

DESIGN STRATEGY
---------------
This script has two stages.

Stage A: FAST SCREEN
    Uses the ACTUAL Participant x ExpDay E/NE count profiles from
    Participant_data.csv. Whole participant profiles (all days together) are
    bootstrap-sampled. It analytically propagates:
        - participant random-slope variance
        - participant-day random-intercept variance
        - residual variance
        - actual day-specific E/NE imbalance
    to locate the approximate N at which P(BF_direction,0 >= 3) >= 0.80 or P(BF_direction,0 >= 10) >= 0.80.

Stage B: FULL PYMC VALIDATION
    Around the Stage-A crossing, it:
        1) bootstraps whole participant/day count profiles,
        2) generates every synthetic epoch from the exact final hierarchical
           data-generating model,
        3) fits the exact final PyMC model,
        4) computes the directional Savage-Dickey BF from beta_state,
        5) estimates P(BF >= 3) and P(BF >= 10) from the same simulated BF values.

The Stage-B result is the final model-aligned design estimate.

COMPUTATIONAL NOTE
------------------
Full PyMC validation is expensive. The defaults below are deliberately modest
for a test run. For a manuscript-quality final run, increase:

    FULL_N_SIM
    FULL_DRAWS
    FULL_TUNE

and inspect MCSE, R-hat, ESS, and divergences.

INPUTS
------
ROIpower_by_epoch.csv
    Pilot P5/P11 epoch data.

Participant_data.csv
    REQUIRED columns:
        Participant_ID, ExpDay, experience, no-experience

    Each row is one valid Participant x ExpDay and both experience and
    no-experience must be > 0.

OUTPUTS
-------
pilot_exact_nuisance_parameters.csv
sample_size_dayprofile_fast_screen.csv
full_pymc_design_raw.csv
sample_size_dayprofile_full_pymc_validation.csv
"""

from pathlib import Path
import math
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm, gaussian_kde

import arviz as az


# =============================================================================
# Files
# =============================================================================

ROI_FILE = Path("ROIpower_by_epoch_pilot.csv")
PARTICIPANT_FILE = Path("Participant_data.csv")

# One permanent RAW master, independent of the BF threshold.
# A first run can import existing BF3/BF10 RAW files into this master.
FULL_RAW_OUT = Path("full_pymc_design_raw_master.csv")
LEGACY_RAW_FILES = []

# The pilot parameter file must be the same as for the historical simulations.
PILOT_PARAM_OUT = Path("pilot_exact_nuisance_parameters.csv")
FAST_OUT = Path("sample_size_fast_screen_BF3_BF10.csv")
# Both threshold summaries in one CSV (BF_threshold = 3 or 10).
FULL_SUMMARY_OUT = Path("sample_size_full_pymc_validation_BF3_BF10.csv")
MINIMUM_N_OUT = Path("sample_size_minimum_N_BF3_BF10.csv")


# =============================================================================
# Hypothesis
# =============================================================================

PILOT_PARTICIPANTS = [5, 11]

PRIMARY_HYPOTHESES = {
    "H1_posterior_delta": {
        "column": "Posterior_delta",
        "direction": "negative",  # Experience < No-experience
        "siclari_t": -2.98,
        "siclari_n": 32,
    },
    "H2_posterior_gamma": {
        "column": "Posterior_gamma",
        "direction": "positive",  # Experience > No-experience
        "siclari_t": +3.46,
        "siclari_n": 32,
    },
}


# =============================================================================
# Priors -- must match the final confirmatory model
# =============================================================================

ALPHA_PRIOR_SD = 1.0
BETA_PRIOR_SD = 0.50

PARTICIPANT_INTERCEPT_SD_PRIOR = 0.50
PARTICIPANT_SLOPE_SD_PRIOR = 0.50
DAY_SD_PRIOR = 0.50
RESIDUAL_SD_PRIOR = 1.00


# =============================================================================
# Design scenario
# =============================================================================

EFFECT_ATTENUATIONS = [1.00, 0.75, 0.50]

# Applied to the pilot posterior mean of sigma_participant_slope.
SLOPE_HETEROGENEITY_MULTIPLIERS = [0.50, 1.00, 1.50]

# A single dataset/BF value is evaluated against BOTH thresholds.
BF_THRESHOLDS = (3.0, 10.0)
TARGET_SUCCESS_PROB = 0.80

N_MIN = 5
N_MAX = 250

# Fast screening Monte Carlo
FAST_N_SIM_SCAN = 5000
FAST_N_SIM_REFINE = 30000

# -------------------------------------------------------------------------
# Full PyMC validation
# -------------------------------------------------------------------------
# Running full validation for every 3x3 scenario is very expensive.
# By default, validate the central scenario:
#   100% Siclari effect, 1.0x pilot slope heterogeneity.
#
# Set FULL_VALIDATE_ALL_SCENARIOS = True to validate all scenarios.
FULL_VALIDATE_ALL_SCENARIOS = False
FULL_EFFECT_ATTENUATIONS = [1.00]
FULL_SLOPE_MULTIPLIERS = [1.00]

# Full validation candidate range around the two Fast Screen crossings.
# "auto_window": tests all integer Ns from (smaller crossing - 4)
# to (larger crossing + 4). Reports minimum among tested Ns only.
# "exhaustive": tests ALL integer Ns from N_MIN to upper crossing + 4.
# Exhaustive can require thousands of PyMC fits; use for final verification.
FULL_N_MODE = "auto_window"  # or "exhaustive"
FULL_N_BELOW = 4
FULL_N_ABOVE = 4
# Override the automatic range for a targeted check, e.g. [18,19,20,21].
# None means determine Ns automatically from both fast-screen crossings.
#EXPLICIT_CANDIDATE_NS = None
#EXPLICIT_CANDIDATE_NS = [28, 29, 30, 31]
EXPLICIT_CANDIDATE_NS = [33]


# Test-run defaults. For final manuscript analysis, consider >= 200-500
# simulated datasets per N, depending on computational feasibility.
#FULL_N_SIM = 30
FULL_N_SIM = 500  
#FULL_N_SIM = 500 

# MCMC per simulated dataset. Increase for the final analysis if needed.
FULL_CHAINS = 2
FULL_TUNE = 3000
FULL_DRAWS = 2000
FULL_TARGET_ACCEPT = 0.99


# Pilot nuisance-model MCMC
PILOT_CHAINS = 4
PILOT_DRAWS = 3000
PILOT_TUNE = 3000
PILOT_TARGET_ACCEPT = 0.95

SEED = 20260921

# False: aggregate the existing master RAW without running PyMC or reading pilot files.
# True: run Fast Screen + PyMC (resuming already-completed simulations).
RUN_SIMULATIONS = True
#RUN_SIMULATIONS = False

# =============================================================================
# Data loading / Checks
# =============================================================================

def load_data():
    roi = pd.read_csv(ROI_FILE)
    pdata = pd.read_csv(PARTICIPANT_FILE)

    roi_required = {
        "Participant", "ExpDay", "State",
        "Posterior_delta", "Posterior_gamma",
    }
    pdata_required = {
        "Participant_ID", "ExpDay", "experience", "no-experience"
    }

    miss = roi_required - set(roi.columns)
    if miss:
        raise ValueError(f"Missing ROI columns: {sorted(miss)}")

    miss = pdata_required - set(pdata.columns)
    if miss:
        raise ValueError(f"Missing Participant_data columns: {sorted(miss)}")

    roi = roi[roi["Participant"].isin(PILOT_PARTICIPANTS)].copy()
    roi["State"] = roi["State"].astype(str).str.strip().str.lower()

    bad = set(roi["State"].dropna().unique()) - {
        "experience", "no-experience"
    }
    if bad:
        raise ValueError(f"Unexpected pilot State values: {sorted(bad)}")

    pdata = pdata.copy()
    if pdata.duplicated(["Participant_ID", "ExpDay"]).any():
        dup = pdata.loc[
            pdata.duplicated(["Participant_ID", "ExpDay"], keep=False)
        ]
        raise ValueError(
            "Participant_data contains duplicate Participant_ID x ExpDay rows:\n"
            + dup.to_string(index=False)
        )

    if (pdata[["experience", "no-experience"]] <= 0).any().any():
        raise ValueError(
            "Every Participant x ExpDay must have both Experience and "
            "No-experience (>0), consistent with the confirmatory inclusion rule."
        )

    pdata = pdata.sort_values(
        ["Participant_ID", "ExpDay"]
    ).reset_index(drop=True)

    return roi, pdata


# =============================================================================
# Fixed pilot scaling
# =============================================================================

def pilot_fixed_scale(roi, outcome):
    """
    Fixed scale determined ONLY from P5/P11.

    center:
        pilot grand mean.

    residual_sd:
        pooled within Participant x ExpDay x State residual SD.
        This is the externally fixed scaling constant for the confirmatory model.
    """
    cell = ["Participant", "ExpDay", "State"]

    center = float(roi[outcome].mean())

    residual = (
        roi[outcome]
        - roi.groupby(cell)[outcome].transform("mean")
    )

    n_cells = roi.groupby(cell).ngroups
    df_resid = len(residual) - n_cells
    if df_resid <= 0:
        raise ValueError(f"Insufficient pilot residual df for {outcome}")

    residual_sd = math.sqrt(float(np.sum(residual ** 2)) / df_resid)

    return center, residual_sd


# =============================================================================
# Fina PyMC model
# =============================================================================

def build_and_sample_final_model(
    dat,
    draws,
    tune,
    chains,
    target_accept,
    random_seed,
    progressbar=False,
):
    """
    Exact final model:

        y_scaled ~ State
                   + (1 + Sate || Participant)
                   + (1 | Participant:Day)

    dat must contain:
        Participant
        ParticipantDay
        State_c
        y_scaled
    """

    # Only the fitting path requires PyMC (summary-only runs do not).
    import pymc as pm

    dat = dat.reset_index(drop=True).copy()

    p_codes, p_levels = pd.factorize(dat["Participant"], sort=True)
    d_codes, d_levels = pd.factorize(dat["ParticipantDay"], sort=True)

    x = dat["State_c"].to_numpy(float)
    y = dat["y_scaled"].to_numpy(float)

    coords = {
        "participant": [str(x) for x in p_levels],
        "participant_day": [str(x) for x in d_levels],
        "obs_id": np.arange(len(dat)),
    }

    with pm.Model(coords=coords) as model:

        alpha = pm.Normal(
            "alpha",
            mu=0.0,
            sigma=ALPHA_PRIOR_SD,
        )

        beta_state = pm.Normal(
            "beta_state",
            mu=0.0,
            sigma=BETA_PRIOR_SD,
        )

        # Participant random intercept
        sigma_participant_intercept = pm.HalfNormal(
            "sigma_participant_intercept",
            sigma=PARTICIPANT_INTERCEPT_SD_PRIOR,
        )
        z_pi = pm.Normal(
            "z_participant_intercept",
            0.0, 1.0,
            dims="participant",
        )
        participant_intercept = pm.Deterministic(
            "participant_intercept",
            z_pi * sigma_participant_intercept,
            dims="participant",
        )

        # Participant random State slope, INDEPENDENT of intercept
        sigma_participant_slope = pm.HalfNormal(
            "sigma_participant_slope",
            sigma=PARTICIPANT_SLOPE_SD_PRIOR,
        )
        z_ps = pm.Normal(
            "z_participant_slope",
            0.0, 1.0,
            dims="participant",
        )
        participant_slope = pm.Deterministic(
            "participant_slope",
            z_ps * sigma_participant_slope,
            dims="participant",
        )

        # Participant:Day random intercept
        sigma_day = pm.HalfNormal(
            "sigma_day",
            sigma=DAY_SD_PRIOR,
        )
        z_day = pm.Normal(
            "z_day",
            0.0, 1.0,
            dims="participant_day",
        )
        day_intercept = pm.Deterministic(
            "day_intercept",
            z_day * sigma_day,
            dims="participant_day",
        )

        sigma_resid = pm.HalfNormal(
            "sigma_resid",
            sigma=RESIDUAL_SD_PRIOR,
        )

        mu = (
            alpha
            + beta_state * x
            + participant_intercept[p_codes]
            + participant_slope[p_codes] * x
            + day_intercept[d_codes]
        )

        pm.Normal(
            "y_obs",
            mu=mu,
            sigma=sigma_resid,
            observed=y,
            dims="obs_id",
        )

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,  # safer inside repeated simulation loops
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=progressbar,
            compute_convergence_checks=True,
        )

    return idata


# =============================================================================
# Pilot nuisance parameters from the same model
# =============================================================================

def fit_pilot_nuisance_parameters(roi):
    """
    Fits the SAME hierarchical model to P5/P11 for each outcome after fixed
    pilot scaling. The resulting SD posterior means are only anchors for the
    design simulation, not precise population estimates.

    If PILOT_PARAM_OUT already exists, it is reused.
    """
    if PILOT_PARAM_OUT.exists():
        print(f"Using existing {PILOT_PARAM_OUT}")
        return pd.read_csv(PILOT_PARAM_OUT)

    rows = []

    for h_index, (h_name, h) in enumerate(
        PRIMARY_HYPOTHESES.items(), start=1
    ):
        outcome = h["column"]

        center, residual_sd_fixed = pilot_fixed_scale(roi, outcome)

        dat = roi[
            ["Participant", "ExpDay", "State", outcome]
        ].dropna().copy()

        dat["State_c"] = dat["State"].map({
            "no-experience": -0.5,
            "experience": +0.5,
        }).astype(float)

        dat["ParticipantDay"] = (
            dat["Participant"].astype(str)
            + "_day_"
            + dat["ExpDay"].astype(str)
        )

        dat["y_scaled"] = (
            dat[outcome] - center
        ) / residual_sd_fixed

        print("\n" + "=" * 78)
        print(f"Pilot nuisance fit: {h_name}")
        print("=" * 78)

        idata = build_and_sample_final_model(
            dat=dat,
            draws=PILOT_DRAWS,
            tune=PILOT_TUNE,
            chains=PILOT_CHAINS,
            target_accept=PILOT_TARGET_ACCEPT,
            random_seed=SEED + h_index * 1000,
            progressbar=True,
        )

        s = az.summary(
            idata,
            var_names=[
                "beta_state",
                "sigma_participant_intercept",
                "sigma_participant_slope",
                "sigma_day",
                "sigma_resid",
            ],
            hdi_prob=0.95,
        )

        rows.append({
            "hypothesis": h_name,
            "outcome": outcome,
            "pilot_center_raw": center,
            "pilot_residual_sd_raw_fixed": residual_sd_fixed,

            "pilot_beta_state_mean":
                float(s.loc["beta_state", "mean"]),

            "sigma_participant_intercept_anchor":
                float(s.loc["sigma_participant_intercept", "mean"]),

            "sigma_participant_slope_anchor":
                float(s.loc["sigma_participant_slope", "mean"]),

            "sigma_day_anchor":
                float(s.loc["sigma_day", "mean"]),

            "sigma_resid_anchor":
                float(s.loc["sigma_resid", "mean"]),

            "pilot_beta_rhat":
                float(s.loc["beta_state", "r_hat"]),

            "pilot_beta_ess_bulk":
                float(s.loc["beta_state", "ess_bulk"]),
        })

    out = pd.DataFrame(rows)
    out.to_csv(PILOT_PARAM_OUT, index=False)
    return out


# =============================================================================
# Actual participant * day profiles
# =============================================================================

def make_participant_profiles(pdata):
    """
    Returns a dict:
        original participant ID -> dataframe of that participant's valid days.

    Whole profiles are bootstrap-sampled, preserving:
        - number of valid days per participant
        - day-specific nE
        - day-specific nNE
    """
    profiles = {}

    for pid, g in pdata.groupby("Participant_ID", sort=True):
        profiles[pid] = (
            g[["ExpDay", "experience", "no-experience"]]
            .sort_values("ExpDay")
            .reset_index(drop=True)
        )

    return profiles


# =============================================================================
# Effect calibration using day-specific counts
# =============================================================================

def participant_aggregate_difference_variance(
    profile,
    sigma_slope,
    sigma_day,
    sigma_resid,
):
    """
    Variance of the participant-level aggregate E-NE difference implied by
    the exact data-generating model.

    Participant intercept cancels.

    Random slope contributes:
        sigma_slope^2

    Day intercept contributes because E and NE may be distributed differently
    across days:
        sigma_day^2 * sum_d (nE_d/N_E - nNE_d/N_NE)^2

    Residual contributes:
        sigma_resid^2 * (1/N_E + 1/N_NE)
    """
    nE = profile["experience"].to_numpy(float)
    nNE = profile["no-experience"].to_numpy(float)

    NE = float(nE.sum())
    NNE = float(nNE.sum())

    a_day = nE / NE - nNE / NNE

    var_slope = sigma_slope ** 2
    var_day = sigma_day ** 2 * float(np.sum(a_day ** 2))
    var_resid = sigma_resid ** 2 * (1.0 / NE + 1.0 / NNE)

    return var_slope + var_day + var_resid


def calibrate_beta_from_external_dz(
    target_dz,
    profiles,
    sigma_slope_reference,
    sigma_day,
    sigma_resid,
):
    """
    Calibrate beta_true so that the standardized participant-level E-NE
    effect equals target_dz under the reference nuisance structure.

        beta_true = target_dz * sqrt(mean participant contrast variance)

    This is done using the actual participant-day count profiles.
    """
    vars_ = [
        participant_aggregate_difference_variance(
            profile=p,
            sigma_slope=sigma_slope_reference,
            sigma_day=sigma_day,
            sigma_resid=sigma_resid,
        )
        for p in profiles.values()
    ]

    reference_sd = math.sqrt(float(np.mean(vars_)))
    beta_true = target_dz * reference_sd

    return beta_true, reference_sd


# =============================================================================
# Fast screening BF
# =============================================================================

def directional_log_bf_normal(
    beta_hat,
    likelihood_var,
    prior_sd,
    direction,
):
    """
    Directional BF for beta vs H0=0 under Normal likelihood and Normal prior.
    """
    s2 = np.asarray(likelihood_var, float)
    t2 = prior_sd ** 2
    beta_hat = np.asarray(beta_hat, float)

    log_bf_full = (
        0.5 * np.log(s2 / (s2 + t2))
        + 0.5 * beta_hat ** 2
        * (1.0 / s2 - 1.0 / (s2 + t2))
    )

    post_var = 1.0 / (1.0 / s2 + 1.0 / t2)
    post_mean = post_var * beta_hat / s2
    post_sd = np.sqrt(post_var)

    z0 = (0.0 - post_mean) / post_sd

    if direction == "negative":
        p_dir = norm.cdf(z0)
    elif direction == "positive":
        p_dir = 1.0 - norm.cdf(z0)
    else:
        raise ValueError(direction)

    p_dir = np.clip(p_dir, 1e-300, 1.0)

    return (
        math.log(2.0)
        + log_bf_full
        + np.log(p_dir)
    )


def fast_success_probability(
    N,
    beta_true,
    sigma_slope,
    sigma_day,
    sigma_resid,
    profiles,
    direction,
    n_sim,
    rng,
    bf_threshold,
):
    """
    Fast screen using the exact day-profile-implied participant contrast
    variance. Whole participant profiles are bootstrap-sampled.
    """

    profile_list = list(profiles.values())
    n_profiles = len(profile_list)

    # Precompute each profile's non-population contrast variance.
    var_by_profile = np.array([
        participant_aggregate_difference_variance(
            p,
            sigma_slope=sigma_slope,
            sigma_day=sigma_day,
            sigma_resid=sigma_resid,
        )
        for p in profile_list
    ])

    idx = rng.integers(
        0, n_profiles, size=(n_sim, N)
    )

    V = var_by_profile[idx]

    D = rng.normal(
        loc=beta_true,
        scale=np.sqrt(V),
    )

    W = 1.0 / V
    Wsum = W.sum(axis=1)

    beta_hat = (W * D).sum(axis=1) / Wsum
    likelihood_var = 1.0 / Wsum

    logbf = directional_log_bf_normal(
        beta_hat=beta_hat,
        likelihood_var=likelihood_var,
        prior_sd=BETA_PRIOR_SD,
        direction=direction,
    )

    success = logbf >= math.log(bf_threshold)

    p = float(success.mean())
    mcse = math.sqrt(p * (1.0 - p) / n_sim)

    return p, mcse


def find_fast_required_N(
    beta_true,
    sigma_slope,
    sigma_day,
    sigma_resid,
    profiles,
    direction,
    seed,
    bf_threshold,
):
    first = None

    for N in range(N_MIN, N_MAX + 1):
        rng = np.random.default_rng(seed + N)

        p, _ = fast_success_probability(
            N=N,
            beta_true=beta_true,
            sigma_slope=sigma_slope,
            sigma_day=sigma_day,
            sigma_resid=sigma_resid,
            profiles=profiles,
            direction=direction,
            n_sim=FAST_N_SIM_SCAN,
            rng=rng,
            bf_threshold=bf_threshold,
        )

        if p >= TARGET_SUCCESS_PROB:
            first = N
            break

    if first is None:
        return None, np.nan, np.nan

    lo = max(N_MIN, first - 3)
    hi = min(N_MAX, first + 3)

    for N in range(lo, hi + 1):
        rng = np.random.default_rng(seed + 100000 + N)

        p, mcse = fast_success_probability(
            N=N,
            beta_true=beta_true,
            sigma_slope=sigma_slope,
            sigma_day=sigma_day,
            sigma_resid=sigma_resid,
            profiles=profiles,
            direction=direction,
            n_sim=FAST_N_SIM_REFINE,
            rng=rng,
            bf_threshold=bf_threshold,
        )

        if p >= TARGET_SUCCESS_PROB:
            return N, p, mcse

    return first, np.nan, np.nan


# =============================================================================
# Synthetic epoch generation: Exact final data-generating model
# =============================================================================

def bootstrap_profiles_for_N(profiles, N, rng):
    """
    Whole-participant bootstrap.
    A sampled participant contributes all of their valid day profiles.
    """
    keys = list(profiles.keys())
    chosen = rng.choice(keys, size=N, replace=True)

    sampled = []

    for new_pid, source_pid in enumerate(chosen, start=1):
        p = profiles[source_pid].copy()
        p["SimParticipant"] = new_pid
        p["SourceProfile"] = source_pid
        sampled.append(p)

    return pd.concat(sampled, ignore_index=True)


def simulate_exact_epoch_dataset(
    profiles,
    N,
    beta_true,
    sigma_participant_intercept,
    sigma_participant_slope,
    sigma_day,
    sigma_resid,
    rng,
):
    """
    Generate every epoch from:

        y = beta*x
            + participant_intercept
            + participant_slope*x
            + participant_day_intercept
            + residual

    alpha_true is set to 0 because the outcome is fixed-centered by the pilot
    center. This does not affect the State effect.
    """

    sampled_days = bootstrap_profiles_for_N(
        profiles=profiles,
        N=N,
        rng=rng,
    )

    rows = []

    # Participant random effects
    u0 = rng.normal(
        0.0,
        sigma_participant_intercept,
        size=N,
    )
    u1 = rng.normal(
        0.0,
        sigma_participant_slope,
        size=N,
    )

    for _, row in sampled_days.iterrows():
        pid = int(row["SimParticipant"])
        expday = int(row["ExpDay"])

        day_re = rng.normal(0.0, sigma_day)

        for state, n_obs, x in [
            ("experience", int(row["experience"]), +0.5),
            ("no-experience", int(row["no-experience"]), -0.5),
        ]:
            mu = (
                beta_true * x
                + u0[pid - 1]
                + u1[pid - 1] * x
                + day_re
            )

            y = rng.normal(
                loc=mu,
                scale=sigma_resid,
                size=n_obs,
            )

            for value in y:
                rows.append({
                    "Participant": pid,
                    "ExpDay": expday,
                    "ParticipantDay": f"{pid}_day_{expday}",
                    "State": state,
                    "State_c": x,
                    "y_scaled": float(value),
                })

    return pd.DataFrame(rows)


# =============================================================================
# Savage-Dickey BF from full PyMC fit
# =============================================================================

def directional_bf_from_idata(idata, direction):
    beta = (
        idata.posterior["beta_state"]
        .values
        .reshape(-1)
    )

    prior_density_0 = norm.pdf(
        0.0, loc=0.0, scale=BETA_PRIOR_SD
    )

    kde = gaussian_kde(beta)
    posterior_density_0 = float(
        kde.evaluate([0.0])[0]
    )

    bf_two = prior_density_0 / posterior_density_0

    if direction == "negative":
        p_dir = float(np.mean(beta < 0))
    else:
        p_dir = float(np.mean(beta > 0))

    bf_dir = 2.0 * bf_two * p_dir

    return bf_dir, p_dir


# =============================================================================
# Full PyMC validation
# =============================================================================

def run_one_full_sim(
    sim_index,
    N,
    beta_true,
    sigma_participant_intercept,
    sigma_participant_slope,
    sigma_day,
    sigma_resid,
    profiles,
    direction,
    seed,
):
    rng = np.random.default_rng(seed)

    dat = simulate_exact_epoch_dataset(
        profiles=profiles,
        N=N,
        beta_true=beta_true,
        sigma_participant_intercept=sigma_participant_intercept,
        sigma_participant_slope=sigma_participant_slope,
        sigma_day=sigma_day,
        sigma_resid=sigma_resid,
        rng=rng,
    )

    idata = build_and_sample_final_model(
        dat=dat,
        draws=FULL_DRAWS,
        tune=FULL_TUNE,
        chains=FULL_CHAINS,
        target_accept=FULL_TARGET_ACCEPT,
        random_seed=seed + 999,
        progressbar=False,
    )

    bf_dir, p_dir = directional_bf_from_idata(
        idata, direction
    )

    summary = az.summary(
        idata,
        var_names=["beta_state"],
    )

    rhat = float(summary.loc["beta_state", "r_hat"])
    ess = float(summary.loc["beta_state", "ess_bulk"])

    divergences = int(
        idata.sample_stats["diverging"].values.sum()
    )

    return {
        "sim_index": sim_index,
        "N": N,
        "n_epochs": len(dat),
        "BF_direction_vs_H0": bf_dir,
        "posterior_probability_predicted_direction": p_dir,
        # BF is the threshold-independent result; success is derived at summary time.
        "rhat_beta": rhat,
        "ess_bulk_beta": ess,
        "divergences": divergences,
    }


def should_full_validate(attenuation, slope_mult):
    if FULL_VALIDATE_ALL_SCENARIOS:
        return True

    return (
        attenuation in FULL_EFFECT_ATTENUATIONS
        and slope_mult in FULL_SLOPE_MULTIPLIERS
    )


# =============================================================================
# Master RAW file: safe one-time legacy import and atomic saving
# =============================================================================

RAW_KEY = ["hypothesis", "effect_attenuation", "slope_multiplier", "N", "sim_index"]


def save_master_raw(df):
    """Atomically replace the one master CSV, keeping the previous version until ready."""
    temp = FULL_RAW_OUT.with_name(FULL_RAW_OUT.name + ".tmp")
    df.to_csv(temp, index=False)
    os.replace(temp, FULL_RAW_OUT)


def load_master_raw():
    """Use existing master or consolidate legacy files once. Never mix different fits silently."""
    if FULL_RAW_OUT.exists() and FULL_RAW_OUT.stat().st_size > 0:
        return pd.read_csv(FULL_RAW_OUT)

    frames = []
    for path in LEGACY_RAW_FILES:
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            part = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if len(part):
            frames.append(part)
            print(f"Importing {len(part)} rows from {path}")

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    if not set(RAW_KEY + ["BF_direction_vs_H0"]).issubset(merged.columns):
        raise ValueError("Existing RAW files are missing required key/BF columns")

    # Duplicated keys represent the SAME simulation only if BF/true effect agree.
    for _, g in merged.groupby(RAW_KEY, dropna=False):
        if len(g) <= 1:
            continue
        for col in ["BF_direction_vs_H0", "beta_true_scaled"]:
            v = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy(float)
            if len(v) > 1 and not np.allclose(v, v[0], rtol=1e-7, atol=1e-10):
                raise ValueError(
                    "Conflicting simulations share the same key. "
                    f"Do not merge different generation settings: {g[RAW_KEY].iloc[0].to_dict()}, "
                    f"column={col}"
                )

    # On duplicate keys, favor records with a calculated BF over failed records.
    merged["_bf_valid"] = merged["BF_direction_vs_H0"].notna().astype(int)
    merged = merged.sort_values("_bf_valid").drop_duplicates(RAW_KEY, keep="last")
    merged = merged.drop(columns="_bf_valid").reset_index(drop=True)

    # Old threshold-specific success columns are NOT authoritative in the master.
    old_success = [c for c in merged if c.startswith("success_BF")]
    merged = merged.drop(columns=old_success)
    save_master_raw(merged)
    print(f"Created {FULL_RAW_OUT}: {len(merged)} unique simulations")
    return merged


# =============================================================================
# Main
# =============================================================================



def choose_candidate_ns(fast_rows_for_scenario):
    """Return N candidates around both BF3/BF10 fast-screen crossings.

    Note: short-window mode cannot establish the absolute minimum N.
    """
    if EXPLICIT_CANDIDATE_NS is not None:
        candidates = sorted({int(n) for n in EXPLICIT_CANDIDATE_NS})
        if any(n < N_MIN or n > N_MAX for n in candidates):
            raise ValueError("EXPLICIT_CANDIDATE_NS contains N outside N_MIN..N_MAX")
        return candidates

    crossings = [
        int(r["fast_required_N"])
        for r in fast_rows_for_scenario
        if pd.notna(r["fast_required_N"])
    ]
    if not crossings:
        warnings.warn("No fast-screen crossing found for this scenario; "
                      "set EXPLICIT_CANDIDATE_NS to test specific Ns.")
        return []

    lo = max(N_MIN, min(crossings) - FULL_N_BELOW)
    hi = min(N_MAX, max(crossings) + FULL_N_ABOVE)
    if FULL_N_MODE == "exhaustive":
        lo = N_MIN
    elif FULL_N_MODE != "auto_window":
        raise ValueError("FULL_N_MODE must be 'auto_window' or 'exhaustive'")
    return list(range(lo, hi + 1))


def summarize_master_raw(full_raw, minimum_repetitions):
    """Aggregate same BF samples at BOTH thresholds, retaining failed counts."""
    if full_raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    required = set(RAW_KEY + ["BF_direction_vs_H0", "outcome", "direction"])
    missing = required - set(full_raw.columns)
    if missing:
        raise ValueError(f"Master RAW lacks columns: {sorted(missing)}")
    if full_raw.duplicated(RAW_KEY).any():
        raise ValueError("Duplicate RAW simulation keys: inspect master before summarizing")

    data = full_raw.copy()
    data["BF_direction_vs_H0"] = pd.to_numeric(
        data["BF_direction_vs_H0"], errors="coerce"
    )
    # non-finite BF estimates cannot be classified reliably.
    data.loc[~np.isfinite(data["BF_direction_vs_H0"]), "BF_direction_vs_H0"] = np.nan

    group_cols = ["hypothesis", "outcome", "direction",
                  "effect_attenuation", "slope_multiplier", "N"]
    rows = []
    for keys, g in data.groupby(group_cols, dropna=False):
        v = g.loc[g["BF_direction_vs_H0"].notna()].copy()
        n_total, n = len(g), len(v)
        for th in BF_THRESHOLDS:
            k = int((v["BF_direction_vs_H0"] >= th).sum())
            p = k / n if n else np.nan
            mcse = math.sqrt(p * (1 - p) / n) if n else np.nan
            rows.append({
                **dict(zip(group_cols, keys)),
                "BF_threshold": th,
                "n_total_simulations": n_total,
                "n_valid_simulations": n,
                "n_failed_or_invalid": n_total - n,
                "n_BF_successes": k,
                "P_BF_full_PyMC": p,
                "MCSE_full_PyMC": mcse,
                "mean_rhat_beta": float(v["rhat_beta"].mean()) if n and "rhat_beta" in v else np.nan,
                "max_rhat_beta": float(v["rhat_beta"].max()) if n and "rhat_beta" in v else np.nan,
                "min_ess_bulk_beta": float(v["ess_bulk_beta"].min()) if n and "ess_bulk_beta" in v else np.nan,
                "total_divergences": float(v["divergences"].sum()) if n and "divergences" in v else np.nan,
                "meets_80pct": bool(n >= minimum_repetitions and p >= TARGET_SUCCESS_PROB),
                "enough_repetitions": bool(n >= minimum_repetitions),
            })

    summary = pd.DataFrame(rows).sort_values(
        ["hypothesis", "effect_attenuation", "slope_multiplier", "BF_threshold", "N"]
    ).reset_index(drop=True)

    # First passing N among the Ns with enough repetitions. If any smaller N
    # was not fully simulated, we DO NOT call it a proven global minimum.
    minimum_rows = []
    scenario = ["hypothesis", "effect_attenuation", "slope_multiplier", "BF_threshold"]
    for keys, g in summary.groupby(scenario):
        good = g.loc[g["meets_80pct"]].sort_values("N")
        if good.empty:
            n_candidate = np.nan
            missing_lower = np.nan
            status = "No qualifying N among tested Ns"
        else:
            n_candidate = int(good.iloc[0]["N"])
            all_lower = set(range(N_MIN, n_candidate))
            sufficiently_tested = set(
                g.loc[(g["N"] < n_candidate) & g["enough_repetitions"], "N"].astype(int)
            )
            missing_lower = len(all_lower - sufficiently_tested)
            status = ("All lower Ns tested" if missing_lower == 0
                      else "PROVISIONAL: smaller Ns not fully tested")
        minimum_rows.append({
            **dict(zip(scenario, keys)),
            "first_N_meeting_80pct_among_tested": n_candidate,
            "number_of_lower_Ns_not_fully_tested": missing_lower,
            "status": status,
            "minimum_valid_simulations_per_N": minimum_repetitions,
        })
    #minimum = pd.DataFrame(minimum_rows)
    #return summary, minimum
    
    minimum = pd.DataFrame(minimum_rows)
    minimum = minimum.drop(columns=["status"], errors="ignore")
    return summary, minimum


def main():
    if not RUN_SIMULATIONS:
        full_raw = load_master_raw()
        summary, minimum = summarize_master_raw(
            full_raw, minimum_repetitions=FULL_N_SIM
        )
        if summary.empty:
            print("No simulation results found in the master RAW file.")
            return
        summary.to_csv(FULL_SUMMARY_OUT, index=False)
        minimum.to_csv(MINIMUM_N_OUT, index=False)
        print("Existing RAW results grouped by BF >= 3 and BF >= 10:")
        print(summary.to_string(index=False))
        print("\nProvisional minimum Ns (among sufficiently sampled Ns):")
        print(minimum.to_string(index=False))
        print("\nSaved:", FULL_SUMMARY_OUT, MINIMUM_N_OUT)
        return

    roi, pdata = load_data()
    profiles = make_participant_profiles(pdata)
    print(f"Participant profiles: {pdata['Participant_ID'].nunique()} participants / "
          f"{len(pdata)} valid participant-days")

    pilot_pars = fit_pilot_nuisance_parameters(roi)
    print("\nPilot nuisance anchors\n", pilot_pars.to_string(index=False))

    # Stage A: retain the ORIGINAL scenario row order so that old sim seeds
    # (SEED + (row_index+1)*10_000_000 + N*10_000 + sim_index) remain reproducible.
    fast_rows = []
    for h_index, (h_name, h) in enumerate(PRIMARY_HYPOTHESES.items(), start=1):
        pp = pilot_pars.loc[pilot_pars["hypothesis"] == h_name].iloc[0]
        sigma_pi = float(pp["sigma_participant_intercept_anchor"])
        sigma_slope_anchor = float(pp["sigma_participant_slope_anchor"])
        sigma_day = float(pp["sigma_day_anchor"])
        sigma_resid = float(pp["sigma_resid_anchor"])
        siclari_dz = float(h["siclari_t"]) / math.sqrt(float(h["siclari_n"]))

        for attenuation in EFFECT_ATTENUATIONS:
            target_dz = siclari_dz * attenuation
            beta_true, reference_sd = calibrate_beta_from_external_dz(
                target_dz=target_dz, profiles=profiles,
                sigma_slope_reference=sigma_slope_anchor,
                sigma_day=sigma_day, sigma_resid=sigma_resid,
            )
            for slope_mult in SLOPE_HETEROGENEITY_MULTIPLIERS:
                sigma_slope_true = sigma_slope_anchor * slope_mult
                profile_vars = [
                    participant_aggregate_difference_variance(
                        p, sigma_slope_true, sigma_day, sigma_resid
                    ) for p in profiles.values()
                ]
                effective_dz = beta_true / math.sqrt(float(np.mean(profile_vars)))
                row = {
                    "hypothesis": h_name,
                    "outcome": h["column"],
                    "direction": h["direction"],
                    "siclari_dz": siclari_dz,
                    "effect_attenuation": attenuation,
                    "target_dz_at_1x_slope": target_dz,
                    "beta_true_scaled": beta_true,
                    "reference_participant_difference_SD": reference_sd,
                    "sigma_participant_intercept_true": sigma_pi,
                    "slope_multiplier": slope_mult,
                    "sigma_participant_slope_true": sigma_slope_true,
                    "sigma_day_true": sigma_day,
                    "sigma_resid_true": sigma_resid,
                    "effective_dz_this_scenario": effective_dz,
                }
                base_seed = (SEED + h_index * 1_000_000
                             + int(attenuation * 1000) * 100
                             + int(slope_mult * 100))
                for bf_threshold in BF_THRESHOLDS:
                    n_fast, p_fast, mcse_fast = find_fast_required_N(
                        beta_true=beta_true,
                        sigma_slope=sigma_slope_true,
                        sigma_day=sigma_day,
                        sigma_resid=sigma_resid,
                        profiles=profiles,
                        direction=h["direction"],
                        seed=base_seed,
                        bf_threshold=bf_threshold,
                    )
                    tag = str(int(bf_threshold))
                    row[f"fast_required_N_BF{tag}"] = n_fast
                    row[f"fast_P_BFge{tag}_at_N"] = p_fast
                    row[f"fast_MCSE_BF{tag}"] = mcse_fast
                fast_rows.append(row)

    fast_df = pd.DataFrame(fast_rows)
    fast_df.to_csv(FAST_OUT, index=False)
    print("\nStage A (FAST) - candidate crossings:")
    print(fast_df[["hypothesis", "effect_attenuation", "slope_multiplier",
                   "fast_required_N_BF3", "fast_required_N_BF10"]].to_string(index=False))

    # Stage B: use one BF value to evaluate BOTH thresholds; do not rerun based
    # on which threshold the user requests.
    full_raw = load_master_raw()
    for row_index, row in fast_df.iterrows():
        attenuation = float(row["effect_attenuation"])
        slope_mult = float(row["slope_multiplier"])
        if not should_full_validate(attenuation, slope_mult):
            continue
        h_name = row["hypothesis"]
        direction = row["direction"]
        crossings = [
            {"fast_required_N": row.get(f"fast_required_N_BF{int(th)}", np.nan)}
            for th in BF_THRESHOLDS
        ]
        candidate_ns = choose_candidate_ns(crossings)
        print(f"\nStage B: {h_name}, candidate N={candidate_ns}, "
              f"target repeats per N={FULL_N_SIM}")

        for N in candidate_ns:
            if len(full_raw):
                mask = (
                    (full_raw["hypothesis"] == h_name)
                    & np.isclose(full_raw["effect_attenuation"], attenuation)
                    & np.isclose(full_raw["slope_multiplier"], slope_mult)
                    & (full_raw["N"] == N)
                )
                old = full_raw.loc[mask]
                old_betas = pd.to_numeric(old["beta_true_scaled"], errors="coerce").dropna()
                if len(old_betas) and not np.allclose(
                    old_betas.to_numpy(float), float(row["beta_true_scaled"]),
                    rtol=1e-7, atol=1e-10,
                ):
                    raise ValueError(
                        f"{h_name} N={N}: historical beta_true_scaled differs! "
                        "Do not mix distinct pilot calibration settings."
                    )
                completed = set(old.loc[
                    np.isfinite(pd.to_numeric(old["BF_direction_vs_H0"], errors="coerce")),
                    "sim_index",
                ].astype(int))
            else:
                completed = set()

            for sim_index in range(FULL_N_SIM):
                if sim_index in completed:
                    continue
                # Keep the original master script seed scheme unchanged.
                seed = SEED + (row_index + 1) * 10_000_000 + N * 10_000 + sim_index
                print(f"FULL PyMC: {h_name}, N={N}, sim={sim_index+1}/{FULL_N_SIM}")
                try:
                    res = run_one_full_sim(
                        sim_index=sim_index, N=N,
                        beta_true=float(row["beta_true_scaled"]),
                        sigma_participant_intercept=float(row["sigma_participant_intercept_true"]),
                        sigma_participant_slope=float(row["sigma_participant_slope_true"]),
                        sigma_day=float(row["sigma_day_true"]),
                        sigma_resid=float(row["sigma_resid_true"]),
                        profiles=profiles, direction=direction, seed=seed,
                    )
                except Exception as exc:
                    warnings.warn(f"Simulation failed for {h_name}, N={N}, sim={sim_index}: {exc}")
                    res = {
                        "sim_index": sim_index, "N": N, "n_epochs": np.nan,
                        "BF_direction_vs_H0": np.nan,
                        "posterior_probability_predicted_direction": np.nan,
                        "rhat_beta": np.nan, "ess_bulk_beta": np.nan,
                        "divergences": np.nan,
                        "error": str(exc),
                    }
                res.update({
                    "hypothesis": h_name,
                    "outcome": row["outcome"],
                    "direction": direction,
                    "effect_attenuation": attenuation,
                    "slope_multiplier": slope_mult,
                    "beta_true_scaled": float(row["beta_true_scaled"]),
                    "sigma_participant_intercept_true": float(row["sigma_participant_intercept_true"]),
                    "sigma_participant_slope_true": float(row["sigma_participant_slope_true"]),
                    "sigma_day_true": float(row["sigma_day_true"]),
                    "sigma_resid_true": float(row["sigma_resid_true"]),
                    "full_chains": FULL_CHAINS,
                    "full_tune": FULL_TUNE,
                    "full_draws": FULL_DRAWS,
                    "full_target_accept": FULL_TARGET_ACCEPT,
                })
                if len(full_raw):
                    same = (
                        (full_raw["hypothesis"] == h_name)
                        & np.isclose(full_raw["effect_attenuation"], attenuation)
                        & np.isclose(full_raw["slope_multiplier"], slope_mult)
                        & (full_raw["N"] == N)
                        & (full_raw["sim_index"] == sim_index)
                    )
                    full_raw = full_raw.loc[~same].copy()
                full_raw = pd.concat([full_raw, pd.DataFrame([res])], ignore_index=True)
                save_master_raw(full_raw)  # atomic replacement after EACH simulation

    summary, minimum = summarize_master_raw(full_raw, minimum_repetitions=FULL_N_SIM)
    if len(summary):
        summary.to_csv(FULL_SUMMARY_OUT, index=False)
        minimum.to_csv(MINIMUM_N_OUT, index=False)
        print("\nStage B full PyMC BF >= 3 and BF >= 10:\n",
              summary.to_string(index=False))
        print("\nFirst N meeting 80% among N values sufficiently tested:\n",
              minimum.to_string(index=False))
        print("\nNote: A candidate is provisional unless all smaller Ns from N_MIN "
              "were also tested with enough repetitions. Near 80%, use 500+ sims.")

    print("\nSaved:\n", FAST_OUT, "\n", FULL_RAW_OUT,
          "\n", FULL_SUMMARY_OUT, "\n", MINIMUM_N_OUT)


if __name__ == "__main__":
    main()
