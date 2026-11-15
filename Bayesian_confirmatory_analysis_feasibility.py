"""
Confirmatory Bayesian linear mixed-effects analysis for an MEG Study of the Neural Correlates of Consciousness

Model:
    y_scaled ~ State
               + (1 + State || Participant)
               + (1 | Participant:Day)

State coding:
    no-experience = -0.5
    experience    = +0.5

Thus:
    beta_state = E[y_scaled | Experience] - E[y_scaled | No-experience]

Fixed pilot-based scaling:
    y_scaled = (y_raw - pilot_center) / pilot_residual_sd

pilot_center and pilot_residual_sd are estimated ONLY from pilot participants
P5/P11. The confirmatory sample's own mean/SD are never used for scaling.

Primary prior:
    beta_state ~ Normal(0, 0.5^2)

Sensitivity priors:
    beta_state ~ Normal(0, 0.35^2)
    beta_state ~ Normal(0, 0.75^2)
    beta_state ~ Normal(0, 1.00^2)

Other priors:
    alpha                       ~ Normal(0, 1^2)
    sigma_participant_intercept ~ HalfNormal(0.5)
    sigma_participant_slope     ~ HalfNormal(0.5)
    sigma_day                   ~ HalfNormal(0.5)
    sigma_resid                 ~ HalfNormal(1.0)

Directional Bayes factors:
    BF_-0 = 2 * BF10_two_sided * P(beta_state < 0 | data)
    BF_+0 = 2 * BF10_two_sided * P(beta_state > 0 | data)

Input:
    ROIpower_by_epoch_pilot.csv

Required columns:
    Participant, ExpDay, State, Posterior_delta, Posterior_gamma

Optional secondary outcomes:
    Frontal_delta, Frontal_gamma

Only Participant x ExpDay cells containing BOTH experience and no-experience
are retained in the confirmatory analysis.
"""

from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm

import pymc as pm
import arviz as az


# =============================================================================
# Settings
# =============================================================================

DATA_FILE = Path("ROIpower_by_epoch_pilot.csv")

PILOT_PARTICIPANTS = [5, 11]

# For feasibility
CONFIRMATORY_PARTICIPANTS = [5, 11]

# Settings for the confirmatory analysis
# None = use all non-pilot participants.
# Or specify IDs explicitly, e.g.:
#CONFIRMATORY_PARTICIPANTS = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12]


# Expected count is checked with a warning, not an analysis-stopping error.
EXPECTED_CONFIRMATORY_N = 10

HYPOTHESES = {
    "H1_posterior_delta": {
        "column": "Posterior_delta",
        "direction": "negative",
        "region": "posterior",
        "band": "delta",
        "primary": True,
    },
    "H2_posterior_gamma": {
        "column": "Posterior_gamma",
        "direction": "positive",
        "region": "posterior",
        "band": "low-gamma",
        "primary": True,
    },
    "H3_frontal_delta": {
        "column": "Frontal_delta",
        "direction": "negative",
        "region": "frontal",
        "band": "delta",
        "primary": False,
    },
    "H4_frontal_gamma": {
        "column": "Frontal_gamma",
        "direction": "positive",
        "region": "frontal",
        "band": "low-gamma",
        "primary": False,
    },
}

PRIMARY_BETA_PRIOR_SD = 0.50
SENSITIVITY_BETA_PRIOR_SDS = [0.35, 0.75, 1.00]

ALPHA_PRIOR_SD = 1.00
PARTICIPANT_INTERCEPT_SD_PRIOR = 0.50
PARTICIPANT_SLOPE_SD_PRIOR = 0.50
DAY_SD_PRIOR = 0.50
RESIDUAL_SD_PRIOR = 1.00

CHAINS = 4
CORES = 4
TUNE = 6000
DRAWS = 6000
TARGET_ACCEPT = 0.99

RHAT_MAX = 1.01
ESS_BULK_MIN = 400
ESS_TAIL_MIN = 400
MAX_DIVERGENCES = 0

SEED = 20260923

SCALING_OUT = Path("feasibility_pilot_scaling.csv")
VALID_COUNTS_OUT = Path("feasibility_valid_day_counts.csv")
OUTCOME_COUNTS_OUT = Path("feasibility_outcome_sample_counts.csv")
PRIMARY_RESULTS_OUT = Path("feasibility_primary_results.csv")
SENSITIVITY_RESULTS_OUT = Path("feasibility_sensitivity_results.csv")

TRACE_TEMPLATE = "feasibility_{hypothesis}_priorSD{prior_sd}_{fit_label}.nc"
PLOT_TEMPLATE = "feasibility_{hypothesis}_priorSD{prior_sd}_{fit_label}.png"


# =============================================================================
# Data loading and valid-day filtering
# =============================================================================

def load_data():
    df = pd.read_csv(DATA_FILE)

    required = {"Participant", "ExpDay", "State"}
    required |= {
        h["column"] for h in HYPOTHESES.values() if h["primary"]
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Keep missing values as missing instead of converting NaN to the text "nan".
    # Empty / whitespace-only fields are also considered missing.  Rows without
    # a known State cannot contribute to the Experience vs No-experience model.
    df["State"] = (
        df["State"].astype("string").str.strip().str.lower()
    )
    missing_state = (
        df["State"].isna()
        | df["State"].eq("").fillna(False)
    )
    n_missing_state = int(missing_state.sum())
    if n_missing_state:
        warnings.warn(
            f"Excluding {n_missing_state} rows with missing/blank State "
            "before selecting pilot and confirmatory data."
        )
        df = df.loc[~missing_state].copy()

    # Preserve validation: an unexpected non-missing label is an input error,
    # not a reason to silently drop an observation.
    allowed = {"experience", "no-experience"}
    unexpected = set(df["State"].unique()) - allowed
    if unexpected:
        raise ValueError(f"Unexpected State values: {sorted(unexpected)}")

    pilot = df[df["Participant"].isin(PILOT_PARTICIPANTS)].copy()
    if pilot.empty:
        raise ValueError(
            f"No pilot data found for participants {PILOT_PARTICIPANTS}."
        )

    if CONFIRMATORY_PARTICIPANTS is None:
        confirm = df[
            ~df["Participant"].isin(PILOT_PARTICIPANTS)
        ].copy()
    else:
        # overlap = set(CONFIRMATORY_PARTICIPANTS) & set(PILOT_PARTICIPANTS)
        # if overlap:
        #     raise ValueError(
        #         f"Confirmatory participants overlap with pilot: {sorted(overlap)}"
        #     )

        confirm = df[
            df["Participant"].isin(CONFIRMATORY_PARTICIPANTS)
        ].copy()

        found = set(confirm["Participant"].unique())
        missing_ids = set(CONFIRMATORY_PARTICIPANTS) - found
        if missing_ids:
            warnings.warn(
                "No rows found for these preregistered confirmatory "
                f"participants: {sorted(missing_ids)}. "
                "They will not be replaced; their absence will be recorded "
                "in the outcome-specific sample-count report."
            )

    if confirm.empty:
        raise ValueError("No confirmatory data remain after participant filtering.")

    # Confirmatory inclusion rule:
    # retain only Participant x ExpDay cells containing BOTH states.
    day_counts = (
        confirm
        .groupby(["Participant", "ExpDay", "State"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["experience", "no-experience"]:
        if col not in day_counts.columns:
            day_counts[col] = 0

    day_counts["valid_day"] = (
        (day_counts["experience"] > 0)
        & (day_counts["no-experience"] > 0)
    )

    day_counts.to_csv(VALID_COUNTS_OUT, index=False)

    valid_keys = day_counts.loc[
        day_counts["valid_day"], ["Participant", "ExpDay"]
    ]

    confirm = confirm.merge(
        valid_keys,
        on=["Participant", "ExpDay"],
        how="inner",
    )

    n_confirm = confirm["Participant"].nunique()

    if EXPECTED_CONFIRMATORY_N is not None and n_confirm != EXPECTED_CONFIRMATORY_N:
        warnings.warn(
            f"Expected {EXPECTED_CONFIRMATORY_N} confirmatory participants "
            f"after valid-day filtering, but found {n_confirm}. "
            "The analysis will proceed with the available participants; "
            "actual N will be recorded separately for each outcome."
        )

    print("\nConfirmatory valid-day epoch counts")
    print(day_counts.loc[day_counts["valid_day"]].to_string(index=False))
    print(f"\nConfirmatory participants: {n_confirm}")
    print(
        "Valid participant-days: "
        f"{confirm[['Participant', 'ExpDay']].drop_duplicates().shape[0]}"
    )

    return pilot, confirm


# =============================================================================
# Fixed pilot scaling
# =============================================================================

def pilot_fixed_scale(pilot, outcome):
    """
    pilot_center:
        pilot grand mean.

    pilot_residual_sd:
        pooled within Participant x ExpDay x State residual SD.
    """
    dat = pilot[
        ["Participant", "ExpDay", "State", outcome]
    ].dropna().copy()

    if dat.empty:
        raise ValueError(f"No pilot observations for {outcome}.")

    center = float(dat[outcome].mean())

    cell = ["Participant", "ExpDay", "State"]
    residual = (
        dat[outcome]
        - dat.groupby(cell)[outcome].transform("mean")
    )

    n_cells = dat.groupby(cell).ngroups
    df_resid = len(residual) - n_cells
    if df_resid <= 0:
        raise ValueError(
            f"Insufficient pilot residual degrees of freedom for {outcome}."
        )

    residual_sd = math.sqrt(
        float(np.sum(np.square(residual))) / df_resid
    )

    if not np.isfinite(residual_sd) or residual_sd <= 0:
        raise ValueError(
            f"Invalid pilot residual SD for {outcome}: {residual_sd}"
        )

    return center, residual_sd, len(dat), n_cells, df_resid


def compute_and_save_scaling(pilot):
    rows = []

    for h_name, h in HYPOTHESES.items():
        outcome = h["column"]

        if outcome not in pilot.columns:
            if h["primary"]:
                raise ValueError(f"Required primary outcome missing: {outcome}")
            warnings.warn(
                f"Skipping {h_name}: optional pilot column {outcome} not found."
            )
            continue

        center, residual_sd, n_obs, n_cells, df_resid = (
            pilot_fixed_scale(pilot, outcome)
        )

        rows.append({
            "hypothesis": h_name,
            "outcome": outcome,
            "primary": h["primary"],
            "direction": h["direction"],
            "pilot_center_raw": center,
            "pilot_residual_sd_raw_fixed": residual_sd,
            "pilot_n_observations": n_obs,
            "pilot_n_participant_day_state_cells": n_cells,
            "pilot_residual_df": df_resid,
            "primary_beta_prior_SD_scaled": PRIMARY_BETA_PRIOR_SD,
            "primary_beta_prior_SD_raw":
                PRIMARY_BETA_PRIOR_SD * residual_sd,
        })

    out = pd.DataFrame(rows)
    out.to_csv(SCALING_OUT, index=False)

    print("\nFixed pilot scaling")
    print(out.to_string(index=False))

    return out


# =============================================================================
# Prepare confirmatory outcome
# =============================================================================

def prepare_confirmatory_outcome(
    confirm,
    outcome,
    pilot_center,
    pilot_residual_sd,
):
    dat = confirm[
        ["Participant", "ExpDay", "State", outcome]
    ].dropna().copy()

    # Reapply the BOTH-states rule after outcome-specific neural-data QC.
    # A participant-day can become invalid when all epochs for one State are
    # missing for this particular outcome, even if its interview data are valid.
    if not dat.empty:
        outcome_day_counts = (
            dat.groupby(["Participant", "ExpDay", "State"])
            .size().unstack(fill_value=0)
        )
        for state in ("experience", "no-experience"):
            if state not in outcome_day_counts.columns:
                outcome_day_counts[state] = 0
        valid_outcome_days = outcome_day_counts.index[
            (outcome_day_counts["experience"] > 0)
            & (outcome_day_counts["no-experience"] > 0)
        ]
        keys = pd.MultiIndex.from_frame(dat[["Participant", "ExpDay"]])
        dat = dat.loc[keys.isin(valid_outcome_days)].copy()

    if dat.empty:
        return dat  # main() warns, records N=0 and skips an impossible model fit.

    dat["state_c"] = dat["State"].map({
        "no-experience": -0.5,
        "experience": +0.5,
    }).astype(float)

    dat["ParticipantDay"] = (
        dat["Participant"].astype(str)
        + "_day_"
        + dat["ExpDay"].astype(str)
    )

    # External scaling fixed exclusively from pilot data.
    dat["y_scaled"] = (
        dat[outcome] - pilot_center
    ) / pilot_residual_sd

    return dat


# =============================================================================
# Final Pymc model
# =============================================================================

def build_and_sample_model(
    dat,
    beta_prior_sd,
    tune,
    draws,
    chains,
    target_accept,
    random_seed,
    progressbar=True,
):
    dat = dat.reset_index(drop=True).copy()

    p_codes, p_levels = pd.factorize(
        dat["Participant"], sort=True
    )
    d_codes, d_levels = pd.factorize(
        dat["ParticipantDay"], sort=True
    )

    x = dat["state_c"].to_numpy(float)
    y = dat["y_scaled"].to_numpy(float)

    coords = {
        "participant": [str(v) for v in p_levels],
        "participant_day": [str(v) for v in d_levels],
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
            sigma=beta_prior_sd,
        )

        # Participant random intercept (non-centered).
        sigma_participant_intercept = pm.HalfNormal(
            "sigma_participant_intercept",
            sigma=PARTICIPANT_INTERCEPT_SD_PRIOR,
        )
        z_pi = pm.Normal(
            "z_participant_intercept",
            mu=0.0,
            sigma=1.0,
            dims="participant",
        )
        participant_intercept = pm.Deterministic(
            "participant_intercept",
            z_pi * sigma_participant_intercept,
            dims="participant",
        )

        # Participant random State slope, independent of intercept.
        sigma_participant_slope = pm.HalfNormal(
            "sigma_participant_slope",
            sigma=PARTICIPANT_SLOPE_SD_PRIOR,
        )
        z_ps = pm.Normal(
            "z_participant_slope",
            mu=0.0,
            sigma=1.0,
            dims="participant",
        )
        participant_slope = pm.Deterministic(
            "participant_slope",
            z_ps * sigma_participant_slope,
            dims="participant",
        )

        # Participant:Day random intercept (non-centered).
        sigma_day = pm.HalfNormal(
            "sigma_day",
            sigma=DAY_SD_PRIOR,
        )
        z_day = pm.Normal(
            "z_day",
            mu=0.0,
            sigma=1.0,
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
            cores=min(CORES, chains),
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=progressbar,
            compute_convergence_checks=True,
        )

    return idata


# =============================================================================
# Diagnotics
# =============================================================================

SCALAR_DIAGNOSTIC_VARS = [
    "alpha",
    "beta_state",
    "sigma_participant_intercept",
    "sigma_participant_slope",
    "sigma_day",
    "sigma_resid",
]


def model_diagnostics(idata):
    s = az.summary(
        idata,
        var_names=SCALAR_DIAGNOSTIC_VARS,
        hdi_prob=0.95,
    )

    max_rhat = (
        float(s["r_hat"].max())
        if "r_hat" in s.columns
        else np.nan
    )
    min_ess_bulk = (
        float(s["ess_bulk"].min())
        if "ess_bulk" in s.columns
        else np.nan
    )
    min_ess_tail = (
        float(s["ess_tail"].min())
        if "ess_tail" in s.columns
        else np.nan
    )

    divergences = int(
        idata.sample_stats["diverging"].values.sum()
    )

    rhat_ok = np.isfinite(max_rhat) and max_rhat < RHAT_MAX
    ess_bulk_ok = (
        np.isfinite(min_ess_bulk)
        and min_ess_bulk >= ESS_BULK_MIN
    )
    ess_tail_ok = (
        np.isfinite(min_ess_tail)
        and min_ess_tail >= ESS_TAIL_MIN
    )
    divergence_ok = divergences <= MAX_DIVERGENCES

    passed = all([
        rhat_ok,
        ess_bulk_ok,
        ess_tail_ok,
        divergence_ok,
    ])

    return {
        "diagnostics_passed": passed,
        "max_rhat_scalar": max_rhat,
        "min_ess_bulk_scalar": min_ess_bulk,
        "min_ess_tail_scalar": min_ess_tail,
        "divergences": divergences,
        "rhat_ok": rhat_ok,
        "ess_bulk_ok": ess_bulk_ok,
        "ess_tail_ok": ess_tail_ok,
        "divergence_ok": divergence_ok,
    }


# =============================================================================
# Posteruir synnarues / Savage-Dickey BF
# =============================================================================

def hdi_interval(samples, prob=0.95):
    x = np.asarray(samples, dtype=float).reshape(-1)
    h = np.asarray(
        az.hdi(x, hdi_prob=prob)
    ).reshape(-1)
    return float(h[0]), float(h[1])


def directional_savage_dickey(
    beta_samples,
    prior_sd,
    direction,
):
    beta = np.asarray(beta_samples, dtype=float).reshape(-1)

    prior_density_0 = float(
        norm.pdf(0.0, loc=0.0, scale=prior_sd)
    )

    if np.std(beta, ddof=1) <= 0:
        return {
            "prior_density_beta0": prior_density_0,
            "posterior_density_beta0_KDE": np.nan,
            "BF10_two_sided_SavageDickey_approx": np.nan,
            "P_preregistered_direction": np.nan,
            "BF_direction_vs_H0_SavageDickey_approx": np.nan,
        }

    kde = gaussian_kde(beta)
    posterior_density_0 = float(
        kde.evaluate([0.0])[0]
    )

    if (
        posterior_density_0 <= 0
        or not np.isfinite(posterior_density_0)
    ):
        bf_two = np.nan
    else:
        bf_two = prior_density_0 / posterior_density_0

    if direction == "negative":
        p_direction = float(np.mean(beta < 0))
    elif direction == "positive":
        p_direction = float(np.mean(beta > 0))
    else:
        raise ValueError(
            "direction must be 'negative' or 'positive'"
        )

    bf_direction = (
        2.0 * bf_two * p_direction
        if np.isfinite(bf_two)
        else np.nan
    )

    return {
        "prior_density_beta0": prior_density_0,
        "posterior_density_beta0_KDE": posterior_density_0,
        "BF10_two_sided_SavageDickey_approx": bf_two,
        "P_preregistered_direction": p_direction,
        "BF_direction_vs_H0_SavageDickey_approx": bf_direction,
    }


def summarize_fit(
    idata,
    dat,
    h_name,
    h,
    beta_prior_sd,
    pilot_center,
    pilot_residual_sd,
    fit_label,
    mcmc_settings,
):
    beta_scaled = (
        idata.posterior["beta_state"]
        .values
        .reshape(-1)
    )

    beta_raw = beta_scaled * pilot_residual_sd

    hdi_scaled = hdi_interval(beta_scaled, 0.95)
    hdi_raw = hdi_interval(beta_raw, 0.95)

    p_negative = float(np.mean(beta_scaled < 0))
    p_positive = float(np.mean(beta_scaled > 0))

    bf = directional_savage_dickey(
        beta_samples=beta_scaled,
        prior_sd=beta_prior_sd,
        direction=h["direction"],
    )

    diag = model_diagnostics(idata)

    result = {
        "hypothesis": h_name,
        "outcome": h["column"],
        "region": h["region"],
        "band": h["band"],
        "primary_hypothesis": h["primary"],
        "direction": h["direction"],
        "fit_label": fit_label,

        "n_participants": dat["Participant"].nunique(),
        "n_participant_days":
            dat[["Participant", "ExpDay"]].drop_duplicates().shape[0],
        "n_epochs": len(dat),
        "n_experience":
            int((dat["State"] == "experience").sum()),
        "n_no_experience":
            int((dat["State"] == "no-experience").sum()),

        "pilot_center_raw": pilot_center,
        "pilot_residual_sd_raw_fixed": pilot_residual_sd,

        "beta_prior_SD_scaled": beta_prior_sd,
        "beta_prior_SD_raw":
            beta_prior_sd * pilot_residual_sd,

        "beta_scaled_mean": float(np.mean(beta_scaled)),
        "beta_scaled_median": float(np.median(beta_scaled)),
        "beta_scaled_HDI95_low": hdi_scaled[0],
        "beta_scaled_HDI95_high": hdi_scaled[1],

        "beta_raw_mean_E_minus_NE": float(np.mean(beta_raw)),
        "beta_raw_median_E_minus_NE": float(np.median(beta_raw)),
        "beta_raw_HDI95_low": hdi_raw[0],
        "beta_raw_HDI95_high": hdi_raw[1],

        "P_beta_lt_0": p_negative,
        "P_beta_gt_0": p_positive,

        **bf,
        **diag,
        **mcmc_settings,
    }

    result["strong_directional_evidence_BFgt10"] = int(
        np.isfinite(
            result["BF_direction_vs_H0_SavageDickey_approx"]
        )
        and (
            result["BF_direction_vs_H0_SavageDickey_approx"]
            > 10.0
        )
    )

    return result


# =============================================================================
# Save trace / Plot
# =============================================================================

def safe_prior_tag(prior_sd):
    return str(prior_sd).replace(".", "p")


def save_trace_and_plot(
    idata,
    h_name,
    prior_sd,
    fit_label,
):
    prior_tag = safe_prior_tag(prior_sd)

    trace_path = Path(
        TRACE_TEMPLATE.format(
            hypothesis=h_name,
            prior_sd=prior_tag,
            fit_label=fit_label,
        )
    )

    try:
        idata.to_netcdf(trace_path)
    except Exception as exc:
        warnings.warn(
            f"Could not save NetCDF trace {trace_path}: {exc}"
        )

    try:
        import matplotlib.pyplot as plt

        az.plot_posterior(
            idata,
            var_names=["beta_state"],
            hdi_prob=0.95,
            ref_val=0,
        )
        plt.suptitle(
            f"{h_name}: beta_state = Experience - No-experience "
            f"(pilot-scaled; prior SD={prior_sd})",
            y=1.02,
        )
        plt.tight_layout()

        plot_path = Path(
            PLOT_TEMPLATE.format(
                hypothesis=h_name,
                prior_sd=prior_tag,
                fit_label=fit_label,
            )
        )
        plt.savefig(
            plot_path,
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    except Exception as exc:
        warnings.warn(
            f"Could not save posterior plot for {h_name}: {exc}"
        )


# =============================================================================
# Main
# =============================================================================

def main():
    pilot, confirm = load_data()
    scaling = compute_and_save_scaling(pilot)

    primary_rows = []
    sensitivity_rows = []
    outcome_count_rows = []

    for h_index, (h_name, h) in enumerate(
        HYPOTHESES.items(),
        start=1,
    ):
        outcome = h["column"]

        if outcome not in confirm.columns:
            if h["primary"]:
                raise ValueError(
                    f"Required primary outcome missing: {outcome}"
                )
            warnings.warn(
                f"Skipping secondary analysis {h_name}: {outcome} not found."
            )
            continue

        sp = scaling.loc[
            scaling["hypothesis"] == h_name
        ]

        if sp.empty:
            warnings.warn(
                f"Skipping {h_name}: no pilot scaling information."
            )
            continue

        sp = sp.iloc[0]
        pilot_center = float(sp["pilot_center_raw"])
        pilot_residual_sd = float(
            sp["pilot_residual_sd_raw_fixed"]
        )

        dat = prepare_confirmatory_outcome(
            confirm=confirm,
            outcome=outcome,
            pilot_center=pilot_center,
            pilot_residual_sd=pilot_residual_sd,
        )

        # Record actual outcome-specific N *after* missing neural data and the
        # both-States-within-day inclusion rule have been applied.
        expected_ids = set(CONFIRMATORY_PARTICIPANTS or confirm["Participant"].unique())
        actual_ids = set(dat["Participant"].unique())
        missing_ids = sorted(expected_ids - actual_ids)
        if missing_ids:
            warnings.warn(
                f"{h_name}: {len(actual_ids)}/{len(expected_ids)} participants "
                f"have analyzable outcome data. Excluded/missing IDs: {missing_ids}. "
                "No replacement participants will be added."
            )
        outcome_count_rows.append({
            "hypothesis": h_name,
            "outcome": outcome,
            "n_preregistered_participants": len(expected_ids),
            "n_participants": len(actual_ids),
            "n_participant_days": dat[["Participant", "ExpDay"]].drop_duplicates().shape[0],
            "n_epochs": len(dat),
            "n_experience": int((dat["State"] == "experience").sum()),
            "n_no_experience": int((dat["State"] == "no-experience").sum()),
            "included_participant_ids": ",".join(map(str, sorted(actual_ids))),
            "excluded_or_missing_participant_ids": ",".join(map(str, missing_ids)),
        })
        pd.DataFrame(outcome_count_rows).to_csv(OUTCOME_COUNTS_OUT, index=False)
        if dat.empty:
            warnings.warn(f"{h_name}: No analyzable observations; model fit skipped.")
            continue

        # Primary preregistered prior.
        seed_primary = SEED + h_index * 1_000_000

        mcmc_settings = {
            "mcmc_chains": CHAINS,
            "mcmc_tune": TUNE,
            "mcmc_draws": DRAWS,
            "mcmc_target_accept": TARGET_ACCEPT,
        }

        idata = build_and_sample_model(
            dat=dat,
            beta_prior_sd=PRIMARY_BETA_PRIOR_SD,
            tune=TUNE,
            draws=DRAWS,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=seed_primary,
        )

        fit_label = "primary_sampling"

        diag = model_diagnostics(idata)
        if not diag["diagnostics_passed"]:
            warnings.warn(
                f"{h_name}: MCMC diagnostics failed: {diag}"
            )
        # A failed diagnostic triggers a warning, NOT a refit or an exception.
        # Its diagnostic flags and posterior samples are still saved below.

        primary_result = summarize_fit(
            idata=idata,
            dat=dat,
            h_name=h_name,
            h=h,
            beta_prior_sd=PRIMARY_BETA_PRIOR_SD,
            pilot_center=pilot_center,
            pilot_residual_sd=pilot_residual_sd,
            fit_label=fit_label,
            mcmc_settings=mcmc_settings,
        )

        primary_rows.append(primary_result)

        save_trace_and_plot(
            idata=idata,
            h_name=h_name,
            prior_sd=PRIMARY_BETA_PRIOR_SD,
            fit_label=fit_label,
        )

        # Save after each model, even when its MCMC diagnostics failed.
        pd.DataFrame(primary_rows).to_csv(
            PRIMARY_RESULTS_OUT,
            index=False,
        )

        # Prior-sensitivity analyses.
        for s_index, prior_sd in enumerate(
            SENSITIVITY_BETA_PRIOR_SDS,
            start=1,
        ):
            seed_s = (
                SEED
                + h_index * 1_000_000
                + s_index * 100_000
            )

            idata_s = build_and_sample_model(
                dat=dat,
                beta_prior_sd=prior_sd,
                tune=TUNE,
                draws=DRAWS,
                chains=CHAINS,
                target_accept=TARGET_ACCEPT,
                random_seed=seed_s,
            )

            fit_label_s = "primary_sampling"
            mcmc_settings_s = mcmc_settings.copy()

            diag_s = model_diagnostics(idata_s)
            if not diag_s["diagnostics_passed"]:
                warnings.warn(
                    f"{h_name}, prior SD={prior_sd}: "
                    f"MCMC diagnostics failed: {diag_s}"
                )
            # Save sensitivity results regardless of diagnostic outcome.
            sensitivity_result = summarize_fit(
                idata=idata_s,
                dat=dat,
                h_name=h_name,
                h=h,
                beta_prior_sd=prior_sd,
                pilot_center=pilot_center,
                pilot_residual_sd=pilot_residual_sd,
                fit_label=fit_label_s,
                mcmc_settings=mcmc_settings_s,
            )

            sensitivity_rows.append(sensitivity_result)

            save_trace_and_plot(
                idata=idata_s,
                h_name=h_name,
                prior_sd=prior_sd,
                fit_label=fit_label_s,
            )

            pd.DataFrame(sensitivity_rows).to_csv(
                SENSITIVITY_RESULTS_OUT,
                index=False,
            )

    primary_results = pd.DataFrame(primary_rows)
    sensitivity_results = pd.DataFrame(sensitivity_rows)

    if len(primary_results):
        cols = [
            "hypothesis",
            "outcome",
            "n_participants",
            "n_participant_days",
            "n_epochs",
            "beta_raw_mean_E_minus_NE",
            "beta_raw_HDI95_low",
            "beta_raw_HDI95_high",
            "P_preregistered_direction",
            "BF_direction_vs_H0_SavageDickey_approx",
            "strong_directional_evidence_BFgt10",
            "max_rhat_scalar",
            "min_ess_bulk_scalar",
            "min_ess_tail_scalar",
            "divergences",
            "fit_label",
        ]

        print("\n\n===== CONFIRMATORY PRIMARY-PRIOR RESULTS =====")
        print(primary_results[cols].to_string(index=False))

    if len(sensitivity_results):
        cols_s = [
            "hypothesis",
            "beta_prior_SD_scaled",
            "BF_direction_vs_H0_SavageDickey_approx",
            "P_preregistered_direction",
            "beta_raw_mean_E_minus_NE",
            "beta_raw_HDI95_low",
            "beta_raw_HDI95_high",
            "diagnostics_passed",
            "fit_label",
        ]

        print("\n\n===== PRIOR-SENSITIVITY RESULTS =====")
        print(
            sensitivity_results[cols_s].to_string(index=False)
        )

    print("\nSaved:")
    print(f"  {SCALING_OUT}")
    print(f"  {VALID_COUNTS_OUT}")
    print(f"  {OUTCOME_COUNTS_OUT}")
    print(f"  {PRIMARY_RESULTS_OUT}")
    print(f"  {SENSITIVITY_RESULTS_OUT}")

    print("\nInterpretation:")
    print(
        "  beta_state < 0 means Experience has lower power than No-experience."
    )
    print(
        "  beta_state > 0 means Experience has higher power than No-experience."
    )
    print(
        "  Preregistered conclusions use the primary beta prior SD = 0.50."
    )
    print(
        "  Sensitivity-prior fits are robustness analyses only."
    )
    print(
        "  Confirmatory data are never used to estimate the scaling center/SD."
    )


if __name__ == "__main__":
    main()
