"""
Bayesian single-participant random-effects analysis of pilot MEG ROI power.

Input:
    ROIpower_by_epoch_pilot.csv

Outcomes:
    Posterior_delta
    Posterior_gamma
    Frontal_delta
    Frontal_gamma

Model for a single participant:
    y_jd = alpha + beta * State_jd
           + u_day[d]
           + epsilon_jd

where State is coded:
    no-experience = -0.5
    experience    = +0.5

Thus:
    beta = mean(Experience) - mean(No-experience)

Random effect:
    experimental-day random intercept


Bayesian evidence:
------------------
The script reports:
  * posterior mean/median of beta
  * 95% HDI
  * P(beta < 0) and P(beta > 0)
  * directional posterior probability for the preregistered direction
      delta: beta < 0
      gamma: beta > 0
  * approximate Savage-Dickey BF10 for beta != 0 vs beta = 0
  * approximate directional BF for the preregistered direction vs beta = 0

The BF is prior-sensitive. The default beta prior is Normal(0, 0.5) on the
outcome-z scale. Run prior sensitivity analyses before using BF values as a
confirmatory criterion.

Required packages:
    pip install -U pymc arviz pandas numpy scipy matplotlib

Recommended environment:
    a fresh conda/venv with mutually compatible current PyMC/ArviZ versions.
"""

from pathlib import Path
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

# Change this value to analyze another participant.
PARTICIPANT = 11

OUTCOMES = {
    "Posterior_delta": "negative",  # H: Experience < No-experience
    "Posterior_gamma": "positive",  # H: Experience > No-experience
    "Frontal_delta": "negative",    # H: Experience < No-experience
    "Frontal_gamma": "positive",    # H: Experience > No-experience
}

# Prior on beta after outcome standardization.
# beta is the standardized Experience - No-experience difference.
BETA_PRIOR_SD = 0.50

# Weakly regularizing priors for day random-effect/residual SDs on z scale.
DAY_SD_PRIOR = 0.50
RESIDUAL_SD_PRIOR = 1.00

DRAWS = 6000
TUNE = 6000
CHAINS = 4
CORES = 4
TARGET_ACCEPT = 0.99
SEED = 20260921

RESULT_FILE = Path(f"Bayesian_single_participant_P{PARTICIPANT}_results.csv")
TRACE_FILE_TEMPLATE = f"posterior_P{PARTICIPANT}_{{outcome}}.nc"
PLOT_FILE_TEMPLATE = f"posterior_P{PARTICIPANT}_{{outcome}}.png"


# =============================================================================
# Helpers
# =============================================================================

def load_data():
    df = pd.read_csv(DATA_FILE)

    required = {
        "Participant", "ExpDay", "State",
        "Posterior_delta", "Posterior_gamma",
        "Frontal_delta", "Frontal_gamma",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Keep only the selected participant.
    df = df[df["Participant"] == PARTICIPANT].copy()

    if df.empty:
        raise ValueError(f"No data found for Participant {PARTICIPANT}.")

    # Normalize State labels while preserving missing values.
    df["State"] = df["State"].astype("string").str.strip().str.lower()

    missing_state = df["State"].isna() | df["State"].eq("").fillna(False)
    n_missing_state = int(missing_state.sum())
    if n_missing_state:
        warnings.warn(
            f"Excluding {n_missing_state} rows with missing/blank State."
        )
        df = df.loc[~missing_state].copy()

    allowed = {"experience", "no-experience"}
    unexpected = set(df["State"].unique()) - allowed
    if unexpected:
        raise ValueError(f"Unexpected State values: {sorted(unexpected)}")

    # Retain only ExpDay cells containing BOTH States, matching the
    # confirmatory-analysis inclusion rule.
    day_counts = (
        df.groupby(["ExpDay", "State"])
        .size()
        .unstack(fill_value=0)
    )
    for state in ("experience", "no-experience"):
        if state not in day_counts.columns:
            day_counts[state] = 0

    valid_days = day_counts.index[
        (day_counts["experience"] > 0)
        & (day_counts["no-experience"] > 0)
    ]

    invalid_days = day_counts.index.difference(valid_days)
    if len(invalid_days):
        warnings.warn(
            "Excluding ExpDay(s) without both experience and no-experience: "
            + ", ".join(map(str, invalid_days.tolist()))
        )

    df = df[df["ExpDay"].isin(valid_days)].copy()
    if df.empty:
        raise ValueError(
            f"Participant {PARTICIPANT}: no ExpDay contains both States."
        )

    print(f"\nEpoch counts for Participant {PARTICIPANT}: ExpDay x State")
    print(day_counts.loc[valid_days])

    return df


def _hdi(samples, prob=0.95):
    """Return lower/upper HDI robustly across ArviZ versions."""
    x = np.asarray(samples, dtype=float).reshape(-1)
    try:
        h = np.asarray(az.hdi(x, hdi_prob=prob)).reshape(-1)
        return float(h[0]), float(h[1])
    except Exception:
        # Fallback to equal-tailed interval if ArviZ API differs.
        a = (1.0 - prob) / 2.0
        return float(np.quantile(x, a)), float(np.quantile(x, 1.0 - a))


def savage_dickey_bf(beta_samples, prior_sd, direction):
    """
    Approximate BF via Savage-Dickey density ratio.

    BF10_two_sided = prior density at beta=0 / posterior density at beta=0.

    For a symmetric prior, the directional alternative has prior probability
    0.5, so:

        BF_direction,0
            = 2 * BF10_two_sided * P(direction | data, H1_full)

    This uses KDE for the posterior density at zero and is therefore an
    approximation. Report prior sensitivity analyses if BF is used formally.
    """
    beta_samples = np.asarray(beta_samples, dtype=float).reshape(-1)

    prior_density_0 = norm.pdf(0.0, loc=0.0, scale=prior_sd)

    if np.std(beta_samples, ddof=1) == 0:
        return np.nan, np.nan, np.nan, np.nan

    kde = gaussian_kde(beta_samples)
    posterior_density_0 = float(kde.evaluate([0.0])[0])

    if posterior_density_0 <= 0 or not np.isfinite(posterior_density_0):
        return np.nan, np.nan, prior_density_0, posterior_density_0

    bf10_two_sided = prior_density_0 / posterior_density_0

    if direction == "negative":
        p_direction = float(np.mean(beta_samples < 0))
    elif direction == "positive":
        p_direction = float(np.mean(beta_samples > 0))
    else:
        raise ValueError("direction must be 'negative' or 'positive'")

    bf_directional = 2.0 * bf10_two_sided * p_direction

    return bf10_two_sided, bf_directional, prior_density_0, posterior_density_0


def fit_outcome(df, outcome, direction):
    d = df[["Participant", "ExpDay", "State", outcome]].dropna().copy()

    if d.empty:
        raise ValueError(f"{outcome}: no valid observations after dropping missing values.")

    # Reapply the BOTH-States rule after outcome-specific missing values are
    # removed. This mirrors the confirmatory analysis and protects against a
    # day becoming single-State for a particular outcome.
    outcome_day_counts = (
        d.groupby(["ExpDay", "State"])
        .size()
        .unstack(fill_value=0)
    )
    for state in ("experience", "no-experience"):
        if state not in outcome_day_counts.columns:
            outcome_day_counts[state] = 0

    valid_outcome_days = outcome_day_counts.index[
        (outcome_day_counts["experience"] > 0)
        & (outcome_day_counts["no-experience"] > 0)
    ]
    d = d[d["ExpDay"].isin(valid_outcome_days)].copy()

    if d.empty:
        raise ValueError(
            f"{outcome}: no ExpDay contains both States after dropping missing values."
        )

    # Coding makes beta exactly Experience - No-experience.
    d["state_code"] = d["State"].map({
        "no-experience": -0.5,
        "experience": +0.5,
    }).astype(float)

    # For a single participant, ExpDay itself is sufficient as the day ID.
    day_levels = sorted(d["ExpDay"].unique())
    day_map = {v: i for i, v in enumerate(day_levels)}

    day_idx = d["ExpDay"].map(day_map).to_numpy(dtype=int)
    x = d["state_code"].to_numpy(dtype=float)

    # Standardize outcome for stable priors/sampling.
    y_raw = d[outcome].to_numpy(dtype=float)
    y_mean = float(y_raw.mean())
    y_sd = float(y_raw.std(ddof=1))

    if y_sd <= 0:
        raise ValueError(f"{outcome}: outcome SD is zero.")

    y = (y_raw - y_mean) / y_sd

    coords = {
        "obs_id": np.arange(len(d)),
        "exp_day": [str(v) for v in day_levels],
    }

    with pm.Model(coords=coords) as model:
        # Data containers
        x_data = pm.Data("x", x, dims="obs_id")
        day_data = pm.Data("day_idx", day_idx, dims="obs_id")

        # Fixed effects
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
        beta = pm.Normal("beta", mu=0.0, sigma=BETA_PRIOR_SD)

        # Experimental-day random intercept: non-centered
        sigma_day = pm.HalfNormal("sigma_day", sigma=DAY_SD_PRIOR)

        z_day = pm.Normal(
            "z_day",
            mu=0.0,
            sigma=1.0,
            dims="exp_day",
        )

        u_day = pm.Deterministic(
            "u_day",
            z_day * sigma_day,
            dims="exp_day",
        )

        # Residual SD
        sigma = pm.HalfNormal("sigma", sigma=RESIDUAL_SD_PRIOR)

        # Linear predictor
        mu = (
            alpha
            + beta * x_data
            + u_day[day_data]
        )

        # Likelihood
        pm.Normal(
            "y_obs",
            mu=mu,
            sigma=sigma,
            observed=y,
            dims="obs_id",
        )

        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            cores=CORES,
            target_accept=TARGET_ACCEPT,
            random_seed=SEED,
            return_inferencedata=True,
        )

    # Save full posterior.
    trace_path = Path(TRACE_FILE_TEMPLATE.format(outcome=outcome.lower()))
    try:
        idata.to_netcdf(trace_path)
    except Exception as exc:
        warnings.warn(f"Could not save NetCDF trace: {exc}")

    # Posterior draws for State effect.
    beta_z = np.asarray(idata.posterior["beta"]).reshape(-1)
    beta_raw = beta_z * y_sd

    hdi_z = _hdi(beta_z, 0.95)
    hdi_raw = _hdi(beta_raw, 0.95)

    p_negative = float(np.mean(beta_z < 0))
    p_positive = float(np.mean(beta_z > 0))
    p_direction = p_negative if direction == "negative" else p_positive

    bf10_two, bf_directional, prior_d0, post_d0 = savage_dickey_bf(
        beta_z,
        prior_sd=BETA_PRIOR_SD,
        direction=direction,
    )

    # Diagnostics for key scalar parameters.
    summary = az.summary(
        idata,
        var_names=["alpha", "beta", "sigma_day", "sigma"],
        hdi_prob=0.95,
    )

    print(f"\n===== Participant {PARTICIPANT}: {outcome} =====")
    print(summary)

    # Extract beta diagnostics robustly.
    beta_rhat = (
        float(summary.loc["beta", "r_hat"])
        if "r_hat" in summary.columns
        else np.nan
    )
    beta_ess_bulk = (
        float(summary.loc["beta", "ess_bulk"])
        if "ess_bulk" in summary.columns
        else np.nan
    )
    beta_ess_tail = (
        float(summary.loc["beta", "ess_tail"])
        if "ess_tail" in summary.columns
        else np.nan
    )

    result = {
        "participant": PARTICIPANT,
        "outcome": outcome,
        "directional_hypothesis": direction,
        "n_epochs": len(d),
        "n_exp_days": len(day_levels),
        "n_experience": int((d["State"] == "experience").sum()),
        "n_no_experience": int((d["State"] == "no-experience").sum()),
        "outcome_raw_mean": y_mean,
        "outcome_raw_sd": y_sd,
        "beta_prior_SD_zscale": BETA_PRIOR_SD,
        "beta_z_mean": float(np.mean(beta_z)),
        "beta_z_median": float(np.median(beta_z)),
        "beta_z_HDI95_low": hdi_z[0],
        "beta_z_HDI95_high": hdi_z[1],
        "beta_raw_mean_E_minus_NE": float(np.mean(beta_raw)),
        "beta_raw_median_E_minus_NE": float(np.median(beta_raw)),
        "beta_raw_HDI95_low": hdi_raw[0],
        "beta_raw_HDI95_high": hdi_raw[1],
        "P_beta_lt_0": p_negative,
        "P_beta_gt_0": p_positive,
        "P_preregistered_direction": p_direction,
        "BF10_two_sided_SavageDickey_approx": bf10_two,
        "BF_direction_vs_H0_SavageDickey_approx": bf_directional,
        "prior_density_beta0": prior_d0,
        "posterior_density_beta0_KDE": post_d0,
        "beta_rhat": beta_rhat,
        "beta_ess_bulk": beta_ess_bulk,
        "beta_ess_tail": beta_ess_tail,
    }

    # Posterior plot for beta.
    try:
        import matplotlib.pyplot as plt

        az.plot_posterior(
            idata,
            var_names=["beta"],
            hdi_prob=0.95,
            ref_val=0,
        )
        plt.suptitle(
            f"P{PARTICIPANT} {outcome}: "
            "beta = Experience - No-experience (z scale)",
            y=1.02,
        )
        plt.tight_layout()

        plot_path = Path(PLOT_FILE_TEMPLATE.format(outcome=outcome.lower()))
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()

    except Exception as exc:
        warnings.warn(f"Could not save posterior plot: {exc}")

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    df = load_data()

    rows = []
    for outcome, direction in OUTCOMES.items():
        rows.append(fit_outcome(df, outcome, direction))

    results = pd.DataFrame(rows)
    results.to_csv(RESULT_FILE, index=False)

    display_cols = [
        "participant",
        "outcome",
        "beta_raw_mean_E_minus_NE",
        "beta_raw_HDI95_low",
        "beta_raw_HDI95_high",
        "P_preregistered_direction",
        "BF10_two_sided_SavageDickey_approx",
        "BF_direction_vs_H0_SavageDickey_approx",
        "beta_rhat",
    ]

    print("\n\n===== FINAL SUMMARY =====")
    print(results[display_cols].to_string(index=False))
    print(f"\nSaved results to: {RESULT_FILE}")

    print("\nInterpretation:")
    print("  beta < 0 means Experience has lower power than No-experience.")
    print("  beta > 0 means Experience has higher power than No-experience.")
    print("  For posterior delta, inspect P(beta < 0).")
    print("  For posterior gamma, inspect P(beta > 0).")
    print(
        "  BF values are prior-sensitive approximations; "
        "run prior sensitivity analyses."
    )


if __name__ == "__main__":
    main()
