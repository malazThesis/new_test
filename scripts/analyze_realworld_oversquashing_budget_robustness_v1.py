from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from scipy import stats


BASE = Path(
    "runs/"
    "realworld_oversquashing_rewiring_v1/"
    "realworld_oversquashing_master.csv"
)

B005 = Path(
    "runs/"
    "realworld_oversquashing_budget005_v1/"
    "realworld_oversquashing_0p5_master.csv"
)

B020 = Path(
    "runs/"
    "realworld_oversquashing_budget020_v1/"
    "realworld_oversquashing_2p0_master.csv"
)

OUT = Path(
    "runs/"
    "realworld_oversquashing_budget_robustness_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD
# ============================================================

base = pd.read_csv(BASE)
b005 = pd.read_csv(B005)
b020 = pd.read_csv(B020)

# 1% intervention rows from original Test 15
b010 = base[
    base["variant"].isin(
        ["targeted", "random"]
    )
].copy()

b010["budget_fraction"] = 0.01
b010["budget_label"] = "1.0%"

# Normalize budget metadata
b005["budget_fraction"] = 0.005
b005["budget_label"] = "0.5%"

b020["budget_fraction"] = 0.02
b020["budget_label"] = "2.0%"

interventions = pd.concat(
    [
        b005,
        b010,
        b020,
    ],
    ignore_index=True,
)

assert len(interventions) == 360

assert set(
    interventions["variant"]
) == {
    "targeted",
    "random",
}

assert set(
    interventions[
        "budget_fraction"
    ]
) == {
    0.005,
    0.01,
    0.02,
}


# ============================================================
# ORIGINAL BASELINE
#
# Original graph does not depend on edge budget and is
# therefore taken only once from the audited 1% experiment.
# ============================================================

original = base[
    base["variant"]
    == "original"
].copy()

assert len(original) == 60


# ============================================================
# PRIMARY UNIT:
# average over the three initialization seeds first.
#
# Leaves 5 paired replicate seeds per condition.
# ============================================================

group_base = [
    "dataset",
    "model",
    "replicate_seed",
]

original_rep = (
    original.groupby(
        group_base,
        as_index=False,
    )["test_acc"]
    .mean()
    .rename(
        columns={
            "test_acc":
                "original"
        }
    )
)

intervention_rep = (
    interventions.groupby(
        group_base
        + [
            "budget_fraction",
            "budget_label",
            "variant",
        ],
        as_index=False,
    )["test_acc"]
    .mean()
)

init_counts = (
    interventions.groupby(
        group_base
        + [
            "budget_fraction",
            "variant",
        ]
    )
    .size()
)

assert init_counts.eq(3).all()

wide = (
    intervention_rep.pivot(
        index=
            group_base
            + [
                "budget_fraction",
                "budget_label",
            ],
        columns="variant",
        values="test_acc",
    )
    .reset_index()
)

wide = wide.merge(
    original_rep,
    on=group_base,
    how="left",
    validate="many_to_one",
)

assert not wide[
    [
        "original",
        "random",
        "targeted",
    ]
].isna().any().any()

assert len(wide) == (
    2 * 2 * 5 * 3
)


# ============================================================
# EFFECTS
# ============================================================

wide[
    "delta_targeted_vs_original"
] = (
    wide["targeted"]
    - wide["original"]
)

wide[
    "delta_random_vs_original"
] = (
    wide["random"]
    - wide["original"]
)

wide[
    "delta_targeted_vs_random"
] = (
    wide["targeted"]
    - wide["random"]
)

wide.to_csv(
    OUT
    / "test15_budget_replicate_level.csv",
    index=False,
)


# ============================================================
# EXACT SIGN-FLIP TEST
# ============================================================

def summarize_delta(values):
    x = np.asarray(
        values,
        dtype=float,
    )

    n = len(x)

    assert n == 5

    mean = float(
        x.mean()
    )

    sd = float(
        x.std(ddof=1)
    )

    sem = (
        sd
        / np.sqrt(n)
    )

    tcrit = stats.t.ppf(
        0.975,
        df=n - 1,
    )

    low = (
        mean
        - tcrit * sem
    )

    high = (
        mean
        + tcrit * sem
    )

    observed = abs(
        mean
    )

    null = []

    for signs in itertools.product(
        [-1.0, 1.0],
        repeat=n,
    ):
        signs = np.asarray(
            signs
        )

        null.append(
            abs(
                np.mean(
                    signs * x
                )
            )
        )

    null = np.asarray(
        null
    )

    p = float(
        np.mean(
            null
            >= observed - 1e-15
        )
    )

    return {
        "n_replicates":
            n,

        "mean":
            mean,

        "sd":
            sd,

        "ci95_low":
            low,

        "ci95_high":
            high,

        "positive":
            int(
                (x > 0).sum()
            ),

        "negative":
            int(
                (x < 0).sum()
            ),

        "zero":
            int(
                (x == 0).sum()
            ),

        "exact_signflip_p":
            p,
    }


# ============================================================
# PRIMARY EFFECT TABLE
# ============================================================

effects = []

contrasts = [
    "delta_targeted_vs_original",
    "delta_random_vs_original",
    "delta_targeted_vs_random",
]

for (
    dataset,
    model,
    budget_fraction,
    budget_label,
), g in wide.groupby(
    [
        "dataset",
        "model",
        "budget_fraction",
        "budget_label",
    ]
):
    for contrast in contrasts:
        s = summarize_delta(
            g[contrast].to_numpy()
        )

        effects.append({
            "dataset":
                dataset,

            "model":
                model,

            "budget_fraction":
                budget_fraction,

            "budget_label":
                budget_label,

            "contrast":
                contrast,

            **s,
        })

effects = pd.DataFrame(
    effects
)

for c in [
    "mean",
    "sd",
    "ci95_low",
    "ci95_high",
]:
    effects[
        c + "_pp"
    ] = (
        100
        * effects[c]
    )

effects = effects.sort_values(
    [
        "dataset",
        "model",
        "budget_fraction",
        "contrast",
    ]
)

effects.to_csv(
    OUT
    / "test15_budget_primary_effects.csv",
    index=False,
)


# ============================================================
# ACCURACY TABLE
# ============================================================

acc_rows = []

for (
    dataset,
    model,
), g0 in original_rep.groupby(
    [
        "dataset",
        "model",
    ]
):
    vals = (
        g0["original"]
        .to_numpy()
    )

    acc_rows.append({
        "dataset":
            dataset,

        "model":
            model,

        "budget_fraction":
            0.0,

        "budget_label":
            "Original",

        "variant":
            "original",

        "n_replicates":
            len(vals),

        "mean_accuracy":
            float(
                vals.mean()
            ),

        "sd_accuracy":
            float(
                vals.std(
                    ddof=1
                )
            ),
    })

for (
    dataset,
    model,
    budget_fraction,
    budget_label,
    variant,
), g in intervention_rep.groupby(
    [
        "dataset",
        "model",
        "budget_fraction",
        "budget_label",
        "variant",
    ]
):
    vals = (
        g["test_acc"]
        .to_numpy()
    )

    acc_rows.append({
        "dataset":
            dataset,

        "model":
            model,

        "budget_fraction":
            budget_fraction,

        "budget_label":
            budget_label,

        "variant":
            variant,

        "n_replicates":
            len(vals),

        "mean_accuracy":
            float(
                vals.mean()
            ),

        "sd_accuracy":
            float(
                vals.std(
                    ddof=1
                )
            ),
    })

accuracy = pd.DataFrame(
    acc_rows
)

accuracy[
    "mean_accuracy_pct"
] = (
    100
    * accuracy[
        "mean_accuracy"
    ]
)

accuracy[
    "sd_accuracy_pct"
] = (
    100
    * accuracy[
        "sd_accuracy"
    ]
)

accuracy.to_csv(
    OUT
    / "test15_budget_accuracy_summary.csv",
    index=False,
)


# ============================================================
# PAIRNORM INTERACTIONS
#
# Difference in intervention effect:
#
# PairNorm effect - GraphSAGE effect
# ============================================================

interaction_rows = []

for (
    dataset,
    budget_fraction,
    budget_label,
), g in wide.groupby(
    [
        "dataset",
        "budget_fraction",
        "budget_label",
    ]
):
    pivot = (
        g.set_index(
            [
                "replicate_seed",
                "model",
            ]
        )
    )

    for contrast in contrasts:
        pn = (
            pivot.xs(
                "GraphSAGEPairNorm",
                level="model",
            )[contrast]
        )

        gs = (
            pivot.xs(
                "GraphSAGE",
                level="model",
            )[contrast]
        )

        pn = pn.sort_index()
        gs = gs.sort_index()

        assert np.array_equal(
            pn.index.to_numpy(),
            gs.index.to_numpy(),
        )

        interaction = (
            pn.to_numpy()
            - gs.to_numpy()
        )

        s = summarize_delta(
            interaction
        )

        interaction_rows.append({
            "dataset":
                dataset,

            "budget_fraction":
                budget_fraction,

            "budget_label":
                budget_label,

            "contrast":
                contrast,

            **s,
        })

interaction = pd.DataFrame(
    interaction_rows
)

for c in [
    "mean",
    "sd",
    "ci95_low",
    "ci95_high",
]:
    interaction[
        c + "_pp"
    ] = (
        100
        * interaction[c]
    )

interaction.to_csv(
    OUT
    / "test15_budget_pairnorm_interactions.csv",
    index=False,
)


# ============================================================
# STRUCTURAL SUMMARY
# ============================================================

struct = (
    interventions[
        [
            "dataset",
            "budget_fraction",
            "budget_label",
            "variant",
            "replicate_seed",
            "edge_budget",
            "cross_cut",
            "fixed_phi",
            "lambda2",
            "homophily",
        ]
    ]
    .drop_duplicates()
)

assert len(struct) == (
    2 * 3 * 2 * 5
)

struct_summary = (
    struct.groupby(
        [
            "dataset",
            "budget_fraction",
            "budget_label",
            "variant",
        ],
        as_index=False,
    )
    .agg(
        edge_budget_mean=(
            "edge_budget",
            "mean",
        ),

        cross_cut_mean=(
            "cross_cut",
            "mean",
        ),

        fixed_phi_mean=(
            "fixed_phi",
            "mean",
        ),

        fixed_phi_sd=(
            "fixed_phi",
            "std",
        ),

        lambda2_mean=(
            "lambda2",
            "mean",
        ),

        homophily_mean=(
            "homophily",
            "mean",
        ),
    )
)

struct_summary.to_csv(
    OUT
    / "test15_budget_structural_summary.csv",
    index=False,
)


# ============================================================
# COMPACT CONSOLE OUTPUT
# ============================================================

print()
print("=" * 110)
print("A. ACCURACY BY BUDGET [%]")
print("=" * 110)

print(
    accuracy
    .sort_values(
        [
            "dataset",
            "model",
            "budget_fraction",
            "variant",
        ]
    )[
        [
            "dataset",
            "model",
            "budget_label",
            "variant",
            "mean_accuracy_pct",
            "sd_accuracy_pct",
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)


print()
print("=" * 110)
print("B. TARGETED - ORIGINAL [pp]")
print("=" * 110)

x = effects[
    effects["contrast"]
    == "delta_targeted_vs_original"
]

print(
    x[
        [
            "dataset",
            "model",
            "budget_label",
            "mean_pp",
            "sd_pp",
            "ci95_low_pp",
            "ci95_high_pp",
            "positive",
            "negative",
            "exact_signflip_p",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 110)
print("C. RANDOM - ORIGINAL [pp]")
print("=" * 110)

x = effects[
    effects["contrast"]
    == "delta_random_vs_original"
]

print(
    x[
        [
            "dataset",
            "model",
            "budget_label",
            "mean_pp",
            "sd_pp",
            "ci95_low_pp",
            "ci95_high_pp",
            "positive",
            "negative",
            "exact_signflip_p",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 110)
print("D. TARGETED - RANDOM [pp] — PRIMARY STRUCTURAL CONTROL")
print("=" * 110)

x = effects[
    effects["contrast"]
    == "delta_targeted_vs_random"
]

print(
    x[
        [
            "dataset",
            "model",
            "budget_label",
            "mean_pp",
            "sd_pp",
            "ci95_low_pp",
            "ci95_high_pp",
            "positive",
            "negative",
            "exact_signflip_p",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 110)
print("E. STRUCTURAL MANIPULATION")
print("=" * 110)

print(
    struct_summary[
        [
            "dataset",
            "budget_label",
            "variant",
            "edge_budget_mean",
            "cross_cut_mean",
            "fixed_phi_mean",
            "lambda2_mean",
            "homophily_mean",
        ]
    ]
    .round(6)
    .to_string(
        index=False
    )
)


print()
print("=" * 110)
print("F. PAIRNORM INTERACTION FOR TARGETED - RANDOM")
print("=" * 110)

x = interaction[
    interaction["contrast"]
    == "delta_targeted_vs_random"
]

print(
    x[
        [
            "dataset",
            "budget_label",
            "mean_pp",
            "sd_pp",
            "ci95_low_pp",
            "ci95_high_pp",
            "positive",
            "negative",
            "exact_signflip_p",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print()
print(
    "OUTPUT:",
    OUT,
)

print()
print(
    "TEST 15 CROSS-BUDGET ANALYSIS: PASS"
)
