from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(
    "runs/"
    "realworld_oversquashing_budget_robustness_v1"
)

OUT = Path(
    "figures/"
    "oversquashing_test15"
)

TABLES = Path(
    "tables/"
    "oversquashing_test15"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

TABLES.mkdir(
    parents=True,
    exist_ok=True,
)


ACC = pd.read_csv(
    ROOT
    / "test15_budget_accuracy_summary.csv"
)

EFF = pd.read_csv(
    ROOT
    / "test15_budget_primary_effects.csv"
)

STRUCT = pd.read_csv(
    ROOT
    / "test15_budget_structural_summary.csv"
)


DATASET_LABEL = {
    "pubmed": "PubMed",
    "roman_empire": "Roman-Empire",
}

MODEL_LABEL = {
    "GraphSAGE": "GraphSAGE",
    "GraphSAGEPairNorm":
        "GraphSAGE + PairNorm",
}

VARIANT_LABEL = {
    "original": "Original",
    "random": "Random",
    "targeted": "Targeted",
}

BUDGET_X = {
    "Original": 0.0,
    "0.5%": 0.5,
    "1.0%": 1.0,
    "2.0%": 2.0,
}


# ============================================================
# 1. ACCURACY VS BUDGET
# One figure per dataset.
#
# Visual convention:
# - GraphSAGE = blue
# - GraphSAGE + PairNorm = orange
# - Original = dashed horizontal line
# - Random = solid line with circles
# - Targeted = solid line with squares
# ============================================================

MODEL_COLOR = {
    "GraphSAGE": "tab:blue",
    "GraphSAGEPairNorm": "tab:orange",
}

for dataset in [
    "pubmed",
    "roman_empire",
]:
    x = ACC[
        ACC["dataset"]
        == dataset
    ].copy()

    fig, ax = plt.subplots(
        figsize=(7.2, 4.6)
    )

    for model in [
        "GraphSAGE",
        "GraphSAGEPairNorm",
    ]:
        color = MODEL_COLOR[
            model
        ]

        # ----------------------------------------------------
        # Original baseline
        # ----------------------------------------------------

        base = x[
            (
                x["model"]
                == model
            )
            &
            (
                x["variant"]
                == "original"
            )
        ]

        assert len(base) == 1

        baseline = float(
            base[
                "mean_accuracy_pct"
            ].iloc[0]
        )

        ax.axhline(
            baseline,
            color=color,
            linestyle="--",
            linewidth=1.6,
            label=(
                MODEL_LABEL[model]
                + " — Original"
            ),
        )

        # ----------------------------------------------------
        # Random rewiring
        # ----------------------------------------------------

        g = x[
            (
                x["model"]
                == model
            )
            &
            (
                x["variant"]
                == "random"
            )
        ].copy()

        g["budget_x"] = (
            g["budget_label"]
            .map(BUDGET_X)
        )

        g = g.sort_values(
            "budget_x"
        )

        ax.errorbar(
            g["budget_x"],
            g["mean_accuracy_pct"],
            yerr=g[
                "sd_accuracy_pct"
            ],
            color=color,
            marker="o",
            linestyle=":",
            capsize=3,
            linewidth=1.5,
            markersize=5,
            label=(
                MODEL_LABEL[model]
                + " — Random"
            ),
        )

        # ----------------------------------------------------
        # Targeted rewiring
        # ----------------------------------------------------

        g = x[
            (
                x["model"]
                == model
            )
            &
            (
                x["variant"]
                == "targeted"
            )
        ].copy()

        g["budget_x"] = (
            g["budget_label"]
            .map(BUDGET_X)
        )

        g = g.sort_values(
            "budget_x"
        )

        ax.errorbar(
            g["budget_x"],
            g["mean_accuracy_pct"],
            yerr=g[
                "sd_accuracy_pct"
            ],
            color=color,
            marker="s",
            linestyle="-",
            capsize=3,
            linewidth=1.5,
            markersize=5,
            label=(
                MODEL_LABEL[model]
                + " — Targeted"
            ),
        )

    ax.set_xlabel(
        "Added-edge budget [% of original edges]"
    )

    ax.set_ylabel(
        "Test accuracy [%]"
    )

    ax.set_title(
        DATASET_LABEL[dataset]
        + ": accuracy across rewiring budgets"
    )

    ax.set_xticks(
        [0.5, 1.0, 2.0]
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    # ----------------------------------------------------
    # Legend outside the plotting area.
    #
    # A separate white legend box is placed below the axes
    # so that none of the data are covered.
    # ----------------------------------------------------

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.20),
        fontsize=8,
        frameon=True,
        facecolor="white",
        edgecolor="0.75",
        framealpha=1.0,
        ncol=2,
        borderpad=0.8,
        columnspacing=1.4,
        handlelength=2.4,
    )

    # Reserve explicit space for the legend underneath
    # the actual plotting area.
    fig.subplots_adjust(
        bottom=0.31
    )

    stem = (
        OUT
        / (
            "test15_accuracy_budget_"
            + dataset
        )
    )

    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 2. TARGETED - RANDOM EFFECT
# Primary causal/control comparison.
# One figure per dataset.
# ============================================================

primary = EFF[
    EFF["contrast"]
    == "delta_targeted_vs_random"
].copy()

for dataset in [
    "pubmed",
    "roman_empire",
]:
    x = primary[
        primary["dataset"]
        == dataset
    ].copy()

    fig, ax = plt.subplots(
        figsize=(6.8, 4.4)
    )

    for model, marker in [
        ("GraphSAGE", "o"),
        (
            "GraphSAGEPairNorm",
            "s",
        ),
    ]:
        g = x[
            x["model"]
            == model
        ].copy()

        g = g.sort_values(
            "budget_fraction"
        )

        budget_pct = (
            100
            * g[
                "budget_fraction"
            ]
        )

        y = g["mean_pp"]

        lower = (
            y
            - g["ci95_low_pp"]
        )

        upper = (
            g["ci95_high_pp"]
            - y
        )

        ax.errorbar(
            budget_pct,
            y,
            yerr=np.vstack(
                [
                    lower,
                    upper,
                ]
            ),
            marker=marker,
            capsize=4,
            linewidth=1.5,
            label=MODEL_LABEL[model],
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Added-edge budget [% of original edges]"
    )

    ax.set_ylabel(
        "Targeted − Random test accuracy [pp]"
    )

    ax.set_title(
        DATASET_LABEL[dataset]
        + ": targeted vs. degree-matched random rewiring"
    )

    ax.set_xticks(
        [0.5, 1.0, 2.0]
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        frameon=False,
    )

    fig.tight_layout()

    stem = (
        OUT
        / (
            "test15_targeted_minus_random_"
            + dataset
        )
    )

    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 3. FIXED BOTTLENECK CONDUCTANCE VS BUDGET
# Demonstrates successful structural intervention.
# ============================================================

for dataset in [
    "pubmed",
    "roman_empire",
]:
    x = STRUCT[
        STRUCT["dataset"]
        == dataset
    ].copy()

    fig, ax = plt.subplots(
        figsize=(6.6, 4.3)
    )

    # Original reference from known baseline:
    if dataset == "pubmed":
        original_phi = (
            0.03146451312315577
        )
    else:
        original_phi = (
            0.000851003474930856
        )

    ax.axhline(
        original_phi,
        linestyle=":",
        linewidth=1.3,
        label="Original",
    )

    for variant, marker in [
        ("random", "o"),
        ("targeted", "s"),
    ]:
        g = x[
            x["variant"]
            == variant
        ].copy()

        g = g.sort_values(
            "budget_fraction"
        )

        ax.plot(
            100
            * g[
                "budget_fraction"
            ],
            g[
                "fixed_phi_mean"
            ],
            marker=marker,
            linewidth=1.7,
            label=VARIANT_LABEL[
                variant
            ],
        )

    ax.set_xlabel(
        "Added-edge budget [% of original edges]"
    )

    ax.set_ylabel(
        "Fixed-cut conductance"
    )

    ax.set_title(
        DATASET_LABEL[dataset]
        + ": structural bottleneck intervention"
    )

    ax.set_xticks(
        [0.5, 1.0, 2.0]
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        frameon=False,
    )

    fig.tight_layout()

    stem = (
        OUT
        / (
            "test15_fixed_cut_conductance_"
            + dataset
        )
    )

    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 4. TABLE: ACCURACY
# ============================================================

rows = []

for dataset in [
    "pubmed",
    "roman_empire",
]:
    for model in [
        "GraphSAGE",
        "GraphSAGEPairNorm",
    ]:
        x = ACC[
            (
                ACC["dataset"]
                == dataset
            )
            &
            (
                ACC["model"]
                == model
            )
        ]

        base = x[
            x["variant"]
            == "original"
        ].iloc[0]

        row = {
            "Dataset":
                DATASET_LABEL[
                    dataset
                ],

            "Model":
                MODEL_LABEL[
                    model
                ],

            "Original":
                (
                    f"{base['mean_accuracy_pct']:.2f}"
                    f" $\\pm$ "
                    f"{base['sd_accuracy_pct']:.2f}"
                ),
        }

        for budget in [
            "0.5%",
            "1.0%",
            "2.0%",
        ]:
            for variant in [
                "random",
                "targeted",
            ]:
                z = x[
                    (
                        x["budget_label"]
                        == budget
                    )
                    &
                    (
                        x["variant"]
                        == variant
                    )
                ].iloc[0]

                key = (
                    budget
                    + " "
                    + VARIANT_LABEL[
                        variant
                    ]
                )

                row[key] = (
                    f"{z['mean_accuracy_pct']:.2f}"
                    f" $\\pm$ "
                    f"{z['sd_accuracy_pct']:.2f}"
                )

        rows.append(row)

tab_acc = pd.DataFrame(
    rows
)

tab_acc.to_csv(
    TABLES
    / "test15_accuracy_table.csv",
    index=False,
)

with open(
    TABLES
    / "test15_accuracy_table.tex",
    "w",
) as f:
    f.write(
        tab_acc.to_latex(
            index=False,
            escape=False,
        )
    )


# ============================================================
# 5. TABLE: PRIMARY TARGETED - RANDOM EFFECT
# ============================================================

rows = []

for _, r in (
    primary.sort_values(
        [
            "dataset",
            "model",
            "budget_fraction",
        ]
    )
    .iterrows()
):
    rows.append({
        "Dataset":
            DATASET_LABEL[
                r["dataset"]
            ],

        "Model":
            MODEL_LABEL[
                r["model"]
            ],

        "Budget":
            r[
                "budget_label"
            ],

        "Targeted - Random [pp]":
            f"{r['mean_pp']:.2f}",

        "95% CI":
            (
                f"[{r['ci95_low_pp']:.2f}, "
                f"{r['ci95_high_pp']:.2f}]"
            ),

        "Positive replicates":
            int(
                r["positive"]
            ),

        "Negative replicates":
            int(
                r["negative"]
            ),

        "Exact sign-flip p":
            f"{r['exact_signflip_p']:.4f}",
    })

tab_eff = pd.DataFrame(
    rows
)

tab_eff.to_csv(
    TABLES
    / "test15_targeted_random_effects.csv",
    index=False,
)

with open(
    TABLES
    / "test15_targeted_random_effects.tex",
    "w",
) as f:
    f.write(
        tab_eff.to_latex(
            index=False,
            escape=False,
        )
    )


# ============================================================
# 6. TABLE: STRUCTURAL MANIPULATION
# ============================================================

rows = []

for _, r in (
    STRUCT.sort_values(
        [
            "dataset",
            "budget_fraction",
            "variant",
        ]
    )
    .iterrows()
):
    rows.append({
        "Dataset":
            DATASET_LABEL[
                r["dataset"]
            ],

        "Budget":
            r[
                "budget_label"
            ],

        "Variant":
            VARIANT_LABEL[
                r["variant"]
            ],

        "Added edges":
            int(
                round(
                    r[
                        "edge_budget_mean"
                    ]
                )
            ),

        "Cross-cut edges":
            f"{r['cross_cut_mean']:.1f}",

        "Fixed-cut conductance":
            f"{r['fixed_phi_mean']:.6f}",

        "Homophily":
            f"{r['homophily_mean']:.4f}",
    })

tab_struct = pd.DataFrame(
    rows
)

tab_struct.to_csv(
    TABLES
    / "test15_structural_table.csv",
    index=False,
)

with open(
    TABLES
    / "test15_structural_table.tex",
    "w",
) as f:
    f.write(
        tab_struct.to_latex(
            index=False,
            escape=False,
        )
    )


print(
    "FIGURES:",
    OUT,
)

print(
    "TABLES:",
    TABLES,
)

print()
print(
    "TEST 15 FINAL PLOTS/TABLES: PASS"
)
