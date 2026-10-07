from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# PATHS
# ============================================================

INPUT = Path(
    "runs/"
    "realworld_oversquashing_budget_robustness_v1/"
    "test15_budget_accuracy_summary.csv"
)

OUT_DIR = Path(
    "figures/"
    "oversquashing_test15"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_PDF = (
    OUT_DIR
    / "test15_accuracy_budgets_combined.pdf"
)

OUT_PNG = (
    OUT_DIR
    / "test15_accuracy_budgets_combined.png"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT
)

required_columns = {
    "dataset",
    "model",
    "budget_fraction",
    "budget_label",
    "variant",
    "mean_accuracy_pct",
    "sd_accuracy_pct",
}

missing = (
    required_columns
    - set(df.columns)
)

assert not missing, (
    "Missing columns: "
    + str(missing)
)


# ============================================================
# LABELS
# ============================================================

DATASET_LABEL = {
    "pubmed":
        "PubMed",

    "roman_empire":
        "Roman-Empire",
}


# ============================================================
# UNIQUE STYLE FOR EACH SERIES
#
# Every legend entry receives its own color.
#
# Original:
#   dashed
#
# Random:
#   dotted + circle
#
# Targeted:
#   solid + square
# ============================================================

STYLE = {
    (
        "GraphSAGE",
        "original",
    ): {
        "color":
            "#1f77b4",

        "linestyle":
            "--",

        "marker":
            None,

        "label":
            "GS Original",
    },

    (
        "GraphSAGE",
        "random",
    ): {
        "color":
            "#2ca02c",

        "linestyle":
            ":",

        "marker":
            "o",

        "label":
            "GS Random",
    },

    (
        "GraphSAGE",
        "targeted",
    ): {
        "color":
            "#d62728",

        "linestyle":
            "-",

        "marker":
            "s",

        "label":
            "GS Targeted",
    },

    (
        "GraphSAGEPairNorm",
        "original",
    ): {
        "color":
            "#9467bd",

        "linestyle":
            "--",

        "marker":
            None,

        "label":
            "PN Original",
    },

    (
        "GraphSAGEPairNorm",
        "random",
    ): {
        "color":
            "#ff7f0e",

        "linestyle":
            ":",

        "marker":
            "o",

        "label":
            "PN Random",
    },

    (
        "GraphSAGEPairNorm",
        "targeted",
    ): {
        "color":
            "#8c564b",

        "linestyle":
            "-",

        "marker":
            "s",

        "label":
            "PN Targeted",
    },
}


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        12.4,
        5.8,
    ),
)


for ax, dataset in zip(
    axes,
    [
        "pubmed",
        "roman_empire",
    ],
):
    data = df[
        df["dataset"]
        == dataset
    ].copy()

    # ========================================================
    # ORIGINAL BASELINES
    #
    # Original does not depend on rewiring budget.
    # Therefore it is plotted as a horizontal dashed line.
    # ========================================================

    for model in [
        "GraphSAGE",
        "GraphSAGEPairNorm",
    ]:
        original = data[
            (
                data["model"]
                == model
            )
            &
            (
                data["variant"]
                == "original"
            )
        ]

        assert len(original) == 1, (
            dataset,
            model,
            len(original),
        )

        value = float(
            original[
                "mean_accuracy_pct"
            ].iloc[0]
        )

        style = STYLE[
            (
                model,
                "original",
            )
        ]

        ax.axhline(
            y=value,
            color=style[
                "color"
            ],
            linestyle=style[
                "linestyle"
            ],
            linewidth=2.0,
            zorder=2,
        )


    # ========================================================
    # RANDOM + TARGETED
    # ========================================================

    for model in [
        "GraphSAGE",
        "GraphSAGEPairNorm",
    ]:
        for variant in [
            "random",
            "targeted",
        ]:
            g = data[
                (
                    data["model"]
                    == model
                )
                &
                (
                    data["variant"]
                    == variant
                )
            ].copy()

            assert len(g) == 3, (
                dataset,
                model,
                variant,
                len(g),
            )

            g = g.sort_values(
                "budget_fraction"
            )

            x = (
                100
                * g[
                    "budget_fraction"
                ].to_numpy()
            )

            y = g[
                "mean_accuracy_pct"
            ].to_numpy()

            yerr = g[
                "sd_accuracy_pct"
            ].to_numpy()

            style = STYLE[
                (
                    model,
                    variant,
                )
            ]

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                color=style[
                    "color"
                ],
                linestyle=style[
                    "linestyle"
                ],
                marker=style[
                    "marker"
                ],
                linewidth=2.0,
                markersize=6,
                markeredgewidth=0.8,
                capsize=3.5,
                elinewidth=1.2,
                zorder=3,
            )


    # ========================================================
    # AXES
    # ========================================================

    ax.set_title(
        DATASET_LABEL[
            dataset
        ],
        fontsize=12,
    )

    ax.set_xlabel(
        "Added-edge budget [% of original edges]",
        fontsize=10,
    )

    ax.set_ylabel(
        "Test accuracy [%]",
        fontsize=10,
    )

    ax.set_xticks(
        [
            0.5,
            1.0,
            2.0,
        ]
    )

    ax.set_xlim(
        0.35,
        2.15,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
        linewidth=0.8,
    )

    ax.tick_params(
        axis="both",
        labelsize=9,
    )


# ============================================================
# CUSTOM LEGEND
#
# We construct the six entries manually to guarantee that
# every label receives the correct color/style.
# ============================================================

legend_order = [
    (
        "GraphSAGE",
        "original",
    ),
    (
        "GraphSAGE",
        "random",
    ),
    (
        "GraphSAGE",
        "targeted",
    ),
    (
        "GraphSAGEPairNorm",
        "original",
    ),
    (
        "GraphSAGEPairNorm",
        "random",
    ),
    (
        "GraphSAGEPairNorm",
        "targeted",
    ),
]

legend_handles = []

for key in legend_order:
    style = STYLE[
        key
    ]

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=style[
                "color"
            ],
            linestyle=style[
                "linestyle"
            ],
            marker=style[
                "marker"
            ],
            linewidth=2.0,
            markersize=6,
            label=style[
                "label"
            ],
        )
    )


# ============================================================
# WHITE LEGEND BOX BELOW THE PLOTS
#
# It is inside the overall figure canvas but completely
# outside both plotting areas.
# ============================================================

legend = fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(
        0.5,
        0.025,
    ),
    ncol=3,
    fontsize=9,
    frameon=True,
    facecolor="white",
    edgecolor="0.70",
    framealpha=1.0,
    fancybox=False,
    borderpad=0.9,
    columnspacing=2.0,
    handlelength=3.0,
    handletextpad=0.7,
)

legend.get_frame().set_linewidth(
    0.8
)


# ============================================================
# LAYOUT
#
# bottom=0.25 explicitly reserves room for the legend box.
# Therefore the legend cannot cover the plots.
# ============================================================

fig.subplots_adjust(
    left=0.08,
    right=0.98,
    top=0.92,
    bottom=0.25,
    wspace=0.25,
)


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
)

fig.savefig(
    OUT_PNG,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print()
print(
    "SAVED PDF:",
    OUT_PDF,
)

print(
    "SAVED PNG:",
    OUT_PNG,
)

print()
print(
    "TEST 15 ACCURACY-BUDGET FIGURE: PASS"
)
