from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNS = Path("runs")

GRAPH_FILE = (
    RUNS
    / "longrange_bottleneck_2x2_graph_means.csv"
)

ROOT = Path(
    "plots/oversquashing_longrange_bottleneck_final"
)

DIR_ACCURACY = ROOT / "01_accuracy"
DIR_SPECIFICITY = ROOT / "02_source_target_specificity"
DIR_EFFECTS = ROOT / "03_intervention_effects"
DIR_HEATMAPS = ROOT / "04_bottleneck_landscape"
DIR_DATA = ROOT / "05_plot_data"

for directory in [
    DIR_ACCURACY,
    DIR_SPECIFICITY,
    DIR_EFFECTS,
    DIR_HEATMAPS,
    DIR_DATA,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


MODEL_ORDER = [
    "GraphSAGE",
    "GraphSAGEPairNorm",
    "GraphSAGERewired",
    "GraphSAGEPairNormRewired",
]

MODEL_LABEL = {
    "GraphSAGE":
        "GraphSAGE",
    "GraphSAGEPairNorm":
        "GraphSAGE + PairNorm",
    "GraphSAGERewired":
        "GraphSAGE + Rewiring",
    "GraphSAGEPairNormRewired":
        "PairNorm + Rewiring",
}

MARKERS = {
    "GraphSAGE":
        "o",
    "GraphSAGEPairNorm":
        "s",
    "GraphSAGERewired":
        "^",
    "GraphSAGEPairNormRewired":
        "D",
}

LINESTYLES = {
    "GraphSAGE":
        "-",
    "GraphSAGEPairNorm":
        "--",
    "GraphSAGERewired":
        "-.",
    "GraphSAGEPairNormRewired":
        ":",
}

DISTANCES = [
    2,
    4,
    6,
    8,
    10,
]

WIDTHS = [
    1,
    2,
    4,
    8,
]


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 350,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save_png(
    fig,
    directory,
    name,
):
    path = (
        directory
        / f"{name}.png"
    )

    fig.savefig(
        path,
        dpi=350,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(
        "saved:",
        path,
    )


df = pd.read_csv(
    GRAPH_FILE
)

assert len(df) == 400

assert set(
    df["model"]
) == set(
    MODEL_ORDER
)

assert df[
    "n_init"
].eq(
    3
).all()


cell_counts = (
    df.groupby(
        [
            "model",
            "distance",
            "width",
        ]
    )[
        "graph_seed"
    ]
    .nunique()
)

assert cell_counts.eq(
    5
).all()


summary = (
    df.groupby(
        [
            "model",
            "distance",
            "width",
        ],
        as_index=False,
    )
    .agg(
        n_graphs=(
            "graph_seed",
            "nunique",
        ),
        accuracy_mean=(
            "test_acc",
            "mean",
        ),
        accuracy_sd=(
            "test_acc",
            "std",
        ),
        matched_influence_mean=(
            "matched_influence",
            "mean",
        ),
        matched_influence_sd=(
            "matched_influence",
            "std",
        ),
        specificity_mean=(
            "specificity",
            "mean",
        ),
        specificity_sd=(
            "specificity",
            "std",
        ),
    )
)

assert len(summary) == 80
assert summary[
    "n_graphs"
].eq(
    5
).all()

summary.to_csv(
    DIR_DATA
    / "longrange_bottleneck_cell_summary.csv",
    index=False,
)


def get_model_cell(
    model,
    width,
):
    x = (
        summary[
            summary[
                "model"
            ].eq(model)
            & summary[
                "width"
            ].eq(width)
        ]
        .sort_values(
            "distance"
        )
    )

    assert len(x) == 5

    return x


fig, axes = plt.subplots(
    2,
    2,
    figsize=(
        10.8,
        7.8,
    ),
    sharex=True,
    sharey=True,
)

for ax, width in zip(
    axes.flat,
    WIDTHS,
):
    for model in MODEL_ORDER:
        x = get_model_cell(
            model,
            width,
        )

        ax.errorbar(
            x["distance"],
            100.0
            * x[
                "accuracy_mean"
            ],
            yerr=(
                100.0
                * x[
                    "accuracy_sd"
                ]
            ),
            marker=MARKERS[
                model
            ],
            linestyle=LINESTYLES[
                model
            ],
            linewidth=1.9,
            markersize=5.5,
            capsize=3,
            label=MODEL_LABEL[
                model
            ],
        )

    ax.axhline(
        50,
        color="0.55",
        linewidth=1,
        linestyle=":",
    )

    ax.set_title(
        f"Bottleneck width = {width}"
    )

    ax.set_xticks(
        DISTANCES
    )

    ax.set_ylim(
        48,
        102,
    )

    ax.grid(
        axis="y",
        alpha=0.18,
    )

for ax in axes[
    1,
    :
]:
    ax.set_xlabel(
        "Source-target distance"
    )

for ax in axes[
    :,
    0
]:
    ax.set_ylabel(
        "Test accuracy (%)"
    )

handles, labels = (
    axes[
        0,
        0
    ].get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(
        0.5,
        -0.01,
    ),
)

fig.suptitle(
    "Long-range bottleneck benchmark: predictive performance",
    fontsize=13,
)

fig.subplots_adjust(
    bottom=0.14,
    top=0.91,
    wspace=0.13,
    hspace=0.20,
)

save_png(
    fig,
    DIR_ACCURACY,
    "accuracy_vs_distance_by_bottleneck_width",
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        10.6,
        4.3,
    ),
    sharey=True,
)

for ax, model, title in [
    (
        axes[0],
        "GraphSAGE",
        "GraphSAGE",
    ),
    (
        axes[1],
        "GraphSAGEPairNorm",
        "GraphSAGE + PairNorm",
    ),
]:
    for distance in DISTANCES:
        x = (
            summary[
                summary[
                    "model"
                ].eq(model)
                & summary[
                    "distance"
                ].eq(distance)
            ]
            .sort_values(
                "width"
            )
        )

        assert len(x) == 4

        ax.errorbar(
            x["width"],
            100.0
            * x[
                "accuracy_mean"
            ],
            yerr=(
                100.0
                * x[
                    "accuracy_sd"
                ]
            ),
            marker="o",
            linewidth=1.8,
            markersize=5,
            capsize=3,
            label=(
                f"d = {distance}"
            ),
        )

    ax.set_xscale(
        "log",
        base=2,
    )

    ax.set_xticks(
        WIDTHS
    )

    ax.set_xticklabels(
        WIDTHS
    )

    ax.set_xlabel(
        "Bottleneck width"
    )

    ax.set_title(
        title
    )

    ax.set_ylim(
        48,
        102,
    )

    ax.grid(
        axis="y",
        alpha=0.18,
    )

axes[
    0
].set_ylabel(
    "Test accuracy (%)"
)

axes[
    1
].legend(
    title="Distance",
    frameon=False,
    bbox_to_anchor=(
        1.02,
        1.0,
    ),
)

fig.suptitle(
    "Effect of bottleneck width at fixed source-target distance",
    fontsize=13,
)

fig.subplots_adjust(
    left=0.08,
    right=0.86,
    bottom=0.15,
    top=0.84,
    wspace=0.12,
)

save_png(
    fig,
    DIR_ACCURACY,
    "accuracy_vs_bottleneck_width",
)


fig, axes = plt.subplots(
    2,
    2,
    figsize=(
        10.8,
        7.8,
    ),
    sharex=True,
    sharey=True,
)

for ax, width in zip(
    axes.flat,
    WIDTHS,
):
    for model in MODEL_ORDER:
        x = get_model_cell(
            model,
            width,
        )

        ax.errorbar(
            x["distance"],
            x[
                "specificity_mean"
            ],
            yerr=x[
                "specificity_sd"
            ],
            marker=MARKERS[
                model
            ],
            linestyle=LINESTYLES[
                model
            ],
            linewidth=1.9,
            markersize=5.5,
            capsize=3,
            label=MODEL_LABEL[
                model
            ],
        )

    ax.axhline(
        0.5,
        color="0.55",
        linewidth=1,
        linestyle=":",
    )

    ax.set_title(
        f"Bottleneck width = {width}"
    )

    ax.set_xticks(
        DISTANCES
    )

    ax.set_ylim(
        0.45,
        1.02,
    )

    ax.grid(
        axis="y",
        alpha=0.18,
    )

for ax in axes[
    1,
    :
]:
    ax.set_xlabel(
        "Source-target distance"
    )

for ax in axes[
    :,
    0
]:
    ax.set_ylabel(
        "Source-target specificity"
    )

handles, labels = (
    axes[
        0,
        0
    ].get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(
        0.5,
        -0.01,
    ),
)

fig.suptitle(
    "Long-range bottleneck benchmark: source-target specificity",
    fontsize=13,
)

fig.subplots_adjust(
    bottom=0.14,
    top=0.91,
    wspace=0.13,
    hspace=0.20,
)

save_png(
    fig,
    DIR_SPECIFICITY,
    "specificity_vs_distance_by_bottleneck_width",
)


def model_frame(
    model,
):
    x = df[
        df[
            "model"
        ].eq(model)
    ].copy()

    return x[
        [
            "distance",
            "width",
            "graph_seed",
            "test_acc",
            "specificity",
        ]
    ]


base = model_frame(
    "GraphSAGE"
).rename(
    columns={
        "test_acc":
            "acc_base",
        "specificity":
            "spec_base",
    }
)

pn = model_frame(
    "GraphSAGEPairNorm"
).rename(
    columns={
        "test_acc":
            "acc_pn",
        "specificity":
            "spec_pn",
    }
)

rew = model_frame(
    "GraphSAGERewired"
).rename(
    columns={
        "test_acc":
            "acc_rew",
        "specificity":
            "spec_rew",
    }
)

pn_rew = model_frame(
    "GraphSAGEPairNormRewired"
).rename(
    columns={
        "test_acc":
            "acc_pn_rew",
        "specificity":
            "spec_pn_rew",
    }
)


KEYS = [
    "distance",
    "width",
    "graph_seed",
]

effects = (
    base.merge(
        pn,
        on=KEYS,
        validate="one_to_one",
    )
    .merge(
        rew,
        on=KEYS,
        validate="one_to_one",
    )
    .merge(
        pn_rew,
        on=KEYS,
        validate="one_to_one",
    )
)

assert len(
    effects
) == 100

effects[
    "pairnorm_gain_pp"
] = (
    100.0
    * (
        effects[
            "acc_pn"
        ]
        - effects[
            "acc_base"
        ]
    )
)

effects[
    "rewiring_gain_pp"
] = (
    100.0
    * (
        effects[
            "acc_rew"
        ]
        - effects[
            "acc_base"
        ]
    )
)

effects[
    "rewiring_after_pairnorm_pp"
] = (
    100.0
    * (
        effects[
            "acc_pn_rew"
        ]
        - effects[
            "acc_pn"
        ]
    )
)

effects[
    "pairnorm_after_rewiring_pp"
] = (
    100.0
    * (
        effects[
            "acc_pn_rew"
        ]
        - effects[
            "acc_rew"
        ]
    )
)

effects.to_csv(
    DIR_DATA
    / "longrange_bottleneck_graph_level_effects.csv",
    index=False,
)


effect_summary = (
    effects.groupby(
        [
            "distance",
            "width",
        ],
        as_index=False,
    )
    .agg(
        pairnorm_gain_mean=(
            "pairnorm_gain_pp",
            "mean",
        ),
        pairnorm_gain_sd=(
            "pairnorm_gain_pp",
            "std",
        ),
        rewiring_gain_mean=(
            "rewiring_gain_pp",
            "mean",
        ),
        rewiring_gain_sd=(
            "rewiring_gain_pp",
            "std",
        ),
        rewiring_after_pairnorm_mean=(
            "rewiring_after_pairnorm_pp",
            "mean",
        ),
        rewiring_after_pairnorm_sd=(
            "rewiring_after_pairnorm_pp",
            "std",
        ),
        pairnorm_after_rewiring_mean=(
            "pairnorm_after_rewiring_pp",
            "mean",
        ),
        pairnorm_after_rewiring_sd=(
            "pairnorm_after_rewiring_pp",
            "std",
        ),
    )
)

effect_summary.to_csv(
    DIR_DATA
    / "longrange_bottleneck_effect_summary.csv",
    index=False,
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        10.8,
        4.4,
    ),
    sharey=True,
)

for ax, mean_col, sd_col, title in [
    (
        axes[0],
        "pairnorm_gain_mean",
        "pairnorm_gain_sd",
        "PairNorm vs GraphSAGE",
    ),
    (
        axes[1],
        "rewiring_gain_mean",
        "rewiring_gain_sd",
        "Rewiring vs GraphSAGE",
    ),
]:
    for width in WIDTHS:
        x = (
            effect_summary[
                effect_summary[
                    "width"
                ].eq(width)
            ]
            .sort_values(
                "distance"
            )
        )

        ax.errorbar(
            x[
                "distance"
            ],
            x[
                mean_col
            ],
            yerr=x[
                sd_col
            ],
            marker="o",
            linewidth=1.8,
            markersize=5,
            capsize=3,
            label=(
                f"width = {width}"
            ),
        )

    ax.axhline(
        0,
        color="0.5",
        linestyle=":",
        linewidth=1,
    )

    ax.set_xticks(
        DISTANCES
    )

    ax.set_xlabel(
        "Source-target distance"
    )

    ax.set_title(
        title
    )

    ax.grid(
        axis="y",
        alpha=0.18,
    )

axes[
    0
].set_ylabel(
    "Test accuracy gain (percentage points)"
)

axes[
    1
].legend(
    frameon=False,
    bbox_to_anchor=(
        1.02,
        1.0,
    ),
)

fig.suptitle(
    "Recovery from the structural bottleneck",
    fontsize=13,
)

fig.subplots_adjust(
    left=0.08,
    right=0.85,
    bottom=0.15,
    top=0.82,
    wspace=0.14,
)

save_png(
    fig,
    DIR_EFFECTS,
    "pairnorm_vs_rewiring_accuracy_gain",
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        10.8,
        4.4,
    ),
    sharey=True,
)

for ax, mean_col, sd_col, title in [
    (
        axes[0],
        "rewiring_after_pairnorm_mean",
        "rewiring_after_pairnorm_sd",
        "Additional rewiring gain after PairNorm",
    ),
    (
        axes[1],
        "pairnorm_after_rewiring_mean",
        "pairnorm_after_rewiring_sd",
        "Additional PairNorm gain after rewiring",
    ),
]:
    for width in WIDTHS:
        x = (
            effect_summary[
                effect_summary[
                    "width"
                ].eq(width)
            ]
            .sort_values(
                "distance"
            )
        )

        ax.errorbar(
            x[
                "distance"
            ],
            x[
                mean_col
            ],
            yerr=x[
                sd_col
            ],
            marker="o",
            linewidth=1.8,
            markersize=5,
            capsize=3,
            label=(
                f"width = {width}"
            ),
        )

    ax.axhline(
        0,
        color="0.5",
        linestyle=":",
        linewidth=1,
    )

    ax.set_xticks(
        DISTANCES
    )

    ax.set_xlabel(
        "Source-target distance"
    )

    ax.set_title(
        title
    )

    ax.grid(
        axis="y",
        alpha=0.18,
    )

axes[
    0
].set_ylabel(
    "Additional accuracy gain (percentage points)"
)

axes[
    1
].legend(
    frameon=False,
    bbox_to_anchor=(
        1.02,
        1.0,
    ),
)

fig.suptitle(
    "Residual intervention effects near the performance ceiling",
    fontsize=13,
)

fig.subplots_adjust(
    left=0.08,
    right=0.85,
    bottom=0.15,
    top=0.82,
    wspace=0.14,
)

save_png(
    fig,
    DIR_EFFECTS,
    "residual_2x2_intervention_effects",
)


baseline_summary = (
    summary[
        summary[
            "model"
        ].eq(
            "GraphSAGE"
        )
    ]
    .copy()
)


def heatmap_matrix(
    value_col,
):
    matrix = (
        baseline_summary
        .pivot(
            index="distance",
            columns="width",
            values=value_col,
        )
        .reindex(
            index=DISTANCES,
            columns=WIDTHS,
        )
    )

    return matrix


accuracy_matrix = (
    100.0
    * heatmap_matrix(
        "accuracy_mean"
    )
)

specificity_matrix = heatmap_matrix(
    "specificity_mean"
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        9.6,
        4.4,
    ),
)

im1 = axes[
    0
].imshow(
    accuracy_matrix.to_numpy(),
    aspect="auto",
    vmin=50,
    vmax=100,
)

axes[
    0
].set_title(
    "GraphSAGE test accuracy (%)"
)

axes[
    0
].set_xticks(
    np.arange(
        len(
            WIDTHS
        )
    )
)

axes[
    0
].set_xticklabels(
    WIDTHS
)

axes[
    0
].set_yticks(
    np.arange(
        len(
            DISTANCES
        )
    )
)

axes[
    0
].set_yticklabels(
    DISTANCES
)

axes[
    0
].set_xlabel(
    "Bottleneck width"
)

axes[
    0
].set_ylabel(
    "Source-target distance"
)

for i in range(
    len(
        DISTANCES
    )
):
    for j in range(
        len(
            WIDTHS
        )
    ):
        axes[
            0
        ].text(
            j,
            i,
            f"{accuracy_matrix.iloc[i, j]:.1f}",
            ha="center",
            va="center",
            fontsize=8,
        )

fig.colorbar(
    im1,
    ax=axes[
        0
    ],
    fraction=0.046,
    pad=0.04,
)


im2 = axes[
    1
].imshow(
    specificity_matrix.to_numpy(),
    aspect="auto",
    vmin=0.5,
    vmax=1.0,
)

axes[
    1
].set_title(
    "GraphSAGE source-target specificity"
)

axes[
    1
].set_xticks(
    np.arange(
        len(
            WIDTHS
        )
    )
)

axes[
    1
].set_xticklabels(
    WIDTHS
)

axes[
    1
].set_yticks(
    np.arange(
        len(
            DISTANCES
        )
    )
)

axes[
    1
].set_yticklabels(
    DISTANCES
)

axes[
    1
].set_xlabel(
    "Bottleneck width"
)

axes[
    1
].set_ylabel(
    "Source-target distance"
)

for i in range(
    len(
        DISTANCES
    )
):
    for j in range(
        len(
            WIDTHS
        )
    ):
        axes[
            1
        ].text(
            j,
            i,
            f"{specificity_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8,
        )

fig.colorbar(
    im2,
    ax=axes[
        1
    ],
    fraction=0.046,
    pad=0.04,
)

fig.suptitle(
    "Structural bottleneck landscape without intervention",
    fontsize=13,
)

fig.subplots_adjust(
    left=0.08,
    right=0.97,
    bottom=0.15,
    top=0.82,
    wspace=0.32,
)

save_png(
    fig,
    DIR_HEATMAPS,
    "graphsage_bottleneck_landscape",
)


print()
print("=" * 100)
print("OVERSQUASHING TEST 1 PLOTS")
print("=" * 100)

for path in sorted(
    ROOT.rglob(
        "*.png"
    )
):
    print(
        path
    )

print()
print(
    "PNG count:",
    len(
        list(
            ROOT.rglob(
                "*.png"
            )
        )
    )
)

print(
    "PDF count:",
    len(
        list(
            ROOT.rglob(
                "*.pdf"
            )
        )
    )
)
