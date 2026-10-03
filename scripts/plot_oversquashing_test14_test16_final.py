from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# PATHS
# ============================================================

TEST14_ROOTS = [
    Path(
        "runs/"
        "longrange_bottleneck_replicated_v2"
    ),
    Path(
        "runs/"
        "longrange_bottleneck_pairnorm_replicated_v2"
    ),
]

TEST16_ROOT = Path(
    "runs/"
    "longrange_bottleneck_jacobian_v1"
)

FIG14 = Path(
    "figures/"
    "oversquashing_test14"
)

TAB14 = Path(
    "tables/"
    "oversquashing_test14"
)

FIG16 = Path(
    "figures/"
    "oversquashing_test16"
)

TAB16 = Path(
    "tables/"
    "oversquashing_test16"
)

for p in [
    FIG14,
    TAB14,
    FIG16,
    TAB16,
]:
    p.mkdir(
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
        "PairNorm",

    "GraphSAGERewired":
        "Rewired",

    "GraphSAGEPairNormRewired":
        "PairNorm + Rewired",
}

MODEL_SHORT = {
    "GraphSAGE": "GS",
    "GraphSAGEPairNorm": "PN",
    "GraphSAGERewired": "RW",
    "GraphSAGEPairNormRewired": "PN+RW",
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def pick(
    obj,
    names,
    path,
    required=True,
    default=None,
):
    for name in names:
        if name in obj:
            value = obj[name]

            if value is not None:
                return value

    if required:
        raise KeyError(
            f"{path}: none of {names} found"
        )

    return default


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def specificity(
    matched,
    distractor,
):
    matched = float(matched)
    distractor = float(distractor)

    denominator = (
        matched + distractor
    )

    if denominator <= 0:
        return np.nan

    return (
        matched
        / denominator
    )


def save_figure(
    fig,
    stem,
):
    fig.tight_layout()

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
# TEST 14 — LOAD 1200 RUNS
# ============================================================

rows14 = []

for root in TEST14_ROOTS:
    files = sorted(
        root.rglob(
            "*_summary.json"
        )
    )

    print(
        "TEST14",
        root,
        "summary JSONs:",
        len(files),
    )

    for path in files:
        j = load_json(path)

        model = str(
            pick(
                j,
                [
                    "model",
                    "model_name",
                    "model_type",
                ],
                path,
            )
        )

        if model not in MODEL_ORDER:
            continue

        distance = int(
            pick(
                j,
                [
                    "distance",
                    "source_target_distance",
                    "path_length",
                ],
                path,
            )
        )

        width = int(
            pick(
                j,
                [
                    "width",
                    "bottleneck_width",
                    "bottleneck_size",
                ],
                path,
            )
        )

        graph_seed = int(
            pick(
                j,
                [
                    "graph_seed",
                    "seed_graph",
                ],
                path,
            )
        )

        init_seed = int(
            pick(
                j,
                [
                    "init_seed",
                    "seed",
                    "model_seed",
                ],
                path,
            )
        )

        accuracy = float(
            pick(
                j,
                [
                    "best_test_accuracy_at_best_val",
                    "best_test_acc_at_best_val",
                    "test_acc",
                ],
                path,
            )
        )

        matched = float(
            pick(
                j,
                [
                    "matched_influence_mean",
                    "matched_influence",
                ],
                path,
            )
        )

        distractor = float(
            pick(
                j,
                [
                    "distractor_influence_mean",
                    "distractor_influence",
                ],
                path,
            )
        )

        rows14.append({
            "distance":
                distance,

            "width":
                width,

            "model":
                model,

            "graph_seed":
                graph_seed,

            "init_seed":
                init_seed,

            "accuracy":
                accuracy,

            "specificity":
                specificity(
                    matched,
                    distractor,
                ),

            "matched_influence":
                matched,

            "distractor_influence":
                distractor,

            "summary_path":
                str(path),
        })


t14 = pd.DataFrame(
    rows14
)

print()
print(
    "TEST14 loaded rows:",
    len(t14),
)

assert len(t14) == 1200

key14 = [
    "distance",
    "width",
    "model",
    "graph_seed",
    "init_seed",
]

assert (
    t14.duplicated(
        key14
    ).sum()
    == 0
)

assert set(
    t14["distance"]
) == {
    2, 4, 6, 8, 10,
}

assert set(
    t14["width"]
) == {
    1, 2, 4, 8,
}

assert set(
    t14["model"]
) == set(
    MODEL_ORDER
)

counts14 = (
    t14.groupby(
        [
            "distance",
            "width",
            "model",
        ]
    )
    .size()
)

assert counts14.eq(15).all()

assert np.isfinite(
    t14[
        [
            "accuracy",
            "specificity",
        ]
    ].to_numpy()
).all()

print(
    "TEST14 LOAD AUDIT: PASS"
)


# ============================================================
# TEST 14 FIGURE A
# GraphSAGE accuracy heatmap
# ============================================================

gs = (
    t14[
        t14["model"]
        == "GraphSAGE"
    ]
    .groupby(
        [
            "distance",
            "width",
        ]
    )["accuracy"]
    .mean()
    .mul(100)
    .unstack(
        "width"
    )
    .reindex(
        index=[
            2, 4, 6, 8, 10,
        ],
        columns=[
            1, 2, 4, 8,
        ],
    )
)

fig, ax = plt.subplots(
    figsize=(6.4, 4.8)
)

im = ax.imshow(
    gs.to_numpy(),
    aspect="auto",
)

ax.set_xticks(
    np.arange(
        len(gs.columns)
    )
)

ax.set_xticklabels(
    gs.columns
)

ax.set_yticks(
    np.arange(
        len(gs.index)
    )
)

ax.set_yticklabels(
    gs.index
)

ax.set_xlabel(
    "Bottleneck width"
)

ax.set_ylabel(
    "Source–target distance / GNN depth"
)

ax.set_title(
    "Controlled bottleneck benchmark: GraphSAGE accuracy"
)

for i in range(
    len(gs.index)
):
    for j in range(
        len(gs.columns)
    ):
        value = float(
            gs.iloc[i, j]
        )

        ax.text(
            j,
            i,
            f"{value:.1f}",
            ha="center",
            va="center",
        )

cbar = fig.colorbar(
    im,
    ax=ax,
)

cbar.set_label(
    "Test accuracy [%]"
)

save_figure(
    fig,
    FIG14
    / "test14_graphsage_accuracy_heatmap",
)


# ============================================================
# TEST 14 FIGURE B/C
#
# Hard regime:
# distance >= 6 and width <= 2
#
# Average init seeds first, then structural cells within
# graph seed, leaving n=5 graph-seed replicates.
# ============================================================

hard = t14[
    (
        t14["distance"]
        >= 6
    )
    &
    (
        t14["width"]
        <= 2
    )
].copy()

hard_init = (
    hard.groupby(
        [
            "model",
            "graph_seed",
            "distance",
            "width",
        ],
        as_index=False,
    )
    .agg(
        accuracy=(
            "accuracy",
            "mean",
        ),
        specificity=(
            "specificity",
            "mean",
        ),
    )
)

hard_graph = (
    hard_init.groupby(
        [
            "model",
            "graph_seed",
        ],
        as_index=False,
    )
    .agg(
        accuracy=(
            "accuracy",
            "mean",
        ),
        specificity=(
            "specificity",
            "mean",
        ),
    )
)

assert (
    hard_graph.groupby(
        "model"
    )
    .size()
    .eq(5)
    .all()
)

hard_summary = (
    hard_graph.groupby(
        "model"
    )
    .agg(
        accuracy_mean=(
            "accuracy",
            "mean",
        ),
        accuracy_sd=(
            "accuracy",
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
    .reindex(
        MODEL_ORDER
    )
    .reset_index()
)

# Accuracy
fig, ax = plt.subplots(
    figsize=(7.0, 4.4)
)

x = np.arange(
    len(
        hard_summary
    )
)

ax.bar(
    x,
    100
    * hard_summary[
        "accuracy_mean"
    ],
    yerr=(
        100
        * hard_summary[
            "accuracy_sd"
        ]
    ),
    capsize=4,
)

ax.set_xticks(x)

ax.set_xticklabels(
    [
        MODEL_SHORT[m]
        for m
        in hard_summary[
            "model"
        ]
    ]
)

ax.set_ylabel(
    "Test accuracy [%]"
)

ax.set_title(
    "Hard long-range regime: predictive recovery"
)

ax.grid(
    axis="y",
    alpha=0.25,
)

save_figure(
    fig,
    FIG14
    / "test14_hard_regime_accuracy",
)

# Specificity
fig, ax = plt.subplots(
    figsize=(7.0, 4.4)
)

ax.bar(
    x,
    hard_summary[
        "specificity_mean"
    ],
    yerr=hard_summary[
        "specificity_sd"
    ],
    capsize=4,
)

ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1,
)

ax.set_xticks(x)

ax.set_xticklabels(
    [
        MODEL_SHORT[m]
        for m
        in hard_summary[
            "model"
        ]
    ]
)

ax.set_ylabel(
    "Matched-source specificity"
)

ax.set_title(
    "Hard long-range regime: source-specific influence"
)

ax.set_ylim(
    0,
    1.05,
)

ax.grid(
    axis="y",
    alpha=0.25,
)

save_figure(
    fig,
    FIG14
    / "test14_hard_regime_specificity",
)


# ============================================================
# TEST 14 TABLE A — baseline accuracy matrix
# ============================================================

gs_table = gs.copy()

gs_table.index.name = (
    "Distance"
)

gs_table.columns = [
    f"B={x}"
    for x in gs_table.columns
]

gs_table = (
    gs_table
    .round(2)
)

gs_table.to_csv(
    TAB14
    / "test14_graphsage_accuracy_matrix.csv"
)

with open(
    TAB14
    / "test14_graphsage_accuracy_matrix.tex",
    "w",
) as f:
    f.write(
        gs_table.to_latex(
            escape=False,
        )
    )


# ============================================================
# TEST 14 TABLE B — hard regime
# ============================================================

hard_table = []

for _, r in (
    hard_summary.iterrows()
):
    hard_table.append({
        "Variant":
            MODEL_LABEL[
                r["model"]
            ],

        "Accuracy [\\%]":
            (
                f"{100*r['accuracy_mean']:.2f}"
                f" $\\pm$ "
                f"{100*r['accuracy_sd']:.2f}"
            ),

        "Specificity":
            (
                f"{r['specificity_mean']:.4f}"
                f" $\\pm$ "
                f"{r['specificity_sd']:.4f}"
            ),
    })

hard_table = pd.DataFrame(
    hard_table
)

hard_table.to_csv(
    TAB14
    / "test14_hard_regime_summary.csv",
    index=False,
)

with open(
    TAB14
    / "test14_hard_regime_summary.tex",
    "w",
) as f:
    f.write(
        hard_table.to_latex(
            index=False,
            escape=False,
        )
    )


# ============================================================
# TEST 16 — LOAD 240 RUNS
# ============================================================

rows16 = []

files16 = sorted(
    TEST16_ROOT.rglob(
        "*_summary.json"
    )
)

print()
print(
    "TEST16 summary JSONs:",
    len(files16),
)

for path in files16:
    j = load_json(path)

    model = str(
        pick(
            j,
            [
                "model",
                "model_name",
                "model_type",
            ],
            path,
        )
    )

    if model not in MODEL_ORDER:
        continue

    distance = int(
        pick(
            j,
            [
                "distance",
                "source_target_distance",
                "path_length",
            ],
            path,
        )
    )

    width = int(
        pick(
            j,
            [
                "width",
                "bottleneck_width",
                "bottleneck_size",
            ],
            path,
        )
    )

    graph_seed = int(
        pick(
            j,
            [
                "graph_seed",
                "seed_graph",
            ],
            path,
        )
    )

    init_seed = int(
        pick(
            j,
            [
                "init_seed",
                "seed",
                "model_seed",
            ],
            path,
        )
    )

    accuracy = float(
        pick(
            j,
            [
                "best_test_accuracy_at_best_val",
                "best_test_acc_at_best_val",
                "test_acc",
            ],
            path,
        )
    )

    fd_m = float(
        pick(
            j,
            [
                "matched_influence_mean",
                "matched_influence",
            ],
            path,
        )
    )

    fd_d = float(
        pick(
            j,
            [
                "distractor_influence_mean",
                "distractor_influence",
            ],
            path,
        )
    )

    sig_m = float(
        pick(
            j,
            [
                "jacobian_matched_signal_mean",
            ],
            path,
        )
    )

    sig_d = float(
        pick(
            j,
            [
                "jacobian_distractor_signal_mean",
            ],
            path,
        )
    )

    fro_m = float(
        pick(
            j,
            [
                "jacobian_matched_fro_mean",
            ],
            path,
        )
    )

    fro_d = float(
        pick(
            j,
            [
                "jacobian_distractor_fro_mean",
            ],
            path,
        )
    )

    rows16.append({
        "distance":
            distance,

        "width":
            width,

        "model":
            model,

        "graph_seed":
            graph_seed,

        "init_seed":
            init_seed,

        "accuracy":
            accuracy,

        "fd_specificity":
            specificity(
                fd_m,
                fd_d,
            ),

        "signal_jac_specificity":
            specificity(
                sig_m,
                sig_d,
            ),

        "fro_jac_specificity":
            specificity(
                fro_m,
                fro_d,
            ),

        "summary_path":
            str(path),
    })


t16 = pd.DataFrame(
    rows16
)

print(
    "TEST16 loaded rows:",
    len(t16),
)

assert len(t16) == 240

key16 = [
    "distance",
    "width",
    "model",
    "graph_seed",
    "init_seed",
]

assert (
    t16.duplicated(
        key16
    ).sum()
    == 0
)

assert set(
    zip(
        t16["distance"],
        t16["width"],
    )
) == {
    (2, 1),
    (2, 8),
    (8, 1),
    (8, 8),
}

counts16 = (
    t16.groupby(
        [
            "distance",
            "width",
            "model",
        ]
    )
    .size()
)

assert counts16.eq(15).all()

assert np.isfinite(
    t16[
        [
            "accuracy",
            "fd_specificity",
            "signal_jac_specificity",
            "fro_jac_specificity",
        ]
    ].to_numpy()
).all()

print(
    "TEST16 LOAD AUDIT: PASS"
)


# ============================================================
# TEST 16 CONDITION MEANS
# ============================================================

cond16 = (
    t16.groupby(
        [
            "distance",
            "width",
            "model",
        ],
        as_index=False,
    )
    .agg(
        accuracy=(
            "accuracy",
            "mean",
        ),
        fd_specificity=(
            "fd_specificity",
            "mean",
        ),
        signal_jac_specificity=(
            "signal_jac_specificity",
            "mean",
        ),
        fro_jac_specificity=(
            "fro_jac_specificity",
            "mean",
        ),
    )
)

assert len(cond16) == 16


# ============================================================
# TEST 16 FIGURE A
# FD vs signal-coordinate Jacobian
# ============================================================

pearson = stats.pearsonr(
    cond16[
        "fd_specificity"
    ],
    cond16[
        "signal_jac_specificity"
    ],
)

spearman = stats.spearmanr(
    cond16[
        "fd_specificity"
    ],
    cond16[
        "signal_jac_specificity"
    ],
)

fig, ax = plt.subplots(
    figsize=(6.2, 5.4)
)

markers = {
    "GraphSAGE": "o",
    "GraphSAGEPairNorm": "s",
    "GraphSAGERewired": "^",
    "GraphSAGEPairNormRewired": "D",
}

for model in MODEL_ORDER:
    g = cond16[
        cond16["model"]
        == model
    ]

    ax.scatter(
        g[
            "fd_specificity"
        ],
        g[
            "signal_jac_specificity"
        ],
        marker=markers[
            model
        ],
        s=55,
        label=MODEL_SHORT[
            model
        ],
    )

ax.plot(
    [0.45, 1.01],
    [0.45, 1.01],
    linestyle="--",
    linewidth=1,
)

ax.set_xlim(
    0.45,
    1.01,
)

ax.set_ylim(
    0.45,
    1.01,
)

ax.set_xlabel(
    "Finite-difference specificity"
)

ax.set_ylabel(
    "Signal-Jacobian specificity"
)

ax.set_title(
    "Agreement of direct source–target sensitivity diagnostics"
)

ax.text(
    0.47,
    0.97,
    (
        f"Pearson r = {pearson.statistic:.3f}\n"
        f"Spearman ρ = {spearman.statistic:.3f}"
    ),
    va="top",
)

ax.grid(
    alpha=0.25,
)

ax.legend(
    frameon=False,
)

save_figure(
    fig,
    FIG16
    / "test16_fd_vs_signal_jacobian",
)


# ============================================================
# TEST 16 FIGURE B
# Signal-Jacobian specificity across structural cells
# ============================================================

cells = [
    (2, 1),
    (2, 8),
    (8, 1),
    (8, 8),
]

cell_labels = [
    "d2 / B1",
    "d2 / B8",
    "d8 / B1",
    "d8 / B8",
]

x = np.arange(
    len(cells)
)

bar_width = 0.19

fig, ax = plt.subplots(
    figsize=(8.0, 4.6)
)

for i, model in enumerate(
    MODEL_ORDER
):
    vals = []

    for d, b in cells:
        z = cond16[
            (
                cond16[
                    "distance"
                ]
                == d
            )
            &
            (
                cond16[
                    "width"
                ]
                == b
            )
            &
            (
                cond16[
                    "model"
                ]
                == model
            )
        ]

        assert len(z) == 1

        vals.append(
            float(
                z[
                    "signal_jac_specificity"
                ].iloc[0]
            )
        )

    ax.bar(
        x
        + (
            i - 1.5
        )
        * bar_width,
        vals,
        width=bar_width,
        label=MODEL_SHORT[
            model
        ],
    )

ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1,
)

ax.set_xticks(x)

ax.set_xticklabels(
    cell_labels
)

ax.set_ylabel(
    "Signal-Jacobian specificity"
)

ax.set_title(
    "Direct long-range source specificity"
)

ax.set_ylim(
    0,
    1.05,
)

ax.grid(
    axis="y",
    alpha=0.25,
)

ax.legend(
    frameon=False,
    ncol=4,
)

save_figure(
    fig,
    FIG16
    / "test16_signal_jacobian_specificity",
)


# ============================================================
# TEST 16 TABLE A
# ============================================================

table16 = cond16.copy()

table16[
    "Model"
] = table16[
    "model"
].map(
    MODEL_LABEL
)

table16[
    "Accuracy [\\%]"
] = (
    100
    * table16[
        "accuracy"
    ]
).map(
    lambda x:
        f"{x:.2f}"
)

table16[
    "FD specificity"
] = table16[
    "fd_specificity"
].map(
    lambda x:
        f"{x:.4f}"
)

table16[
    "Signal-Jac specificity"
] = table16[
    "signal_jac_specificity"
].map(
    lambda x:
        f"{x:.4f}"
)

table16[
    "Frobenius-Jac specificity"
] = table16[
    "fro_jac_specificity"
].map(
    lambda x:
        f"{x:.4f}"
)

table16 = (
    table16[
        [
            "distance",
            "width",
            "Model",
            "Accuracy [\\%]",
            "FD specificity",
            "Signal-Jac specificity",
            "Frobenius-Jac specificity",
        ]
    ]
    .rename(
        columns={
            "distance":
                "Distance",

            "width":
                "Width",
        }
    )
    .sort_values(
        [
            "Distance",
            "Width",
            "Model",
        ]
    )
)

table16.to_csv(
    TAB16
    / "test16_direct_sensitivity_summary.csv",
    index=False,
)

with open(
    TAB16
    / "test16_direct_sensitivity_summary.tex",
    "w",
) as f:
    f.write(
        table16.to_latex(
            index=False,
            escape=False,
        )
    )


# ============================================================
# TEST 16 TABLE B — diagnostic agreement
# ============================================================

corr_rows = []

corr_rows.append({
    "Scope":
        "All condition means",

    "Pearson r":
        pearson.statistic,

    "Spearman rho":
        spearman.statistic,
})

for model in MODEL_ORDER:
    g = cond16[
        cond16["model"]
        == model
    ]

    pr = stats.pearsonr(
        g[
            "fd_specificity"
        ],
        g[
            "signal_jac_specificity"
        ],
    )

    sr = stats.spearmanr(
        g[
            "fd_specificity"
        ],
        g[
            "signal_jac_specificity"
        ],
    )

    corr_rows.append({
        "Scope":
            MODEL_LABEL[
                model
            ],

        "Pearson r":
            pr.statistic,

        "Spearman rho":
            sr.statistic,
    })

corr = pd.DataFrame(
    corr_rows
)

corr[
    "Pearson r"
] = corr[
    "Pearson r"
].map(
    lambda x:
        f"{x:.4f}"
)

corr[
    "Spearman rho"
] = corr[
    "Spearman rho"
].map(
    lambda x:
        f"{x:.4f}"
)

corr.to_csv(
    TAB16
    / "test16_diagnostic_agreement.csv",
    index=False,
)

with open(
    TAB16
    / "test16_diagnostic_agreement.tex",
    "w",
) as f:
    f.write(
        corr.to_latex(
            index=False,
            escape=False,
        )
    )


# ============================================================
# FINAL
# ============================================================

print()
print(
    "TEST14 FIGURES:",
    FIG14,
)

print(
    "TEST14 TABLES:",
    TAB14,
)

print(
    "TEST16 FIGURES:",
    FIG16,
)

print(
    "TEST16 TABLES:",
    TAB16,
)

print()
print(
    "TEST 14 / TEST 16 FINAL PLOTS/TABLES: PASS"
)
