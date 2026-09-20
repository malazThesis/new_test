from pathlib import Path

import numpy as np
import pandas as pd


RUNS = Path("runs")

SOURCE = (
    RUNS
    / "longrange_bottleneck_2x2_graph_means.csv"
)

OUT = Path(
    "tables/oversquashing_longrange_bottleneck_final"
)

OUT.mkdir(
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
        "+ PairNorm",
    "GraphSAGERewired":
        "+ Rewiring",
    "GraphSAGEPairNormRewired":
        "PairNorm + Rewiring",
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


def mean_sd(
    mean,
    sd,
    digits=2,
):
    return (
        f"{mean:.{digits}f} ± "
        f"{sd:.{digits}f}"
    )


def write_table(
    df,
    stem,
    caption,
    label,
):
    csv_path = (
        OUT
        / f"{stem}.csv"
    )

    tex_path = (
        OUT
        / f"{stem}.tex"
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    latex = df.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        position="htbp",
    )

    tex_path.write_text(
        latex
    )

    print(
        "saved:",
        csv_path,
    )

    print(
        "saved:",
        tex_path,
    )


df = pd.read_csv(
    SOURCE
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


counts = (
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

assert counts.eq(
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
        specificity_mean=(
            "specificity",
            "mean",
        ),
        specificity_sd=(
            "specificity",
            "std",
        ),
        influence_mean=(
            "matched_influence",
            "mean",
        ),
        influence_sd=(
            "matched_influence",
            "std",
        ),
    )
)

assert summary[
    "n_graphs"
].eq(
    5
).all()


accuracy_rows = []

for distance in DISTANCES:
    for width in WIDTHS:
        row = {
            "Distance":
                distance,
            "Width":
                width,
        }

        for model in MODEL_ORDER:
            x = summary[
                summary[
                    "model"
                ].eq(model)
                & summary[
                    "distance"
                ].eq(distance)
                & summary[
                    "width"
                ].eq(width)
            ]

            assert len(x) == 1

            x = x.iloc[
                0
            ]

            row[
                MODEL_LABEL[
                    model
                ]
            ] = mean_sd(
                100.0
                * x[
                    "accuracy_mean"
                ],
                100.0
                * x[
                    "accuracy_sd"
                ],
                2,
            )

        accuracy_rows.append(
            row
        )


accuracy_table = pd.DataFrame(
    accuracy_rows
)

write_table(
    accuracy_table,
    "01_accuracy_2x2",
    (
        "Test accuracy in the long-range bottleneck benchmark. "
        "Values are mean $\\pm$ SD over five graph seeds after averaging "
        "three initialization seeds per graph."
    ),
    "tab:longrange_accuracy",
)


specificity_rows = []

for distance in DISTANCES:
    for width in WIDTHS:
        row = {
            "Distance":
                distance,
            "Width":
                width,
        }

        for model in MODEL_ORDER:
            x = summary[
                summary[
                    "model"
                ].eq(model)
                & summary[
                    "distance"
                ].eq(distance)
                & summary[
                    "width"
                ].eq(width)
            ]

            assert len(x) == 1

            x = x.iloc[
                0
            ]

            row[
                MODEL_LABEL[
                    model
                ]
            ] = mean_sd(
                x[
                    "specificity_mean"
                ],
                x[
                    "specificity_sd"
                ],
                3,
            )

        specificity_rows.append(
            row
        )


specificity_table = pd.DataFrame(
    specificity_rows
)

write_table(
    specificity_table,
    "02_source_target_specificity_2x2",
    (
        "Source-to-target specificity in the long-range bottleneck benchmark. "
        "A value of 0.5 indicates equal sensitivity to the matched source and "
        "a distractor, while values approaching 1 indicate target-specific "
        "source sensitivity."
    ),
    "tab:longrange_specificity",
)


def select_model(
    model,
    acc_name,
    spec_name,
):
    x = df[
        df[
            "model"
        ].eq(model)
    ][
        [
            "distance",
            "width",
            "graph_seed",
            "test_acc",
            "specificity",
        ]
    ].copy()

    return x.rename(
        columns={
            "test_acc":
                acc_name,
            "specificity":
                spec_name,
        }
    )


base = select_model(
    "GraphSAGE",
    "acc_base",
    "spec_base",
)

pn = select_model(
    "GraphSAGEPairNorm",
    "acc_pn",
    "spec_pn",
)

rew = select_model(
    "GraphSAGERewired",
    "acc_rew",
    "spec_rew",
)

pn_rew = select_model(
    "GraphSAGEPairNormRewired",
    "acc_pn_rew",
    "spec_pn_rew",
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


effect_summary = (
    effects.groupby(
        [
            "distance",
            "width",
        ],
        as_index=False,
    )
    .agg(
        pn_mean=(
            "pairnorm_gain_pp",
            "mean",
        ),
        pn_sd=(
            "pairnorm_gain_pp",
            "std",
        ),
        rew_mean=(
            "rewiring_gain_pp",
            "mean",
        ),
        rew_sd=(
            "rewiring_gain_pp",
            "std",
        ),
        rew_after_pn_mean=(
            "rewiring_after_pairnorm_pp",
            "mean",
        ),
        rew_after_pn_sd=(
            "rewiring_after_pairnorm_pp",
            "std",
        ),
        pn_after_rew_mean=(
            "pairnorm_after_rewiring_pp",
            "mean",
        ),
        pn_after_rew_sd=(
            "pairnorm_after_rewiring_pp",
            "std",
        ),
    )
)


effect_rows = []

for r in effect_summary.itertuples():
    effect_rows.append(
        {
            "Distance":
                int(
                    r.distance
                ),
            "Width":
                int(
                    r.width
                ),
            "PairNorm gain (pp)":
                mean_sd(
                    r.pn_mean,
                    r.pn_sd,
                    2,
                ),
            "Rewiring gain (pp)":
                mean_sd(
                    r.rew_mean,
                    r.rew_sd,
                    2,
                ),
            "Rewiring after PairNorm (pp)":
                mean_sd(
                    r.rew_after_pn_mean,
                    r.rew_after_pn_sd,
                    2,
                ),
            "PairNorm after rewiring (pp)":
                mean_sd(
                    r.pn_after_rew_mean,
                    r.pn_after_rew_sd,
                    2,
                ),
        }
    )


effect_table = pd.DataFrame(
    effect_rows
)

write_table(
    effect_table,
    "03_intervention_effects",
    (
        "Accuracy effects of PairNorm and targeted rewiring. "
        "Values are percentage-point differences relative to the matched "
        "graph-level condition."
    ),
    "tab:longrange_intervention_effects",
)


hard = df[
    df[
        "distance"
    ].ge(
        6
    )
    & df[
        "width"
    ].le(
        2
    )
].copy()

assert len(
    hard
) == (
    4
    * 3
    * 2
    * 5
)


hard_rows = []

for model in MODEL_ORDER:
    x = hard[
        hard[
            "model"
        ].eq(model)
    ]

    hard_rows.append(
        {
            "Model":
                MODEL_LABEL[
                    model
                ],
            "Accuracy (%)":
                mean_sd(
                    100.0
                    * x[
                        "test_acc"
                    ].mean(),
                    100.0
                    * x[
                        "test_acc"
                    ].std(
                        ddof=1
                    ),
                    2,
                ),
            "Specificity":
                mean_sd(
                    x[
                        "specificity"
                    ].mean(),
                    x[
                        "specificity"
                    ].std(
                        ddof=1
                    ),
                    3,
                ),
            "Graph-condition means":
                len(
                    x
                ),
        }
    )


hard_table = pd.DataFrame(
    hard_rows
)

write_table(
    hard_table,
    "04_hard_regime_summary",
    (
        "Descriptive summary of the difficult long-range regime "
        "($d \\geq 6$, bottleneck width $\\leq 2$). "
        "Each entry summarizes graph-level means after averaging "
        "three initialization seeds."
    ),
    "tab:longrange_hard_regime",
)


baseline = summary[
    summary[
        "model"
    ].eq(
        "GraphSAGE"
    )
].copy()


trend_rows = []

for distance in DISTANCES:
    x = (
        baseline[
            baseline[
                "distance"
            ].eq(
                distance
            )
        ]
        .sort_values(
            "width"
        )
    )

    rho = x[
        "width"
    ].corr(
        x[
            "accuracy_mean"
        ],
        method="spearman",
    )

    trend_rows.append(
        {
            "Analysis":
                "Width effect",
            "Fixed condition":
                f"distance = {distance}",
            "Spearman rho":
                float(
                    rho
                ),
            "Direction":
                (
                    "higher width -> higher accuracy"
                    if rho > 0
                    else
                    "higher width -> lower accuracy"
                ),
        }
    )


for width in WIDTHS:
    x = (
        baseline[
            baseline[
                "width"
            ].eq(
                width
            )
        ]
        .sort_values(
            "distance"
        )
    )

    rho = x[
        "distance"
    ].corr(
        x[
            "accuracy_mean"
        ],
        method="spearman",
    )

    trend_rows.append(
        {
            "Analysis":
                "Distance effect",
            "Fixed condition":
                f"width = {width}",
            "Spearman rho":
                float(
                    rho
                ),
            "Direction":
                (
                    "higher distance -> higher accuracy"
                    if rho > 0
                    else
                    "higher distance -> lower accuracy"
                ),
        }
    )


trend_table = pd.DataFrame(
    trend_rows
)

trend_table[
    "Spearman rho"
] = trend_table[
    "Spearman rho"
].map(
    lambda x:
        f"{x:+.3f}"
)

write_table(
    trend_table,
    "05_graphsage_structural_trends",
    (
        "Descriptive Spearman trends for baseline GraphSAGE across "
        "bottleneck width and source-target distance."
    ),
    "tab:longrange_structural_trends",
)


effects.to_csv(
    OUT
    / "raw_graph_level_intervention_effects.csv",
    index=False,
)

summary.to_csv(
    OUT
    / "raw_cell_summary.csv",
    index=False,
)


print()
print("=" * 100)
print("FINAL TABLE FILES")
print("=" * 100)

for path in sorted(
    OUT.iterdir()
):
    print(
        path
    )
