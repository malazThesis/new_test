from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNS = Path("runs")
ROOT = Path("plots/thesis_pairnorm_final")

DIR_COMPONENT = ROOT / "01_component_ablation"
DIR_EFFECTS = ROOT / "02_component_effects"
DIR_LR = ROOT / "03_lr_sensitivity"
DIR_PAIRWISE = ROOT / "04_supplementary_pairwise"
DIR_DATA = ROOT / "05_plot_data"

for p in [
    DIR_COMPONENT,
    DIR_EFFECTS,
    DIR_LR,
    DIR_PAIRWISE,
    DIR_DATA,
]:
    p.mkdir(
        parents=True,
        exist_ok=True,
    )


RW_FILE = (
    RUNS
    / "thesis_realworld_component_ablation_matched25.csv"
)

CSBM_FILE = (
    RUNS
    / "thesis_csbm_component_ablation_lr003_matched25.csv"
)

LR_FILE = (
    RUNS
    / "realworld_homophily_l8_lr_sensitivity_h01_all400.csv"
)

SELECTED_FILE = (
    RUNS
    / "realworld_homophily_l8_lr_sensitivity_h01_validation_selected.csv"
)


ORDER = [
    "baseline",
    "center",
    "scale",
    "full",
]

LABELS = [
    "Baseline",
    "Center only",
    "Scale only",
    "Full PairNorm",
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


def save_png(fig, directory, name):
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

    print(
        "saved:",
        path,
    )

    plt.close(fig)


def graph_means(df):
    out = (
        df.groupby(
            [
                "dataset",
                "variant",
                "graph_seed",
            ],
            as_index=False,
        )
        .agg(
            n_init=(
                "init_seed",
                "nunique",
            ),
            test_acc=(
                "test_acc",
                "mean",
            ),
            nde=(
                "normalized_dirichlet_energy",
                "mean",
            ),
            effective_rank=(
                "effective_rank",
                "mean",
            ),
            effective_rank_ratio=(
                "effective_rank_ratio",
                "mean",
            ),
            pairwise=(
                "pairwise",
                "mean",
            ),
        )
    )

    if not out[
        "n_init"
    ].eq(5).all():
        raise RuntimeError(
            out.to_string(
                index=False
            )
        )

    return out


def paired_panel(
    ax,
    frame,
    metric,
    ylabel,
    percent=False,
    ylim=None,
):
    pivot = (
        frame.pivot(
            index="graph_seed",
            columns="variant",
            values=metric,
        )
        .reindex(
            columns=ORDER
        )
        .sort_index()
    )

    if pivot.isna().any().any():
        raise RuntimeError(
            pivot.to_string()
        )

    values = pivot.to_numpy(
        dtype=float
    )

    if percent:
        values = (
            100.0
            * values
        )

    x = np.arange(
        len(ORDER)
    )

    for row in values:
        ax.plot(
            x,
            row,
            marker="o",
            markersize=3,
            linewidth=1.0,
            alpha=0.45,
            color="0.55",
            zorder=1,
        )

    mean = values.mean(
        axis=0
    )

    sd = values.std(
        axis=0,
        ddof=1,
    )

    ax.errorbar(
        x,
        mean,
        yerr=sd,
        marker="o",
        markersize=6,
        linewidth=2.2,
        capsize=4,
        color="black",
        zorder=10,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        LABELS,
        rotation=18,
        ha="right",
    )

    ax.set_ylabel(
        ylabel
    )

    if ylim is not None:
        ax.set_ylim(
            *ylim
        )

    ax.grid(
        axis="y",
        alpha=0.18,
    )


def component_figure(
    df,
    dataset,
    title,
    filename,
):
    frame = graph_means(
        df[
            df["dataset"].eq(
                dataset
            )
        ].copy()
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            11.6,
            3.8,
        ),
        constrained_layout=True,
    )

    paired_panel(
        axes[0],
        frame,
        "test_acc",
        "Test accuracy (%)",
        percent=True,
        ylim=(
            0,
            100,
        ),
    )

    paired_panel(
        axes[1],
        frame,
        "nde",
        "Normalized Dirichlet energy",
    )

    axes[1].set_ylim(
        bottom=0
    )

    paired_panel(
        axes[2],
        frame,
        "effective_rank",
        "Effective rank",
    )

    axes[2].set_ylim(
        bottom=0
    )

    axes[0].set_title(
        "(a) Predictive performance"
    )

    axes[1].set_title(
        "(b) Dirichlet energy"
    )

    axes[2].set_title(
        "(c) Effective rank"
    )

    fig.suptitle(
        title,
        fontsize=12,
    )

    save_png(
        fig,
        DIR_COMPONENT,
        filename,
    )


def pairwise_figure(
    df,
    dataset,
    title,
    filename,
):
    frame = graph_means(
        df[
            df["dataset"].eq(
                dataset
            )
        ].copy()
    )

    fig, ax = plt.subplots(
        figsize=(
            5.8,
            3.9,
        ),
        constrained_layout=True,
    )

    paired_panel(
        ax,
        frame,
        "pairwise",
        "Mean pairwise cosine distance",
        ylim=(
            0,
            1.06,
        ),
    )

    ax.set_title(
        title
    )

    save_png(
        fig,
        DIR_PAIRWISE,
        filename,
    )


def realworld_effects(df):
    graph = graph_means(
        df
    )

    output_rows = []

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            9.7,
            4.0,
        ),
        constrained_layout=True,
    )

    conditions = [
        (
            axes[0],
            "PubMed",
            "PubMed, h=0.1",
        ),
        (
            axes[1],
            "Roman-Empire",
            "Roman-Empire, h=0.9",
        ),
    ]

    for ax, dataset, title in conditions:
        sub = graph[
            graph[
                "dataset"
            ].eq(dataset)
        ]

        pivot = (
            sub.pivot(
                index="graph_seed",
                columns="variant",
                values="test_acc",
            )
            .reindex(
                columns=ORDER
            )
            .sort_index()
        )

        effects = pd.DataFrame(
            {
                "center":
                    100.0
                    * (
                        pivot["center"]
                        - pivot["baseline"]
                    ),
                "scale":
                    100.0
                    * (
                        pivot["scale"]
                        - pivot["baseline"]
                    ),
                "full":
                    100.0
                    * (
                        pivot["full"]
                        - pivot["baseline"]
                    ),
            },
            index=pivot.index,
        )

        x = np.arange(
            3
        )

        for graph_seed, row in effects.iterrows():
            values = row.to_numpy(
                dtype=float
            )

            ax.plot(
                x,
                values,
                marker="o",
                markersize=3,
                linewidth=1.0,
                alpha=0.45,
                color="0.55",
            )

            for variant, value in zip(
                [
                    "center",
                    "scale",
                    "full",
                ],
                values,
            ):
                output_rows.append(
                    {
                        "dataset":
                            dataset,
                        "graph_seed":
                            graph_seed,
                        "variant":
                            variant,
                        "accuracy_delta_pp":
                            value,
                    }
                )

        mean = effects.mean(
            axis=0
        ).to_numpy()

        sd = effects.std(
            axis=0,
            ddof=1,
        ).to_numpy()

        ax.errorbar(
            x,
            mean,
            yerr=sd,
            marker="o",
            markersize=6,
            linewidth=2.2,
            capsize=4,
            color="black",
            zorder=10,
        )

        ax.axhline(
            0,
            linewidth=1,
            color="0.3",
        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            [
                "Center",
                "Scale",
                "Full PairNorm",
            ]
        )

        ax.set_ylabel(
            "Test accuracy change vs baseline (pp)"
        )

        ax.set_title(
            title
        )

        ax.grid(
            axis="y",
            alpha=0.18,
        )

    fig.suptitle(
        "PairNorm component effects relative to matched GraphSAGE baseline\n"
        "GraphSAGE L8, lr=0.01; each gray line is one graph averaged over five initializations",
        fontsize=12,
    )

    save_png(
        fig,
        DIR_EFFECTS,
        "realworld_component_effects_vs_baseline",
    )

    pd.DataFrame(
        output_rows
    ).to_csv(
        DIR_DATA
        / "realworld_component_graph_level_effects.csv",
        index=False,
    )


def find_col(
    df,
    names,
):
    for name in names:
        if name in df.columns:
            return name

    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    for name in names:
        if name.lower() in lower:
            return lower[
                name.lower()
            ]

    raise RuntimeError(
        f"Missing columns {names}; "
        f"available={list(df.columns)}"
    )


def lr_sensitivity():
    if not LR_FILE.exists():
        print(
            "SKIP LR plot:",
            LR_FILE,
            "not found",
        )
        return

    df = pd.read_csv(
        LR_FILE
    )

    dataset_col = find_col(
        df,
        [
            "dataset",
        ],
    )

    model_col = find_col(
        df,
        [
            "model",
        ],
    )

    lr_col = find_col(
        df,
        [
            "lr",
            "learning_rate",
        ],
    )

    graph_col = find_col(
        df,
        [
            "graph_seed",
        ],
    )

    test_col = find_col(
        df,
        [
            "test_best",
            "best_test_acc_at_best_val",
            "test_acc",
        ],
    )

    graph = (
        df.groupby(
            [
                dataset_col,
                model_col,
                lr_col,
                graph_col,
            ],
            as_index=False,
        )
        .agg(
            test=(
                test_col,
                "mean",
            )
        )
    )

    summary = (
        graph.groupby(
            [
                dataset_col,
                model_col,
                lr_col,
            ],
            as_index=False,
        )
        .agg(
            mean=(
                "test",
                "mean",
            ),
            sd=(
                "test",
                "std",
            ),
            n_graphs=(
                graph_col,
                "nunique",
            ),
        )
    )

    if not summary[
        "n_graphs"
    ].eq(5).all():
        raise RuntimeError(
            summary.to_string(
                index=False
            )
        )

    selected = None

    if SELECTED_FILE.exists():
        selected = pd.read_csv(
            SELECTED_FILE
        )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            9.8,
            4.0,
        ),
        sharey=True,
        constrained_layout=True,
    )

    models = [
        (
            "GraphSAGE",
            "GraphSAGE",
            "o",
        ),
        (
            "GraphSAGEPairNorm",
            "GraphSAGE + PairNorm",
            "s",
        ),
    ]

    for ax, dataset in zip(
        axes,
        [
            "PubMed",
            "Roman-Empire",
        ],
    ):
        for model, label, marker in models:
            sub = summary[
                summary[
                    dataset_col
                ].eq(dataset)
                & summary[
                    model_col
                ].eq(model)
            ].sort_values(
                lr_col
            )

            if sub.empty:
                raise RuntimeError(
                    f"No LR rows for "
                    f"{dataset} {model}"
                )

            x = sub[
                lr_col
            ].to_numpy(
                dtype=float
            )

            y = (
                100.0
                * sub[
                    "mean"
                ].to_numpy(
                    dtype=float
                )
            )

            err = (
                100.0
                * sub[
                    "sd"
                ].to_numpy(
                    dtype=float
                )
            )

            line = ax.errorbar(
                x,
                y,
                yerr=err,
                marker=marker,
                linewidth=2.0,
                markersize=6,
                capsize=3,
                label=label,
            )

            if selected is not None:
                sel_dataset_col = find_col(
                    selected,
                    [
                        "dataset",
                    ],
                )

                sel_model_col = find_col(
                    selected,
                    [
                        "model",
                    ],
                )

                sel_lr_col = find_col(
                    selected,
                    [
                        "lr",
                        "learning_rate",
                    ],
                )

                sel = selected[
                    selected[
                        sel_dataset_col
                    ].eq(dataset)
                    & selected[
                        sel_model_col
                    ].eq(model)
                ]

                if len(sel) == 1:
                    lr_sel = float(
                        sel.iloc[0][
                            sel_lr_col
                        ]
                    )

                    match = sub[
                        np.isclose(
                            sub[
                                lr_col
                            ].astype(float),
                            lr_sel,
                        )
                    ]

                    if len(match) == 1:
                        y_sel = (
                            100.0
                            * float(
                                match.iloc[0][
                                    "mean"
                                ]
                            )
                        )

                        ax.scatter(
                            [
                                lr_sel
                            ],
                            [
                                y_sel
                            ],
                            marker="*",
                            s=120,
                            color=line[
                                0
                            ].get_color(),
                            zorder=20,
                        )

        ax.set_xscale(
            "log"
        )

        ax.set_xticks(
            [
                0.001,
                0.003,
                0.01,
                0.03,
            ]
        )

        ax.set_xticklabels(
            [
                ".001",
                ".003",
                ".01",
                ".03",
            ]
        )

        ax.set_xlabel(
            "Learning rate"
        )

        ax.set_title(
            dataset
        )

        ax.grid(
            axis="y",
            alpha=0.18,
        )

    axes[0].set_ylabel(
        "Test accuracy at best validation checkpoint (%)"
    )

    axes[0].set_ylim(
        0,
        100,
    )

    axes[0].legend(
        frameon=False,
    )

    fig.suptitle(
        "Learning-rate sensitivity under matched controlled conditions\n"
        "h=0.1, GraphSAGE L8, five graphs × five initializations; stars denote validation-selected learning rates",
        fontsize=12,
    )

    save_png(
        fig,
        DIR_LR,
        "pubmed_roman_h01_lr_sensitivity",
    )

    summary.to_csv(
        DIR_DATA
        / "lr_sensitivity_graph_level_summary.csv",
        index=False,
    )


rw = pd.read_csv(
    RW_FILE
)

csbm = pd.read_csv(
    CSBM_FILE
)

assert len(
    rw
) == 200

assert len(
    csbm
) == 100

rw_counts = (
    rw.groupby(
        [
            "dataset",
            "variant",
        ]
    )
    .size()
)

csbm_counts = (
    csbm.groupby(
        [
            "dataset",
            "variant",
        ]
    )
    .size()
)

assert rw_counts.eq(
    25
).all()

assert csbm_counts.eq(
    25
).all()


graph_means(
    rw
).to_csv(
    DIR_DATA
    / "realworld_component_graph_means.csv",
    index=False,
)

graph_means(
    csbm
).to_csv(
    DIR_DATA
    / "csbm_component_graph_means.csv",
    index=False,
)


component_figure(
    rw,
    "PubMed",
    (
        "PubMed: PairNorm component ablation\n"
        "h=0.1, lr=0.01, GraphSAGE L8, epoch 2400; "
        "5 graphs × 5 initializations"
    ),
    "pubmed_component_ablation_h01_lr001",
)


component_figure(
    rw,
    "Roman-Empire",
    (
        "Roman-Empire: PairNorm component ablation\n"
        "h=0.9, lr=0.01, GraphSAGE L8, epoch 2400; "
        "5 graphs × 5 initializations"
    ),
    "roman_component_ablation_h09_lr001",
)


component_figure(
    csbm,
    "cSBM",
    (
        "cSBM: PairNorm component ablation\n"
        "h=0.1, feature signal=0.50, lr=0.03, GraphSAGE L8, epoch 200; "
        "5 graphs × 5 initializations"
    ),
    "csbm_component_ablation_h01_fs050_lr003",
)


realworld_effects(
    rw
)


lr_sensitivity()


pairwise_figure(
    rw,
    "PubMed",
    (
        "PubMed: supporting pairwise-distance diagnostic\n"
        "h=0.1, lr=0.01, GraphSAGE L8, epoch 2400"
    ),
    "pubmed_pairwise_h01_lr001",
)


pairwise_figure(
    rw,
    "Roman-Empire",
    (
        "Roman-Empire: supporting pairwise-distance diagnostic\n"
        "h=0.9, lr=0.01, GraphSAGE L8, epoch 2400"
    ),
    "roman_pairwise_h09_lr001",
)


pairwise_figure(
    csbm,
    "cSBM",
    (
        "cSBM: supporting pairwise-distance diagnostic\n"
        "h=0.1, feature signal=0.50, lr=0.03, GraphSAGE L8"
    ),
    "csbm_pairwise_h01_fs050_lr003",
)


print()
print("=" * 100)
print("FINAL PNG FILES")
print("=" * 100)

for p in sorted(
    ROOT.rglob(
        "*.png"
    )
):
    print(
        p
    )

print()
print("PDF count:")

print(
    len(
        list(
            ROOT.rglob(
                "*.pdf"
            )
        )
    )
)
