from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


RUNS = Path("runs")

REFERENCE = (
    RUNS
    / "realworld_pairnorm_component_reference_matched_lr01_v2"
)

ABLATION = (
    RUNS
    / "realworld_pairnorm_component_ablation_matched_lr01_v2"
)

FACTORIAL = (
    RUNS
    / "realworld_homophily_graph_init_factorial_v2"
    / "factorial_results_600.csv"
)

OUT = (
    RUNS
    / "thesis_realworld_component_ablation_matched25.csv"
)

SUMMARY = (
    RUNS
    / "thesis_realworld_component_ablation_matched25_summary.csv"
)


MODEL_VARIANT = {
    "GraphSAGE":
        "baseline",
    "GraphSAGECenterNorm":
        "center",
    "GraphSAGEScaleNorm":
        "scale",
    "GraphSAGEPairNorm":
        "full",
}


def clean_dataset(value):
    s = str(value).lower()

    if "pubmed" in s:
        return "PubMed"

    if "roman" in s:
        return "Roman-Empire"

    raise ValueError(
        value
    )


def get_graph_seed(
    summary,
    path,
):
    value = summary.get(
        "graph_seed",
        None,
    )

    if value is not None:
        try:
            return int(value)
        except Exception:
            pass

    m = re.search(
        r"graph_seed(\d+)",
        str(path),
    )

    if m is not None:
        return int(
            m.group(1)
        )

    data_path = str(
        summary.get(
            "data_path",
            "",
        )
    )

    m = re.search(
        r"seed(\d+)\.pt$",
        data_path,
    )

    if m is not None:
        return int(
            m.group(1)
        )

    raise RuntimeError(
        f"Cannot determine graph seed: {path}"
    )


def get_lr(summary):
    for key in [
        "learning_rate",
        "lr",
    ]:
        if key in summary:
            try:
                return float(
                    summary[key]
                )
            except Exception:
                pass

    raise RuntimeError(
        f"No learning rate in summary: {summary.keys()}"
    )


def final_hidden(
    summary,
    summary_path,
):
    oversmoothing = (
        summary_path.with_name(
            summary_path.name.replace(
                "_summary.json",
                "_oversmoothing.csv",
            )
        )
    )

    if not oversmoothing.exists():
        value = summary.get(
            "oversmoothing_file",
            None,
        )

        if value:
            candidate = Path(
                value
            )

            if candidate.exists():
                oversmoothing = (
                    candidate
                )

    if not oversmoothing.exists():
        raise FileNotFoundError(
            oversmoothing
        )

    df = pd.read_csv(
        oversmoothing
    )

    x = df[
        (
            pd.to_numeric(
                df["epoch"],
                errors="coerce",
            )
            == 2400
        )
        & (
            pd.to_numeric(
                df["embedding_dim"],
                errors="coerce",
            )
            == 128
        )
    ].copy()

    if x.empty:
        raise RuntimeError(
            f"No epoch-2400 128-d hidden rows: "
            f"{oversmoothing}"
        )

    x["_layer"] = pd.to_numeric(
        x["layer_index"],
        errors="raise",
    )

    x = x[
        x["_layer"]
        == x["_layer"].max()
    ].copy()

    if len(x) != 1:
        raise RuntimeError(
            f"Expected exactly one final hidden row: "
            f"{oversmoothing}\n{x}"
        )

    return x.iloc[0]


def collect(
    root,
    allowed,
):
    rows = []

    files = sorted(
        root.rglob(
            "*_summary.json"
        )
    )

    print(
        root,
        "summary files:",
        len(files),
    )

    for path in files:
        with open(path) as f:
            s = json.load(f)

        model = str(
            s.get(
                "model",
                "",
            )
        )

        if model not in allowed:
            continue

        if model not in MODEL_VARIANT:
            continue

        if int(
            s.get(
                "num_layers",
                -1,
            )
        ) != 8:
            continue

        lr = get_lr(s)

        if not np.isclose(
            lr,
            0.01,
        ):
            continue

        dataset = clean_dataset(
            s.get(
                "dataset",
                s.get(
                    "data_path",
                    "",
                ),
            )
        )

        wanted_h = (
            0.1
            if dataset == "PubMed"
            else 0.9
        )

        row = final_hidden(
            s,
            path,
        )

        if (
            "effective_rank"
            in row.index
            and pd.notna(
                row["effective_rank"]
            )
        ):
            effective_rank = float(
                row["effective_rank"]
            )
        else:
            effective_rank = (
                float(
                    row[
                        "effective_rank_ratio"
                    ]
                )
                * 128.0
            )

        rows.append(
            {
                "dataset":
                    dataset,
                "h":
                    wanted_h,
                "variant":
                    MODEL_VARIANT[
                        model
                    ],
                "model":
                    model,
                "graph_seed":
                    get_graph_seed(
                        s,
                        path,
                    ),
                "init_seed":
                    int(
                        s["seed"]
                    ),
                "lr":
                    lr,
                "test_acc":
                    float(
                        s[
                            "best_test_acc_at_best_val"
                        ]
                    ),
                "epoch":
                    2400,
                "layer_name":
                    str(
                        row[
                            "layer_name"
                        ]
                    ),
                "embedding_dim":
                    int(
                        row[
                            "embedding_dim"
                        ]
                    ),
                "normalized_dirichlet_energy":
                    float(
                        row[
                            "normalized_dirichlet_energy"
                        ]
                    ),
                "effective_rank":
                    effective_rank,
                "effective_rank_ratio":
                    float(
                        row[
                            "effective_rank_ratio"
                        ]
                    ),
                "pairwise":
                    float(
                        row[
                            "mean_pairwise_cosine_distance"
                        ]
                    ),
                "source":
                    root.name,
            }
        )

    return pd.DataFrame(
        rows
    )


reference = collect(
    REFERENCE,
    {
        "GraphSAGE",
        "GraphSAGEPairNorm",
    },
)

ablation = collect(
    ABLATION,
    {
        "GraphSAGECenterNorm",
        "GraphSAGEScaleNorm",
    },
)

rw = pd.concat(
    [
        reference,
        ablation,
    ],
    ignore_index=True,
)

rw = rw.drop_duplicates(
    [
        "dataset",
        "variant",
        "graph_seed",
        "init_seed",
    ]
)

rw = rw.sort_values(
    [
        "dataset",
        "variant",
        "graph_seed",
        "init_seed",
    ]
).reset_index(
    drop=True
)


print()
print("=" * 100)
print("MATCHED-25 COUNTS")
print("=" * 100)

counts = (
    rw.groupby(
        [
            "dataset",
            "variant",
        ]
    )
    .size()
)

print(
    counts.to_string()
)


expected = {
    (
        dataset,
        variant,
    )
    for dataset in [
        "PubMed",
        "Roman-Empire",
    ]
    for variant in [
        "baseline",
        "center",
        "scale",
        "full",
    ]
}

assert set(
    counts.index
) == expected

assert counts.eq(
    25
).all()

assert len(
    rw
) == 200


cell_counts = (
    rw.groupby(
        [
            "dataset",
            "variant",
            "graph_seed",
        ]
    )[
        "init_seed"
    ]
    .nunique()
)

assert cell_counts.eq(
    5
).all()


print()
print("=" * 100)
print("FINAL HIDDEN LAYERS")
print("=" * 100)

print(
    rw.groupby(
        [
            "dataset",
            "variant",
            "layer_name",
            "embedding_dim",
        ]
    )
    .size()
    .to_string()
)


print()
print("=" * 100)
print("REFERENCE PERFORMANCE REPLICATION")
print("=" * 100)

factorial = pd.read_csv(
    FACTORIAL
)

factorial = factorial[
    factorial["model"].isin(
        [
            "GraphSAGE",
            "GraphSAGEPairNorm",
        ]
    )
    & factorial[
        "depth"
    ].astype(int).eq(
        8
    )
    & factorial[
        "dataset"
    ].isin(
        [
            "RW-PubMed-H0.1-LOW",
            "RW-Roman-empire-H0.9-LOW",
        ]
    )
].copy()

factorial[
    "dataset_clean"
] = factorial[
    "dataset"
].map(
    clean_dataset
)

factorial[
    "variant"
] = factorial[
    "model"
].map(
    MODEL_VARIANT
)

old = factorial[
    [
        "dataset_clean",
        "variant",
        "graph_seed",
        "init_seed",
        "test_best",
    ]
].rename(
    columns={
        "dataset_clean":
            "dataset",
        "test_best":
            "test_old",
    }
)

new = reference[
    [
        "dataset",
        "variant",
        "graph_seed",
        "init_seed",
        "test_acc",
    ]
].rename(
    columns={
        "test_acc":
            "test_new",
    }
)

check = old.merge(
    new,
    on=[
        "dataset",
        "variant",
        "graph_seed",
        "init_seed",
    ],
    how="inner",
    validate="one_to_one",
)

assert len(
    check
) == 100

check[
    "abs_diff"
] = (
    check[
        "test_new"
    ]
    - check[
        "test_old"
    ]
).abs()

print(
    "matched cells:",
    len(check),
)

print(
    "max abs test difference:",
    check[
        "abs_diff"
    ].max(),
)

print(
    "mean abs test difference:",
    check[
        "abs_diff"
    ].mean(),
)


summary = (
    rw.groupby(
        [
            "dataset",
            "variant",
        ],
        as_index=False,
    )
    .agg(
        n=(
            "test_acc",
            "size",
        ),
        test_mean=(
            "test_acc",
            "mean",
        ),
        test_std=(
            "test_acc",
            "std",
        ),
        nde_mean=(
            "normalized_dirichlet_energy",
            "mean",
        ),
        nde_std=(
            "normalized_dirichlet_energy",
            "std",
        ),
        effective_rank_mean=(
            "effective_rank",
            "mean",
        ),
        effective_rank_std=(
            "effective_rank",
            "std",
        ),
        rank_ratio_mean=(
            "effective_rank_ratio",
            "mean",
        ),
        pairwise_mean=(
            "pairwise",
            "mean",
        ),
        pairwise_std=(
            "pairwise",
            "std",
        ),
    )
)


order = {
    "baseline": 0,
    "center": 1,
    "scale": 2,
    "full": 3,
}

summary["_order"] = (
    summary[
        "variant"
    ].map(order)
)

summary = (
    summary.sort_values(
        [
            "dataset",
            "_order",
        ]
    )
    .drop(
        columns="_order"
    )
)


rw.to_csv(
    OUT,
    index=False,
)

summary.to_csv(
    SUMMARY,
    index=False,
)


print()
print("=" * 100)
print("MATCHED-25 SUMMARY")
print("=" * 100)

print(
    summary.to_string(
        index=False
    )
)

print()
print("saved:", OUT)
print("saved:", SUMMARY)
