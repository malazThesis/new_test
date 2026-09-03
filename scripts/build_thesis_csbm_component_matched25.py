from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


RUNS = Path("runs")

SOURCES = [
    (
        RUNS
        / "csbm_l8_lr_sensitivity_fs050",
        {
            "GraphSAGE":
                "baseline",
        },
    ),
    (
        RUNS
        / "csbm_l8_pairnorm_component_ablation_fs050",
        {
            "GraphSAGECenterNorm":
                "center",
            "GraphSAGEScaleNorm":
                "scale",
        },
    ),
    (
        RUNS
        / "csbm_l8_pairnorm_lr_sensitivity_fs050",
        {
            "GraphSAGEPairNorm":
                "full",
        },
    ),
]

OUT = (
    RUNS
    / "thesis_csbm_component_ablation_lr003_matched25.csv"
)

SUMMARY = (
    RUNS
    / "thesis_csbm_component_ablation_lr003_matched25_summary.csv"
)


def graph_seed(
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

    m = re.search(
        r"-G(\d+)-",
        path.name,
    )

    if m is not None:
        return int(
            m.group(1)
        )

    raise RuntimeError(
        path
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

    return np.nan


def final_hidden(
    summary_path,
):
    p = summary_path.with_name(
        summary_path.name.replace(
            "_summary.json",
            "_oversmoothing.csv",
        )
    )

    if not p.exists():
        raise FileNotFoundError(
            p
        )

    df = pd.read_csv(
        p
    )

    epochs = pd.to_numeric(
        df["epoch"],
        errors="coerce",
    )

    dims = pd.to_numeric(
        df["embedding_dim"],
        errors="coerce",
    )

    x = df[
        (
            epochs == 200
        )
        & (
            dims == 128
        )
    ].copy()

    if x.empty:
        raise RuntimeError(
            f"No epoch200 hidden row: {p}"
        )

    x["_layer"] = pd.to_numeric(
        x["layer_index"],
        errors="raise",
    )

    x = x[
        x["_layer"]
        == x["_layer"].max()
    ]

    if len(x) != 1:
        raise RuntimeError(
            f"Expected one hidden row: {p}"
        )

    return x.iloc[0]


rows = []

for root, models in SOURCES:
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

        if model not in models:
            continue

        if int(
            s.get(
                "num_layers",
                -1,
            )
        ) != 8:
            continue

        lr = get_lr(
            s
        )

        if not np.isclose(
            lr,
            0.03,
        ):
            continue

        dataset_blob = str(
            s.get(
                "dataset",
                "",
            )
        )

        if "H0.1" not in dataset_blob:
            continue

        if (
            "FS0.50"
            not in dataset_blob
        ):
            fs = s.get(
                "feature_signal",
                np.nan,
            )

            try:
                if not np.isclose(
                    float(fs),
                    0.5,
                ):
                    continue
            except Exception:
                continue

        r = final_hidden(
            path
        )

        if (
            "effective_rank"
            in r.index
            and pd.notna(
                r[
                    "effective_rank"
                ]
            )
        ):
            rank = float(
                r[
                    "effective_rank"
                ]
            )
        else:
            rank = (
                float(
                    r[
                        "effective_rank_ratio"
                    ]
                )
                * 128.0
            )

        rows.append(
            {
                "dataset":
                    "cSBM",
                "h":
                    0.1,
                "feature_signal":
                    0.5,
                "variant":
                    models[
                        model
                    ],
                "model":
                    model,
                "graph_seed":
                    graph_seed(
                        s,
                        path,
                    ),
                "init_seed":
                    int(
                        s[
                            "seed"
                        ]
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
                    200,
                "normalized_dirichlet_energy":
                    float(
                        r[
                            "normalized_dirichlet_energy"
                        ]
                    ),
                "effective_rank":
                    rank,
                "effective_rank_ratio":
                    float(
                        r[
                            "effective_rank_ratio"
                        ]
                    ),
                "pairwise":
                    float(
                        r[
                            "mean_pairwise_cosine_distance"
                        ]
                    ),
                "source":
                    root.name,
            }
        )


df = pd.DataFrame(
    rows
)

df = df.drop_duplicates(
    [
        "variant",
        "graph_seed",
        "init_seed",
    ]
)

counts = (
    df.groupby(
        "variant"
    )
    .size()
)

print()
print("=" * 100)
print("CSBM MATCHED COUNTS")
print("=" * 100)

print(
    counts.to_string()
)

assert set(
    counts.index
) == {
    "baseline",
    "center",
    "scale",
    "full",
}

assert counts.eq(
    25
).all()

assert len(
    df
) == 100


summary = (
    df.groupby(
        "variant",
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


df.to_csv(
    OUT,
    index=False,
)

summary.to_csv(
    SUMMARY,
    index=False,
)

print()
print(summary.to_string(index=False))

print()
print("saved:", OUT)
print("saved:", SUMMARY)
