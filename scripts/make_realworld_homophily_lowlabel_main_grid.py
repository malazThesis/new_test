from pathlib import Path
import pandas as pd

OUT = Path(
    "configs/"
    "realworld_homophily_lowlabel_main.csv"
)

SMOKE_OUT = Path(
    "configs/"
    "realworld_homophily_lowlabel_smoke.csv"
)

ROOT = Path(
    "realworld_data/"
    "homophily_controlled"
)

DATASETS = [
    ("PubMed", "pubmed"),
    ("Roman-empire", "roman_empire"),
]

TARGETS = [
    (0.1, "h01"),
    (0.5, "h05"),
    (0.9, "h09"),
]

SEEDS = [1, 2, 3, 4, 5]

MODELS = [
    "GCN",
    "GCNPairNorm",
    "GraphSAGE",
    "GraphSAGEPairNorm",
]

GRAPHSAGE_MODELS = [
    "GraphSAGE",
    "GraphSAGEPairNorm",
]

BASE_DEPTHS = [2, 4, 8]

HIDDEN = 128
LR = 0.01
WEIGHT_DECAY = 0.0005
DROPOUT = 0.5

rows = []


def add_row(
    *,
    dataset_name,
    slug,
    target,
    hslug,
    seed,
    model,
    depth,
):
    path = (
        ROOT
        / slug
        / "lowlabel"
        / (
            f"{slug}_"
            f"{hslug}_"
            f"seed{seed}.pt"
        )
    )

    if not path.exists():
        raise FileNotFoundError(path)

    rows.append(
        {
            "data_path": str(path),
            "dataset": (
                f"RW-{dataset_name}-"
                f"H{target:.1f}-LOW"
            ),
            "model": model,
            "num_layers": depth,
            "hidden_channels": HIDDEN,
            "seed": seed,
            "split_idx": 0,
            "lr": LR,
            "weight_decay":
                WEIGHT_DECAY,
            "dropout": DROPOUT,
        }
    )


for (
    dataset_name,
    slug,
) in DATASETS:

    for (
        target,
        hslug,
    ) in TARGETS:

        for seed in SEEDS:

            # ------------------------------------------------
            # cSBM homophily mechanism:
            # 4 models x L2/L4/L8 x h=.1/.5/.9
            # ------------------------------------------------

            for model in MODELS:
                for depth in BASE_DEPTHS:
                    add_row(
                        dataset_name=
                            dataset_name,
                        slug=slug,
                        target=target,
                        hslug=hslug,
                        seed=seed,
                        model=model,
                        depth=depth,
                    )

            # ------------------------------------------------
            # cSBM lowlabel_graphsage:
            # GraphSAGE / PairNorm
            # L2/L7/L8
            #
            # L2/L8 already above,
            # so only add L7.
            # ------------------------------------------------

            for model in GRAPHSAGE_MODELS:
                add_row(
                    dataset_name=
                        dataset_name,
                    slug=slug,
                    target=target,
                    hslug=hslug,
                    seed=seed,
                    model=model,
                    depth=7,
                )

            # ------------------------------------------------
            # cSBM H0.1 parity:
            # GraphSAGE / PairNorm
            # L5/L6/L7/L8/L9/L10
            #
            # L7/L8 already present.
            # ------------------------------------------------

            if target == 0.1:
                for model in (
                    GRAPHSAGE_MODELS
                ):
                    for depth in [
                        5,
                        6,
                        9,
                        10,
                    ]:
                        add_row(
                            dataset_name=
                                dataset_name,
                            slug=slug,
                            target=target,
                            hslug=hslug,
                            seed=seed,
                            model=model,
                            depth=depth,
                        )


df = pd.DataFrame(rows)

key = [
    "data_path",
    "dataset",
    "model",
    "num_layers",
    "seed",
    "split_idx",
]

duplicates = df.duplicated(
    subset=key,
).sum()

if duplicates:
    raise RuntimeError(
        f"Duplicate runs: {duplicates}"
    )

if len(df) != 500:
    raise RuntimeError(
        f"Expected 500 rows, "
        f"found {len(df)}"
    )

dataset_counts = (
    df.groupby(
        df["dataset"].str.extract(
            r"RW-(.*)-H",
            expand=False,
        )
    )
    .size()
    .to_dict()
)

expected_counts = {
    "PubMed": 250,
    "Roman-empire": 250,
}

if dataset_counts != expected_counts:
    raise RuntimeError(
        f"Unexpected dataset counts: "
        f"{dataset_counts}"
    )

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUT,
    index=False,
)

# ----------------------------------------------------
# Four stress/smoke cases:
# - deep GraphSAGEPairNorm
# - deep GraphSAGE baseline
# - GCN
# - GCNPairNorm
# across both datasets / homophily extremes.
# ----------------------------------------------------

tests = [
    (
        "RW-PubMed-H0.1-LOW",
        "GraphSAGEPairNorm",
        10,
        1,
    ),
    (
        "RW-PubMed-H0.9-LOW",
        "GCN",
        8,
        1,
    ),
    (
        "RW-Roman-empire-H0.1-LOW",
        "GraphSAGE",
        10,
        1,
    ),
    (
        "RW-Roman-empire-H0.9-LOW",
        "GCNPairNorm",
        8,
        1,
    ),
]

smoke_rows = []

for (
    dataset,
    model,
    depth,
    seed,
) in tests:

    selected = df[
        (df["dataset"] == dataset)
        & (df["model"] == model)
        & (
            df["num_layers"]
            == depth
        )
        & (df["seed"] == seed)
    ]

    if len(selected) != 1:
        raise RuntimeError(
            "Smoke selection failed: "
            f"{dataset}, {model}, "
            f"L{depth}, seed={seed}, "
            f"rows={len(selected)}"
        )

    smoke_rows.append(
        selected.iloc[0]
    )

smoke = pd.DataFrame(
    smoke_rows
).reset_index(
    drop=True
)

smoke.to_csv(
    SMOKE_OUT,
    index=False,
)

print()
print("=" * 90)
print("MAIN GRID")
print("=" * 90)
print("rows:", len(df))

print()
print(
    df.groupby(
        [
            "dataset",
            "model",
            "num_layers",
        ]
    )
    .size()
    .to_string()
)

print()
print("saved:", OUT)

print()
print("=" * 90)
print("SMOKE GRID")
print("=" * 90)

print(
    smoke[
        [
            "dataset",
            "model",
            "num_layers",
            "seed",
            "lr",
            "data_path",
        ]
    ].to_string(
        index=True
    )
)

print()
print("saved:", SMOKE_OUT)
