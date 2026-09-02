from pathlib import Path
import hashlib
import sys

import pandas as pd
import torch
from torch_geometric.utils import degree, homophily

ROOT = Path.cwd()

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

from run_realworld import load_dataset_any

from make_realworld_homophily_graphs import (
    number_components,
    unique_undirected_edges,
)


DATA_ROOT = (
    Path("/work/log1")
    / Path.home().name
    / "pyg-data"
)

CONTROLLED_ROOT = Path(
    "realworld_data/"
    "homophily_controlled"
)

OUT = Path(
    "runs/"
    "realworld_homophily_graph_validation.csv"
)

DATASETS = {
    "PubMed": "pubmed",
    "Roman-empire": "roman_empire",
}

TARGETS = {
    0.1: "h01",
    0.5: "h05",
    0.9: "h09",
}

SEEDS = [
    1,
    2,
    3,
    4,
    5,
]

REGIMES = [
    "lowlabel",
    "replicated",
]


def load_pt(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def tensor_hash(tensor):
    t = (
        tensor.detach()
        .cpu()
        .contiguous()
    )

    return hashlib.sha256(
        t.numpy().tobytes()
    ).hexdigest()


def edge_hash(edge_index):
    edges = sorted(
        unique_undirected_edges(
            edge_index
        )
    )

    payload = "\n".join(
        f"{u},{v}"
        for u, v in edges
    ).encode()

    return hashlib.sha256(
        payload
    ).hexdigest()


def mask_validity(data):
    train = data.train_mask
    val = data.val_mask
    test = data.test_mask

    if train.ndim == 1:
        train = train[:, None]

    if val.ndim == 1:
        val = val[:, None]

    if test.ndim == 1:
        test = test[:, None]

    if not (
        train.shape
        == val.shape
        == test.shape
    ):
        return False

    for i in range(
        train.size(1)
    ):
        tr = train[:, i]
        va = val[:, i]
        te = test[:, i]

        if torch.any(
            tr & va
        ):
            return False

        if torch.any(
            tr & te
        ):
            return False

        if torch.any(
            va & te
        ):
            return False

        if not torch.all(
            tr | va | te
        ):
            return False

    return True


rows = []

mask_hashes = {}
topology_hashes = {}


for dataset_name, slug in (
    DATASETS.items()
):
    dataset, original = (
        load_dataset_any(
            dataset_name,
            root=str(DATA_ROOT),
            split_idx=0,
        )
    )

    original = original.cpu()

    y_original = (
        original.y.long()
    )

    x_hash_original = (
        tensor_hash(
            original.x
        )
    )

    y_hash_original = (
        tensor_hash(
            y_original
        )
    )

    original_deg = degree(
        original.edge_index[0],
        num_nodes=(
            original.num_nodes
        ),
        dtype=torch.long,
    )

    original_undirected = (
        len(
            unique_undirected_edges(
                original.edge_index
            )
        )
    )

    original_components = (
        number_components(
            original.edge_index,
            int(original.num_nodes),
        )
    )

    if original_components != 1:
        raise RuntimeError(
            f"{dataset_name}: "
            "original graph is not connected"
        )

    for target, hslug in (
        TARGETS.items()
    ):
        for seed in SEEDS:
            regime_data = {}

            for regime in REGIMES:
                path = (
                    CONTROLLED_ROOT
                    / slug
                    / regime
                    / (
                        f"{slug}_"
                        f"{hslug}_"
                        f"seed{seed}.pt"
                    )
                )

                if not path.exists():
                    raise FileNotFoundError(
                        path
                    )

                data = load_pt(
                    path
                )

                regime_data[
                    regime
                ] = data

                if tensor_hash(
                    data.x
                ) != x_hash_original:
                    raise RuntimeError(
                        f"{path}: "
                        "features changed"
                    )

                if tensor_hash(
                    data.y
                ) != y_hash_original:
                    raise RuntimeError(
                        f"{path}: "
                        "labels changed"
                    )

                new_deg = degree(
                    data.edge_index[0],
                    num_nodes=(
                        data.num_nodes
                    ),
                    dtype=torch.long,
                )

                max_degree_diff = int(
                    (
                        new_deg
                        - original_deg
                    )
                    .abs()
                    .max()
                )

                self_loops = int(
                    (
                        data.edge_index[0]
                        == data.edge_index[1]
                    ).sum()
                )

                undirected_edges = (
                    unique_undirected_edges(
                        data.edge_index
                    )
                )

                num_undirected = len(
                    undirected_edges
                )

                directed_entries = int(
                    data.edge_index.size(1)
                )

                components = (
                    number_components(
                        data.edge_index,
                        int(data.num_nodes),
                    )
                )

                realized_h = float(
                    homophily(
                        data.edge_index,
                        data.y,
                        method="edge",
                    )
                )

                target_error = abs(
                    realized_h
                    - target
                )

                masks_ok = (
                    mask_validity(
                        data
                    )
                )

                if not masks_ok:
                    raise RuntimeError(
                        f"{path}: "
                        "invalid masks"
                    )

                if (
                    max_degree_diff
                    != 0
                ):
                    raise RuntimeError(
                        f"{path}: "
                        "degree changed"
                    )

                if self_loops != 0:
                    raise RuntimeError(
                        f"{path}: "
                        "self loops"
                    )

                if components != 1:
                    raise RuntimeError(
                        f"{path}: "
                        f"{components} components"
                    )

                if (
                    num_undirected
                    != original_undirected
                ):
                    raise RuntimeError(
                        f"{path}: "
                        "edge count changed"
                    )

                if (
                    directed_entries
                    != 2
                    * num_undirected
                ):
                    raise RuntimeError(
                        f"{path}: "
                        "duplicate/asymmetric edges"
                    )

                if target_error > 0.0001:
                    raise RuntimeError(
                        f"{path}: "
                        "homophily target missed"
                    )

                if regime == "lowlabel":
                    expected = (
                        0.05,
                        0.10,
                        0.85,
                    )
                else:
                    expected = (
                        0.60,
                        0.20,
                        0.20,
                    )

                train = (
                    data.train_mask
                )
                val = (
                    data.val_mask
                )
                test = (
                    data.test_mask
                )

                if train.ndim == 1:
                    train = (
                        train[:, None]
                    )

                if val.ndim == 1:
                    val = (
                        val[:, None]
                    )

                if test.ndim == 1:
                    test = (
                        test[:, None]
                    )

                train_fraction = float(
                    train[:, 0]
                    .float()
                    .mean()
                )

                val_fraction = float(
                    val[:, 0]
                    .float()
                    .mean()
                )

                test_fraction = float(
                    test[:, 0]
                    .float()
                    .mean()
                )

                tolerance = (
                    2.0
                    / data.num_nodes
                )

                for observed, exp in zip(
                    (
                        train_fraction,
                        val_fraction,
                        test_fraction,
                    ),
                    expected,
                ):
                    if (
                        abs(
                            observed
                            - exp
                        )
                        > tolerance
                    ):
                        raise RuntimeError(
                            f"{path}: "
                            "split ratio incorrect"
                        )

                key = (
                    dataset_name,
                    regime,
                    seed,
                )

                mh = (
                    tensor_hash(train)
                    + tensor_hash(val)
                    + tensor_hash(test)
                )

                if key in mask_hashes:
                    if (
                        mask_hashes[key]
                        != mh
                    ):
                        raise RuntimeError(
                            f"{dataset_name} "
                            f"{regime} seed={seed}: "
                            "masks differ across "
                            "homophily levels"
                        )
                else:
                    mask_hashes[key] = mh

                ehash = edge_hash(
                    data.edge_index
                )

                topology_hashes[
                    (
                        dataset_name,
                        target,
                        seed,
                    )
                ] = ehash

                rows.append(
                    {
                        "dataset":
                            dataset_name,
                        "target_homophily":
                            target,
                        "seed":
                            seed,
                        "regime":
                            regime,
                        "realized_homophily":
                            realized_h,
                        "target_error":
                            target_error,
                        "num_nodes":
                            int(
                                data.num_nodes
                            ),
                        "undirected_edges":
                            num_undirected,
                        "max_degree_difference":
                            max_degree_diff,
                        "self_loops":
                            self_loops,
                        "components":
                            components,
                        "num_splits":
                            int(
                                train.size(1)
                            ),
                        "train_split0":
                            int(
                                train[
                                    :, 0
                                ].sum()
                            ),
                        "val_split0":
                            int(
                                val[
                                    :, 0
                                ].sum()
                            ),
                        "test_split0":
                            int(
                                test[
                                    :, 0
                                ].sum()
                            ),
                        "edge_hash":
                            ehash,
                        "path":
                            str(path),
                    }
                )

            low = regime_data[
                "lowlabel"
            ]

            rep = regime_data[
                "replicated"
            ]

            if edge_hash(
                low.edge_index
            ) != edge_hash(
                rep.edge_index
            ):
                raise RuntimeError(
                    f"{dataset_name}, "
                    f"h={target}, seed={seed}: "
                    "lowlabel and replicated "
                    "topologies differ"
                )


for dataset_name in DATASETS:
    for target in TARGETS:
        hashes = [
            topology_hashes[
                (
                    dataset_name,
                    target,
                    seed,
                )
            ]
            for seed in SEEDS
        ]

        unique = len(
            set(hashes)
        )

        if unique != len(SEEDS):
            raise RuntimeError(
                f"{dataset_name}, "
                f"h={target}: "
                f"only {unique}/"
                f"{len(SEEDS)} "
                "unique rewired graphs"
            )


frame = pd.DataFrame(rows)

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

frame.to_csv(
    OUT,
    index=False,
)

print()
print("=" * 100)
print("VALIDATION PASSED")
print("=" * 100)

summary = (
    frame.groupby(
        [
            "dataset",
            "target_homophily",
            "regime",
        ]
    )
    .agg(
        n=(
            "seed",
            "count",
        ),
        realized_h_min=(
            "realized_homophily",
            "min",
        ),
        realized_h_max=(
            "realized_homophily",
            "max",
        ),
        max_target_error=(
            "target_error",
            "max",
        ),
        max_degree_difference=(
            "max_degree_difference",
            "max",
        ),
        max_components=(
            "components",
            "max",
        ),
        max_self_loops=(
            "self_loops",
            "max",
        ),
    )
    .reset_index()
)

print(
    summary.to_string(
        index=False,
    )
)

print()
print(
    "Rows:",
    len(frame),
)

print(
    "Expected rows:",
    60,
)

print()
print(
    "Saved:",
    OUT,
)
