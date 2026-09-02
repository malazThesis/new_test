from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

ROOT = Path.cwd()

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

from run_realworld import load_dataset_any


DATASETS = [
    "PubMed",
    "Roman-empire",
]

TARGETS = [
    0.1,
    0.5,
    0.9,
]

DATA_ROOT = (
    Path("/work/log1")
    / Path.home().name
    / "pyg-data"
)

OUT = Path(
    "runs/"
    "realworld_homophily_connected_bounds.csv"
)


def unique_edges(edge_index):
    src = edge_index[0].cpu()
    dst = edge_index[1].cpu()

    keep = src != dst

    src = src[keep]
    dst = dst[keep]

    low = torch.minimum(
        src,
        dst,
    )

    high = torch.maximum(
        src,
        dst,
    )

    pairs = torch.stack(
        [low, high],
        dim=1,
    )

    pairs = torch.unique(
        pairs,
        dim=0,
    )

    return pairs


def spanning_tree(
    pairs,
    y,
    num_nodes,
    mode,
):
    u = pairs[:, 0].numpy()
    v = pairs[:, 1].numpy()

    same = (
        y[pairs[:, 0]]
        == y[pairs[:, 1]]
    ).numpy()

    if mode == "min_homophily":
        weights = np.where(
            same,
            2.0,
            1.0,
        )
    elif mode == "max_homophily":
        weights = np.where(
            same,
            1.0,
            2.0,
        )
    else:
        raise ValueError(mode)

    rows = np.concatenate(
        [u, v]
    )

    cols = np.concatenate(
        [v, u]
    )

    values = np.concatenate(
        [weights, weights]
    )

    graph = coo_matrix(
        (
            values,
            (
                rows,
                cols,
            ),
        ),
        shape=(
            num_nodes,
            num_nodes,
        ),
    ).tocsr()

    tree = minimum_spanning_tree(
        graph
    ).tocoo()

    tree_edges = set()

    for a, b in zip(
        tree.row,
        tree.col,
    ):
        a = int(a)
        b = int(b)

        if a > b:
            a, b = b, a

        tree_edges.add(
            (a, b)
        )

    if len(tree_edges) != (
        num_nodes - 1
    ):
        raise RuntimeError(
            f"Expected {num_nodes - 1} "
            f"tree edges, found "
            f"{len(tree_edges)}"
        )

    homo = 0

    for a, b in tree_edges:
        if int(y[a]) == int(y[b]):
            homo += 1

    hetero = (
        len(tree_edges)
        - homo
    )

    return {
        "tree_edges":
            len(tree_edges),
        "homophilic_tree_edges":
            homo,
        "heterophilic_tree_edges":
            hetero,
    }


rows = []

for dataset_name in DATASETS:
    dataset, data = load_dataset_any(
        dataset_name,
        root=str(DATA_ROOT),
        split_idx=0,
    )

    data = data.cpu()

    y = data.y.long()

    n = int(data.num_nodes)

    pairs = unique_edges(
        data.edge_index
    )

    m = int(
        pairs.size(0)
    )

    natural_same = int(
        (
            y[pairs[:, 0]]
            == y[pairs[:, 1]]
        ).sum()
    )

    natural_h = (
        natural_same
        / m
    )

    min_tree = spanning_tree(
        pairs,
        y,
        n,
        "min_homophily",
    )

    max_tree = spanning_tree(
        pairs,
        y,
        n,
        "max_homophily",
    )

    connected_lower_bound = (
        min_tree[
            "homophilic_tree_edges"
        ]
        / m
    )

    connected_upper_bound = (
        1.0
        - max_tree[
            "heterophilic_tree_edges"
        ]
        / m
    )

    print()
    print("=" * 80)
    print(dataset_name)
    print("=" * 80)

    print(
        "nodes:",
        n,
    )

    print(
        "undirected edges:",
        m,
    )

    print(
        "natural homophily:",
        f"{natural_h:.10f}",
    )

    print()
    print(
        "MIN-HOMOPHILY protected tree"
    )

    print(
        "tree homophilic edges:",
        min_tree[
            "homophilic_tree_edges"
        ],
    )

    print(
        "tree heterophilic edges:",
        min_tree[
            "heterophilic_tree_edges"
        ],
    )

    print()
    print(
        "MAX-HOMOPHILY protected tree"
    )

    print(
        "tree homophilic edges:",
        max_tree[
            "homophilic_tree_edges"
        ],
    )

    print(
        "tree heterophilic edges:",
        max_tree[
            "heterophilic_tree_edges"
        ],
    )

    print()
    print(
        "connectivity-safe lower bound:",
        f"{connected_lower_bound:.10f}",
    )

    print(
        "connectivity-safe upper bound:",
        f"{connected_upper_bound:.10f}",
    )

    print()
    print("targets:")

    for target in TARGETS:
        plausible = (
            connected_lower_bound
            <= target
            <= connected_upper_bound
        )

        direction = (
            "decrease"
            if target < natural_h
            else "increase"
            if target > natural_h
            else "same"
        )

        print(
            f"h={target:.1f}: "
            f"direction={direction}, "
            f"connectivity_safe="
            f"{plausible}"
        )

        rows.append(
            {
                "dataset":
                    dataset_name,
                "natural_homophily":
                    natural_h,
                "target_homophily":
                    target,
                "direction":
                    direction,
                "min_tree_homophilic_edges":
                    min_tree[
                        "homophilic_tree_edges"
                    ],
                "min_tree_heterophilic_edges":
                    min_tree[
                        "heterophilic_tree_edges"
                    ],
                "max_tree_homophilic_edges":
                    max_tree[
                        "homophilic_tree_edges"
                    ],
                "max_tree_heterophilic_edges":
                    max_tree[
                        "heterophilic_tree_edges"
                    ],
                "connectivity_safe_lower_bound":
                    connected_lower_bound,
                "connectivity_safe_upper_bound":
                    connected_upper_bound,
                "target_plausible":
                    plausible,
            }
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
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    frame.to_string(
        index=False,
    )
)

print()
print(
    "Saved:",
    OUT,
)
