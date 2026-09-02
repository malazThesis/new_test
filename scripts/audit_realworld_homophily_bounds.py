from pathlib import Path
import sys
import math

import pandas as pd
import torch
from torch_geometric.utils import degree, homophily

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
    "realworld_homophily_degree_bounds.csv"
)


def adjusted_h(
    raw_h,
    chance_h,
):
    denominator = (
        1.0 - chance_h
    )

    if abs(denominator) < 1e-12:
        return float("nan")

    return (
        raw_h - chance_h
    ) / denominator


rows = []

for dataset_name in DATASETS:
    dataset, data = load_dataset_any(
        dataset_name,
        root=str(DATA_ROOT),
        split_idx=0,
    )

    data = data.cpu()

    y = data.y.long()

    num_nodes = int(
        data.num_nodes
    )

    num_classes = int(
        getattr(
            dataset,
            "num_classes",
            int(y.max()) + 1,
        )
    )

    edge_index = data.edge_index

    src = edge_index[0]
    dst = edge_index[1]

    non_loop = src != dst
    src = src[non_loop]
    dst = dst[non_loop]

    low = torch.minimum(
        src,
        dst,
    )

    high = torch.maximum(
        src,
        dst,
    )

    pairs = torch.stack(
        [
            low,
            high,
        ],
        dim=1,
    )

    pairs = torch.unique(
        pairs,
        dim=0,
    )

    m = int(
        pairs.size(0)
    )

    deg = degree(
        edge_index[0],
        num_nodes=num_nodes,
        dtype=torch.long,
    )

    total_stubs = int(
        deg.sum()
    )

    class_counts = torch.bincount(
        y,
        minlength=num_classes,
    )

    class_degree_stubs = []

    max_internal_edges = 0
    min_internal_edges = 0

    for c in range(
        num_classes
    ):
        nodes = (
            y == c
        )

        n_c = int(
            nodes.sum()
        )

        d_c = int(
            deg[nodes].sum()
        )

        simple_cap = (
            n_c
            * (n_c - 1)
            // 2
        )

        stub_cap = (
            d_c // 2
        )

        max_c = min(
            simple_cap,
            stub_cap,
        )

        forced_internal_stubs = max(
            0,
            2 * d_c
            - total_stubs,
        )

        min_c = math.ceil(
            forced_internal_stubs
            / 2
        )

        max_internal_edges += (
            max_c
        )

        min_internal_edges += (
            min_c
        )

        class_degree_stubs.append(
            {
                "class": c,
                "nodes": n_c,
                "degree_stubs": d_c,
                "stub_fraction":
                    d_c
                    / total_stubs,
            }
        )

    lower_bound = (
        min_internal_edges
        / m
    )

    upper_bound = min(
        1.0,
        max_internal_edges
        / m,
    )

    endpoint_labels = torch.cat(
        [
            y[edge_index[0]],
            y[edge_index[1]],
        ],
        dim=0,
    )

    endpoint_counts = torch.bincount(
        endpoint_labels,
        minlength=num_classes,
    ).float()

    endpoint_probs = (
        endpoint_counts
        / endpoint_counts.sum()
    )

    chance_h = float(
        (
            endpoint_probs ** 2
        ).sum()
    )

    natural_h = float(
        homophily(
            edge_index,
            y,
            method="edge",
        )
    )

    print()
    print("=" * 80)
    print(dataset_name)
    print("=" * 80)

    print(
        "nodes:",
        num_nodes,
    )

    print(
        "undirected edges:",
        m,
    )

    print(
        "total degree stubs:",
        total_stubs,
    )

    print(
        "natural edge homophily:",
        f"{natural_h:.10f}",
    )

    print(
        "chance homophily:",
        f"{chance_h:.10f}",
    )

    print(
        "degree-stub relaxed lower bound:",
        f"{lower_bound:.10f}",
    )

    print(
        "degree-stub relaxed upper bound:",
        f"{upper_bound:.10f}",
    )

    print()
    print(
        "class degree-stub fractions:"
    )

    for row in class_degree_stubs:
        print(
            f"class={row['class']:2d} "
            f"nodes={row['nodes']:5d} "
            f"stubs={row['degree_stubs']:6d} "
            f"fraction="
            f"{row['stub_fraction']:.6f}"
        )

    print()
    print("targets:")

    for target in TARGETS:
        possible_under_relaxation = (
            lower_bound
            <= target
            <= upper_bound
        )

        target_adjusted = (
            adjusted_h(
                target,
                chance_h,
            )
        )

        print(
            f"h={target:.1f}: "
            f"relaxed_possible="
            f"{possible_under_relaxation}, "
            f"adjusted="
            f"{target_adjusted:.6f}"
        )

        rows.append(
            {
                "dataset":
                    dataset_name,
                "num_nodes":
                    num_nodes,
                "undirected_edges":
                    m,
                "natural_edge_homophily":
                    natural_h,
                "chance_homophily":
                    chance_h,
                "relaxed_lower_bound":
                    lower_bound,
                "relaxed_upper_bound":
                    upper_bound,
                "target_homophily":
                    target,
                "target_adjusted_homophily":
                    target_adjusted,
                "possible_under_relaxed_degree_bound":
                    possible_under_relaxation,
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
