from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import (
    degree,
    homophily,
)

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

DATA_ROOT = (
    Path("/work/log1")
    / Path.home().name
    / "pyg-data"
)

OUT = Path(
    "runs/realworld_homophily_precheck.csv"
)


def mask_info(mask):
    if mask is None:
        return {
            "shape": None,
            "splits": 0,
            "split0_count": 0,
        }

    if mask.ndim == 1:
        return {
            "shape": str(tuple(mask.shape)),
            "splits": 1,
            "split0_count": int(mask.sum()),
        }

    return {
        "shape": str(tuple(mask.shape)),
        "splits": int(mask.size(1)),
        "split0_count": int(
            mask[:, 0].sum()
        ),
    }


def count_unique_undirected_edges(
    edge_index,
):
    src = edge_index[0].cpu()
    dst = edge_index[1].cpu()

    non_loop = src != dst

    src = src[non_loop]
    dst = dst[non_loop]

    low = torch.minimum(src, dst)
    high = torch.maximum(src, dst)

    pairs = torch.stack(
        [low, high],
        dim=1,
    )

    return int(
        torch.unique(
            pairs,
            dim=0,
        ).size(0)
    )


def adjusted_edge_homophily(
    edge_index,
    y,
    num_classes,
):
    src = edge_index[0].long()
    dst = edge_index[1].long()

    same = (
        y[src] == y[dst]
    ).float()

    edge_h = float(
        same.mean()
    )

    endpoint_labels = torch.cat(
        [
            y[src],
            y[dst],
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
        torch.sum(
            endpoint_probs ** 2
        )
    )

    denominator = (
        1.0 - chance_h
    )

    if abs(denominator) < 1e-12:
        adjusted_h = float("nan")
    else:
        adjusted_h = (
            edge_h - chance_h
        ) / denominator

    return (
        adjusted_h,
        chance_h,
    )


rows = []

for dataset_name in DATASETS:
    print()
    print("=" * 80)
    print(dataset_name)
    print("=" * 80)

    dataset, data = load_dataset_any(
        dataset_name,
        root=str(DATA_ROOT),
        split_idx=0,
    )

    data = data.cpu()

    y = data.y.long()

    num_nodes = int(data.num_nodes)
    num_edges = int(
        data.edge_index.size(1)
    )

    num_classes = int(
        getattr(
            dataset,
            "num_classes",
            int(y.max()) + 1,
        )
    )

    num_features = int(
        data.num_features
    )

    edge_h = float(
        homophily(
            data.edge_index,
            y,
            method="edge",
        )
    )

    (
        adjusted_h,
        chance_homophily,
    ) = adjusted_edge_homophily(
        data.edge_index,
        y,
        num_classes,
    )

    counts = torch.bincount(
        y,
        minlength=num_classes,
    )

    class_fractions = (
        counts.float()
        / counts.sum()
    )

    majority_fraction = float(
        class_fractions.max()
    )

    deg = degree(
        data.edge_index[0],
        num_nodes=num_nodes,
        dtype=torch.float,
    )

    self_loops = int(
        (
            data.edge_index[0]
            == data.edge_index[1]
        ).sum()
    )

    undirected_edges = (
        count_unique_undirected_edges(
            data.edge_index
        )
    )

    train = mask_info(
        getattr(
            data,
            "train_mask",
            None,
        )
    )

    val = mask_info(
        getattr(
            data,
            "val_mask",
            None,
        )
    )

    test = mask_info(
        getattr(
            data,
            "test_mask",
            None,
        )
    )

    x = data.x

    finite_fraction = float(
        torch.isfinite(x).float().mean()
    )

    print(
        "nodes:",
        num_nodes,
    )
    print(
        "directed edge entries:",
        num_edges,
    )
    print(
        "unique undirected edges:",
        undirected_edges,
    )
    print(
        "self loops:",
        self_loops,
    )
    print(
        "features:",
        num_features,
    )
    print(
        "classes:",
        num_classes,
    )
    print(
        "edge homophily:",
        f"{edge_h:.10f}",
    )
    print(
        "adjusted homophily:",
        f"{adjusted_h:.10f}",
    )
    print(
        "degree-weighted chance homophily:",
        f"{chance_homophily:.10f}",
    )
    print(
        "majority baseline:",
        f"{majority_fraction:.10f}",
    )

    print(
        "degree mean/std/min/max:",
        f"{deg.mean().item():.4f}",
        f"{deg.std(unbiased=False).item():.4f}",
        f"{deg.min().item():.0f}",
        f"{deg.max().item():.0f}",
    )

    print(
        "train mask:",
        train,
    )
    print(
        "val mask:",
        val,
    )
    print(
        "test mask:",
        test,
    )

    print(
        "finite feature fraction:",
        finite_fraction,
    )

    print(
        "class counts:",
        counts.tolist(),
    )

    rows.append(
        {
            "dataset":
                dataset_name,
            "num_nodes":
                num_nodes,
            "directed_edge_entries":
                num_edges,
            "unique_undirected_edges":
                undirected_edges,
            "self_loops":
                self_loops,
            "num_features":
                num_features,
            "num_classes":
                num_classes,
            "edge_homophily":
                edge_h,
            "adjusted_homophily":
                adjusted_h,
            "chance_homophily":
                chance_homophily,
            "majority_baseline":
                majority_fraction,
            "degree_mean":
                float(deg.mean()),
            "degree_std":
                float(
                    deg.std(
                        unbiased=False
                    )
                ),
            "degree_min":
                float(deg.min()),
            "degree_max":
                float(deg.max()),
            "train_mask_shape":
                train["shape"],
            "train_splits":
                train["splits"],
            "train_split0_count":
                train[
                    "split0_count"
                ],
            "val_mask_shape":
                val["shape"],
            "val_splits":
                val["splits"],
            "val_split0_count":
                val[
                    "split0_count"
                ],
            "test_mask_shape":
                test["shape"],
            "test_splits":
                test["splits"],
            "test_split0_count":
                test[
                    "split0_count"
                ],
            "finite_feature_fraction":
                finite_fraction,
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
print("Saved:", OUT)
