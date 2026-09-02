from pathlib import Path
import argparse
import random
import sys

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from torch_geometric.utils import degree, homophily

ROOT = Path.cwd()

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

from run_realworld import load_dataset_any


DATA_ROOT = (
    Path("/work/log1")
    / Path.home().name
    / "pyg-data"
)

DEFAULT_OUT = Path(
    "realworld_data/"
    "homophily_controlled"
)

MANIFEST = Path(
    "runs/"
    "realworld_homophily_rewiring_manifest.csv"
)


class IndexedSet:
    def __init__(self):
        self.items = []
        self.pos = {}

    def __len__(self):
        return len(self.items)

    def add(self, item):
        if item in self.pos:
            return

        self.pos[item] = len(self.items)
        self.items.append(item)

    def remove(self, item):
        idx = self.pos.pop(item)

        last = self.items.pop()

        if idx < len(self.items):
            self.items[idx] = last
            self.pos[last] = idx

    def random_item(self, rng):
        return self.items[
            rng.randrange(
                len(self.items)
            )
        ]

    def random_two(self, rng):
        n = len(self.items)

        if n < 2:
            raise RuntimeError(
                "Need at least two items"
            )

        i = rng.randrange(n)

        j = rng.randrange(n - 1)

        if j >= i:
            j += 1

        return (
            self.items[i],
            self.items[j],
        )


def canonical_edge(u, v):
    if u < v:
        return (u, v)

    return (v, u)


def unique_undirected_edges(
    edge_index,
):
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

    return [
        (
            int(row[0]),
            int(row[1]),
        )
        for row in pairs
    ]


def build_edge_index(edges):
    src = []
    dst = []

    for u, v in edges:
        src.extend(
            [u, v]
        )
        dst.extend(
            [v, u]
        )

    return torch.tensor(
        [src, dst],
        dtype=torch.long,
    )


def edge_same_count(
    edges,
    y,
):
    count = 0

    for u, v in edges:
        if int(y[u]) == int(y[v]):
            count += 1

    return count


def edge_homophily_from_edges(
    edges,
    y,
):
    if not edges:
        return float("nan")

    return (
        edge_same_count(
            edges,
            y,
        )
        / len(edges)
    )


def chance_homophily(
    edge_index,
    y,
    num_classes,
):
    labels = torch.cat(
        [
            y[edge_index[0]],
            y[edge_index[1]],
        ],
        dim=0,
    )

    counts = torch.bincount(
        labels,
        minlength=num_classes,
    ).float()

    probs = (
        counts
        / counts.sum()
    )

    return float(
        torch.sum(
            probs ** 2
        )
    )


def adjusted_homophily(
    raw_h,
    chance_h,
):
    denom = (
        1.0 - chance_h
    )

    if abs(denom) < 1e-12:
        return float("nan")

    return (
        raw_h - chance_h
    ) / denom


def initialize_buckets(
    edges,
    y,
):
    homo = {}
    hetero = {}

    edge_set = set(edges)

    for edge in edges:
        u, v = edge

        a = int(y[u])
        b = int(y[v])

        if a == b:
            if a not in homo:
                homo[a] = IndexedSet()

            homo[a].add(edge)

        else:
            key = tuple(
                sorted(
                    (a, b)
                )
            )

            if key not in hetero:
                hetero[key] = (
                    IndexedSet()
                )

            hetero[key].add(edge)

    return (
        edge_set,
        homo,
        hetero,
    )


def orient_hetero_edge(
    edge,
    y,
    class_a,
    class_b,
):
    u, v = edge

    yu = int(y[u])
    yv = int(y[v])

    if (
        yu == class_a
        and yv == class_b
    ):
        return u, v

    if (
        yu == class_b
        and yv == class_a
    ):
        return v, u

    raise RuntimeError(
        "Edge does not match "
        "heterophily bucket"
    )


def increase_homophily_step(
    edge_set,
    homo,
    hetero,
    y,
    rng,
):
    eligible = [
        key
        for key, bucket
        in hetero.items()
        if len(bucket) >= 2
    ]

    if not eligible:
        return False

    key = rng.choice(
        eligible
    )

    bucket = hetero[key]

    old1, old2 = (
        bucket.random_two(rng)
    )

    class_a, class_b = key

    a1, b1 = orient_hetero_edge(
        old1,
        y,
        class_a,
        class_b,
    )

    a2, b2 = orient_hetero_edge(
        old2,
        y,
        class_a,
        class_b,
    )

    if (
        a1 == a2
        or b1 == b2
    ):
        return False

    new1 = canonical_edge(
        a1,
        a2,
    )

    new2 = canonical_edge(
        b1,
        b2,
    )

    if new1 == new2:
        return False

    old_set = {
        old1,
        old2,
    }

    for edge in (
        new1,
        new2,
    ):
        if (
            edge in edge_set
            and edge not in old_set
        ):
            return False

    bucket.remove(old1)
    bucket.remove(old2)

    edge_set.remove(old1)
    edge_set.remove(old2)

    edge_set.add(new1)
    edge_set.add(new2)

    if class_a not in homo:
        homo[class_a] = (
            IndexedSet()
        )

    if class_b not in homo:
        homo[class_b] = (
            IndexedSet()
        )

    homo[class_a].add(new1)
    homo[class_b].add(new2)

    return True


def decrease_homophily_step(
    edge_set,
    homo,
    hetero,
    y,
    rng,
):
    eligible_classes = [
        c
        for c, bucket
        in homo.items()
        if len(bucket) >= 1
    ]

    if len(
        eligible_classes
    ) < 2:
        return False

    class_a, class_b = (
        rng.sample(
            eligible_classes,
            2,
        )
    )

    old1 = (
        homo[class_a]
        .random_item(rng)
    )

    old2 = (
        homo[class_b]
        .random_item(rng)
    )

    a1, a2 = old1
    b1, b2 = old2

    options = [
        (
            canonical_edge(
                a1,
                b1,
            ),
            canonical_edge(
                a2,
                b2,
            ),
        ),
        (
            canonical_edge(
                a1,
                b2,
            ),
            canonical_edge(
                a2,
                b1,
            ),
        ),
    ]

    rng.shuffle(options)

    chosen = None

    old_set = {
        old1,
        old2,
    }

    for new1, new2 in options:
        if (
            new1 == new2
        ):
            continue

        if (
            new1[0]
            == new1[1]
            or new2[0]
            == new2[1]
        ):
            continue

        conflict = False

        for edge in (
            new1,
            new2,
        ):
            if (
                edge in edge_set
                and edge
                not in old_set
            ):
                conflict = True
                break

        if not conflict:
            chosen = (
                new1,
                new2,
            )
            break

    if chosen is None:
        return False

    new1, new2 = chosen

    homo[class_a].remove(
        old1
    )
    homo[class_b].remove(
        old2
    )

    edge_set.remove(old1)
    edge_set.remove(old2)

    edge_set.add(new1)
    edge_set.add(new2)

    key = tuple(
        sorted(
            (
                class_a,
                class_b,
            )
        )
    )

    if key not in hetero:
        hetero[key] = (
            IndexedSet()
        )

    hetero[key].add(new1)
    hetero[key].add(new2)

    return True


def rewire_to_target(
    original_edges,
    y,
    target_h,
    seed,
    max_proposals,
):
    rng = random.Random(seed)

    (
        edge_set,
        homo,
        hetero,
    ) = initialize_buckets(
        original_edges,
        y,
    )

    m = len(edge_set)

    current_same = (
        edge_same_count(
            edge_set,
            y,
        )
    )

    target_same = int(
        round(
            target_h * m
        )
    )

    proposals = 0

    gross_accepted = 0
    retained_accepted = 0
    rolled_back_swaps = 0
    failed_connectivity_checks = 0

    base_window = 250
    current_window = base_window

    accepted_since_checkpoint = 0

    checkpoint_edges = set(
        edge_set
    )

    checkpoint_same = (
        current_same
    )

    next_report = 2000

    print(
        "start same edges:",
        current_same,
        "/",
        m,
        "=",
        current_same / m,
    )

    print(
        "target same edges:",
        target_same,
        "/",
        m,
        "=",
        target_same / m,
    )

    while (
        abs(
            current_same
            - target_same
        ) > 1
    ):
        proposals += 1

        if (
            proposals
            > max_proposals
        ):
            raise RuntimeError(
                "Maximum proposals "
                "reached before a "
                "connected target graph "
                "was found. "
                f"current_h="
                f"{current_same / m:.8f}, "
                f"target_h="
                f"{target_h:.8f}, "
                f"window="
                f"{current_window}, "
                f"failed_connectivity_checks="
                f"{failed_connectivity_checks}"
            )

        if (
            current_same
            < target_same
        ):
            ok = (
                increase_homophily_step(
                    edge_set,
                    homo,
                    hetero,
                    y,
                    rng,
                )
            )

            if ok:
                current_same += 2

        else:
            ok = (
                decrease_homophily_step(
                    edge_set,
                    homo,
                    hetero,
                    y,
                    rng,
                )
            )

            if ok:
                current_same -= 2

        if not ok:
            continue

        gross_accepted += 1
        accepted_since_checkpoint += 1

        reached_target = (
            abs(
                current_same
                - target_same
            ) <= 1
        )

        need_connectivity_check = (
            accepted_since_checkpoint
            >= current_window
            or reached_target
        )

        if not need_connectivity_check:
            continue

        candidate_edge_index = (
            build_edge_index(
                edge_set
            )
        )

        n_components = (
            number_components(
                candidate_edge_index,
                int(y.numel()),
            )
        )

        if n_components == 1:
            retained_accepted += (
                accepted_since_checkpoint
            )

            checkpoint_edges = set(
                edge_set
            )

            checkpoint_same = (
                current_same
            )

            accepted_since_checkpoint = 0

            if (
                current_window
                < base_window
            ):
                current_window = min(
                    base_window,
                    current_window * 2,
                )

            while (
                retained_accepted
                >= next_report
            ):
                print(
                    "retained:",
                    retained_accepted,
                    "gross accepted:",
                    gross_accepted,
                    "proposals:",
                    proposals,
                    "current_h:",
                    f"{current_same/m:.8f}",
                    "components:",
                    n_components,
                    "window:",
                    current_window,
                )

                next_report += 2000

        else:
            failed_connectivity_checks += 1

            rolled_back_swaps += (
                accepted_since_checkpoint
            )

            edge_set = set(
                checkpoint_edges
            )

            current_same = (
                checkpoint_same
            )

            (
                edge_set,
                homo,
                hetero,
            ) = initialize_buckets(
                list(edge_set),
                y,
            )

            accepted_since_checkpoint = 0

            current_window = max(
                1,
                current_window // 2,
            )

            print(
                "ROLLBACK:",
                "components=",
                n_components,
                "current_h restored=",
                f"{current_same/m:.8f}",
                "new window=",
                current_window,
                "failed checks=",
                failed_connectivity_checks,
            )

    final_edge_index = (
        build_edge_index(
            edge_set
        )
    )

    final_components = (
        number_components(
            final_edge_index,
            int(y.numel()),
        )
    )

    if final_components != 1:
        raise RuntimeError(
            "Final graph is not connected: "
            f"{final_components} components"
        )

    edges = sorted(
        edge_set
    )

    final_h = (
        edge_same_count(
            edges,
            y,
        )
        / m
    )

    print()
    print(
        "final_h:",
        f"{final_h:.10f}",
    )

    print(
        "retained accepted swaps:",
        retained_accepted,
    )

    print(
        "gross accepted swaps:",
        gross_accepted,
    )

    print(
        "rolled-back swaps:",
        rolled_back_swaps,
    )

    print(
        "failed connectivity checks:",
        failed_connectivity_checks,
    )

    print(
        "proposals:",
        proposals,
    )

    print(
        "final components:",
        final_components,
    )

    return (
        edges,
        retained_accepted,
        proposals,
    )

def apportion(
    counts,
    target_total,
):
    counts = np.asarray(
        counts,
        dtype=np.int64,
    )

    total = int(
        counts.sum()
    )

    if target_total <= 0:
        return np.zeros_like(
            counts
        )

    raw = (
        counts
        / total
        * target_total
    )

    alloc = np.floor(
        raw
    ).astype(
        np.int64
    )

    alloc = np.minimum(
        alloc,
        counts,
    )

    remaining = (
        target_total
        - int(alloc.sum())
    )

    remainders = (
        raw - np.floor(raw)
    )

    order = np.argsort(
        -remainders
    )

    while remaining > 0:
        changed = False

        for idx in order:
            if (
                alloc[idx]
                < counts[idx]
            ):
                alloc[idx] += 1
                remaining -= 1
                changed = True

                if remaining == 0:
                    break

        if not changed:
            raise RuntimeError(
                "Could not allocate "
                "stratified counts"
            )

    return alloc


def make_stratified_masks(
    y,
    train_ratio,
    val_ratio,
    num_splits,
    base_seed,
):
    y_np = (
        y.cpu()
        .numpy()
        .astype(
            np.int64
        )
    )

    n = len(y_np)

    classes = np.unique(
        y_np
    )

    counts = np.array(
        [
            np.sum(
                y_np == c
            )
            for c in classes
        ],
        dtype=np.int64,
    )

    train_total = int(
        round(
            train_ratio * n
        )
    )

    val_total = int(
        round(
            val_ratio * n
        )
    )

    train_alloc = apportion(
        counts,
        train_total,
    )

    remaining_counts = (
        counts
        - train_alloc
    )

    val_alloc = apportion(
        remaining_counts,
        val_total,
    )

    train_masks = []
    val_masks = []
    test_masks = []

    for split_idx in range(
        num_splits
    ):
        rng = np.random.default_rng(
            base_seed
            + split_idx
        )

        train_idx = []
        val_idx = []
        test_idx = []

        for class_pos, c in enumerate(
            classes
        ):
            nodes = np.flatnonzero(
                y_np == c
            )

            nodes = nodes.copy()

            rng.shuffle(nodes)

            n_train = int(
                train_alloc[
                    class_pos
                ]
            )

            n_val = int(
                val_alloc[
                    class_pos
                ]
            )

            train_idx.extend(
                nodes[
                    :n_train
                ].tolist()
            )

            val_idx.extend(
                nodes[
                    n_train:
                    n_train
                    + n_val
                ].tolist()
            )

            test_idx.extend(
                nodes[
                    n_train
                    + n_val:
                ].tolist()
            )

        train_mask = torch.zeros(
            n,
            dtype=torch.bool,
        )

        val_mask = torch.zeros(
            n,
            dtype=torch.bool,
        )

        test_mask = torch.zeros(
            n,
            dtype=torch.bool,
        )

        train_mask[
            train_idx
        ] = True

        val_mask[
            val_idx
        ] = True

        test_mask[
            test_idx
        ] = True

        assert not torch.any(
            train_mask
            & val_mask
        )

        assert not torch.any(
            train_mask
            & test_mask
        )

        assert not torch.any(
            val_mask
            & test_mask
        )

        assert torch.all(
            train_mask
            | val_mask
            | test_mask
        )

        train_masks.append(
            train_mask
        )
        val_masks.append(
            val_mask
        )
        test_masks.append(
            test_mask
        )

    return (
        torch.stack(
            train_masks,
            dim=1,
        ),
        torch.stack(
            val_masks,
            dim=1,
        ),
        torch.stack(
            test_masks,
            dim=1,
        ),
    )


def number_components(
    edge_index,
    num_nodes,
):
    src = (
        edge_index[0]
        .cpu()
        .numpy()
    )

    dst = (
        edge_index[1]
        .cpu()
        .numpy()
    )

    values = np.ones(
        len(src),
        dtype=np.int8,
    )

    matrix = coo_matrix(
        (
            values,
            (
                src,
                dst,
            ),
        ),
        shape=(
            num_nodes,
            num_nodes,
        ),
    )

    n_components, _ = (
        connected_components(
            matrix,
            directed=False,
            return_labels=True,
        )
    )

    return int(
        n_components
    )


def validate_graph(
    original_edge_index,
    new_edge_index,
    y,
):
    n = int(
        y.numel()
    )

    original_deg = degree(
        original_edge_index[0],
        num_nodes=n,
        dtype=torch.long,
    )

    new_deg = degree(
        new_edge_index[0],
        num_nodes=n,
        dtype=torch.long,
    )

    degree_difference = (
        new_deg
        - original_deg
    ).abs()

    max_degree_difference = int(
        degree_difference.max()
    )

    self_loops = int(
        (
            new_edge_index[0]
            == new_edge_index[1]
        ).sum()
    )

    edges = (
        unique_undirected_edges(
            new_edge_index
        )
    )

    directed_entries = int(
        new_edge_index.size(1)
    )

    unique_undirected = len(
        edges
    )

    expected_directed = (
        2
        * unique_undirected
    )

    if (
        max_degree_difference
        != 0
    ):
        raise RuntimeError(
            "Degree sequence changed"
        )

    if self_loops != 0:
        raise RuntimeError(
            "Self-loops present"
        )

    if (
        directed_entries
        != expected_directed
    ):
        raise RuntimeError(
            "Duplicate or asymmetric "
            "edge entries detected"
        )

    return {
        "max_degree_difference":
            max_degree_difference,
        "self_loops":
            self_loops,
        "unique_undirected_edges":
            unique_undirected,
        "directed_edge_entries":
            directed_entries,
    }


def dataset_slug(name):
    return (
        name.lower()
        .replace(
            "-",
            "_",
        )
    )


def target_slug(
    target,
):
    return (
        f"h{int(round(target*10)):02d}"
    )


def save_regime(
    base_data,
    output_path,
    train_ratio,
    val_ratio,
    regime,
    graph_seed,
    split_seed,
    metadata,
):
    data = base_data.clone()

    (
        train_mask,
        val_mask,
        test_mask,
    ) = make_stratified_masks(
        data.y,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        num_splits=10,
        base_seed=split_seed,
    )

    data.train_mask = (
        train_mask
    )
    data.val_mask = (
        val_mask
    )
    data.test_mask = (
        test_mask
    )

    data.train_ratio = float(
        train_ratio
    )

    data.val_ratio = float(
        val_ratio
    )

    data.split_regime = regime

    data.graph_seed = int(
        graph_seed
    )

    data.rewiring_seed = int(
        graph_seed
    )

    for key, value in (
        metadata.items()
    ):
        setattr(
            data,
            key,
            value,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        data,
        output_path,
    )

    return {
        "train_split0":
            int(
                train_mask[
                    :, 0
                ].sum()
            ),
        "val_split0":
            int(
                val_mask[
                    :, 0
                ].sum()
            ),
        "test_split0":
            int(
                test_mask[
                    :, 0
                ].sum()
            ),
    }


def update_manifest(row):
    MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    new = pd.DataFrame(
        [row]
    )

    if MANIFEST.exists():
        old = pd.read_csv(
            MANIFEST
        )

        keys = [
            "dataset",
            "target_homophily",
            "graph_seed",
        ]

        mask = pd.Series(
            True,
            index=old.index,
        )

        for key in keys:
            mask &= (
                old[key].astype(str)
                == str(row[key])
            )

        old = old[
            ~mask
        ]

        new = pd.concat(
            [
                old,
                new,
            ],
            ignore_index=True,
        )

    new.to_csv(
        MANIFEST,
        index=False,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "PubMed",
            "Roman-empire",
        ],
    )

    parser.add_argument(
        "--target",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--max-proposals",
        type=int,
        default=5000000,
    )

    parser.add_argument(
        "--out-root",
        default=str(
            DEFAULT_OUT
        ),
    )

    args = parser.parse_args()

    dataset, original = (
        load_dataset_any(
            args.dataset,
            root=str(
                DATA_ROOT
            ),
            split_idx=0,
        )
    )

    original = original.cpu()

    y = original.y.long()

    num_nodes = int(
        original.num_nodes
    )

    num_classes = int(
        getattr(
            dataset,
            "num_classes",
            int(y.max()) + 1,
        )
    )

    original_edges = (
        unique_undirected_edges(
            original.edge_index
        )
    )

    m = len(
        original_edges
    )

    natural_h = (
        edge_homophily_from_edges(
            original_edges,
            y,
        )
    )

    print()
    print("=" * 80)
    print(
        args.dataset,
        "target=",
        args.target,
        "seed=",
        args.seed,
    )
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
        "natural homophily:",
        f"{natural_h:.10f}",
    )

    new_edges, accepted, proposals = (
        rewire_to_target(
            original_edges,
            y,
            target_h=args.target,
            seed=args.seed,
            max_proposals=(
                args.max_proposals
            ),
        )
    )

    new_edge_index = (
        build_edge_index(
            new_edges
        )
    )

    validation = (
        validate_graph(
            original.edge_index,
            new_edge_index,
            y,
        )
    )

    original_components = (
        number_components(
            original.edge_index,
            num_nodes,
        )
    )

    new_components = (
        number_components(
            new_edge_index,
            num_nodes,
        )
    )

    realized_h = float(
        homophily(
            new_edge_index,
            y,
            method="edge",
        )
    )

    chance_h = (
        chance_homophily(
            new_edge_index,
            y,
            num_classes,
        )
    )

    adjusted_h = (
        adjusted_homophily(
            realized_h,
            chance_h,
        )
    )

    original_deg = degree(
        original.edge_index[0],
        num_nodes=num_nodes,
        dtype=torch.float,
    )

    base_data = (
        original.clone()
    )

    base_data.edge_index = (
        new_edge_index
    )

    base_data.dataset_name = (
        f"{args.dataset}-"
        f"H{args.target:.1f}"
    )

    base_data.original_dataset_name = (
        args.dataset
    )

    base_data.target_homophily = (
        float(args.target)
    )

    base_data.realized_homophily = (
        realized_h
    )

    base_data.adjusted_homophily = (
        adjusted_h
    )

    base_data.chance_homophily = (
        chance_h
    )

    base_data.original_homophily = (
        natural_h
    )

    base_data.num_classes = (
        num_classes
    )

    base_data.realized_average_degree = (
        float(
            original_deg.mean()
        )
    )

    base_data.feature_signal = (
        float("nan")
    )

    base_data.feature_signal_type = (
        "natural_realworld"
    )

    slug = dataset_slug(
        args.dataset
    )

    hslug = target_slug(
        args.target
    )

    out_root = Path(
        args.out_root
    )

    split_seed_base = (
        100000
        if args.dataset
        == "PubMed"
        else 200000
    )

    split_seed = (
        split_seed_base
        + args.seed * 100
    )

    metadata = {
        "target_homophily":
            float(
                args.target
            ),
        "realized_homophily":
            realized_h,
        "adjusted_homophily":
            adjusted_h,
        "chance_homophily":
            chance_h,
        "original_homophily":
            natural_h,
        "original_components":
            original_components,
        "rewired_components":
            new_components,
    }

    lowlabel_path = (
        out_root
        / slug
        / "lowlabel"
        / (
            f"{slug}_"
            f"{hslug}_"
            f"seed{args.seed}.pt"
        )
    )

    replicated_path = (
        out_root
        / slug
        / "replicated"
        / (
            f"{slug}_"
            f"{hslug}_"
            f"seed{args.seed}.pt"
        )
    )

    low_counts = save_regime(
        base_data,
        lowlabel_path,
        train_ratio=0.05,
        val_ratio=0.10,
        regime="lowlabel_05_10_85",
        graph_seed=args.seed,
        split_seed=split_seed,
        metadata=metadata,
    )

    replicated_counts = (
        save_regime(
            base_data,
            replicated_path,
            train_ratio=0.60,
            val_ratio=0.20,
            regime="replicated_60_20_20",
            graph_seed=args.seed,
            split_seed=split_seed,
            metadata=metadata,
        )
    )

    target_error = abs(
        realized_h
        - args.target
    )

    if target_error > 0.0001:
        raise RuntimeError(
            "Target homophily "
            "error too large: "
            f"{target_error}"
        )

    row = {
        "dataset":
            args.dataset,
        "graph_seed":
            args.seed,
        "target_homophily":
            args.target,
        "realized_homophily":
            realized_h,
        "target_error":
            target_error,
        "adjusted_homophily":
            adjusted_h,
        "chance_homophily":
            chance_h,
        "natural_homophily":
            natural_h,
        "num_nodes":
            num_nodes,
        "undirected_edges":
            m,
        "accepted_swaps":
            accepted,
        "proposals":
            proposals,
        "max_degree_difference":
            validation[
                "max_degree_difference"
            ],
        "self_loops":
            validation[
                "self_loops"
            ],
        "original_components":
            original_components,
        "rewired_components":
            new_components,
        "lowlabel_train":
            low_counts[
                "train_split0"
            ],
        "lowlabel_val":
            low_counts[
                "val_split0"
            ],
        "lowlabel_test":
            low_counts[
                "test_split0"
            ],
        "replicated_train":
            replicated_counts[
                "train_split0"
            ],
        "replicated_val":
            replicated_counts[
                "val_split0"
            ],
        "replicated_test":
            replicated_counts[
                "test_split0"
            ],
        "lowlabel_path":
            str(
                lowlabel_path
            ),
        "replicated_path":
            str(
                replicated_path
            ),
    }

    update_manifest(row)

    print()
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)

    for key, value in (
        row.items()
    ):
        print(
            f"{key}:",
            value,
        )

    print()
    print(
        "Saved manifest:",
        MANIFEST,
    )


if __name__ == "__main__":
    main()
