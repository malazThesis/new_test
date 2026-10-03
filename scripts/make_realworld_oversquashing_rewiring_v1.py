from pathlib import Path
import argparse
import json
import math
import random

import numpy as np
import pandas as pd
import torch

from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh


def load_data(path):
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


def canonical_edges(edge_index):
    edges = set()

    src = edge_index[0].cpu().tolist()
    dst = edge_index[1].cpu().tolist()

    for u, v in zip(src, dst):
        u = int(u)
        v = int(v)

        if u == v:
            continue

        if u > v:
            u, v = v, u

        edges.add((u, v))

    return edges


def edge_index_from_edges(edges):
    rows = []
    cols = []

    for u, v in sorted(edges):
        rows.extend([u, v])
        cols.extend([v, u])

    return torch.tensor(
        [rows, cols],
        dtype=torch.long,
    )


def adjacency_from_edges(edges, n):
    if not edges:
        return sparse.csr_matrix(
            (n, n),
            dtype=np.float64,
        )

    rows = []
    cols = []

    for u, v in edges:
        rows.extend([u, v])
        cols.extend([v, u])

    vals = np.ones(
        len(rows),
        dtype=np.float64,
    )

    return sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(n, n),
    )


def normalized_laplacian(A):
    deg = np.asarray(
        A.sum(axis=1)
    ).reshape(-1)

    inv_sqrt = np.zeros_like(
        deg,
        dtype=np.float64,
    )

    mask = deg > 0

    inv_sqrt[mask] = (
        1.0
        / np.sqrt(deg[mask])
    )

    D = sparse.diags(
        inv_sqrt
    )

    I = sparse.eye(
        A.shape[0],
        dtype=np.float64,
        format="csr",
    )

    return (
        I
        - D @ A @ D
    ).tocsr()


def fiedler_vector(edges, n):
    A = adjacency_from_edges(
        edges,
        n,
    )

    L = normalized_laplacian(A)

    vals, vecs = eigsh(
        L,
        k=2,
        which="SM",
        tol=1e-5,
        maxiter=max(
            5000,
            10 * n,
        ),
    )

    order = np.argsort(vals)

    vals = vals[order]
    vecs = vecs[:, order]

    return (
        float(vals[1]),
        np.asarray(
            vecs[:, 1],
            dtype=np.float64,
        ),
    )


def spectral_target_edges(
    original_edges,
    fiedler,
    budget,
):
    """
    Add non-edges between opposite Fiedler extremes.

    No labels/features are used.
    """

    n = len(fiedler)

    order = np.argsort(fiedler)

    # Large enough candidate pools while remaining cheap.
    pool_size = min(
        n,
        max(
            256,
            4 * budget,
        ),
    )

    low = [
        int(x)
        for x in order[:pool_size]
    ]

    high = [
        int(x)
        for x in order[-pool_size:][::-1]
    ]

    added = set()

    # Deterministic search over large spectral separation.
    #
    # Different offsets stop us from depending on one
    # particular perfect matching.
    for offset in range(pool_size):
        if len(added) >= budget:
            break

        for i in range(pool_size):
            if len(added) >= budget:
                break

            u = low[i]
            v = high[
                (i + offset)
                % pool_size
            ]

            if u == v:
                continue

            e = (
                (u, v)
                if u < v
                else (v, u)
            )

            if (
                e in original_edges
                or e in added
            ):
                continue

            added.add(e)

    if len(added) != budget:
        raise RuntimeError(
            "Could only construct "
            f"{len(added)} targeted edges "
            f"for requested budget {budget}"
        )

    return added


def random_non_edges(
    original_edges,
    budget,
    n,
    seed,
):
    rng = random.Random(
        int(seed)
    )

    added = set()

    attempts = 0
    max_attempts = max(
        100000,
        budget * 1000,
    )

    while (
        len(added) < budget
        and attempts < max_attempts
    ):
        attempts += 1

        u = rng.randrange(n)
        v = rng.randrange(n)

        if u == v:
            continue

        if u > v:
            u, v = v, u

        e = (u, v)

        if (
            e in original_edges
            or e in added
        ):
            continue

        added.add(e)

    if len(added) != budget:
        raise RuntimeError(
            "Random-control generation "
            f"stopped at {len(added)}/{budget}"
        )

    return added


def edge_homophily(edges, y):
    if not edges:
        return float("nan")

    yy = y.cpu().numpy()

    same = sum(
        int(yy[u] == yy[v])
        for u, v in edges
    )

    return (
        same
        / len(edges)
    )


def degree_stats(edges, n):
    deg = np.zeros(
        n,
        dtype=np.int64,
    )

    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    return {
        "degree_mean":
            float(deg.mean()),
        "degree_std":
            float(deg.std(ddof=0)),
        "degree_min":
            int(deg.min()),
        "degree_max":
            int(deg.max()),
    }


def graph_metrics(
    edges,
    n,
    y,
):
    A = adjacency_from_edges(
        edges,
        n,
    )

    n_components, _ = (
        connected_components(
            A,
            directed=False,
            return_labels=True,
        )
    )

    lambda2, fiedler = (
        fiedler_vector(
            edges,
            n,
        )
    )

    # Fiedler median cut as a simple
    # label-free bottleneck proxy.
    threshold = float(
        np.median(fiedler)
    )

    side = (
        fiedler <= threshold
    )

    deg = np.asarray(
        A.sum(axis=1)
    ).reshape(-1)

    volume_a = float(
        deg[side].sum()
    )

    volume_b = float(
        deg[~side].sum()
    )

    cut_edges = 0

    for u, v in edges:
        if side[u] != side[v]:
            cut_edges += 1

    denom = min(
        volume_a,
        volume_b,
    )

    conductance = (
        float(cut_edges / denom)
        if denom > 0
        else float("nan")
    )

    out = {
        "undirected_edges":
            int(len(edges)),
        "components":
            int(n_components),
        "edge_homophily":
            float(
                edge_homophily(
                    edges,
                    y,
                )
            ),
        "lambda2_norm_laplacian":
            float(lambda2),
        "fiedler_cut_edges":
            int(cut_edges),
        "fiedler_conductance":
            float(conductance),
    }

    out.update(
        degree_stats(
            edges,
            n,
        )
    )

    return out


def save_variant(
    source_data,
    edges,
    out_path,
    dataset,
    graph_seed,
    variant,
    budget,
    added_edges,
    metrics,
):
    data = source_data.clone()

    data.edge_index = (
        edge_index_from_edges(
            edges
        )
    )

    data.oversquashing_variant = (
        variant
    )

    data.oversquashing_dataset = (
        dataset
    )

    data.oversquashing_graph_seed = (
        int(graph_seed)
    )

    data.oversquashing_edge_budget = (
        int(budget)
    )

    data.oversquashing_added_edges = (
        int(added_edges)
    )

    data.oversquashing_rewiring_version = (
        "spectral_bypass_v1"
    )

    for key, value in metrics.items():
        setattr(
            data,
            "oversquashing_" + key,
            value,
        )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        data,
        out_path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-root",
        required=True,
    )

    parser.add_argument(
        "--out-root",
        required=True,
    )

    parser.add_argument(
        "--budget-fraction",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "pubmed",
            "roman_empire",
        ],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
    )

    args = parser.parse_args()

    source_root = Path(
        args.source_root
    )

    out_root = Path(
        args.out_root
    )

    rows = []

    for dataset in args.datasets:
        for graph_seed in args.seeds:
            source_path = (
                source_root
                / dataset
                / "lowlabel"
                / (
                    f"{dataset}_h01_"
                    f"seed{graph_seed}.pt"
                )
            )

            print()
            print("=" * 80)
            print(source_path)
            print("=" * 80)

            if not source_path.exists():
                raise FileNotFoundError(
                    source_path
                )

            data = load_data(
                source_path
            )

            n = int(
                data.x.size(0)
            )

            original_edges = (
                canonical_edges(
                    data.edge_index
                )
            )

            m = len(
                original_edges
            )

            budget = max(
                1,
                int(
                    round(
                        args.budget_fraction
                        * m
                    )
                ),
            )

            print(
                "nodes:",
                n,
                "edges:",
                m,
                "budget:",
                budget,
            )

            # Fiedler vector determined on the
            # original graph only.
            original_lambda2, fiedler = (
                fiedler_vector(
                    original_edges,
                    n,
                )
            )

            targeted_added = (
                spectral_target_edges(
                    original_edges,
                    fiedler,
                    budget,
                )
            )

            random_added = (
                random_non_edges(
                    original_edges,
                    budget,
                    n,
                    seed=(
                        900000
                        + graph_seed
                        + (
                            0
                            if dataset == "pubmed"
                            else 10000
                        )
                    ),
                )
            )

            variants = {
                "original":
                    original_edges,

                "targeted":
                    (
                        original_edges
                        | targeted_added
                    ),

                "random":
                    (
                        original_edges
                        | random_added
                    ),
            }

            for variant, edges in variants.items():
                metrics = graph_metrics(
                    edges,
                    n,
                    data.y,
                )

                if variant == "original":
                    added = 0
                else:
                    added = (
                        len(edges)
                        - len(original_edges)
                    )

                assert (
                    added == 0
                    if variant == "original"
                    else added == budget
                )

                out_path = (
                    out_root
                    / dataset
                    / (
                        f"{dataset}_h01_"
                        f"seed{graph_seed}_"
                        f"{variant}.pt"
                    )
                )

                save_variant(
                    source_data=data,
                    edges=edges,
                    out_path=out_path,
                    dataset=dataset,
                    graph_seed=graph_seed,
                    variant=variant,
                    budget=budget,
                    added_edges=added,
                    metrics=metrics,
                )

                row = {
                    "dataset":
                        dataset,
                    "graph_seed":
                        graph_seed,
                    "variant":
                        variant,
                    "source_path":
                        str(source_path),
                    "output_path":
                        str(out_path),
                    "edge_budget":
                        budget,
                    "added_edges":
                        added,
                    "original_lambda2":
                        original_lambda2,
                    **metrics,
                }

                rows.append(row)

                print(
                    variant,
                    "edges=",
                    metrics[
                        "undirected_edges"
                    ],
                    "h=",
                    f"{metrics['edge_homophily']:.6f}",
                    "lambda2=",
                    f"{metrics['lambda2_norm_laplacian']:.6g}",
                    "conductance=",
                    f"{metrics['fiedler_conductance']:.6g}",
                    "components=",
                    metrics[
                        "components"
                    ],
                )

    manifest = pd.DataFrame(
        rows
    )

    manifest_path = (
        out_root
        / "manifest.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    print()
    print("SAVED:", manifest_path)
    print("ROWS:", len(manifest))


if __name__ == "__main__":
    main()
