from pathlib import Path
import argparse
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lobpcg
from torch_geometric.datasets import Planetoid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPTS_ROOT),
    )

from dataset_loader_realworld_extra import (
    load_extra_dataset,
)


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


def load_natural(
    dataset,
    natural_root,
):
    if dataset == "pubmed":
        ds = Planetoid(
            root=str(
                natural_root
                / "Planetoid"
                / "PubMed"
            ),
            name="PubMed",
        )

        return ds[0].cpu()

    if dataset == "roman_empire":
        _, data = load_extra_dataset(
            name="Roman-empire",
            root=str(natural_root),
            split_idx=0,
        )

        return data.cpu()

    raise ValueError(dataset)


def controlled_path(
    controlled_root,
    dataset,
    seed,
):
    return (
        controlled_root
        / dataset
        / "lowlabel"
        / (
            f"{dataset}_h01_"
            f"seed{seed}.pt"
        )
    )


def canonical_edges(
    edge_index,
):
    edges = set()

    src = (
        edge_index[0]
        .cpu()
        .tolist()
    )

    dst = (
        edge_index[1]
        .cpu()
        .tolist()
    )

    for u, v in zip(
        src,
        dst,
    ):
        u = int(u)
        v = int(v)

        if u == v:
            continue

        if u > v:
            u, v = v, u

        edges.add(
            (u, v)
        )

    return edges


def edge_index_from_edges(
    edges,
):
    rows = []
    cols = []

    for u, v in sorted(
        edges
    ):
        rows.extend(
            [u, v]
        )
        cols.extend(
            [v, u]
        )

    return torch.tensor(
        [rows, cols],
        dtype=torch.long,
    )


def adjacency(
    edges,
    n,
):
    rows = []
    cols = []

    for u, v in edges:
        rows.extend(
            [u, v]
        )
        cols.extend(
            [v, u]
        )

    values = np.ones(
        len(rows),
        dtype=np.float64,
    )

    return sparse.csr_matrix(
        (
            values,
            (
                rows,
                cols,
            ),
        ),
        shape=(n, n),
    )


def fast_fiedler(
    edges,
    n,
    seed=12345,
):
    A = adjacency(
        edges,
        n,
    )

    deg = np.asarray(
        A.sum(axis=1)
    ).reshape(-1)

    inv_sqrt = np.zeros(
        n,
        dtype=np.float64,
    )

    positive = (
        deg > 0
    )

    inv_sqrt[
        positive
    ] = (
        1.0
        / np.sqrt(
            deg[positive]
        )
    )

    D = sparse.diags(
        inv_sqrt
    )

    L = (
        sparse.eye(
            n,
            dtype=np.float64,
            format="csr",
        )
        - D @ A @ D
    ).tocsr()

    trivial = np.sqrt(
        np.maximum(
            deg,
            0,
        )
    ).reshape(-1, 1)

    trivial /= (
        np.linalg.norm(
            trivial
        )
        + 1e-15
    )

    rng = np.random.default_rng(
        int(seed)
    )

    X = rng.normal(
        size=(n, 1)
    )

    X -= (
        trivial
        @ (
            trivial.T
            @ X
        )
    )

    X /= (
        np.linalg.norm(X)
        + 1e-15
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )

        values, vectors = lobpcg(
            L,
            X,
            Y=trivial,
            largest=False,
            tol=1e-5,
            maxiter=400,
        )

    return (
        float(values[0]),
        np.asarray(
            vectors[:, 0]
        ),
        A,
        deg,
    )


def cheeger_sweep_cut(
    edges,
    n,
    fiedler,
    min_nodes,
):
    """
    Sweep along the Fiedler ordering and select
    the lowest-conductance balanced cut.

    min_nodes ensures that both sides are large
    enough for the intervention edge budget.
    """

    neighbors = [
        []
        for _ in range(n)
    ]

    deg = np.zeros(
        n,
        dtype=np.int64,
    )

    for u, v in edges:
        neighbors[u].append(v)
        neighbors[v].append(u)

        deg[u] += 1
        deg[v] += 1

    order = np.argsort(
        fiedler
    )

    inside = np.zeros(
        n,
        dtype=bool,
    )

    total_volume = float(
        deg.sum()
    )

    volume = 0.0
    cut_edges = 0

    best_phi = float("inf")
    best_k = None
    best_cut_edges = None

    for k, u in enumerate(
        order,
        start=1,
    ):
        u = int(u)

        # Before inserting u:
        # edges to existing side stop crossing;
        # edges to outside begin crossing.
        for v in neighbors[u]:
            if inside[v]:
                cut_edges -= 1
            else:
                cut_edges += 1

        inside[u] = True
        volume += float(
            deg[u]
        )

        other_nodes = (
            n - k
        )

        if (
            k < min_nodes
            or other_nodes < min_nodes
        ):
            continue

        other_volume = (
            total_volume
            - volume
        )

        denominator = min(
            volume,
            other_volume,
        )

        if denominator <= 0:
            continue

        phi = (
            cut_edges
            / denominator
        )

        if phi < best_phi:
            best_phi = float(phi)
            best_k = int(k)
            best_cut_edges = int(
                cut_edges
            )

    if best_k is None:
        raise RuntimeError(
            "No valid Cheeger sweep cut "
            f"for min_nodes={min_nodes}"
        )

    side = np.zeros(
        n,
        dtype=bool,
    )

    side[
        order[:best_k]
    ] = True

    threshold = float(
        0.5
        * (
            fiedler[
                order[best_k - 1]
            ]
            + fiedler[
                order[best_k]
            ]
        )
    )

    return {
        "side":
            side,
        "conductance":
            best_phi,
        "cut_edges":
            best_cut_edges,
        "side_a_nodes":
            int(best_k),
        "side_b_nodes":
            int(n - best_k),
        "threshold":
            threshold,
    }


def fixed_cut_metrics(
    edges,
    n,
    side,
):
    deg = np.zeros(
        n,
        dtype=np.int64,
    )

    crossing = 0

    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

        if (
            bool(side[u])
            != bool(side[v])
        ):
            crossing += 1

    vol_a = float(
        deg[side].sum()
    )

    vol_b = float(
        deg[~side].sum()
    )

    denominator = min(
        vol_a,
        vol_b,
    )

    phi = (
        crossing
        / denominator
        if denominator > 0
        else float("nan")
    )

    return {
        "fixed_cut_edges":
            int(crossing),
        "fixed_cut_conductance":
            float(phi),
        "fixed_cut_volume_a":
            vol_a,
        "fixed_cut_volume_b":
            vol_b,
    }


def select_extreme_endpoints(
    fiedler,
    side,
    budget,
):
    a = np.where(
        side
    )[0]

    b = np.where(
        ~side
    )[0]

    if (
        len(a) < budget
        or len(b) < budget
    ):
        raise RuntimeError(
            "Cut side smaller than budget"
        )

    # Side A consists of the low end of the
    # Fiedler sweep. Choose structurally distant
    # nodes on opposite sides.
    a_sorted = sorted(
        (
            int(x)
            for x in a
        ),
        key=lambda x:
            float(fiedler[x]),
    )

    b_sorted = sorted(
        (
            int(x)
            for x in b
        ),
        key=lambda x:
            float(fiedler[x]),
        reverse=True,
    )

    return (
        a_sorted[:budget],
        b_sorted[:budget],
    )


def targeted_cross_cut_matching(
    original_edges,
    a_nodes,
    b_nodes,
    seed,
    max_attempts=5000,
):
    """
    Every selected endpoint is used exactly once.
    Every added edge crosses the detected bottleneck.
    """

    rng = random.Random(
        int(seed)
    )

    a_nodes = list(
        a_nodes
    )

    b_base = list(
        b_nodes
    )

    for _ in range(
        max_attempts
    ):
        b_nodes = list(
            b_base
        )

        rng.shuffle(
            b_nodes
        )

        added = set()

        valid = True

        for u, v in zip(
            a_nodes,
            b_nodes,
        ):
            if u > v:
                e = (v, u)
            else:
                e = (u, v)

            if (
                e in original_edges
                or e in added
            ):
                valid = False
                break

            added.add(e)

        if (
            valid
            and len(added)
            == len(a_nodes)
        ):
            return added

    raise RuntimeError(
        "Could not construct targeted "
        "cross-cut matching"
    )


def random_endpoint_matching(
    original_edges,
    endpoints,
    targeted_added,
    seed,
    max_attempts=20000,
):
    """
    Same endpoints as targeted, each used once,
    but pairings are fully randomized.

    Therefore targeted and random have exactly
    the same per-node added degree.
    """

    rng = random.Random(
        int(seed)
    )

    endpoints = list(
        endpoints
    )

    for _ in range(
        max_attempts
    ):
        shuffled = list(
            endpoints
        )

        rng.shuffle(
            shuffled
        )

        added = set()
        valid = True

        for i in range(
            0,
            len(shuffled),
            2,
        ):
            u = int(
                shuffled[i]
            )

            v = int(
                shuffled[i + 1]
            )

            if u == v:
                valid = False
                break

            if u > v:
                e = (v, u)
            else:
                e = (u, v)

            if (
                e in original_edges
                or e in added
            ):
                valid = False
                break

            added.add(e)

        if not valid:
            continue

        if (
            len(added)
            != len(endpoints) // 2
        ):
            continue

        if added == targeted_added:
            continue

        return added

    raise RuntimeError(
        "Could not construct random "
        "degree-matched matching"
    )


def degree_vector(
    edges,
    n,
):
    deg = np.zeros(
        n,
        dtype=np.int64,
    )

    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    return deg


def edge_homophily(
    edges,
    y,
):
    yy = (
        y.cpu()
        .numpy()
    )

    return (
        sum(
            int(
                yy[u]
                == yy[v]
            )
            for u, v in edges
        )
        / len(edges)
    )


def graph_metrics(
    edges,
    n,
    y,
    original_side,
    solver_seed,
    budget,
):
    lambda2, fiedler, A, deg = (
        fast_fiedler(
            edges,
            n,
            seed=solver_seed,
        )
    )

    components = int(
        connected_components(
            A,
            directed=False,
            return_labels=False,
        )
    )

    sweep = cheeger_sweep_cut(
        edges,
        n,
        fiedler,
        min_nodes=max(
            budget,
            int(
                round(
                    0.05 * n
                )
            ),
        ),
    )

    fixed = fixed_cut_metrics(
        edges,
        n,
        original_side,
    )

    return {
        "undirected_edges":
            int(len(edges)),
        "components":
            components,
        "edge_homophily":
            float(
                edge_homophily(
                    edges,
                    y,
                )
            ),
        "lambda2_norm_laplacian":
            float(lambda2),
        "cheeger_cut_edges":
            int(
                sweep[
                    "cut_edges"
                ]
            ),
        "cheeger_conductance":
            float(
                sweep[
                    "conductance"
                ]
            ),
        "degree_mean":
            float(
                deg.mean()
            ),
        "degree_std":
            float(
                deg.std()
            ),
        "degree_min":
            int(
                deg.min()
            ),
        "degree_max":
            int(
                deg.max()
            ),
        **fixed,
    }


def single_mask(
    mask,
):
    if mask.dim() == 2:
        return (
            mask[:, 0]
            .clone()
            .bool()
        )

    return (
        mask.clone()
        .bool()
    )


def make_output_data(
    natural,
    controlled,
    edge_index,
    dataset,
    split_seed,
    variant,
    budget,
    metrics,
):
    data = natural.clone()

    # Natural features / labels / topology,
    # but frozen Low-Label split from the
    # controlled experiment.
    data.train_mask = single_mask(
        controlled.train_mask
    )

    data.val_mask = single_mask(
        controlled.val_mask
    )

    data.test_mask = single_mask(
        controlled.test_mask
    )

    data.edge_index = (
        edge_index.clone()
    )

    data.oversquashing_dataset = (
        dataset
    )

    data.oversquashing_variant = (
        variant
    )

    data.oversquashing_split_seed = (
        int(split_seed)
    )

    data.oversquashing_edge_budget = (
        int(budget)
    )

    data.oversquashing_rewiring_version = (
        "fiedler_cheeger_endpoint_matched_v3"
    )

    for key, value in (
        metrics.items()
    ):
        setattr(
            data,
            "oversquashing_" + key,
            value,
        )

    return data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--natural-root",
        required=True,
    )

    parser.add_argument(
        "--controlled-root",
        required=True,
    )

    parser.add_argument(
        "--out-root",
        required=True,
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
        default=[
            1, 2, 3, 4, 5,
        ],
    )

    parser.add_argument(
        "--budget-fraction",
        type=float,
        default=0.01,
    )

    args = parser.parse_args()

    natural_root = Path(
        args.natural_root
    )

    controlled_root = Path(
        args.controlled_root
    )

    out_root = Path(
        args.out_root
    )

    rows = []

    for dataset in (
        args.datasets
    ):
        print()
        print("#" * 80)
        print("DATASET:", dataset)
        print("#" * 80)

        natural = load_natural(
            dataset,
            natural_root,
        )

        n = int(
            natural.num_nodes
        )

        original_edges = (
            canonical_edges(
                natural.edge_index
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
        )

        print(
            "original edges:",
            m,
        )

        print(
            "budget:",
            budget,
        )

        t0 = time.time()

        (
            original_lambda2,
            original_fiedler,
            _,
            _,
        ) = fast_fiedler(
            original_edges,
            n,
            seed=12345,
        )

        original_sweep = (
            cheeger_sweep_cut(
                original_edges,
                n,
                original_fiedler,
                min_nodes=max(
                    budget,
                    int(
                        round(
                            0.05 * n
                        )
                    ),
                ),
            )
        )

        original_side = (
            original_sweep[
                "side"
            ]
        )

        print(
            "original lambda2:",
            original_lambda2,
        )

        print(
            "original Cheeger conductance:",
            original_sweep[
                "conductance"
            ],
        )

        print(
            "original Cheeger cut edges:",
            original_sweep[
                "cut_edges"
            ],
        )

        print(
            "cut sides:",
            original_sweep[
                "side_a_nodes"
            ],
            original_sweep[
                "side_b_nodes"
            ],
        )

        (
            a_nodes,
            b_nodes,
        ) = select_extreme_endpoints(
            original_fiedler,
            original_side,
            budget,
        )

        endpoints = (
            list(a_nodes)
            + list(b_nodes)
        )

        for split_seed in (
            args.seeds
        ):
            print()
            print(
                "split/intervention seed:",
                split_seed,
            )

            control_path = (
                controlled_path(
                    controlled_root,
                    dataset,
                    split_seed,
                )
            )

            controlled = load_pt(
                control_path
            ).cpu()

            assert (
                natural.num_nodes
                == controlled.num_nodes
            )

            assert torch.equal(
                natural.y,
                controlled.y,
            )

            assert torch.allclose(
                natural.x,
                controlled.x,
                rtol=0,
                atol=0,
            )

            targeted_added = (
                targeted_cross_cut_matching(
                    original_edges,
                    a_nodes,
                    b_nodes,
                    seed=(
                        100000
                        + split_seed
                        + (
                            0
                            if dataset
                            == "pubmed"
                            else 10000
                        )
                    ),
                )
            )

            random_added = (
                random_endpoint_matching(
                    original_edges,
                    endpoints,
                    targeted_added,
                    seed=(
                        900000
                        + split_seed
                        + (
                            0
                            if dataset
                            == "pubmed"
                            else 10000
                        )
                    ),
                )
            )

            targeted_edges = (
                original_edges
                | targeted_added
            )

            random_edges = (
                original_edges
                | random_added
            )

            # Exact endpoint-degree control.
            d0 = degree_vector(
                original_edges,
                n,
            )

            dt = degree_vector(
                targeted_edges,
                n,
            )

            dr = degree_vector(
                random_edges,
                n,
            )

            assert np.array_equal(
                dt - d0,
                dr - d0,
            )

            assert (
                int(
                    (dt - d0).sum()
                )
                == 2 * budget
            )

            targeted_cross_added = sum(
                int(
                    original_side[u]
                    != original_side[v]
                )
                for u, v
                in targeted_added
            )

            random_cross_added = sum(
                int(
                    original_side[u]
                    != original_side[v]
                )
                for u, v
                in random_added
            )

            assert (
                targeted_cross_added
                == budget
            )

            variants = {
                "original":
                    original_edges,
                "targeted":
                    targeted_edges,
                "random":
                    random_edges,
            }

            for variant, edges in (
                variants.items()
            ):
                start = time.time()

                metrics = graph_metrics(
                    edges,
                    n,
                    natural.y,
                    original_side,
                    solver_seed=(
                        200000
                        + split_seed
                    ),
                    budget=budget,
                )

                if variant == "original":
                    added = 0
                    added_cross = 0

                elif variant == "targeted":
                    added = budget
                    added_cross = (
                        targeted_cross_added
                    )

                else:
                    added = budget
                    added_cross = (
                        random_cross_added
                    )

                metrics[
                    "added_edges"
                ] = int(
                    added
                )

                metrics[
                    "added_edges_cross_original_cut"
                ] = int(
                    added_cross
                )

                metrics[
                    "original_lambda2"
                ] = float(
                    original_lambda2
                )

                metrics[
                    "original_cheeger_conductance"
                ] = float(
                    original_sweep[
                        "conductance"
                    ]
                )

                metrics[
                    "original_cheeger_cut_edges"
                ] = int(
                    original_sweep[
                        "cut_edges"
                    ]
                )

                data = make_output_data(
                    natural,
                    controlled,
                    edge_index_from_edges(
                        edges
                    ),
                    dataset,
                    split_seed,
                    variant,
                    budget,
                    metrics,
                )

                output_path = (
                    out_root
                    / dataset
                    / (
                        f"{dataset}_"
                        f"splitseed{split_seed}_"
                        f"{variant}.pt"
                    )
                )

                output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                torch.save(
                    data,
                    output_path,
                )

                row = {
                    "dataset":
                        dataset,
                    "split_seed":
                        split_seed,
                    "variant":
                        variant,
                    "source_natural_root":
                        str(
                            natural_root
                        ),
                    "source_mask_path":
                        str(
                            control_path
                        ),
                    "output_path":
                        str(
                            output_path
                        ),
                    "edge_budget":
                        budget,
                    **metrics,
                }

                rows.append(
                    row
                )

                print(
                    variant,
                    "added=",
                    added,
                    "cross_added=",
                    added_cross,
                    "lambda2=",
                    f"{metrics['lambda2_norm_laplacian']:.8g}",
                    "fixed_phi=",
                    f"{metrics['fixed_cut_conductance']:.6g}",
                    "cheeger_phi=",
                    f"{metrics['cheeger_conductance']:.6g}",
                    "h=",
                    f"{metrics['edge_homophily']:.6f}",
                    "time=",
                    f"{time.time()-start:.1f}s",
                )

        print(
            "dataset generation time:",
            f"{time.time()-t0:.1f}s",
        )

    manifest = pd.DataFrame(
        rows
    )

    out_root.mkdir(
        parents=True,
        exist_ok=True,
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
    print(
        "SAVED:",
        manifest_path,
    )

    print(
        "ROWS:",
        len(manifest),
    )


if __name__ == "__main__":
    main()
