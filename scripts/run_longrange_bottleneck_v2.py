from pathlib import Path
import argparse
import copy
import csv
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def balanced_channels(n_sources, width, graph_seed):
    rng = np.random.default_rng(
        int(graph_seed) * 100003
        + int(width) * 1009
    )

    perm = rng.permutation(
        n_sources
    )

    channels = np.empty(
        n_sources,
        dtype=np.int64,
    )

    for position, source in enumerate(
        perm
    ):
        channels[
            source
        ] = (
            position
            % width
        )

    return channels


def build_topology(
    distance,
    width,
    n_sources,
    graph_seed,
    rewired,
):
    if distance < 2:
        raise ValueError(
            "distance must be >= 2"
        )

    channels = balanced_channels(
        n_sources=n_sources,
        width=width,
        graph_seed=graph_seed,
    )

    n_channel_layers = (
        distance - 1
    )

    source_start = 0

    channel_start = (
        source_start
        + n_sources
    )

    target_start = (
        channel_start
        + n_channel_layers
        * width
    )

    n_nodes = (
        target_start
        + n_sources
    )

    edges = []

    def channel_node(
        layer,
        channel,
    ):
        return (
            channel_start
            + layer * width
            + channel
        )

    for source in range(
        n_sources
    ):
        channel = int(
            channels[
                source
            ]
        )

        edges.append(
            (
                source,
                channel_node(
                    0,
                    channel,
                ),
            )
        )

    for layer in range(
        n_channel_layers - 1
    ):
        for channel in range(
            width
        ):
            edges.append(
                (
                    channel_node(
                        layer,
                        channel,
                    ),
                    channel_node(
                        layer + 1,
                        channel,
                    ),
                )
            )

    for target in range(
        n_sources
    ):
        channel = int(
            channels[
                target
            ]
        )

        edges.append(
            (
                channel_node(
                    n_channel_layers - 1,
                    channel,
                ),
                target_start
                + target,
            )
        )

    original_edges = len(
        edges
    )

    if rewired:
        for idx in range(
            n_sources
        ):
            edges.append(
                (
                    idx,
                    target_start
                    + idx,
                )
            )

    edge_index = torch.tensor(
        edges,
        dtype=torch.long,
    ).t().contiguous()

    return {
        "edge_index":
            edge_index,
        "channels":
            channels,
        "n_nodes":
            n_nodes,
        "target_start":
            target_start,
        "original_edges":
            original_edges,
        "added_edges":
            (
                n_sources
                if rewired
                else 0
            ),
        "source_target_distance_original":
            distance,
        "source_target_distance_after":
            (
                1
                if rewired
                else distance
            ),
    }


def sample_bits(
    graph_seed,
    split_id,
    sample_id,
    n_sources,
):
    seed = (
        int(graph_seed)
        * 10_000_019
        + int(split_id)
        * 100_003
        + int(sample_id)
        * 101
    )

    rng = np.random.default_rng(
        seed
    )

    return rng.integers(
        0,
        2,
        size=n_sources,
        dtype=np.int64,
    )


def make_graph(
    topology,
    bits,
    n_sources,
):
    input_dim = (
        2 * n_sources
        + 4
    )

    x = torch.zeros(
        (
            topology[
                "n_nodes"
            ],
            input_dim,
        ),
        dtype=torch.float32,
    )

    bit_feature = (
        2 * n_sources
    )

    source_role = (
        bit_feature
        + 1
    )

    channel_role = (
        bit_feature
        + 2
    )

    target_role = (
        bit_feature
        + 3
    )

    for idx in range(
        n_sources
    ):
        signed_value = (
            1.0
            if bits[
                idx
            ] == 1
            else -1.0
        )

        x[
            idx,
            idx,
        ] = signed_value

        x[
            idx,
            bit_feature,
        ] = signed_value

        x[
            idx,
            source_role,
        ] = 1.0

    channel_first = (
        n_sources
    )

    channel_last = (
        topology[
            "target_start"
        ]
    )

    x[
        channel_first:
        channel_last,
        channel_role,
    ] = 1.0

    for idx in range(
        n_sources
    ):
        node = (
            topology[
                "target_start"
            ]
            + idx
        )

        x[
            node,
            n_sources
            + idx,
        ] = 1.0

        x[
            node,
            target_role,
        ] = 1.0

    y = torch.full(
        (
            topology[
                "n_nodes"
            ],
        ),
        -100,
        dtype=torch.long,
    )

    target_mask = torch.zeros(
        topology[
            "n_nodes"
        ],
        dtype=torch.bool,
    )

    for idx in range(
        n_sources
    ):
        node = (
            topology[
                "target_start"
            ]
            + idx
        )

        y[
            node
        ] = int(
            bits[
                idx
            ]
        )

        target_mask[
            node
        ] = True

    data = Data(
        x=x,
        edge_index=topology[
            "edge_index"
        ],
        y=y,
        target_mask=target_mask,
    )

    data.bits = torch.tensor(
        bits,
        dtype=torch.long,
    )

    return data


def make_dataset(
    topology,
    graph_seed,
    split_id,
    n_graphs,
    n_sources,
):
    graphs = []

    for sample_id in range(
        n_graphs
    ):
        bits = sample_bits(
            graph_seed=graph_seed,
            split_id=split_id,
            sample_id=sample_id,
            n_sources=n_sources,
        )

        graphs.append(
            make_graph(
                topology=topology,
                bits=bits,
                n_sources=n_sources,
            )
        )

    return graphs


class BatchPairNorm(nn.Module):
    def __init__(
        self,
        scale=1.0,
        eps=1e-8,
    ):
        super().__init__()

        self.scale = float(
            scale
        )

        self.eps = float(
            eps
        )

    def forward(
        self,
        x,
        batch,
    ):
        if batch is None:
            centered = (
                x
                - x.mean(
                    dim=0,
                    keepdim=True,
                )
            )

            denom = torch.sqrt(
                centered.pow(
                    2
                )
                .sum(
                    dim=1
                )
                .mean()
                + self.eps
            )

            return (
                self.scale
                * centered
                / denom
            )

        n_graphs = int(
            batch.max().item()
        ) + 1

        counts = torch.bincount(
            batch,
            minlength=n_graphs,
        ).to(
            dtype=x.dtype
        )

        sums = torch.zeros(
            (
                n_graphs,
                x.size(
                    1
                ),
            ),
            device=x.device,
            dtype=x.dtype,
        )

        sums.index_add_(
            0,
            batch,
            x,
        )

        means = (
            sums
            / counts.clamp_min(
                1
            ).unsqueeze(
                1
            )
        )

        centered = (
            x
            - means[
                batch
            ]
        )

        squared_norm = (
            centered.pow(
                2
            )
            .sum(
                dim=1
            )
        )

        graph_sq = torch.zeros(
            n_graphs,
            device=x.device,
            dtype=x.dtype,
        )

        graph_sq.index_add_(
            0,
            batch,
            squared_norm,
        )

        graph_sq = (
            graph_sq
            / counts.clamp_min(
                1
            )
        )

        denom = torch.sqrt(
            graph_sq
            + self.eps
        )

        return (
            self.scale
            * centered
            / denom[
                batch
            ].unsqueeze(
                1
            )
        )


class LongRangeGraphSAGE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        dropout,
        pairnorm,
    ):
        super().__init__()

        self.dropout = float(
            dropout
        )

        self.pairnorm_enabled = bool(
            pairnorm
        )

        self.convs = nn.ModuleList()

        self.convs.append(
            SAGEConv(
                input_dim,
                hidden_dim,
            )
        )

        for _ in range(
            num_layers - 1
        ):
            self.convs.append(
                SAGEConv(
                    hidden_dim,
                    hidden_dim,
                )
            )

        if self.pairnorm_enabled:
            self.norms = nn.ModuleList(
                [
                    BatchPairNorm()
                    for _ in range(
                        num_layers
                    )
                ]
            )
        else:
            self.norms = None

        self.classifier = nn.Linear(
            hidden_dim,
            2,
        )

    def forward(
        self,
        x,
        edge_index,
        batch,
    ):
        for layer_idx, conv in enumerate(
            self.convs
        ):
            x = conv(
                x,
                edge_index,
            )

            if self.pairnorm_enabled:
                x = self.norms[
                    layer_idx
                ](
                    x,
                    batch,
                )

            x = F.relu(
                x
            )

            if self.dropout > 0:
                x = F.dropout(
                    x,
                    p=self.dropout,
                    training=self.training,
                )

        return self.classifier(
            x
        )


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    correct = 0
    total = 0
    loss_sum = 0.0

    for batch in loader:
        batch = batch.to(
            device
        )

        logits = model(
            batch.x,
            batch.edge_index,
            batch.batch,
        )

        mask = batch.target_mask

        loss = F.cross_entropy(
            logits[
                mask
            ],
            batch.y[
                mask
            ],
        )

        pred = logits[
            mask
        ].argmax(
            dim=1
        )

        correct += int(
            (
                pred
                == batch.y[
                    mask
                ]
            )
            .sum()
            .item()
        )

        n = int(
            mask.sum().item()
        )

        total += n

        loss_sum += (
            float(
                loss.item()
            )
            * n
        )

    return {
        "loss":
            loss_sum
            / max(
                total,
                1,
            ),
        "accuracy":
            correct
            / max(
                total,
                1,
            ),
    }


@torch.no_grad()
def finite_difference_sensitivity(
    model,
    dataset,
    topology,
    device,
    n_sources,
    graph_seed,
    n_graphs=8,
    n_targets=8,
):
    model.eval()

    rng = np.random.default_rng(
        int(
            graph_seed
        )
        * 17_171
        + 919
    )

    matched_rows = []
    distractor_rows = []

    max_graphs = min(
        n_graphs,
        len(
            dataset
        ),
    )

    for graph_idx in range(
        max_graphs
    ):
        data = copy.deepcopy(
            dataset[
                graph_idx
            ]
        ).to(
            device
        )

        batch = torch.zeros(
            data.x.size(
                0
            ),
            dtype=torch.long,
            device=device,
        )

        base_logits = model(
            data.x,
            data.edge_index,
            batch,
        )

        targets = rng.choice(
            n_sources,
            size=min(
                n_targets,
                n_sources,
            ),
            replace=False,
        )

        channels = topology[
            "channels"
        ]

        for target_id in targets:
            target_id = int(
                target_id
            )

            target_node = (
                topology[
                    "target_start"
                ]
                + target_id
            )

            matched_x = (
                data.x.clone()
            )

            matched_x[
                target_id,
                target_id,
            ] *= -1.0

            matched_logits = model(
                matched_x,
                data.edge_index,
                batch,
            )

            matched_delta = (
                matched_logits[
                    target_node
                ]
                - base_logits[
                    target_node
                ]
            ).norm(
                p=2
            ) / 2.0

            matched_rows.append(
                float(
                    matched_delta.item()
                )
            )

            same_channel = np.where(
                channels
                == channels[
                    target_id
                ]
            )[0]

            same_channel = same_channel[
                same_channel
                != target_id
            ]

            if len(
                same_channel
            ) > 0:
                distractor_id = int(
                    rng.choice(
                        same_channel
                    )
                )

                distractor_x = (
                    data.x.clone()
                )

                distractor_x[
                    distractor_id,
                    distractor_id,
                ] *= -1.0

                distractor_logits = model(
                    distractor_x,
                    data.edge_index,
                    batch,
                )

                distractor_delta = (
                    distractor_logits[
                        target_node
                    ]
                    - base_logits[
                        target_node
                    ]
                ).norm(
                    p=2
                ) / 2.0

                distractor_rows.append(
                    float(
                        distractor_delta.item()
                    )
                )

    matched = np.asarray(
        matched_rows,
        dtype=float,
    )

    distractor = np.asarray(
        distractor_rows,
        dtype=float,
    )

    matched_mean = (
        float(
            matched.mean()
        )
        if matched.size
        else np.nan
    )

    distractor_mean = (
        float(
            distractor.mean()
        )
        if distractor.size
        else np.nan
    )

    ratio = (
        matched_mean
        / max(
            distractor_mean,
            1e-12,
        )
        if np.isfinite(
            matched_mean
        )
        and np.isfinite(
            distractor_mean
        )
        else np.nan
    )

    return {
        "matched_influence_mean":
            matched_mean,
        "matched_influence_std":
            (
                float(
                    matched.std(
                        ddof=1
                    )
                )
                if matched.size > 1
                else 0.0
            ),
        "distractor_influence_mean":
            distractor_mean,
        "distractor_influence_std":
            (
                float(
                    distractor.std(
                        ddof=1
                    )
                )
                if distractor.size > 1
                else 0.0
            ),
        "matched_to_distractor_ratio":
            float(
                ratio
            ),
        "n_matched_measurements":
            int(
                matched.size
            ),
        "n_distractor_measurements":
            int(
                distractor.size
            ),
    }


def read_config(
    grid,
    task_id,
):
    with open(
        grid,
        newline="",
    ) as f:
        rows = list(
            csv.DictReader(
                f
            )
        )

    if task_id < 0 or task_id >= len(
        rows
    ):
        raise IndexError(
            task_id
        )

    return rows[
        task_id
    ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--grid",
        required=True,
    )

    parser.add_argument(
        "--task-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--out-dir",
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    args = parser.parse_args()

    cfg = read_config(
        args.grid,
        args.task_id,
    )

    model_name = cfg[
        "model"
    ]

    distance = int(
        cfg[
            "distance"
        ]
    )

    width = int(
        cfg[
            "bottleneck_width"
        ]
    )

    graph_seed = int(
        cfg[
            "graph_seed"
        ]
    )

    init_seed = int(
        cfg[
            "init_seed"
        ]
    )

    n_sources = int(
        cfg[
            "n_sources"
        ]
    )

    hidden_dim = int(
        cfg[
            "hidden_dim"
        ]
    )

    num_layers = int(
        cfg[
            "num_layers"
        ]
    )

    train_graphs = int(
        cfg[
            "train_graphs"
        ]
    )

    val_graphs = int(
        cfg[
            "val_graphs"
        ]
    )

    test_graphs = int(
        cfg[
            "test_graphs"
        ]
    )

    batch_size = int(
        cfg[
            "batch_size"
        ]
    )

    epochs = int(
        cfg[
            "epochs"
        ]
    )

    patience = int(
        cfg[
            "patience"
        ]
    )

    lr = float(
        cfg[
            "lr"
        ]
    )

    weight_decay = float(
        cfg[
            "weight_decay"
        ]
    )

    dropout = float(
        cfg[
            "dropout"
        ]
    )

    pairnorm = (
        "PairNorm"
        in model_name
    )

    rewired = (
        "Rewired"
        in model_name
    )

    if num_layers < distance:
        raise RuntimeError(
            f"num_layers={num_layers} "
            f"is smaller than distance={distance}"
        )

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        f"LRB-D{distance}"
        f"-B{width}"
        f"-{model_name}"
        f"-G{graph_seed}"
        f"-I{init_seed}"
    )

    summary_path = (
        out_dir
        / f"{stem}_summary.json"
    )

    history_path = (
        out_dir
        / f"{stem}_history.csv"
    )

    if summary_path.exists():
        print(
            "Already completed:",
            summary_path,
        )
        return

    topology = build_topology(
        distance=distance,
        width=width,
        n_sources=n_sources,
        graph_seed=graph_seed,
        rewired=rewired,
    )

    train_dataset = make_dataset(
        topology=topology,
        graph_seed=graph_seed,
        split_id=1,
        n_graphs=train_graphs,
        n_sources=n_sources,
    )

    val_dataset = make_dataset(
        topology=topology,
        graph_seed=graph_seed,
        split_id=2,
        n_graphs=val_graphs,
        n_sources=n_sources,
    )

    test_dataset = make_dataset(
        topology=topology,
        graph_seed=graph_seed,
        split_id=3,
        n_graphs=test_graphs,
        n_sources=n_sources,
    )

    seed_everything(
        init_seed
    )

    generator = torch.Generator()
    generator.manual_seed(
        init_seed
        + 71
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    if (
        args.device.startswith(
            "cuda"
        )
        and torch.cuda.is_available()
    ):
        device = torch.device(
            args.device
        )
    else:
        device = torch.device(
            "cpu"
        )

    model = LongRangeGraphSAGE(
        input_dim=(
            2 * n_sources
            + 4
        ),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        pairnorm=pairnorm,
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val = -1.0
    best_epoch = -1
    best_state = None
    best_test_at_val = np.nan

    wait = 0

    history = []

    for epoch in range(
        1,
        epochs + 1
    ):
        model.train()

        train_correct = 0
        train_total = 0
        train_loss_sum = 0.0

        for batch in train_loader:
            batch = batch.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                batch.x,
                batch.edge_index,
                batch.batch,
            )

            mask = (
                batch.target_mask
            )

            loss = F.cross_entropy(
                logits[
                    mask
                ],
                batch.y[
                    mask
                ],
            )

            loss.backward()

            optimizer.step()

            pred = logits[
                mask
            ].argmax(
                dim=1
            )

            correct = (
                pred
                == batch.y[
                    mask
                ]
            ).sum().item()

            n = int(
                mask.sum().item()
            )

            train_correct += int(
                correct
            )

            train_total += n

            train_loss_sum += (
                float(
                    loss.item()
                )
                * n
            )

        train_acc = (
            train_correct
            / max(
                train_total,
                1,
            )
        )

        train_loss = (
            train_loss_sum
            / max(
                train_total,
                1,
            )
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
        )

        improved = (
            val_metrics[
                "accuracy"
            ]
            > best_val
            + 1e-12
        )

        if improved:
            best_val = float(
                val_metrics[
                    "accuracy"
                ]
            )

            best_epoch = int(
                epoch
            )

            best_state = {
                key:
                    value.detach()
                    .cpu()
                    .clone()
                for key, value
                in model.state_dict().items()
            }

            test_metrics_now = evaluate(
                model,
                test_loader,
                device,
            )

            best_test_at_val = float(
                test_metrics_now[
                    "accuracy"
                ]
            )

            wait = 0

        else:
            wait += 1

        history.append(
            {
                "epoch":
                    epoch,
                "train_loss":
                    train_loss,
                "train_accuracy":
                    train_acc,
                "val_loss":
                    val_metrics[
                        "loss"
                    ],
                "val_accuracy":
                    val_metrics[
                        "accuracy"
                    ],
                "best_val_accuracy":
                    best_val,
                "best_test_accuracy_at_best_val":
                    best_test_at_val,
            }
        )

        if wait >= patience:
            break

    if best_state is None:
        raise RuntimeError(
            "No best checkpoint found"
        )

    model.load_state_dict(
        best_state
    )

    model = model.to(
        device
    )

    final_val = evaluate(
        model,
        val_loader,
        device,
    )

    final_test = evaluate(
        model,
        test_loader,
        device,
    )

    sensitivity = finite_difference_sensitivity(
        model=model,
        dataset=test_dataset,
        topology=topology,
        device=device,
        n_sources=n_sources,
        graph_seed=graph_seed,
        n_graphs=8,
        n_targets=8,
    )

    channel_counts = np.bincount(
        topology[
            "channels"
        ],
        minlength=width,
    )

    summary = {
        "benchmark":
            "longrange_bottleneck_v2",
        "model":
            model_name,
        "pairnorm":
            pairnorm,
        "rewired":
            rewired,
        "distance":
            distance,
        "bottleneck_width":
            width,
        "graph_seed":
            graph_seed,
        "init_seed":
            init_seed,
        "n_sources":
            n_sources,
        "hidden_dim":
            hidden_dim,
        "num_layers":
            num_layers,
        "train_graphs":
            train_graphs,
        "val_graphs":
            val_graphs,
        "test_graphs":
            test_graphs,
        "batch_size":
            batch_size,
        "epochs_requested":
            epochs,
        "epochs_ran":
            len(
                history
            ),
        "patience":
            patience,
        "lr":
            lr,
        "weight_decay":
            weight_decay,
        "dropout":
            dropout,
        "n_nodes_per_graph":
            topology[
                "n_nodes"
            ],
        "n_edges_original":
            topology[
                "original_edges"
            ],
        "n_edges_after":
            int(
                topology[
                    "edge_index"
                ].size(
                    1
                )
            ),
        "added_rewiring_edges":
            topology[
                "added_edges"
            ],
        "source_target_distance_original":
            topology[
                "source_target_distance_original"
            ],
        "source_target_distance_after":
            topology[
                "source_target_distance_after"
            ],
        "sources_per_channel_min":
            int(
                channel_counts.min()
            ),
        "sources_per_channel_max":
            int(
                channel_counts.max()
            ),
        "best_epoch":
            best_epoch,
        "best_val_accuracy":
            best_val,
        "best_test_accuracy_at_best_val":
            best_test_at_val,
        "checkpoint_val_accuracy":
            final_val[
                "accuracy"
            ],
        "checkpoint_test_accuracy":
            final_test[
                "accuracy"
            ],
        **sensitivity,
    }

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False,
    )

    with open(
        summary_path,
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
