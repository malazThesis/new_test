from pathlib import Path
import argparse
import copy
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class PairNorm(nn.Module):
    def __init__(
        self,
        scale=1.0,
        eps=1e-6,
    ):
        super().__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, x):
        mean = x.mean(
            dim=0,
            keepdim=True,
        )

        x = x - mean

        row_norm = torch.sqrt(
            (x ** 2).sum(
                dim=1,
                keepdim=True,
            )
            + self.eps
        )

        mean_norm = (
            row_norm.mean()
        )

        return (
            self.scale
            * x
            / (
                mean_norm
                + self.eps
            )
        )


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
        pairnorm,
    ):
        super().__init__()

        assert num_layers >= 2

        self.convs = (
            nn.ModuleList()
        )

        self.pns = (
            nn.ModuleList()
        )

        self.dropout = float(
            dropout
        )

        self.pairnorm = bool(
            pairnorm
        )

        self.convs.append(
            SAGEConv(
                in_channels,
                hidden_channels,
            )
        )

        for _ in range(
            num_layers - 2
        ):
            self.convs.append(
                SAGEConv(
                    hidden_channels,
                    hidden_channels,
                )
            )

        self.convs.append(
            SAGEConv(
                hidden_channels,
                out_channels,
            )
        )

        if self.pairnorm:
            for _ in range(
                num_layers - 1
            ):
                self.pns.append(
                    PairNorm()
                )

    def forward(
        self,
        x,
        edge_index,
    ):
        for i, conv in enumerate(
            self.convs[:-1]
        ):
            x = conv(
                x,
                edge_index,
            )

            if self.pairnorm:
                x = self.pns[i](
                    x
                )

            x = F.relu(x)

            x = F.dropout(
                x,
                p=self.dropout,
                training=self.training,
            )

        return self.convs[-1](
            x,
            edge_index,
        )


@torch.no_grad()
def evaluate(
    model,
    data,
):
    model.eval()

    logits = model(
        data.x,
        data.edge_index,
    )

    pred = logits.argmax(
        dim=1
    )

    result = {}

    for name, mask in [
        ("train", data.train_mask),
        ("val", data.val_mask),
        ("test", data.test_mask),
    ]:
        mask = mask.bool()

        correct = (
            pred[mask]
            == data.y[mask]
        ).sum().item()

        total = int(
            mask.sum()
        )

        result[name] = (
            correct / total
        )

    return result


def train_epoch(
    model,
    data,
    optimizer,
):
    model.train()

    optimizer.zero_grad()

    logits = model(
        data.x,
        data.edge_index,
    )

    loss = F.cross_entropy(
        logits[
            data.train_mask.bool()
        ],
        data.y[
            data.train_mask.bool()
        ],
    )

    loss.backward()

    optimizer.step()

    return float(
        loss.item()
    )


def load_row(
    grid,
    task_id,
):
    with open(
        grid,
        "r",
        encoding="utf-8",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    if (
        task_id < 0
        or task_id >= len(rows)
    ):
        raise IndexError(
            f"task-id {task_id} "
            f"outside 0..{len(rows)-1}"
        )

    return rows[task_id]


def scalar_attr(
    data,
    name,
    default=None,
):
    if not hasattr(
        data,
        name,
    ):
        return default

    value = getattr(
        data,
        name,
    )

    if torch.is_tensor(
        value
    ):
        if value.numel() == 1:
            return value.item()

        return default

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    return value


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--grid",
        required=True,
    )

    parser.add_argument(
        "--task-id",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--out-dir",
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2400,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    args = parser.parse_args()

    cfg = load_row(
        args.grid,
        args.task_id,
    )

    data_path = Path(
        cfg["data_path"]
    )

    dataset = cfg[
        "dataset"
    ]

    variant = cfg[
        "variant"
    ]

    model_name = cfg[
        "model"
    ]

    replicate_seed = int(
        cfg["replicate_seed"]
    )

    init_seed = int(
        cfg["init_seed"]
    )

    num_layers = int(
        cfg["num_layers"]
    )

    hidden_channels = int(
        cfg["hidden_channels"]
    )

    lr = float(
        cfg["lr"]
    )

    weight_decay = float(
        cfg["weight_decay"]
    )

    dropout = float(
        cfg["dropout"]
    )

    pairnorm = (
        model_name
        == "GraphSAGEPairNorm"
    )

    if model_name not in {
        "GraphSAGE",
        "GraphSAGEPairNorm",
    }:
        raise ValueError(
            model_name
        )

    set_seed(
        init_seed
    )

    data = load_pt(
        data_path
    )

    device = torch.device(
        args.device
        if (
            args.device != "cuda"
            or torch.cuda.is_available()
        )
        else "cpu"
    )

    data = data.to(
        device
    )

    in_channels = int(
        data.x.size(1)
    )

    out_channels = int(
        data.y.max().item()
        + 1
    )

    model = GraphSAGE(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        dropout=dropout,
        pairnorm=pairnorm,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    history = []

    best_val = -1.0
    best_test = -1.0
    best_train = -1.0
    best_epoch = -1
    best_state = None

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        loss = train_epoch(
            model,
            data,
            optimizer,
        )

        scores = evaluate(
            model,
            data,
        )

        history.append({
            "epoch":
                epoch,
            "loss":
                loss,
            "train_acc":
                scores["train"],
            "val_acc":
                scores["val"],
            "test_acc":
                scores["test"],
        })

        if (
            scores["val"]
            > best_val
        ):
            best_val = float(
                scores["val"]
            )

            best_test = float(
                scores["test"]
            )

            best_train = float(
                scores["train"]
            )

            best_epoch = int(
                epoch
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

    assert (
        best_state is not None
    )

    model.load_state_dict(
        best_state
    )

    checkpoint_scores = (
        evaluate(
            model,
            data,
        )
    )

    summary = {
        "benchmark":
            "realworld_oversquashing_rewiring_v1",

        "dataset":
            dataset,

        "variant":
            variant,

        "model":
            model_name,

        "pairnorm":
            pairnorm,

        "replicate_seed":
            replicate_seed,

        "init_seed":
            init_seed,

        "data_path":
            str(data_path),

        "num_layers":
            num_layers,

        "hidden_channels":
            hidden_channels,

        "lr":
            lr,

        "weight_decay":
            weight_decay,

        "dropout":
            dropout,

        "epochs_requested":
            int(args.epochs),

        "best_epoch":
            best_epoch,

        "best_train_acc":
            best_train,

        "best_val_acc":
            best_val,

        "best_test_acc_at_best_val":
            best_test,

        "checkpoint_train_acc":
            checkpoint_scores[
                "train"
            ],

        "checkpoint_val_acc":
            checkpoint_scores[
                "val"
            ],

        "checkpoint_test_acc":
            checkpoint_scores[
                "test"
            ],

        "edge_budget":
            scalar_attr(
                data,
                "oversquashing_edge_budget",
            ),

        "lambda2_norm_laplacian":
            scalar_attr(
                data,
                "oversquashing_lambda2_norm_laplacian",
            ),

        "fixed_cut_conductance":
            scalar_attr(
                data,
                "oversquashing_fixed_cut_conductance",
            ),

        "cheeger_conductance":
            scalar_attr(
                data,
                "oversquashing_cheeger_conductance",
            ),

        "edge_homophily":
            scalar_attr(
                data,
                "oversquashing_edge_homophily",
            ),

        "added_edges":
            scalar_attr(
                data,
                "oversquashing_added_edges",
            ),

        "added_edges_cross_original_cut":
            scalar_attr(
                data,
                "oversquashing_added_edges_cross_original_cut",
            ),

        "components":
            scalar_attr(
                data,
                "oversquashing_components",
            ),
    }

    out_dir = Path(
        args.out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        f"{dataset}_"
        f"{variant}_"
        f"{model_name}_"
        f"L{num_layers}_"
        f"H{hidden_channels}_"
        f"rep{replicate_seed}_"
        f"init{init_seed}"
    )

    history_path = (
        out_dir
        / f"{stem}_history.csv"
    )

    summary_path = (
        out_dir
        / f"{stem}_summary.json"
    )

    import pandas as pd

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False,
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
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
