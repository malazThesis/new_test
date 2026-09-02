from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATASETS = {
    "Amazon-Ratings": {
        "depth": Path(
            "runs/amazon_ratings_fair_lr_800ep/"
            "amazon_ratings_matched_lr_depth_800ep_summary.csv"
        ),
        "oversmoothing": Path(
            "runs/amazon_ratings_matched_lr_oversmoothing_800ep/"
            "amazon_ratings_oversmoothing_pairnorm_effects_epoch800.csv"
        ),
    },
    "Squirrel": {
        "depth": Path(
            "runs/squirrel_fair_lr_800ep/"
            "squirrel_matched_lr_depth_800ep_summary.csv"
        ),
        "oversmoothing": Path(
            "runs/squirrel_matched_lr_oversmoothing_800ep/"
            "squirrel_oversmoothing_pairnorm_effects_epoch800.csv"
        ),
    },
    "Roman-Empire": {
        "depth": Path(
            "runs/roman_empire_fair_lr_800ep/"
            "roman_empire_matched_lr_depth_800ep_summary.csv"
        ),
        "oversmoothing": Path(
            "runs/roman_empire_matched_lr_oversmoothing_800ep/"
            "roman_empire_oversmoothing_pairnorm_effects_epoch800.csv"
        ),
    },
    "Actor": {
        "depth": Path(
            "runs/actor_fair_lr_800ep/"
            "actor_matched_lr_depth_800ep_summary.csv"
        ),
        "oversmoothing": Path(
            "runs/actor_matched_lr_oversmoothing_800ep/"
            "actor_oversmoothing_pairnorm_effects_epoch800.csv"
        ),
    },
}

OUT_DIR = Path("runs/all_realworld_oversmoothing_selected4")
PLOT_DIR = Path("plots/all_realworld_oversmoothing_selected4")

OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for dataset, paths in DATASETS.items():
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{dataset}: missing {name}: {path}"
            )

    depth = pd.read_csv(paths["depth"])
    oversmoothing = pd.read_csv(paths["oversmoothing"])

    interaction = depth[
        depth["comparison"] == "PairNorm depth interaction"
    ]

    if len(interaction) != 1:
        raise RuntimeError(
            f"{dataset}: expected one depth interaction row, "
            f"found {len(interaction)}"
        )

    interaction = interaction.iloc[0]

    l8 = oversmoothing[
        oversmoothing["num_layers"] == 8
    ]

    nde = l8[
        l8["metric"] == "normalized_dirichlet_energy"
    ]

    rank = l8[
        l8["metric"] == "effective_rank_ratio"
    ]

    pairwise = l8[
        l8["metric"] == "mean_pairwise_cosine_distance"
    ]

    edge = l8[
        l8["metric"] == "mean_edge_cosine_distance"
    ]

    if len(nde) != 1 or len(rank) != 1:
        raise RuntimeError(
            f"{dataset}: missing unique L8 NDE/rank rows"
        )

    nde = nde.iloc[0]
    rank = rank.iloc[0]

    pairwise = (
        pairwise.iloc[0]
        if len(pairwise) == 1
        else None
    )

    edge = (
        edge.iloc[0]
        if len(edge) == 1
        else None
    )

    baseline_loss = float(
        interaction["baseline_mean_depth_loss"]
    )

    pairnorm_loss = float(
        interaction["pairnorm_mean_depth_loss"]
    )

    did_pp = float(
        interaction["mean_difference_percentage_points"]
    )

    relative_reduction = np.nan

    if baseline_loss >= 0.5 / 100:
        relative_reduction = (
            baseline_loss - pairnorm_loss
        ) / baseline_loss

    row = {
        "dataset": dataset,
        "learning_rate": float(
            interaction["learning_rate"]
        ),
        "baseline_depth_loss_pp":
            100.0 * baseline_loss,
        "pairnorm_depth_loss_pp":
            100.0 * pairnorm_loss,
        "depth_interaction_pp":
            did_pp,
        "depth_interaction_positive_pairs":
            int(interaction["positive_pairs"]),
        "depth_interaction_negative_pairs":
            int(interaction["negative_pairs"]),
        "depth_interaction_exact_holm_p":
            float(
                interaction[
                    "exact_sign_flip_holm_pvalue"
                ]
            ),
        "depth_interaction_cohen_dz":
            float(interaction["cohen_dz"]),
        "relative_depth_loss_reduction":
            relative_reduction,
        "l8_nde_baseline":
            float(nde["baseline_mean"]),
        "l8_nde_pairnorm":
            float(nde["pairnorm_mean"]),
        "l8_nde_ratio":
            float(nde["pairnorm_mean"])
            / float(nde["baseline_mean"]),
        "l8_nde_positive_pairs":
            int(nde["positive_pairs"]),
        "l8_nde_exact_holm_p":
            float(
                nde[
                    "exact_sign_flip_holm_pvalue"
                ]
            ),
        "l8_rank_ratio_baseline":
            float(rank["baseline_mean"]),
        "l8_rank_ratio_pairnorm":
            float(rank["pairnorm_mean"]),
        "l8_rank_ratio_ratio":
            float(rank["pairnorm_mean"])
            / float(rank["baseline_mean"]),
        "l8_rank_positive_pairs":
            int(rank["positive_pairs"]),
        "l8_rank_exact_holm_p":
            float(
                rank[
                    "exact_sign_flip_holm_pvalue"
                ]
            ),
    }

    if pairwise is not None:
        row.update({
            "l8_pairwise_baseline":
                float(pairwise["baseline_mean"]),
            "l8_pairwise_pairnorm":
                float(pairwise["pairnorm_mean"]),
            "l8_pairwise_ratio":
                float(pairwise["pairnorm_mean"])
                / float(pairwise["baseline_mean"]),
        })

    if edge is not None:
        row.update({
            "l8_edge_baseline":
                float(edge["baseline_mean"]),
            "l8_edge_pairnorm":
                float(edge["pairnorm_mean"]),
            "l8_edge_ratio":
                float(edge["pairnorm_mean"])
                / float(edge["baseline_mean"]),
        })

    rows.append(row)

summary = pd.DataFrame(rows)

summary.to_csv(
    OUT_DIR / "selected4_oversmoothing_summary.csv",
    index=False,
)

print("\n=== CROSS-DATASET SUMMARY ===")
print(summary.to_string(index=False))

labels = summary["dataset"].tolist()
x = np.arange(len(labels))
width = 0.36

fig, ax = plt.subplots(figsize=(11, 6))

ax.bar(
    x - width / 2,
    summary["baseline_depth_loss_pp"],
    width,
    label="GraphSAGE",
)

ax.bar(
    x + width / 2,
    summary["pairnorm_depth_loss_pp"],
    width,
    label="GraphSAGEPairNorm",
)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15)
ax.set_ylabel("L4 to L8 test accuracy loss (pp)")
ax.set_title("Depth degradation at matched learning rate")
ax.legend()
ax.axhline(0, linewidth=1)
fig.tight_layout()

path = PLOT_DIR / "selected4_depth_loss.png"
fig.savefig(path, dpi=200)
plt.close(fig)
print("Saved:", path)

fig, ax = plt.subplots(figsize=(11, 6))

ax.bar(
    labels,
    summary["depth_interaction_pp"],
)

ax.axhline(0, linewidth=1)

ax.set_ylabel(
    "PairNorm depth interaction (pp)"
)
ax.set_title(
    "PairNorm effect on L4-to-L8 depth degradation"
)

plt.xticks(rotation=15)
fig.tight_layout()

path = (
    PLOT_DIR
    / "selected4_depth_interaction.png"
)

fig.savefig(path, dpi=200)
plt.close(fig)
print("Saved:", path)

fig, ax = plt.subplots(figsize=(11, 6))

ax.bar(
    x - width / 2,
    summary["l8_nde_ratio"],
    width,
    label="Normalized Dirichlet Energy",
)

ax.bar(
    x + width / 2,
    summary["l8_rank_ratio_ratio"],
    width,
    label="Effective rank ratio",
)

ax.axhline(1, linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15)

ax.set_ylabel(
    "PairNorm / GraphSAGE ratio at L8"
)

ax.set_title(
    "L8 representation metrics at epoch 800"
)

ax.legend()
fig.tight_layout()

path = (
    PLOT_DIR
    / "selected4_l8_representation_ratios.png"
)

fig.savefig(path, dpi=200)
plt.close(fig)
print("Saved:", path)

fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(
    summary["l8_nde_ratio"],
    summary["depth_interaction_pp"],
    s=90,
)

for _, row in summary.iterrows():
    ax.annotate(
        row["dataset"],
        (
            row["l8_nde_ratio"],
            row["depth_interaction_pp"],
        ),
        xytext=(5, 5),
        textcoords="offset points",
    )

ax.axhline(0, linewidth=1)
ax.axvline(1, linewidth=1)

ax.set_xlabel(
    "L8 normalized Dirichlet Energy ratio "
    "(PairNorm / GraphSAGE)"
)

ax.set_ylabel(
    "Performance depth interaction (pp)"
)

ax.set_title(
    "Representation preservation vs. depth performance"
)

fig.tight_layout()

path = (
    PLOT_DIR
    / "selected4_performance_vs_nde.png"
)

fig.savefig(path, dpi=200)
plt.close(fig)
print("Saved:", path)

fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(
    summary["l8_rank_ratio_ratio"],
    summary["depth_interaction_pp"],
    s=90,
)

for _, row in summary.iterrows():
    ax.annotate(
        row["dataset"],
        (
            row["l8_rank_ratio_ratio"],
            row["depth_interaction_pp"],
        ),
        xytext=(5, 5),
        textcoords="offset points",
    )

ax.axhline(0, linewidth=1)
ax.axvline(1, linewidth=1)

ax.set_xlabel(
    "L8 effective-rank-ratio ratio "
    "(PairNorm / GraphSAGE)"
)

ax.set_ylabel(
    "Performance depth interaction (pp)"
)

ax.set_title(
    "Effective rank preservation vs. depth performance"
)

fig.tight_layout()

path = (
    PLOT_DIR
    / "selected4_performance_vs_rank.png"
)

fig.savefig(path, dpi=200)
plt.close(fig)
print("Saved:", path)
