from pathlib import Path
import pandas as pd

candidates = []

for p in Path("configs").glob("*.csv"):
    name = p.name.lower()

    if (
        "roman" in name
        and "phase1" in name
        and "full" in name
    ):
        candidates.append(p)

if not candidates:
    raise RuntimeError(
        "No existing Roman-Empire Phase 1 Full config found."
    )

source = sorted(candidates)[0]
old = pd.read_csv(source)

dataset = str(old.iloc[0]["dataset"])

print("Source config:", source)
print("Dataset identifier:", dataset)

models = [
    "GraphSAGE",
    "GraphSAGEPairNorm",
]

layers = [4, 8]
hidden_channels = 128
seed = 1
splits = range(10)

learning_rates = [
    0.0003,
    0.001,
    0.003,
    0.005,
    0.01,
    0.03,
]

out_dir = Path(
    "configs/roman_empire_fair_lr"
)

out_dir.mkdir(
    parents=True,
    exist_ok=True,
)

for lr in learning_rates:
    rows = []
    experiment_id = 0

    for depth in layers:
        for model in models:
            for split_idx in splits:

                rows.append({
                    "experiment_id":
                        experiment_id,
                    "dataset":
                        dataset,
                    "model":
                        model,
                    "num_layers":
                        depth,
                    "hidden_channels":
                        hidden_channels,
                    "seed":
                        seed,
                    "split_idx":
                        split_idx,
                    "lr":
                        lr,
                })

                experiment_id += 1

    lr_name = str(lr).replace(
        ".",
        "p",
    )

    output = (
        out_dir
        / f"roman_empire_lr_{lr_name}.csv"
    )

    frame = pd.DataFrame(rows)

    frame.to_csv(
        output,
        index=False,
    )

    print(
        output,
        "runs=",
        len(frame),
    )
