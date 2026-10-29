"""Regenerate images/training_curves.png from the epoch logs saved in VIT_Paper.ipynb."""
import json
import re

import matplotlib.pyplot as plt

LOG_PATTERN = r"train_loss: ([\d.]+) \| train_acc: ([\d.]+) \| test_loss: ([\d.]+) \| test_acc: ([\d.]+)"


def training_logs(nb, marker):
    """Return per-epoch metrics from the first code cell whose source contains `marker`."""
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and marker in "".join(cell["source"]):
            text = "".join("".join(o.get("text", [])) for o in cell.get("outputs", []))
            logs = [tuple(map(float, m)) for m in re.findall(LOG_PATTERN, text)]
            if logs:
                return logs
    raise ValueError(f"No training logs found for {marker!r}")


nb = json.load(open("VIT_Paper.ipynb", encoding="utf8"))
runs = {
    "ViT from scratch": training_logs(nb, "vit_results = engine.train"),
    "Pretrained ViT-B/16": training_logs(nb, "pretrained_vit_results = engine.train"),
}

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for (name, logs), color in zip(runs.items(), ["#d62728", "#1f77b4"]):
    epochs = range(1, len(logs) + 1)
    ax[0].plot(epochs, [x[0] for x in logs], "--", color=color, label=f"{name} (train)")
    ax[0].plot(epochs, [x[2] for x in logs], "-o", color=color, label=f"{name} (test)")
    ax[1].plot(epochs, [x[1] * 100 for x in logs], "--", color=color)
    ax[1].plot(epochs, [x[3] * 100 for x in logs], "-o", color=color)

ax[1].axhline(100 / 3, color="gray", ls=":")
ax[0].plot([], [], ":", color="gray", label="Chance level (33.3%)")
ax[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
ax[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)", ylim=(0, 100))
for a in ax:
    a.grid(alpha=0.3)
    a.set_xticks(range(1, 11))
fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=9, frameon=False)
fig.suptitle("Training dynamics: from scratch vs. transfer learning (pizza/steak/sushi)", fontweight="bold")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("images/training_curves.png", dpi=150)
