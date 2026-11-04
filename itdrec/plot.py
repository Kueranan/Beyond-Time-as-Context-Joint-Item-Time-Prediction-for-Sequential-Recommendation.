"""Plot item loss, time loss, and validation NDCG@10 from saved training histories.

Run: python -m itdrec.plot
Custom runs: python -m itdrec.plot --run-dir outputs/my_run --output-dir outputs/my_plots
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without opening a GUI or requiring a display.
import matplotlib.pyplot as plt


def load_history(run_dir):
    """Read and validate the fields needed for plotting; never change run files."""
    path = Path(run_dir) / "history.json"
    history = json.loads(path.read_text())
    if not isinstance(history, list) or not history:
        raise ValueError(f"{path}: expected a nonempty history list")
    previous_epoch = 0
    for row in history:
        epoch = row["epoch"]
        values = [row["item_loss"], row["time_loss"], row["validation"]["ndcg@10"]]
        if not isinstance(epoch, int) or epoch <= previous_epoch:
            raise ValueError(f"{path}: epochs must be positive and strictly increasing")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}: found a non-finite metric at epoch {epoch}")
        previous_epoch = epoch
    return history


def plot_histories(run_dirs, output_dir):
    """Write one PNG and PDF per metric, with one curve per training run."""
    runs = [(Path(directory), load_history(directory)) for directory in run_dirs]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    panels = [
        ("item_loss", "Training item loss", "Item loss (BCE)"),
        ("time_loss", "Training time loss", "Time loss (before multiplying by alpha)"),
        ("validation_ndcg", "Validation NDCG@10", "NDCG@10"),
    ]
    saved = []
    with plt.style.context("seaborn-v0_8-whitegrid"):
        for key, title, ylabel in panels:
            fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
            for directory, history in runs:
                epochs = [row["epoch"] for row in history]
                values = [row["validation"]["ndcg@10"] if key == "validation_ndcg"
                          else row[key] for row in history]
                label = directory.name
                if key == "validation_ndcg":
                    # Reference selects on six Hit/Recall metrics; legacy selects NDCG.
                    selected = [i for i, row in enumerate(history) if row.get("selected_checkpoint")]
                    best = selected[-1] if selected else max(range(len(values)), key=values.__getitem__)
                    label += f" (checkpoint epoch {epochs[best]}: {values[best]:.4f})"
                line, = ax.plot(epochs, values, linewidth=2, label=label)
                if key == "validation_ndcg":
                    ax.scatter(epochs[best], values[best], marker="*", s=180,
                               color=line.get_color(), edgecolors="black", linewidths=0.6,
                               zorder=5)
            ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
            ax.set_ylim(bottom=0)
            ax.xaxis.get_major_locator().set_params(integer=True)
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            for extension in ("png", "pdf"):
                destination = output_dir / f"{key}.{extension}"
                fig.savefig(destination, dpi=200)
                saved.append(destination)
            plt.close(fig)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, nargs="+",
                        help="Run folders containing history.json; defaults to the three dataset configs")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/plots"))
    args = parser.parse_args()
    run_dirs = args.run_dir
    if run_dirs is None:
        run_dirs = [Path(json.loads(Path(f"configs/{name}.json").read_text())["output_dir"])
                    for name in ("pet", "books", "beauty")]
    for path in plot_histories(run_dirs, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
