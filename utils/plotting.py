"""Plotting functions for compare_runs notebook."""

import json

import matplotlib.colors as mcolors
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_all_metrics(metrics_dir, pattern="*_last.json"):
    all_metrics = []
    for f in metrics_dir.glob(pattern):
        with open(f, "r") as file:
            data = json.load(file)
            data["run_name"] = f.stem
            for key in ["train_loss", "val_loss", "train_accuracy", "val_accuracy"]:
                if isinstance(data.get(key), (int, float)):
                    data[key] = [data[key]]
            all_metrics.append(data)
    return all_metrics


def adjust_color_shade(color, factor=0.5):
    """Darken/lighten a color by factor. factor <1 → lighter, factor>1 → darker"""
    rgb = mcolors.to_rgb(color)
    rgb_adj = [min(max(c * factor, 0), 1) for c in rgb]
    return (
        f"rgb({int(rgb_adj[0] * 255)},{int(rgb_adj[1] * 255)},{int(rgb_adj[2] * 255)})"
    )


def plot_metrics_grid(all_metrics):
    if not all_metrics:
        print("No metrics found!")
        return

    # Assign a base color per run
    colors = px.colors.qualitative.Plotly
    color_map = {
        m["run_name"]: colors[i % len(colors)] for i, m in enumerate(all_metrics)
    }

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, subplot_titles=("Loss", "Accuracy")
    )

    trace_info = []  # store run_name and curve_type for each trace
    for m in all_metrics:
        base_color = color_map[m["run_name"]]
        train_color = base_color
        val_color = adjust_color_shade(base_color, factor=1.5)

        # Loss curves
        fig.add_trace(
            go.Scatter(
                y=m["train_loss"],
                mode="lines+markers",
                name=f"{m['run_name']} - train",
                line=dict(color=train_color),
            ),
            row=1,
            col=1,
        )
        trace_info.append({"run": m["run_name"], "type": "train"})

        fig.add_trace(
            go.Scatter(
                y=m["val_loss"],
                mode="lines+markers",
                name=f"{m['run_name']} - val",
                line=dict(color=val_color, dash="dash"),
            ),
            row=1,
            col=1,
        )
        trace_info.append({"run": m["run_name"], "type": "val"})

        # Accuracy curves
        fig.add_trace(
            go.Scatter(
                y=m["train_accuracy"],
                mode="lines+markers",
                name=f"{m['run_name']} - train",
                line=dict(color=train_color),
            ),
            row=2,
            col=1,
        )
        trace_info.append({"run": m["run_name"], "type": "train"})

        fig.add_trace(
            go.Scatter(
                y=m["val_accuracy"],
                mode="lines+markers",
                name=f"{m['run_name']} - val",
                line=dict(color=val_color, dash="dash"),
            ),
            row=2,
            col=1,
        )
        trace_info.append({"run": m["run_name"], "type": "val"})

    # Buttons for train/val visibility
    def get_visibility(train=True, val=True, selected_run=None):
        vis = []
        for i, info in enumerate(trace_info):
            run_match = selected_run is None or info["run"] == selected_run
            type_match = (info["type"] == "train" and train) or (
                info["type"] == "val" and val
            )
            vis.append(run_match and type_match)
        return vis

    buttons_train_val = [
        dict(
            label="All", method="update", args=[{"visible": get_visibility(True, True)}]
        ),
        dict(
            label="Train only",
            method="update",
            args=[{"visible": get_visibility(True, False)}],
        ),
        dict(
            label="Val only",
            method="update",
            args=[{"visible": get_visibility(False, True)}],
        ),
    ]

    buttons_runs = []
    run_names = [m["run_name"] for m in all_metrics]
    buttons_runs.append(
        dict(
            label="All runs",
            method="update",
            args=[{"visible": get_visibility(True, True, None)}],
        )
    )
    for run_name in run_names:
        buttons_runs.append(
            dict(
                label=run_name,
                method="update",
                args=[{"visible": get_visibility(True, True, run_name)}],
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons_train_val,
                direction="left",
                x=1.0,
                xanchor="left",
                y=1.15,
                yanchor="top",
                showactive=True,
                bgcolor="lightgrey",
                bordercolor="black",
            ),
            dict(
                buttons=buttons_runs,
                direction="down",
                x=1.0,
                xanchor="left",
                y=1.1,
                yanchor="top",
                showactive=True,
                bgcolor="lightgrey",
                bordercolor="black",
            ),
        ],
        height=700,
        width=1000,
        title_text="Training Metrics Comparison Across Runs",
        showlegend=True,
        margin=dict(r=200),  # give space on right for dropdowns
    )

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=2, col=1)

    fig.show()
