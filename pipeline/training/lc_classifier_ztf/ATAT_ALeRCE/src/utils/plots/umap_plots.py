import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from ..data.AlerceDictionaries import ALERCE_TAXONOMY


def plot_umap(ax, umap_result, numeric_labels, num_classes, title,marker_size = 4):
    colors = ALERCE_TAXONOMY.colors

    for i in range(num_classes):
        class_indices = numeric_labels == i
        markers = [
            "o",
            "o",
            "D",
            "o",
            "*",
            "o",
            "D",
            "D",
            "o",
            "*",
            "D",
            "D",
            "D",
            "D",
            "D",
            "D",
            "*",
            "*",
            "*",
            "*",
            "*",
            "*",
            "<",
            "D",
            "p",
        ]
        # if i in list(transient.values()):
        ax.scatter(
            x=umap_result[class_indices, 0],
            y=umap_result[class_indices, 1],
            s=marker_size,
            marker=markers[i],
            label=list(ALERCE_TAXONOMY.all_classes.keys())[i],
            color=colors[i],
            alpha=0.75,
        )

    ax.set_facecolor("black")
    ax.set_title(title)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    plt.legend()


def big_group_plot_umap(ax, umap_result, numeric_labels, num_classes, title,marker_size):
    colors = ["cyan", "yellow", "magenta"]
    for i in range(num_classes):
        class_indices = numeric_labels == i
        color = (
            colors[0]
            if i in ALERCE_TAXONOMY.transient.values()
            else (colors[1] if i in ALERCE_TAXONOMY.stochastic.values() else colors[2])
        )
        ax.scatter(
            x=umap_result[class_indices, 0],
            y=umap_result[class_indices, 1],
            s=marker_size,
            color=color,
            alpha=0.45,
        )

    ax.set_facecolor("black")
    ax.set_title(title)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    # plt.xlim(-1,1)
    # plt.ylim(-1,1)
    plt.legend(["transient", "stochastic", "periodic"])


def plot_umap_3d(ax, umap_result, numeric_labels, num_classes, title,marker_size = 4):
    colors = ALERCE_TAXONOMY.colors
    markers = [
        "o",
        "o",
        "D",
        "o",
        "*",
        "o",
        "D",
        "D",
        "o",
        "*",
        "D",
        "D",
        "D",
        "D",
        "D",
        "D",
        "*",
        "*",
        "*",
        "*",
        "*",
        "*",
        "<",
        "D",
        "p",
    ]  # Repeat markers as needed

    for i in range(num_classes):
        class_indices = numeric_labels == i
        ax.scatter(
            umap_result[class_indices, 0],
            umap_result[class_indices, 1],
            umap_result[class_indices, 2],
            s=marker_size,
            marker=markers[i],
            label=list(ALERCE_TAXONOMY.all_classes.keys())[i],
            color=colors[i],
            alpha=0.75,
        )

    # ax.set_facecolor('black')
    ax.set_title(title)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_zlabel("UMAP Dimension 3")

    # Add legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Optional: set the viewing angle
    ax.view_init(elev=10, azim=45)


def plot_umap_3d_plotly(umap_result, numeric_labels, num_classes, title, output_path):
    import plotly.graph_objects as go
    import plotly.express as px
 
    # Create group labels
    group_labels = []
    for label in numeric_labels:
        if label in ALERCE_TAXONOMY.transient.values():
            group_labels.append("Transient")
        elif label in ALERCE_TAXONOMY.stochastic.values():
            group_labels.append("Stochastic")
        else:
            group_labels.append("Periodic")
    # Create class name mapping
    class_names = ALERCE_TAXONOMY.values_as_keys()
    # Create group labels
    group_labels = []
    for label in numeric_labels:
        if label in ALERCE_TAXONOMY.transient.values():
            group_labels.append("Transient")
        elif label in ALERCE_TAXONOMY.stochastic.values():
            group_labels.append("Stochastic")
        else:
            group_labels.append("Periodic")

    # Create class labels for hover text
    class_labels = [class_names[label] for label in numeric_labels]

    # Create the figure
    fig = go.Figure()

    # Color mapping
    color_map = {"Transient": "green", "Stochastic": "red", "Periodic": "blue"}

    # Add traces for each group
    for group in ["Transient", "Stochastic", "Periodic"]:
        mask = [label == group for label in group_labels]

        fig.add_trace(
            go.Scatter3d(
                x=umap_result[mask, 0],
                y=umap_result[mask, 1],
                z=umap_result[mask, 2],
                mode="markers",
                name=group,
                marker=dict(size=2, color=color_map[group], opacity=0.7),
                text=[
                    class_labels[i] for i in range(len(mask)) if mask[i]
                ],  # Add class names as hover text
                hoverinfo="text",
            )
        )

    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            zaxis_title="UMAP Dimension 3",
        ),
        width=1920,
        height=1080,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    # Show the plot in browser
    # fig.show()

    # Save the plot as HTML file
    fig.write_html(output_path)
    print(f"Plot saved to {output_path}")


def plot_umap_3d_plotly_individual(
    umap_result, numeric_labels, num_classes, title, output_path
):
    import plotly.graph_objects as go

    # Create class name mapping
    class_names = ALERCE_TAXONOMY.values_as_keys()

    # Create the figure
    fig = go.Figure()

    # Generate a color palette for all classes
    colors = ALERCE_TAXONOMY.colors
    # Add traces for each class
    for class_idx in range(num_classes):
        mask = numeric_labels == class_idx

        fig.add_trace(
            go.Scatter3d(
                x=umap_result[mask, 0],
                y=umap_result[mask, 1],
                z=umap_result[mask, 2],
                mode="markers",
                name=class_names[class_idx],
                marker=dict(
                    size=2, color=colors[class_idx % len(colors)], opacity=0.98
                ),
                text=[class_names[class_idx]],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            zaxis_title="UMAP Dimension 3",
        ),
        width=1920,
        height=1080,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    # Save the plot as HTML file
    fig.write_html(output_path)
    print(f"Plot saved to {output_path}")
