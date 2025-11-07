import torch
import plotly.graph_objects as go

from mango.util.util import to_numpy


def get_plotly_figure_from_step_losses(step_losses: torch.Tensor, plot_name):
    steps = list(range(len(step_losses)))  # Corresponding steps or x-axis values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=to_numpy(step_losses), mode='lines', name='Values'))
    fig.update_layout(title=plot_name,
                      xaxis_title="Steps",
                      yaxis_title=plot_name,
                      template="plotly_white")
    return fig
