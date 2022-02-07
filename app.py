import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

import numpy as np
import pandas as pd
from flask_caching import Cache

from simulator import generate_batches, simulate, double_priority

sample_size = 1000
products = ["A", "B", "C", "D"]
totals = [10, 5, 15, 8]
flush_lock = 21
transition_lock = 3

mix = dict(zip(products, totals))

color_map = dict(zip(products + ["full"], px.colors.qualitative.Plotly))

# sequence 1
denominator = sum(totals)
sequence1 = np.random.choice(
    products, size=sample_size, p=[t / denominator for t in totals]
).tolist()

# sequence 2
sequence2 = generate_batches(20, products, totals, sample_size)

# sequence 3
sequence3 = generate_batches(40, products, totals, sample_size)


simulation = simulate(
    5,
    sequence1,
    products,
    totals,
    flush_lock,
    transition_lock,
    double_priority,
    reverse=True,
)


def format_data(data):
    out = []
    for f, frame in enumerate(data):
        tmp = {}
        for p in products:
            tmp[p] = [machine[p] for machine in frame.values()]
        out.append(tmp)
    return out


data = format_data(simulation["data"])


# Application start

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = Dash(__name__)
server = app.server
cache = Cache(
    app.server, config={"CACHE_TYPE": "filesystem", "CACHE_DIR": "cache-directory"}
)
app.config.suppress_callback_exceptions = True
timeout = 20


window_size = 1

app.layout = html.Div(
    [
        dcc.Graph(id="graph-with-slider"),
        dcc.Graph(id="graph2-with-slider"),
        dcc.Slider(
            0,
            len(data),
            value=0,
            step=window_size,
            marks=None,
            id="frame-slider",
        ),
    ]
)


@app.callback(
    [Output("graph-with-slider", "figure"), Output("graph2-with-slider", "figure")],
    Input("frame-slider", "value"),
)
@cache.memoize(timeout=timeout)
def update_figure(selected_frame):
    frame_data = data[selected_frame]

    fig = go.Figure()
    for prod, counts in frame_data.items():
        colors = []
        for c in counts:
            if c == mix[prod]:
                colors.append(color_map["full"])
            else:
                colors.append(color_map[prod])

        fig.add_trace(
            go.Bar(
                x=[f"Machine {i}" for i in range(len(counts))],
                y=counts,
                name=prod,
                marker_color=colors,
                texttemplate="%{y}",
                textposition="inside",
            )
        )
    fig.update_layout(yaxis_range=[0, 21], barmode="group")

    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(
            x=["consumed", "trashed"],
            y=[
                simulation["consumption_data"][selected_frame],
                simulation["trash_data"][selected_frame],
            ],
            marker_color=["blue", "red"],
            texttemplate="%{y}",
            textposition="inside",
        )
    )
    fig2.update_layout(
        yaxis_range=[
            0,
            max(simulation["consumption_data"][-1], simulation["trash_data"][-1]),
        ],
        barmode="stack",
    )

    return fig, fig2


if __name__ == "__main__":
    app.run_server(debug=True)
