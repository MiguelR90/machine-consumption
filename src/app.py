from dash import Dash, dcc, html, Input, Output
import plotly.express as px

import numpy as np
import pandas as pd

from simulator import generate_batches, simulate, double_priority

sample_size = 1000
products = ["A", "B", "C", "D"]
totals = [10, 5, 15, 8]
flush_lock = 21
transition_lock = 3

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
    9,
    sequence1,
    products,
    totals,
    flush_lock,
    transition_lock,
    double_priority,
    reverse=True,
)

data = []
for frame, obj in enumerate(simulation[1]):
    for machine, mix in obj.items():
        for product, count in mix.items():
            data.append(
                {
                    "frame": frame,
                    "machine": f"Machine {machine}",
                    "product": product,
                    "count": count,
                }
            )


# Application start

df = pd.DataFrame().from_records(data)

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = Dash(__name__)

window_size = 25

app.layout = html.Div(
    [
        dcc.Graph(id="graph-with-slider"),
        dcc.Slider(0, window_size, value=0, step=1, id="frame-slider"),
        dcc.Slider(0, df["frame"].max(), value=0, step=window_size, id="frame-range"),
    ]
)


@app.callback(
    [
        Output("frame-slider", "min"),
        Output("frame-slider", "max"),
        Output("frame-slider", "value"),
    ],
    Input("frame-range", "value"),
)
def set_frame_slider(start_frame):
    return (start_frame, start_frame + window_size, start_frame)


@app.callback(Output("graph-with-slider", "figure"), Input("frame-slider", "value"))
def update_figure(selected_frame):
    filtered_df = df[df.frame == selected_frame]

    fig = px.bar(
        filtered_df,
        x="machine",
        y="count",
        color="product",
        text_auto=True,
        barmode="group",
    )

    fig.update_layout(yaxis_range=[0, 21])

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
