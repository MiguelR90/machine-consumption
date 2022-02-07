import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State

import json

import numpy as np

from simulator import generate_batches, simulate, double_priority


def generate_product_sequence(
    sample_size, sequence_type, sequence_batch_size, products, product_mix
):
    if sequence_type == "random":
        denominator = sum(product_mix)
        return np.random.choice(
            products, size=sample_size, p=[t / denominator for t in product_mix]
        ).tolist()
    else:
        return generate_batches(sequence_batch_size, products, product_mix, sample_size)


def format_data(data, products):
    out = []
    for frame in data:
        tmp = {}
        for p in products:
            tmp[p] = [machine[p] for machine in frame.values()]
        out.append(tmp)
    return out


def generate_simulation_data(
    num_of_machines,
    products,
    product_mix,
    flush_lock,
    transition_lock,
    priority_type,
    sample_size,
    sequence_type,
    sequence_batch_size,
):
    # mix = dict(zip(products, product_mix))
    # color_map = dict(zip(products, px.colors.qualitative.Plotly))
    sequence = generate_product_sequence(
        sample_size, sequence_type, sequence_batch_size, products, product_mix
    )

    if priority_type == "double":
        priority_func = double_priority
    else:
        priority_func = double_priority

    simulation = simulate(
        num_of_machines,
        sequence,
        products,
        product_mix,
        flush_lock,
        transition_lock,
        priority_func,
        reverse=True,
    )

    return format_data(simulation["data"], products)


# Application start

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = Dash(__name__)
server = app.server

app.layout = html.Div(
    [
        dcc.Input(id="products-input", type="text", value="A,B,C,D"),
        dcc.Input(id="product_mix-input", type="text", value="10,5,15,8"),
        dcc.Input(id="num_of_machines-input", type="number", value=5),
        dcc.Input(id="sample_size-input", type="number", value=1000),
        html.Button(id="run-button", n_clicks=0, children="Simulate"),
        dcc.Store(id="simulation-data"),
        dcc.Graph(id="graph-with-slider"),
        dcc.Slider(
            0,
            100,
            step=1,
            marks=None,
            id="frame-slider",
        ),
    ]
)


@app.callback(
    Output("simulation-data", "data"),
    Input("run-button", "n_clicks"),
    State("products-input", "value"),
    State("product_mix-input", "value"),
    State("num_of_machines-input", "value"),
    State("sample_size-input", "value"),
)
def clean_data(n_clicks, product_str, product_mix_str, num_of_machines, sample_size):

    products = list(product_str.split(","))
    product_mix = [int(mix) for mix in product_mix_str.split(",")]

    simulation_data = generate_simulation_data(
        num_of_machines,
        products,
        product_mix,
        21,
        0,
        "double",
        sample_size,
        "random",
        40,
    )

    return json.dumps(simulation_data)


@app.callback(
    [
        Output("frame-slider", "max"),
        Output("frame-slider", "value"),
    ],
    Input("simulation-data", "data"),
)
def reset_slider_after_data_update(jsonified_data):
    data = json.loads(jsonified_data)
    return len(data) - 1, 0


@app.callback(
    Output("graph-with-slider", "figure"),
    State("simulation-data", "data"),
    Input("frame-slider", "value"),
)
def update_figure(jsonified_data, selected_frame):
    data = json.loads(jsonified_data)
    frame_data = data[selected_frame]

    fig = go.Figure()
    for prod, counts in frame_data.items():

        fig.add_trace(
            go.Bar(
                x=[f"Machine {i}" for i in range(len(counts))],
                y=counts,
                name=prod,
                texttemplate="%{y}",
                textposition="inside",
            )
        )
    fig.update_layout(yaxis_range=[0, 21], barmode="group")

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
