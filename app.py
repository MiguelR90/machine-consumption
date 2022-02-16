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
        dcc.Store(id="graph-with-slider-store"),
        dcc.Slider(
            0,
            100,
            step=1,
            marks=None,
            id="frame-slider",
        ),
        html.Hr(),
        html.Details(
            [
                html.Summary("Contents of figure storage"),
                dcc.Markdown(id="clientside-figure-json"),
            ]
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
    Output("frame-slider", "max"),
    Output("frame-slider", "value"),
    Input("simulation-data", "data"),
)
def reset_slider_after_data_update(jsonified_data):
    data = json.loads(jsonified_data)
    return len(data) - 1, 0


@app.callback(
    Output("graph-with-slider-store", "data"),
    Input("simulation-data", "data"),
)
def update_figure(jsonified_data):
    data = json.loads(jsonified_data)[0]

    fig = go.Figure()
    for prod, counts in data.items():

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


app.clientside_callback(
    """
    function(figure, frame, data_str) {

        if(figure === undefined) {
            return {'data': [], 'layout': {}};
        }

        let data = JSON.parse(data_str)[frame];

        let fig = Object.assign({}, figure);

        for (let i = 0; i < fig.data.length; i++) {
            let product = fig.data[i].name;
            fig.data[i].y = data[product];
        }

        return fig;
    }
    """,
    Output("graph-with-slider", "figure"),
    Input("graph-with-slider-store", "data"),
    Input("frame-slider", "value"),
    State("simulation-data", "data"),
)


@app.callback(
    Output("clientside-figure-json", "children"),
    Input("graph-with-slider", "figure"),
)
def generated_px_figure_json(data):
    return "```\n" + json.dumps(data, indent=2) + "\n```"


if __name__ == "__main__":
    app.run_server(debug=True)
