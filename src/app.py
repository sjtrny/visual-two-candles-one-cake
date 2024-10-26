import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, callback, ctx, dcc, html
from dash.exceptions import PreventUpdate
from scipy.stats import beta, uniform

num_points_restricted = 50000
num_points_full = 20000

distribution_defaults = {
    "Uniform": {"class": uniform, "parameters": {}},
    "Beta": {
        "class": beta,
        "parameters": {
            "a": {"min": 0, "max": 10, "step": 1, "value": 2},
            "b": {"min": 0, "max": 10, "step": 1, "value": 2},
        },
    },
}

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Two Candles, One Cake - Visualised"
server = app.server

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader([
                                    "Joint Density",
                                    dcc.Checklist(
                                        options=["Restrict to Integration Region"],
                                        value=["Restrict to Integration Region"],
                                        inline=True,
                                        id="integration-region",
                                    )
                                ], className='d-flex justify-content-between'),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            id="joint-plot",
                                            config={"displayModeBar": False},
                                        )
                                    ]
                                ),
                            ]
                        )
                    ],
                    lg=6, className="mt-3",
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Marginal Densities"),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            id="marginal-plot",
                                            config={"displayModeBar": False},
                                        )
                                    ]
                                ),
                            ]
                        )
                    ],
                    lg=6, className="mt-3",
                ),
            ],

        ),
        dbc.Row(html.Hr(className='mt-3 mb-0')),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Knife Settings"),
                            dbc.CardBody(
                                [
                                    dbc.Form(
                                        [
                                            dbc.Label(
                                                "Support",
                                            ),
                                            dcc.RangeSlider(
                                                min=0,
                                                max=10,
                                                step=1,
                                                value=[0, 10],
                                                pushable=1,
                                                marks=dict(
                                                    zip(
                                                        range(11),
                                                        [
                                                            f"{i / 10:0.1g}"
                                                            for i in range(11)
                                                        ],
                                                    )
                                                ),
                                                id="distribution1-range",
                                                className="mb-2",
                                            ),
                                            dbc.Label(
                                                "Distribution",
                                                html_for="distribution1-dropdown",
                                            ),
                                            dcc.Dropdown(
                                                options=list(
                                                    distribution_defaults.keys()
                                                ),
                                                value="Uniform",
                                                id="distribution1-dropdown",
                                                className="mb-2",
                                            ),
                                            html.Div(id="distribution1-dynamic"),
                                        ]
                                    )
                                ]
                            ),
                        ]
                    ),
                    lg=4, className="mt-3",
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Candle 1 Settings"),
                            dbc.CardBody(
                                [
                                    dbc.Form(
                                        [
                                            dbc.Label(
                                                "Support",
                                            ),
                                            dcc.RangeSlider(
                                                min=0,
                                                max=10,
                                                step=1,
                                                value=[0, 10],
                                                pushable=1,
                                                marks=dict(
                                                    zip(
                                                        range(11),
                                                        [
                                                            f"{i / 10:0.1g}"
                                                            for i in range(11)
                                                        ],
                                                    )
                                                ),
                                                id="distribution2-range",
                                                className="mb-2",
                                            ),
                                            dbc.Label(
                                                "Distribution",
                                                html_for="distribution2-dropdown",
                                            ),
                                            dcc.Dropdown(
                                                options=list(
                                                    distribution_defaults.keys()
                                                ),
                                                value="Uniform",
                                                id="distribution2-dropdown",
                                                className="mb-2",
                                            ),
                                            html.Div(id="distribution2-dynamic"),
                                        ]
                                    )
                                ]
                            ),
                        ]
                    ),
                    lg=4, className="mt-3",
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Candle 2 Settings"),
                            dbc.CardBody(
                                [
                                    dbc.Form(
                                        [
                                            dbc.Label(
                                                "Support",
                                            ),
                                            dcc.RangeSlider(
                                                min=0,
                                                max=10,
                                                step=1,
                                                value=[0, 10],
                                                pushable=1,
                                                marks=dict(
                                                    zip(
                                                        range(11),
                                                        [
                                                            f"{i / 10:0.1g}"
                                                            for i in range(11)
                                                        ],
                                                    )
                                                ),
                                                id="distribution3-range",
                                                className="mb-2",
                                            ),
                                            dbc.Label(
                                                "Distribution",
                                                html_for="distribution3-dropdown",
                                            ),
                                            dcc.Dropdown(
                                                options=list(
                                                    distribution_defaults.keys()
                                                ),
                                                value="Uniform",
                                                id="distribution3-dropdown",
                                                className="mb-2",
                                            ),
                                            html.Div(id="distribution3-dynamic"),
                                        ]
                                    )
                                ]
                            ),
                        ]
                    ),
                    lg=4, className="mt-3",
                ),
            ]
        ),
    ]
)


@callback(
    Output("distribution1-dynamic", "children"),
    Input("distribution1-dropdown", "value"),
)
def update_distribution(distribution1_dropdown):
    param_dict = distribution_defaults[distribution1_dropdown]["parameters"]

    children = []
    for param_name, param_defaults in param_dict.items():
        children.append(dbc.Label(param_name))
        children.append(
            dcc.Slider(
                id={"type": "distribution1-param", "index": param_name},
                className="mb-2",
                **param_defaults,
            )
        )

    return children


@callback(
    Output("distribution2-dynamic", "children"),
    Input("distribution2-dropdown", "value"),
)
def update_distribution(distribution2_dropdown):
    param_dict = distribution_defaults[distribution2_dropdown]["parameters"]

    children = []
    for param_name, param_defaults in param_dict.items():
        children.append(dbc.Label(param_name))
        children.append(
            dcc.Slider(
                id={"type": "distribution2-param", "index": param_name},
                className="mb-2",
                **param_defaults,
            )
        )

    return children


@callback(
    Output("distribution3-dynamic", "children"),
    Input("distribution3-dropdown", "value"),
)
def update_distribution(distribution3_dropdown):
    param_dict = distribution_defaults[distribution3_dropdown]["parameters"]

    children = []
    for param_name, param_defaults in param_dict.items():
        children.append(dbc.Label(param_name))
        children.append(
            dcc.Slider(
                id={"type": "distribution3-param", "index": param_name},
                className="mb-2",
                **param_defaults,
            )
        )

    return children


@callback(
    Output("joint-plot", "figure"),
    Output("marginal-plot", "figure"),
    inputs=dict(
        integration_region=Input("integration-region", "value"),
        dist1=dict(
            dropdown=Input("distribution1-dropdown", "value"),
            range=Input("distribution1-range", "value"),
            params=Input({"type": "distribution1-param", "index": ALL}, "value"),
        ),
        dist2=dict(
            dropdown=Input("distribution2-dropdown", "value"),
            range=Input("distribution2-range", "value"),
            params=Input({"type": "distribution2-param", "index": ALL}, "value"),
        ),
        dist3=dict(
            dropdown=Input("distribution3-dropdown", "value"),
            range=Input("distribution3-range", "value"),
            params=Input({"type": "distribution3-param", "index": ALL}, "value"),
        ),
    ),
)
def update_graph(integration_region, dist1, dist2, dist3):
    if (
        ctx.args_grouping.dist1["dropdown"]["triggered"] == True
        and len(ctx.args_grouping.dist1["params"]) == 0
    ):
        raise PreventUpdate
    x_dist = distribution_defaults[dist1["dropdown"]]["class"](
        loc=dist1["range"][0] / 10,
        scale=dist1["range"][1] / 10 - dist1["range"][0] / 10,
        **{
            param["id"]["index"]: param["value"]
            for param in ctx.args_grouping.dist1["params"]
        },
    )


    if (
        ctx.args_grouping.dist2["dropdown"]["triggered"] == True
        and len(ctx.args_grouping.dist2["params"]) == 0
    ):
        raise PreventUpdate
    y_dist = distribution_defaults[dist2["dropdown"]]["class"](
        loc=dist2["range"][0] / 10,
        scale=dist2["range"][1] / 10 - dist2["range"][0] / 10,
        **{
            param["id"]["index"]: param["value"]
            for param in ctx.args_grouping.dist2["params"]
        },
    )


    if (
        ctx.args_grouping.dist3["dropdown"]["triggered"] == True
        and len(ctx.args_grouping.dist3["params"]) == 0
    ):
        raise PreventUpdate
    z_dist = distribution_defaults[dist3["dropdown"]]["class"](
        loc=dist3["range"][0] / 10,
        scale=dist3["range"][1] / 10 - dist3["range"][0] / 10,
        **{
            param["id"]["index"]: param["value"]
            for param in ctx.args_grouping.dist3["params"]
        },
    )

    if integration_region:
        x = x_dist.rvs(size=num_points_restricted)
        y = y_dist.rvs(size=num_points_restricted)
        z = z_dist.rvs(size=num_points_restricted)
        condition = ((x > y) & (x < z)) | ((x < y) & (x > z))

        x_region = x[condition]
        y_region = y[condition]
        z_region = z[condition]
    else:
        x_region = x_dist.rvs(size=num_points_full)
        y_region = y_dist.rvs(size=num_points_full)
        z_region = z_dist.rvs(size=num_points_full)

    joint_plot = go.Figure(
        data=[
            go.Scatter3d(
                x=x_region,
                y=y_region,
                z=z_region,
                mode="markers",
                marker=dict(size=2, color="blue", opacity=0.2),
            )
        ]
    )

    joint_plot.update_layout(
        hovermode=False,
        scene=dict(
            aspectmode="cube",
            xaxis_title="Knife",
            yaxis_title="Candle 1",
            zaxis_title="Candle 2",
            xaxis=dict(range=[0, 1], showspikes=False),
            yaxis=dict(range=[0, 1], showspikes=False),
            zaxis=dict(range=[0, 1], showspikes=False),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=400,
    )

    x_range = np.linspace(dist1["range"][0] / 10, dist1["range"][1] / 10, num=100, endpoint=True)
    y_range = np.linspace(dist2["range"][0] / 10, dist2["range"][1] / 10, num=100, endpoint=True)
    z_range = np.linspace(dist3["range"][0] / 10, dist3["range"][1] / 10, num=100, endpoint=True)

    margin_plot = go.Figure(
        data=[
            go.Scatter(
                x=np.concatenate(
                    ([dist3["range"][0] / 10], z_range, [dist3["range"][1] / 10])
                ),
                y=np.concatenate(([0], z_dist.pdf(z_range), [0])),
                line=dict(dash="dash"),
                name="Candle 2",
            ),
            go.Scatter(
                x=np.concatenate(
                    ([dist2["range"][0] / 10], y_range, [dist2["range"][1] / 10])
                ),
                y=np.concatenate(([0], y_dist.pdf(y_range), [0])),
                line=dict(dash="longdash"),
                name="Candle 1",
            ),
            go.Scatter(
                x=np.concatenate(
                    ([dist1["range"][0] / 10], x_range, [dist1["range"][1] / 10])
                ),
                y=np.concatenate(([0], x_dist.pdf(x_range), [0])),
                line=dict(dash="dot"),
                name="Knife",
            ),
        ]
    )

    margin_plot.update_layout(
        hovermode=False,
        xaxis=dict(range=[-0.2, 1.2]),
        margin=dict(l=0, r=0, t=0, b=0),
        height=400,
    )

    return joint_plot, margin_plot


if __name__ == "__main__":
    app.run(debug=True)
