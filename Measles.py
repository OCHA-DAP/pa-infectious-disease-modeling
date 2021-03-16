# UN OCHA COVID-19 INTERNATIONAL MODELING - Task 6
# 2020 The Johns Hopkins University Applied Physics Laboratory LLC

# Controlling measles using supplemental immunization activities: a mathematical model to inform optimal policy
# https://pubmed.ncbi.nlm.nih.gov/25541214/
# https://www.sciencedirect.com/science/article/pii/S0264410X14016077
# This implementation uses a modified version of the system of equations in Section 1 of Appendix A.

import numpy as np
import pandas as pd
import os
import argparse

from scipy.integrate import solve_ivp

import plotly.express as px
import plotly.io as pio
import plotly

# plotly.io.orca.config.executable = "~/anaconda3/bin/orca"

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

DEFAULT_COLORS = px.colors.qualitative.T10
COLORS = [
    DEFAULT_COLORS[1],
    DEFAULT_COLORS[2],
    DEFAULT_COLORS[3],
    DEFAULT_COLORS[5],
    DEFAULT_COLORS[7],
    DEFAULT_COLORS[4],
    DEFAULT_COLORS[8],
    DEFAULT_COLORS[6],
    DEFAULT_COLORS[0],
]
COLUMN_NAMES = [
    "Susceptible Adult",
    "Infected Adult",
    "Recovered Adult",
    "Susceptible Child",
    "Infected Child",
    "Recovered Child",
    "Vaccinated Susceptible",
    "Vaccinated Infected",
    "Vaccinated Recovered",
]


# currently a global variable, because solve_ivp() doesn't accept custom parameters in its function signature
PARAMS = {}


# the paper does not describe how to calculate the basic reproduction number
def calculate_R0(params):
    return "n/a"


# calculate the number of Susceptible adults wrt time
def dSa_dt(params, Sa, Ia, Ic, VI):
    return (
        params["lambda"] * (Ia + Ic + VI) * Sa
        - (params["mu"] * Sa)
        - (params["kappa1"] * Sa)
    )


# calculate the number of Infected adults wrt time
def dIa_dt(params, Sa, Ia, Ic, VI):
    return (
        params["lambda"] * (Ia + Ic + VI) * Sa
        - (params["mu"] * Ia)
        - (params["kappa1"] * Ia)
        - (params["delta"] * Ia)
    )


# calculate the number of Recovered adults wrt time
def dRa_dt(params, Ia, Ra):
    return (params["delta"] * Ia) - (params["mu"] * Ra) - (params["kappa1"] * Ra)


# calculate the number of Susceptible children wrt time
def dSc_dt(params, Sc, Ia, Ic, VI):
    return (
        params["nu"]
        - params["lambda"] * (Ia + Ic + VI) * Sc
        - (params["mu"] * Sc)
        - (params["kappa1"] * Sc)
    )


# calculate the number of Infected children wrt time
def dIc_dt(params, Sc, Ia, Ic, VI):
    return (
        params["lambda"] * (Ia + Ic + VI) * Sc
        - (params["mu"] * Ic)
        - (params["delta"] * Ic)
        - (params["kappa1"] * Ic)
    )


# calculate the number of Recovered children wrt time
def dRc_dt(params, Ic, Rc):
    return (params["delta"] * Ic) - (params["mu"] * Rc) - (params["kappa1"] * Rc)


# calculate the number of vaccinated individuals that are still Susceptible
def dVS_dt(params, Sa, Sc, Ia, Ic, VS, VI):
    return (
        -params["lambda"] * (Ia + Ic + VI) * VS
        - (params["mu"] * VS)
        + ((1 - params["tau"]) * params["kappa1"] * (Sa + Sc))
    )


# calculate the number of vaccinated individuals that are Infected
def dVI_dt(params, Ia, Ic, VS, VI):
    return (
        params["lambda"] * (Ia + Ic + VI) * VS
        - (params["mu"] * VI)
        + params["kappa1"] * (Ia + Ic)
        - (params["delta"] * VI)
    )


# calculate the number of vaccinated individuals that have Recovered
def dVR_dt(params, Sa, Sc, Ra, Rc, VS, VI, VR):
    return (
        (params["delta"] * VI)
        - (params["mu"] * VR)
        + (params["tau"] * params["kappa1"] * (Sa + Sc))
        + (params["kappa1"] * (Ra + Rc))
    )


# the system of differential equations for all populations
def dP_dt(t, init_conditions):

    # initial conditions
    Sa, Ia, Ra, Sc, Ic, Rc, VS, VI, VR = init_conditions

    # temp fix: update the model parameters using the global variable
    # (should be a function parameter)
    global PARAMS

    return [
        dSa_dt(PARAMS, Sa, Ia, Ic, VI),
        dIa_dt(PARAMS, Sa, Ia, Ic, VI),
        dRa_dt(PARAMS, Ia, Ra),
        dSc_dt(PARAMS, Sc, Ia, Ic, VI),
        dIc_dt(PARAMS, Sc, Ia, Ic, VI),
        dRc_dt(PARAMS, Ic, Rc),
        dVS_dt(PARAMS, Sa, Sc, Ia, Ic, VS, VI),
        dVI_dt(PARAMS, Ia, Ic, VS, VI),
        dVR_dt(PARAMS, Sa, Sc, Ra, Rc, VS, VI, VR),
    ]


# integrate across a single interval of integration
def integrate(parameters, interval, init_conditions, step_size=1, population_error=0):

    # update the global parameters with what is passed into the function
    global PARAMS
    PARAMS = parameters

    # get all time steps
    ts = np.linspace(
        interval[0], interval[1], 1 + int((interval[1] - interval[0]) / step_size)
    )

    # solve the system
    solution = solve_ivp(dP_dt, interval, init_conditions, t_eval=ts, method="RK45")

    if population_error:
        for sa, ia, ra, sc, ic, rc, vs, vi, vr in zip(
            solution["y"][0],
            solution["y"][1],
            solution["y"][2],
            solution["y"][3],
            solution["y"][4],
            solution["y"][5],
            solution["y"][6],
            solution["y"][7],
            solution["y"][8],
        ):
            assert (
                1 - population_error
                <= sum(sa, ia, ra, sc, ic, rc, vs, vi, vr)
                <= 1 + population_error
            )

    df = pd.DataFrame(solution["y"].T, columns=COLUMN_NAMES)
    df["time"] = ts
    df = df.set_index("time")

    # update the "initial" conditions with the current state variables
    state = {
        "sa0": solution["y"][0][-1],
        "ia0": solution["y"][1][-1],
        "ra0": solution["y"][2][-1],
        "sc0": solution["y"][3][-1],
        "ic0": solution["y"][4][-1],
        "rc0": solution["y"][5][-1],
        "vs0": solution["y"][6][-1],
        "vi0": solution["y"][7][-1],
        "vr0": solution["y"][8][-1],
    }

    return df, state


def piecewise_integration(time_series_params):

    # initialize
    first = sorted(time_series_params.items(), key=lambda kv: kv[0])[0][1]
    init_conditions = [
        first["sa0"],
        first["ia0"],
        first["ra0"],
        first["sc0"],
        first["ic0"],
        first["rc0"],
        first["vs0"],
        first["vi0"],
        first["vr0"],
    ]
    results = pd.DataFrame()
    prev_time_step = 0

    # integrate over each distinct time interval, using its parameters
    for time_interval_str, param in time_series_params.items():

        # integrate over the interval of integration
        time_interval = (prev_time_step, int(time_interval_str) - 1)
        print("\t", time_interval)
        df, state = integrate(param, time_interval, init_conditions)
        df = (df * 100).round(2)
        results = df if results.empty else pd.concat([results, df])
        # results.drop_duplicates(inplace=True)
        R0 = calculate_R0(param)

        # update the "initial" conditions
        init_conditions = list(state.values())
        prev_time_step = int(time_interval_str)

    return results, R0


# plot the results
def draw(results_df, R0):

    fig = px.line(results_df[COLUMN_NAMES], color_discrete_sequence=COLORS)

    fig.update_layout(height=540, width=880, title_font=dict(size=20))

    fig.update_layout(
        title="Measles",
        xaxis_title="Time",
        yaxis_title="% of Population",
        legend_title="",
        font=dict(
            size=18,
        ),
        hovermode="x unified",
    )

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Measles model.")
    parser.add_argument("input_file", help=f"Path to an Excel file of input parameters")
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Destination directory",
        default=os.path.join(os.path.dirname(os.path.realpath(__file__))),
        type=str,
    )
    parser.add_argument(
        "-t", "--time", help="total number of time steps, if plotting with Dash)", default=100, type=int
    )
    parser.add_argument("-i", "--image", help="save screenshots", action="store_true"),
    parser.add_argument("--dash", help="launch interactive Dash plot in a web browser", action="store_true")
    parser.add_argument("-p", "--port", help="port number to use for Dash", default=8052, type=int)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # read input Excel file and iterate through each tabbed sheet (geographic region)
    input_df_dict = pd.read_excel(args.input_file, sheet_name=None)
    for sheet_name, input_df in input_df_dict.items():
        print(sheet_name)
        input_df = input_df.set_index("Parameter")
        param_description = input_df["Description"].to_dict()
        df = input_df.drop(columns=["Description"], errors="ignore")

        # integrate
        results_df, R0 = piecewise_integration(df.to_dict())

        # output the results
        output_name = os.path.join(args.output_dir, f"Measles")
        with pd.ExcelWriter(
            f"{output_name}.xlsx", engine="openpyxl", mode="w"
        ) as writer:
            results_df.to_excel(writer, sheet_name=sheet_name)
        fig = draw(results_df, R0)
        if args.image:
            pio.write_image(fig, f"{output_name}_{sheet_name}.png")

        # update the parameters for Dash
        dash_params = sorted(df.to_dict().items(), key=lambda kv: kv[0])[0][1]

    # display in the interactive Dash web app
    if args.dash:
        print("starting Dash...")
        app = dash.Dash(__name__, external_stylesheets=["bWLwgP.css"])

        inputs = [
            dcc.Input(
                id=f"input-{param}",
                type="number",
                value=default_val,
                style={
                    "margin-bottom": "20px",
                    "margin-left": "5px",
                    "margin-right": "20px",
                },
            )
            for param, default_val in dash_params.items()
        ]
        divs = [
            html.Div(id=f"div-{param}") for param, default_val in dash_params.items()
        ]
        input_forms = []
        for param, description, dash_input, div in zip(
            dash_params.keys(), param_description.values(), inputs, divs
        ):
            input_forms.append(f"{param}: ")
            input_forms.append(description)
            input_forms.append(dash_input)
            input_forms.append(div)
        STATES = [State(f"input-{param}", "value") for param in dash_params]

        app.layout = html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(
                            id="graph",
                            style={"width": "49%", "display": "inline-block"},
                        )
                    ]
                ),
                html.Div(
                    [
                        html.Button(
                            id="submit-button-state",
                            style={"margin-bottom": "10px"},
                            n_clicks=0,
                            children="Run Simulation",
                        )
                    ]
                ),
                html.Div(
                    [
                        "Number of Time Steps: ",
                        dcc.Input(id="num_ts", type="number", value=args.time),
                    ],
                    style={
                        "margin-bottom": "20px",
                        "margin-left": "5px",
                        "margin-right": "20px",
                    },
                ),
                html.Div(
                    input_forms, style={"width": "49%", "display": "inline-block"}
                ),
            ]
        )

        @app.callback(
            Output("graph", "figure"),
            Input("submit-button-state", "n_clicks"),
            State("num_ts", "value"),
            STATES,
        )
        def update_output(n_clicks, num_ts, *args):
            params = dict(zip(param_description.keys(), args))
            results_df, R0 = piecewise_integration({num_ts: params})
            return draw(results_df, R0)

        app.run_server(debug=False, port=args.port, host="0.0.0.0")
