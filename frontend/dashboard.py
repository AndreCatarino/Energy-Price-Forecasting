# dependencies
import sys
sys.path.append('../src')
import utils
import data_processor
import deeplearning_build
import dash
import plotly.graph_objs as go
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import joblib
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import base64
import io
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Prepare data and deep learning models

# Load processed data
processed_data: pd.DataFrame = utils.load_preprocessed_data()

# Data preprocessing object
processor = data_processor.DataProcessor(processed_data)
X, y = processor.create_feature_matrix_and_target_vector(target_column="price actual")

# Target variable should be the last column (for compatibility with deeplearning_build module)
processed_data = processed_data.drop(columns=["price actual"])
processed_data["price actual"] = y

# Split data into train, validation and test sets (80%, 20%)
_, df_test = train_test_split(processed_data, test_size=0.2, random_state=0, shuffle=False)

# instantiate deepL class
deepL = deeplearning_build.deepL()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    children=[
        html.Div(
            [
                html.H1("Energy Price Forecasting Dashboard", style={'text-align': 'center'}),
                html.Hr(),
                html.H4(id='output_container', children=[], style={'text-align': 'center'}),
                html.Br(),
                html.Div("Select a model to evaluate its performance on the test set:", style={'text-align': 'center'}),
                dcc.Dropdown(
                    id="select_model",
                    options=[
                        {"label": "BiLSTM", "value": 1},
                        {"label": "CNN-BiLSTM", "value": 2},
                        {"label": "CNN-BiLSTM-Attention", "value": 3}
                    ],
                    multi=False,
                    value=1,
                    style={"text-align": 'center'}
                ),
                html.Br(),
                html.Div(
                    id='performance_table_container',
                    children=[],
                    style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
                ),
                html.Br(),
                dcc.Loading(
                    id="loading-graph",
                    children=dcc.Graph(id="model_accuracy", figure={}),
                    type="default"
                ),
                html.Br(),
                html.Div([
                    dcc.Graph(id='error_distribution', figure={}),
                ]),
                html.Div(id='residuals-container', children=[]),
                html.Div(id='output_plots', children=[]),
            ],
            style={'backgroundColor': '#f2f2f2', 'padding': '30px'}  # Light gray background color with padding
        )
    ]
)

def create_performance_table(model_name, mae, mse, rmse, r2):
    performance_metrics = pd.DataFrame({
        'Metric': [
            #"Model chosen",
            "Mean Absolute Error",
            "Mean Squared Error",
            "Root Mean Squared Error",
            "R2 Score"
        ],
        'Value': [
            #model_name,
            round(mae, 2),
            round(mse, 2),
            round(rmse, 2),
            round(r2, 2)
        ]
    })

    performance_table = dbc.Table.from_dataframe(
        performance_metrics,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True
    )

    return performance_table


def model_processing(df_test, model_name):
    deepL.load_model(model_name) 
        
    test_set = deepL.prepare_sequential_window(df_test, window_size=15, classification=False)
    y_pred = deepL.predict(model_name, df_test)
    y_pred = y_pred.reshape(-1, 1)

    y_batch_list = []
    for _, y_batch in test_set:
        y_batch_list.append(y_batch.numpy())

    y_batch_list = np.array(y_batch_list)
    y_batch_list = y_batch_list.reshape(-1, 1)

    # load scaler
    scaler = joblib.load("../artifacts/scaler.pkl")
    pred = scaler.inverse_transform(y_pred)
    original_target = scaler.inverse_transform(y_batch_list)
    
    # Calculate the errors (residuals)
    errors = original_target - pred

    mae = mean_absolute_error(original_target, pred)
    mse = mean_squared_error(original_target, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(original_target, pred)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(0, len(original_target)), y=original_target.flatten(), mode='lines',
                             name='Actual Price'))
    fig.add_trace(go.Scatter(x=np.arange(0, len(original_target)), y=pred.flatten(), mode='lines',
                                name='Predicted Price'))
    fig.update_layout(title=f"Actual vs Predicted Price ({model_name})",
                        xaxis_title="Time",
                        yaxis_title="Price ($)")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    fig.update_layout(legend_title_text='Legend')
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    container = f"Multivariate Sequence to Vector Modeling: Next hour price prediction using {model_name} model"

    return container, fig, errors, mae, mse, rmse, r2

# Connect the plotly graphs with Dash Components
@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='model_accuracy', component_property='figure')],
     Output(component_id='error_distribution', component_property='figure'),
     Output(component_id='residuals-container', component_property='children'),
     Output(component_id='output_plots', component_property='children'),
     Output(component_id='performance_table_container', component_property='children'),
    [Input(component_id='select_model', component_property='value')] 
)


def update_graph(option_selected):
    models = {
        1: "BiLSTM",
        2: "CNN-BiLSTM",
        3: "CNN-BiLSTM-Attention"
    }

    if option_selected in models:
        model_name = models[option_selected]
        loading = True
        container, fig, errors, mae, mse, rmse, r2 = model_processing(df_test, model_name)
        loading = False

        # Plot error distribution
        error_fig = go.Figure()
        error_fig.add_trace(go.Histogram(x=errors.flatten(), histnorm='probability density'))
        error_fig.update_layout(title="Error Distribution",
                            xaxis_title="Error",
                            yaxis_title="Density",
                            showlegend=False)

        # Perform autocorrelation test
        
        # Reshape errors array if needed
        errors = errors.flatten()

        # Perform autocorrelation test
        fig_acf, ax_acf = plt.subplots()
        fig_pacf, ax_pacf = plt.subplots()

        plot_acf(errors, ax=ax_acf)
        plot_pacf(errors, ax=ax_pacf)

        # Save the plots as base64 encoded strings in assets folder
        buffer_acf = io.BytesIO()
        fig_acf.savefig(buffer_acf, format='png')
        buffer_acf.seek(0)
        buffer_pacf = io.BytesIO()
        fig_pacf.savefig(buffer_pacf, format='png')
        buffer_pacf.seek(0)

        # Encode images to base64 strings
        encoded_acf = base64.b64encode(buffer_acf.read()).decode('utf-8')
        encoded_pacf = base64.b64encode(buffer_pacf.read()).decode('utf-8')

        # Create HTML images
        acf_plot = html.Img(src=f"data:image/png;base64,{encoded_acf}")
        pacf_plot = html.Img(src=f"data:image/png;base64,{encoded_pacf}")

        performance_table = create_performance_table(model_name, mae, mse, rmse, r2)

        return container, fig, error_fig, acf_plot, pacf_plot, performance_table

if __name__ == "__main__":
    app.run_server(mode='inline', host = '127.0.0.1', port='8050', debug=True)      
