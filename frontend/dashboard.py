# dependencies
import sys
sys.path.append('../src')
import utils
import data_processor
import deeplearning_build
import dash
import plotly.graph_objs as go
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

# App layout
app.layout = html.Div([
    html.H1("Energy Price Forecasting Dashboard", style={'text-align': 'center'}),
    dcc.Dropdown(id="select_model",
                 options=[
                     {"label": "BiLSTM", "value": 1},
                     {"label": "CNN-BiLSTM", "value": 2},
                     {"label": "CNN-BiLSTM-Attention", "value": 3}],
                 multi=False,
                 value=1,
                 style={'width': "40%"}
                 ),
    html.Div(id='output_container', children=[]),
    html.Br(),
    # wrapping the graph in a loading component
    dcc.Loading(
        id="loading-graph",
        children=dcc.Graph(id="model_accuracy", figure={}),
        type="default",
    ),
])

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

    # Display model performance metrics 
    container = f"The model chosen by user was: {model_name}\n"
    container += "Model Performance Metrics: \n"
    container += f"Mean Absolute Error: {round(mae, 2)};\n"
    container += f"Mean Squared Error: {round(mse, 2)};\n"
    container += f"Root Mean Squared Error: {round(rmse, 2)}; \n"
    container += f"R2 Score: {round(r2, 2)} \n"

    return container, fig
    
# Connect the plotly graphs with Dash Components
@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='model_accuracy', component_property='figure')],
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
        container, fig = model_processing(df_test, model_name)
        loading = False
        return container, fig
    else:
        container = "Please select a valid model"
        fig = go.Figure()
        return container, fig    

if __name__ == "__main__":
    app.run_server(mode='inline', host = '127.0.0.1', port='8050', debug=True)        
