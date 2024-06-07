import pandas as pd
from io import StringIO, BytesIO
import requests
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
import uuid

def load_dataset(file=None, url=None):
    if file:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file, delimiter=';')
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        elif file.filename.endswith('.json'):
            df = pd.read_json(file)
        else:
            raise ValueError("Unsupported file format")
    elif url:
        response = requests.get(url)
        if response.status_code == 200:
            if url.endswith('.csv'):
                df = pd.read_csv(StringIO(response.text), delimiter=';')
            elif url.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(BytesIO(response.content))
            elif url.endswith('.json'):
                df = pd.read_json(StringIO(response.text))
            else:
                raise ValueError("Unsupported file format")
        else:
            raise ValueError("Could not retrieve the file from the URL")
    else:
        raise ValueError("No file or URL provided")

    return df

def process_regression_data(dataset, input_cols, output_col):
    X = dataset[input_cols]
    y = dataset[output_col]
    return X, y

def create_plot(train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=train_scores_mean,
        error_y=dict(type='data', array=train_scores_std),
        mode='lines+markers',
        name='Training score'
    ))

    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=test_scores_mean,
        error_y=dict(type='data', array=test_scores_std),
        mode='lines+markers',
        name='Cross-validation score'
    ))

    fig.update_layout(
        title="Curvas de Aprendizaje",
        xaxis_title="Tamaño del conjunto de entrenamiento",
        yaxis_title="Score"
    )

    return pio.to_html(fig, full_html=False)

def train_regression_model(X_train, y_train, model_type, **kwargs):
    print("Entering train_regression_model function")
    
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'polynomial':
        degree = kwargs.get('degree', 2)
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    elif model_type == 'ridge':
        alpha = kwargs.get('alpha', 1.0)
        model = Ridge(alpha=alpha)
    elif model_type == 'lasso':
        alpha = kwargs.get('alpha', 1.0)
        model = Lasso(alpha=alpha)
    elif model_type == 'neural_network':
        neurons = kwargs.get('neurons', 10)
        hidden_activation = kwargs.get('hidden_activation', 'relu')
        output_activation = kwargs.get('output_activation', 'linear')
        layers = kwargs.get('layers', 1)
        epochs = kwargs.get('epochs', 100)
        
        model = Sequential()
        model.add(Dense(neurons, input_shape=(X_train.shape[1],), activation=hidden_activation))
        for _ in range(layers - 1):
            model.add(Dense(neurons, activation=hidden_activation))
        model.add(Dense(1, activation=output_activation))
        model.compile(optimizer='adam', loss='mean_squared_error')

        tensorboard_callback = TensorBoard(log_dir="./logs")

        history = model.fit(np.array(X_train), np.array(y_train), epochs=epochs, verbose=0, callbacks=[tensorboard_callback])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history.history['loss'],
            mode='lines',
            name='Model Loss'
        ))

        fig.update_layout(
            title='Model Loss',
            xaxis_title='Epoch',
            yaxis_title='Loss'
        )

        plot_div = pio.to_html(fig, full_html=False)
        return model, history, plot_div, '/results'  # Agregar la URL de redirección

    model.fit(X_train, y_train)
    
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plot_div = create_plot(train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std)
    return model, None, plot_div, '/results'  # Agregar la URL de redirección
