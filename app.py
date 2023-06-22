# Name: Musa Khumalo
# Student ID: 410921335
# Data Science Final Project
import dash
from dash import Dash, html, dcc, callback, Output, Input
from dash import dash_table
from jupyter_dash import JupyterDash
from dash.dependencies import Input, Output, State

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.graph_objects as go
import plotly.express as px

from sklearn import linear_model, neighbors
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import Select


app = JupyterDash(__name__)

areas_taiwan = {
    'North': ['Keelung City', 'Taipei City', 'New Taipei City', 'Taoyuan City', 'Hsinchu City', 'Hsinchu County'],
    'Chu-Miao': ['Miaoli County'],
    'Central': ['Taichung City', 'Changhua County', 'Nantou County'],
    'Yun-Chia-Nan': ['Yunlin County', 'Chiayi City', 'Chiayi County', 'Tainan City'],
    'Kao-Ping': ['Kaohsiung City', 'Pingtung County'],
    'Yilan': ['Yilan County'],
    'Hua-Tung': ['Hualien County', 'Taitung County'],
    'Island': ['Penghu County', 'Kinmen County', 'Lienchiang County']
}

# Create the dropdown menu options
features = ['AQI', 'AVPM25', 'AVPM10', 'AVO3', 'AVCO', 'SO2', 'NO2', 'PM25', 'PM10', 'O3', 'CO']

models = {'Regression': linear_model.LinearRegression,
          'KNN': neighbors.KNeighborsRegressor}

# Define the Dash layout
app.layout = html.Div([
    # Application title
    html.H1("Taiwan Air Quality Monitoring Predictor"),
    html.Label('Select an Area:'),
    dcc.Dropdown(
        id='area-dropdown',
        options=[{'label': area, 'value': area} for area in areas_taiwan.keys()],
        value=list(areas_taiwan.keys())[0]
    ),
    html.Label('Select a County:'),
    dcc.Dropdown(
        id='county-dropdown',
        value=areas_taiwan[list(areas_taiwan.keys())[0]][0]  # Set default value to the first county of the selected area
    ),
    html.Label('Select a Feature:'),
    dcc.Dropdown(
        id='feature-dropdown',
        options=[{'label': feature, 'value': feature} for feature in features],  # Replace `features` with your list of available features
        value=features[0]  # Set default value to the first feature in the list
    ),
    html.Button('Generate Graph', id='graph-button', n_clicks=0),
    html.Div(id='graph-output'),
    html.H3("Predicting Air Quality Index"),
    html.P("Select model:"),
    dcc.Dropdown(
        id='dropdown',
        options=["Regression", "KNN"],
        value='Regression',
        clearable=False
    ),
    dcc.Graph(id="graph"),
    dcc.Graph(id="graph2"),
    dcc.Graph(id="graph3")
])


def wrangle(df):
    # Remove missing values from features
    df = df[df['SO2'] != 'Equipment Error']
    df = df[df['SO2'] != 'Equipment Calibration']
    df = df[df['NO2'] != 'Equipment Error']
    df = df[df['SO2'] != 'Equipment Calibration']
    df = df[df['O3'] != 'Equipment Error']
    df = df[df['O3'] != 'Equipment Calibration']
    df = df[df['CO'] != 'Equipment Error']
    df = df[df['CO'] != 'Equipment Calibration']
    df = df[df['PM25'] != 'Equipment Error']
    df = df[df['PM25'] != 'Equipment Calibration']
    df = df[df['PM10'] != 'Equipment Error']
    df = df[df['PM10'] != 'Equipment Calibration']
    
    # Set type as float
    df = df.astype(float)
    # Set index name to `date`
    df.index.name = "date"
    
    return df

def web_scraping_function(area, county):
    
    DRIVER_PATH = './/chromedriver.exe'
    URL = 'https://airtw.epa.gov.tw/ENG/EnvMonitoring/Central/CentralMonitoring.aspx'
    s = Service(executable_path=DRIVER_PATH)
    browser = webdriver.Chrome(service=s)   
    browser.get(URL) 
    
    AQI = []
    AVPM25 = []
    AVPM10 = []
    AVO3 = []
    AVCO = []
    SO2 = []
    NO2 = []
    PM25 = []
    PM10 = []
    O3 = []
    CO = []

    area_tag = browser.find_element(By.XPATH, '//select[@id="ddl_Area"]')
    areas = Select(area_tag)
    areas.select_by_visible_text(area)
    county_tag = browser.find_element(By.XPATH, '//select[@id="ddl_County"]')
    counties = Select(county_tag)
    counties.select_by_visible_text(county)
    time_tag = browser.find_element(By.XPATH, '//*[@id="ddl_Time"]')
    times = Select(time_tag)

    timestamps = []

    for option in times.options:
        time = option.text
        times.select_by_visible_text(time)
        timestamps.append(time)

        button = browser.find_element(By.XPATH, '//*[@id="btn_search"]')
        button.click()
        
        soup = BeautifulSoup(browser.page_source, 'html.parser')

        aqi = soup.find('p', class_='num', id='AQI').text
        avpm25 = soup.find('p', class_='num', id='AVPM25').text
        avpm10 = soup.find('p', class_='num', id='AVPM10').text
        avo3 = soup.find('p', class_='num', id='AVO3').text
        avco = soup.find('p', class_='num', id='AVCO').text
        so2 = soup.find('p', class_='num', id='SO2').text
        no2 = soup.find('p', class_='num', id='NO2').text
        pm25 = soup.find('p', class_='num', id='PM25').text
        pm10 = soup.find('p', class_='num', id='PM10').text
        o3 = soup.find('p', class_='num', id='O3').text
        co = soup.find('p', class_='num', id='CO').text

        AQI.append(aqi)
        AVPM25.append(avpm25)
        AVPM10.append(avpm10)
        AVO3.append(avo3)
        AVCO.append(avco)
        SO2.append(so2)
        NO2.append(no2)
        PM25.append(pm25)
        PM10.append(pm10)
        O3.append(o3)
        CO.append(co)
        
    browser.quit() 
    
    data = {
    'AQI': AQI,
    'AVPM25': AVPM25,
    'AVPM10': AVPM10,
    'AVO3': AVO3,
    'AVCO': AVCO,
    'SO2': SO2,
    'NO2': NO2,
    'PM25': PM25,
    'PM10': PM10,
    'O3': O3,
    'CO': CO
    }
    timestamps = pd.to_datetime(timestamps)

    analysis_df = pd.DataFrame(data, index=timestamps)
    
    analysis_df = wrangle(analysis_df)
    return analysis_df



@app.callback(
    Output('county-dropdown', 'options'),
    Input('area-dropdown', 'value')
)
def update_counties(area):
    counties = areas_taiwan.get(area, [])  # Get the counties for the selected area
    county_options = [{'label': county, 'value': county} for county in counties]
    return county_options

# Define an empty data cache
data_cache = {}


@app.callback(
    Output('graph-output', 'children'),
    Input('graph-button', 'n_clicks'),
    State('area-dropdown', 'value'),
    State('county-dropdown', 'value'),
    State('feature-dropdown', 'value'),
    State('graph-output', 'children')
)
def generate_graph(n_clicks, area, county, feature, graph_output):
    if n_clicks is not None and n_clicks > 0:
        # Check if the data for the selected area and county is already in the cache
        if (area, county) in data_cache:
            analysis_df = data_cache[(area, county)]
        else:
            # Retrieve the data using the web_scraping_function
            # NOTE: This takes a while, about 4 minutes, so you may consider using the csv file saved from past data of North, Keelung City
            # analysis_df = web_scraping_function(area, county)
            analysis_df = pd.read_csv('data.csv', index_col=0)
            
            # Store the retrieved data in the cache
            data_cache[(area, county)] = analysis_df
        
        categories = list(analysis_df.index)
        values = list(analysis_df[feature])  # Use the selected feature for the graph

        data = [
            go.Scatter(x=categories, y=values, mode='lines', marker=dict(color='#EF553B'))
        ]

        layout = go.Layout(
            title='Analysis Result for {} - {}'.format(county, feature),
            xaxis=dict(title='Time'),
            yaxis=dict(title=feature)  # Use the selected feature as the y-axis label
        )
        analysis_df = analysis_df.reset_index()  # Reset the index
        fig = go.Figure(data=data, layout=layout)

        graph = dcc.Graph(figure=fig)

        table = dash_table.DataTable(
            data=analysis_df.iloc[:10].to_dict('records'),
            columns=[{'name': col, 'id': col} for col in analysis_df.columns],
            style_data={'whiteSpace': 'normal', 'height': 'auto'},
            style_table={'overflowX': 'auto'},
            page_size=10
        )

        return [graph, html.H4('Top 10 Records'), table]
    else:
        return graph_output

@app.callback(
    Output("graph2", "figure"),
    Output("graph", "figure"),
    Output("graph3", "figure"),
    Input('dropdown', "value"),
    Input('graph-button', 'n_clicks'),
    State('area-dropdown', 'value'),
    State('county-dropdown', 'value'),
    State('feature-dropdown', 'value')
)
def train_and_display(name, n_clicks, area, county, feature):
    if n_clicks is not None and n_clicks > 0:
        if (area, county) in data_cache:
            df = data_cache[(area, county)]
        else:
            # Retrieve the data using the web_scraping_function
            # NOTE: This takes a while, about 4 minutes, so you may consider using the csv file saved from past data of North, Keelung City
            # df = web_scraping_function(area, county)
            df = pd.read_csv('data.csv', index_col=0)
            data_cache[(area, county)] = df
        
        target = 'AQI'
        X = df.drop(columns=target)
        y = df[target]
        
        correlation = X.corrwith(y)
        correlation
        # Choosing the best features with coefficients > 0.1
        features_to_drop = []
        for key, val in dict(correlation).items():
            if val < 0.1:
                features_to_drop.append(key)

        X = X.drop(columns=features_to_drop)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = models[name]()
        model.fit(X_train, y_train)
        # Model coefficients graph
        colors = ['Positive' if c > 0 else 'Negative' for c in model.coef_]

        fig = px.bar(
            x=X.columns, y=model.coef_, color=colors,
            color_discrete_sequence=['red', 'blue'],
            labels=dict(x='Feature', y='Linear coefficient'),
            title='Weight of each feature for predicting Air Quality Index'
        )
        # Correlation matrix
        fig2 = px.imshow(df.corr(), title='Correlation Coefficient Matrix')
        
        # Evaluation scores
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)*100

        # Labels for the scores
        labels = ['MSE', 'MAE', 'R-squared']

        # Scores values
        scores = [mse, mae, r2]

        # Define colors for each bar
        colors = ['blue', 'blue', '#EF553B']

        # Create the bar chart
        fig3 = go.Figure(data=go.Bar(x=labels, y=scores, marker=dict(color=colors)))

        # Set the chart title and axes labels
        fig3.update_layout(
            title='Evaluation Scores',
            xaxis_title='Metric',
            yaxis_title='Score'
        )

        return fig, fig2, fig3


    else:
        raise dash.exceptions.PreventUpdate


if __name__ == '__main__':
    app.run_server(debug=True, port=5002)
