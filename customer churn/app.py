import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from dash import no_update

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

import os
import sys
import copy
import time
# print(os.getcwd())

from src.navbar import get_navbar   # navbar function
from src.graphs import df, layout, ohe, cat_features, svm_model  
from content import tab_prediction_content, tab_analysis_content

# Creating the app

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    external_stylesheets = [dbc.themes.CYBORG,'/assets/styles.css']
) 

server=app.server

# Tabs Content

tabs = dbc.Tabs(
    [
        dbc.Tab(tab_prediction_content, label="Prediction"),
        dbc.Tab(tab_analysis_content, label="Data Analysis"),
    ]
)



#-------------------------------------------------------------------------------
# APPLICATION LAYOUT
#-------------------------------------------------------------------------------

app.layout = html.Div(
    [
        get_navbar(),
        html.Br(),
        html.Br(),
        html.Div(
            dbc.Row(dbc.Col(tabs, width=12)),
            id="mainContainer",
            style={"display": "flex", "flex-direction": "column"}
        ),
        html.P("🎬💻☎📞", className="footer")
    ],
)

#-------------------------------------------------------------------------------
# CALLBACKS
#-------------------------------------------------------------------------------

# Data Analysis Tab - Categorical Bar Chart 

@app.callback(
    Output("categorical_bar_graph", "figure"),
    [
        Input("categorical_dropdown", "value"),
    ],
)

def bar_categorical(feature):

    time.sleep(0.2)

    temp = df.groupby([feature, 'Churn']).count()['customerID'].reset_index()
    
    fig = px.bar(temp, x=feature, y="customerID",
             color=temp['Churn'].map({'Yes': 'Churn', 'No': 'NoChurn'}),
             color_discrete_map={"Churn": "#47acb1", "NoChurn": "#f26522"},
             barmode='group')
    
    layout_count = copy.deepcopy(layout)
    fig.update_layout(layout_count)
    
    _title = (feature[0].upper() + feature[1:]) + " Distribution by Churn"
    
    fig.update_layout(
        title = {'text': _title, 'x': 0.5},
        #xaxis_visible=False,
        xaxis_title="",
        yaxis_title="Count",
        legend_title_text="",
        legend = {'x': 0.16}
    )
    return fig

@app.callback(
    Output("categorical_pie_graph", "figure"),
    [
        Input("categorical_dropdown", "value"),
    ],
)

# Data Analysis Tab - Donut Chart 

def donut_categorical(feature):

    time.sleep(0.2)

    temp = df.groupby([feature]).count()['customerID'].reset_index()

    fig = px.pie(temp, values="customerID", names=feature, hole=.5)

    layout_count = copy.deepcopy(layout)
    fig.update_layout(layout_count)
    
    _title = (feature[0].upper() + feature[1:]) + " Percentage"

    if(df[feature].nunique() == 2):
        _x = 0.3
    elif(df[feature].nunique() == 3):
        _x = 0.16
    else:
        _x = 0

    fig.update_layout(
        title = {'text': _title, 'x': 0.5},
        legend = {'x': _x}
    )

    return fig

# Prediction Tab - Predict and Update the Result Cards 

@app.callback(
    [dash.dependencies.Output(component_id='svm_result', component_property='children')],
    

    [dash.dependencies.Input('btn_predict', 'n_clicks')],

    [dash.dependencies.State('ft_gender', 'value'),
     dash.dependencies.State('ft_partner', 'value'),
     dash.dependencies.State('ft_dependents', 'value'),
     dash.dependencies.State('ft_phoneService', 'value'),
     dash.dependencies.State('ft_multipleLines', 'value'),
     dash.dependencies.State('ft_internetService', 'value'),
     dash.dependencies.State('ft_onlineSecurity', 'value'),
     dash.dependencies.State('ft_onlineBackup', 'value'),
     dash.dependencies.State('ft_deviceProtection', 'value'),
     dash.dependencies.State('ft_techSupport', 'value'),
     dash.dependencies.State('ft_streamingTv', 'value'),
     dash.dependencies.State('ft_streamingMovies', 'value'),
     dash.dependencies.State('ft_contract', 'value'),
     dash.dependencies.State('ft_paperlessBilling', 'value'),
     dash.dependencies.State('ft_paymentMethod', 'value'),
     dash.dependencies.State('ft_seniorCitizen', 'value'),
     dash.dependencies.State('ft_monthlyCharges', 'value'),
     dash.dependencies.State('ft_totalCharges', 'value'),
     dash.dependencies.State('ft_tenure', 'value')]
)

def predict_churn(n_clicks, ft_gender, ft_partner, ft_dependents, ft_phoneService, ft_multipleLines,
                            ft_internetService, ft_onlineSecurity, ft_onlineBackup, ft_deviceProtection,
                            ft_techSupport, ft_streamingTv, ft_streamingMovies, ft_contract,
                            ft_paperlessBilling, ft_paymentMehod, ft_seniorCitizen, ft_monthlyCharges,
                            ft_totalCharges, ft_tenure):

    time.sleep(0.4)

    sample = {'gender': ft_gender, 'Partner': ft_partner, 'Dependents': ft_dependents,
              'PhoneService': ft_phoneService, 'MultipleLines': ft_multipleLines,
              'InternetService': ft_internetService, 'OnlineSecurity': ft_onlineSecurity, 'OnlineBackup': ft_onlineBackup,
              'DeviceProtection': ft_deviceProtection, 'TechSupport': ft_techSupport, 'StreamingTV': ft_streamingTv,
              'StreamingMovies': ft_streamingMovies, 'Contract': ft_contract, 'PaperlessBilling': ft_paperlessBilling,
              'PaymentMethod': ft_paymentMehod, 'TotalCharges': float(ft_totalCharges), 'MonthlyCharges': float(ft_monthlyCharges),
              'tenure': int(ft_tenure), 'SeniorCitizen': int(ft_seniorCitizen)}

    sample_df = pd.DataFrame(sample, index=[0])
    # print(sample_df)
    sample_df_enc = ohe.transform(sample_df[cat_features])
    # print('Yes2')
    feature_names = ohe.get_feature_names_out(cat_features)
    # print('Yes3')
    print(feature_names)
    
    sample_df_enc = pd.DataFrame(sample_df_enc,columns=feature_names)
    # print('Yes4')
    print(sample_df_enc)
    sam = pd.concat([sample_df_enc, sample_df[['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'tenure']]], axis=1)
    # print('Yes5')
    print(sam)
    svm_prediction = svm_model.predict(sam)
    # print('HEYYYYYYYY')
    print(svm_prediction)
    

    def churn_to_text(num):
        if(num == 0):
            return ["Predicted: Not Churn(STAY)"]
        elif(num == 1):
            return ["Predicted: Churn(EXIT)"]

    

    if(n_clicks):
        return churn_to_text(svm_prediction)
    else:
        return no_update



#-------------------------------------------------------------------------------
# MAIN FUNCTION
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)