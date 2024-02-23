import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

def get_navbar():
   
    nav_item_home = dbc.NavItem(dbc.NavLink(html.A(
                        [
                            html.Img(src="/assets/signal.png", className="hub-logo")])))
    nav_item_about = dbc.NavItem(html.H4("Telecom Customer Churn Analysis and Prediction"),
    className="cover")
    logo=dbc.Navbar(dbc.Container( 
         dbc.Nav(
                        [
                            
                            nav_item_about,
                            nav_item_home]
    )))
    return logo
    