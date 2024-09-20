import pickle
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from google.cloud import bigquery
from cachetools import cached, TTLCache
import setup
from setup import get_category_for_sku

# Initialisation de l'application Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

with open('data/df_load_data.pickle','rb') as modelFile:
     df = pickle.load(modelFile)

# Chargement des modèles
def load_models():
    models = {
        'rf_op': pickle.load(open('data/fitted_model/fitted_model_rf_op.pickle', 'rb')),
        'rf_hors_op': pickle.load(open('data/fitted_model/fitted_model_rf_hors_op.pickle', 'rb')),
        'lr_op': pickle.load(open('data/fitted_model/fitted_model_lr_op.pickle', 'rb')),
        'lr_hors_op': pickle.load(open('data/fitted_model/fitted_model_lr_hors_op.pickle', 'rb')),
        'gbc_op': pickle.load(open('data/fitted_model/fitted_model_gbc_op.pickle', 'rb')),
        'gbc_hors_op': pickle.load(open('data/fitted_model/fitted_model_gbc_hors_op.pickle', 'rb')),
    }
    return models

models = load_models()

# Liste des product_id pour le dropdown
product_id_list = df['product_id'].unique()
product_id_options = [{'label': str(id), 'value': str(id)} for id in product_id_list]

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Dashboard de Prédiction E-commerce"))
    ]),
    
    dbc.Row([
        # Sidebar
        dbc.Col([
            html.H2("Filtres"),
            dbc.Form([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("ID du Produit"),
                        dcc.Dropdown(
                            options=product_id_options,
                            placeholder="Entrez l'ID du produit",
                            searchable=True,
                            clearable=True,
                        )
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Modèle de prédiction"),
                        dcc.Dropdown(
                            options=[
                                {'label': 'Random Forest', 'value': 'rf'},
                                {'label': 'Linear Regression', 'value': 'lr'},
                                {'label': 'Gradient Boosting', 'value': 'gbc'}
                            ],
                            placeholder="Sélectionnez un modèle",
                        )
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Catégorie"),
                        dbc.Input(
                            id="category-input",
                            type="text",
                            placeholder="Catégorie",
                            readOnly=True
                            )
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Prix Moyen"),
                        dbc.Input(
                            id="avg-price-input",
                            type="number",
                            placeholder="Entrez le prix moyen",
                            )
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Indice Prix Moyen"),
                        dbc.Input(
                            id="indice-avg-price-input",
                            type="number",
                            placeholder="Entrez l'indice prix moyen",
                            )
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Impressions"),
                        dcc.Slider(
                            id='impression-gs-slider',
                            min=0,
                            max=df['impression_gs'].max(),
                            step=10000,
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("En Page d'Accueil"),
                        dcc.Dropdown(
                            id="on-front-dropdown",
                            options=[
                                {'label': 'Oui', 'value': 1},
                                {'label': 'Non', 'value': 0}
                            ],
                            placeholder="Sélectionnez Oui ou Non"
                        )
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Date de Lancement"),
                        dcc.DatePickerSingle(
                            id='launch-date-picker',
                            placeholder='Sélectionnez une date',
                            display_format='YYYY-MM-DD',
                            clearable=True,
                            with_portal=True
                        )
                    ]),
                ]),
                dbc.Button("Prédire", id="predict-button", color="primary")
            ])
        ], md=3, lg=2, className="sidebar"),
        
        # Main content
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.H2("Prévisions de Performance"),
                    dbc.Row([
                        dbc.Col([
                            html.H3("En Opération Commerciale"),
                            html.Div(id="prediction-op", children="En attente...", className="prediction-value")
                        ]),
                        dbc.Col([
                            html.H3("Hors Opération Commerciale", className="h6 mb-0"),
                            html.Div(id="prediction-no-op", children="En attente...", className="prediction-value")
                        ]),
                    ]),
                ]),
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H2("Historique des Performances"),
                    dcc.Graph(
                        id="performance-graph",
                        config={
                            'responsive': True,
                            'displayModeBar': False,
                        },
                        style={"height": "400", "width": "100%"}
                    )
                ]),
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H2("Historique des Campagnes"),
                    html.Div(id="campaign-history", className="campaign-history", style={"maxHeight": "150px", "overflowY": "auto"})
                ]),
            ]),
        ], md=9, lg=10, className="dashboard-content")
    ])
], fluid=True)

@app.callback(
    [Output("category-input", "value"),
     Output("avg-price-input", "value"),
     Output("indice-avg-price-input", "value"),
     Output("on-front-dropdown", "value")],
    [Input("product-id-input", "value")]
)
def update_sidebar(product_id):
    if product_id:
        product_data = df[df['product_id'] == product_id].iloc[0]
        category = product_data['Category_1']
        avg_price = product_data['avg_price']
        indice_avg_price = product_data['indice_avg_price']
        on_front = product_data['on_front']
        return category, avg_price, indice_avg_price, on_front
    return "Catégorie non disponible", "Prix non disponible", "Indice non disponible", None

@app.callback(
    [Output("prediction-op", "children"),
     Output("prediction-no-op", "children")],
    [Input("predict-button", "n_clicks")],
    [State("model-dropdown", "value"),
     State("product-id-input", "value"),
     State("avg-price-input", "value"),
     State("indice-avg-price-input", "value"),
     State("impression-gs-slider", "value"),
     State("on-front-dropdown", "value"),
     State("launch-date-picker", "date")]
)
def update_predictions(n_clicks, model_op, product_id, avg_price, indice_avg_price, impression_gs, on_front, launch_date):
    if n_clicks > 0:
        if not model_op or not product_id:
            return "Erreur: Modèle ou produit non sélectionné", "Erreur: Modèle ou produit non sélectionné"
        
        try:
            if launch_date is None:
                raise ValueError("La date de lancement est requise.")
        
            model_op_key = f"{model_op}_op"
            model_no_op_key = f"{model_op}_hors_op"

            if model_op_key not in models or model_no_op_key not in models:
                return "Erreur: Modèle sélectionné inconnu", "Erreur: Modèle sélectionné inconnu"
            
            category_for_sku = get_category_for_sku(df, product_id)

            input_data = setup.load_data_model(
                df,
                product_id,
                avg_price,
                indice_avg_price,
                impression_gs,
                on_front,
                launch_date
            )

            prediction_op = models[model_op_key].predict(input_data)[0]
            prediction_no_op = models[model_no_op_key].predict(input_data)[0]

            return f"{prediction_op:.0f} nouveaux clients", f"{prediction_no_op:.0f} nouveaux clients"
        
        except Exception as e:
            return f"Erreur: {str(e)}", f"Erreur: {str(e)}"

    return "En attente...", "En attente..."

@app.callback(
    [Output("campaign-history", "children"),
     Output("performance-graph", "figure")],
    [Input("product-id-input", "value"),
     Input("model-dropdown", "value"),
     Input("avg-price-input", "value"),
     Input("indice-avg-price-input", "value"),
     Input("impression-gs-slider", "value"),
     Input("on-front-dropdown", "value"),
     Input("launch-date-picker", "date")]
)
def update_history_and_graph(product_id, model_op, avg_price, indice_avg_price, impression_gs, on_front, launch_date):
    if product_id:
        try:
            campaigns = df[(df['product_id'] == product_id) & (df['on_operation'] == 1)][['operation_name', 'startdate_op', 'enddate_op']].drop_duplicates()
            if campaigns.empty:
                campaign_history = ["Aucun historique disponible"]
            else:
                campaign_history = [
                    html.Div([
                        html.Strong(row['operation_name']),
                        html.Span(f" ({row['startdate_op']} - {row['enddate_op']})"),
                    ], className="campaign-item")
                    for _, row in campaigns.iterrows()
                ]

            performance_data = df[df['product_id'] == product_id]
            fig = go.Figure()

            if not performance_data.empty:
                fig.add_trace(go.Bar(
                    x=performance_data['order_date'], 
                    y=performance_data['nb_new_customers'],
                    name='Nouveaux clients',
                    marker_color='blue'
                ))

                fig.add_trace(go.Bar(
                    x=performance_data['order_date'], 
                    y=performance_data['total_customers'],
                    name='Clients totaux',
                    marker_color='green'
                ))

                if model_op and launch_date:
                    try:
                        model_op_key = f"{model_op}_op"
                        model_no_op_key = f"{model_op}_hors_op"

                        if model_op_key in models and model_no_op_key in models:
                            input_data = setup.load_data_model(
                                df,
                                product_id,
                                avg_price,
                                indice_avg_price,
                                impression_gs,
                                on_front,
                                launch_date
                            )

                            prediction_op = models[model_op_key].predict(input_data)[0]
                            prediction_no_op = models[model_no_op_key].predict(input_data)[0]

                            fig.add_trace(go.Scatter(
                                x=[pd.to_datetime(launch_date)],
                                y=[prediction_op],
                                mode='markers+text',
                                name='Prédiction avec Promo',
                                text=['Prédiction avec Promo'],
                                textposition='top center',
                                marker=dict(size=12, color='red')
                            ))

                            fig.add_trace(go.Scatter(
                                x=[pd.to_datetime(launch_date)],
                                y=[prediction_no_op],
                                mode='markers+text',
                                name='Prédiction Hors Promo',
                                text=['Prédiction Hors Promo'],
                                textposition='top center',
                                marker=dict(size=12, color='orange')
                            ))

                    except Exception as e:
                        print(f"Erreur lors de la prédiction: {str(e)}")

                fig.update_layout(
                    barmode='group',
                    title='Performance du produit au fil du temps',
                    xaxis_title='Date',
                    yaxis_title='Nombre de clients',
                    xaxis_tickformat='%d<br>%B',
                    legend_title_text='Type de clients',
                    # margin=dict(l=40, r=20, t=40, b=20),
                    autosize=True,  # Rend le graphique responsive
                    height=400,  # Hauteur de base, s'ajustera en fonction de l'écran
                    hovermode='closest'
                )

                # Configuration pour rendre le graphique responsive
                fig.update_layout(
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
            else:
                fig = go.Figure()

            # Configuration pour rendre le graphique responsive
            fig.update_layout(
                autosize=True,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="LightSteelBlue",
            )

            return campaign_history, fig
        
        except Exception as e:
            return ["Erreur dans l'historique des campagnes"], go.Figure()

    return ["En attente..."], go.Figure()

if __name__ == '__main__':
    app.run_server(debug=False)
