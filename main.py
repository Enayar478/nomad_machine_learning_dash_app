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

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Dashboard de Prédiction E-commerce", className="dashboard-title"),
    ], className="header"),
    
    # Conteneur principal
    html.Div([
        # Sidebar
        html.Div([
            html.H2("Filtres", className="sidebar-title"),

            # Dropdown pour sélectionner l'ID du produit
            html.Label("ID du Produit"),
            dcc.Dropdown(
                id="product-id-input",
                options=product_id_options,
                placeholder="Entrez l'ID du produit",
                searchable=True,
                clearable=True,
                className="input-box"
            ),

            # Dropdown pour sélectionner un modèle
            html.Br(),
            html.Label("Modèle de prédiction"),
            dcc.Dropdown(
                id="model-dropdown", 
                options=[
                    {'label': 'Random Forest', 'value': 'rf'},
                    {'label': 'Linear Regression', 'value': 'lr'},
                    {'label': 'Gradient Boosting', 'value': 'gbc'}
                ],
                placeholder="Sélectionnez un modèle",
                className="input-box"
            ),
    
            # Champ affichant automatiquement la catégorie après sélection du produit
            html.Br(),
            html.Label("Catégorie"),
            dcc.Input(
                id="category-input",
                type="text",
                placeholder="Catégorie",
                readOnly=True,
                className="input-box"
            ),

            # Champ pour afficher le prix moyen (avg_price) que l'utilisateur peut remplir
            html.Br(),
            html.Label("Prix Moyen"),
            dcc.Input(
                id="avg-price-input",
                type="number",
                placeholder="Entrez le prix moyen",
                className="input-box"
            ),

            # Champ pour afficher l'indice du prix moyen (indice_avg_price) que l'utilisateur peut remplir
            html.Br(),
            html.Label("Indice Prix Moyen"),
            dcc.Input(
                id="indice-avg-price-input",
                type="number",
                placeholder="Entrez l'indice prix moyen",
                className="input-box"
            ),

            # Slider pour sélectionner les impressions (impression_gs)
            html.Br(),
            html.Label("Impressions"),
            dcc.Slider(
                id='impression-gs-slider',
                min=0,
                max=df['impression_gs'].max(),
                step=10000,
                tooltip={"placement": "bottom", "always_visible": True},
                marks=None,
                value=10000,
                className="slider"
            ),
            html.Div(id="impression-gs-output", className="output-box"),

            # Sélecteur pour afficher si le produit est en page d'accueil (on_front)
            html.Br(),
            html.Label("En Page d'Accueil"),
            dcc.Dropdown(
                id="on-front-dropdown",
                options=[
                    {'label': 'Oui', 'value': 1},
                    {'label': 'Non', 'value': 0}
                ],
                placeholder="Sélectionnez Oui ou Non",
                className="input-box"
            ),

            # Sélection de la date de lancement de l'opération (calendrier)
            html.Br(),
            html.Label("Date de Lancement de l'Opération"),
            dcc.DatePickerSingle(
                id='launch-date-picker',
                placeholder='Sélectionnez une date',
                display_format='YYYY-MM-DD',
                className="input-box"
            ),
    
            # Bouton pour prédire
            html.Br(),
            html.Button("Prédire", id="predict-button", n_clicks=0, className="predict-button"),
        ], className="sidebar"),
        
        # Contenu principal
        html.Div([
            html.Div([
                html.H2("Prévisions de Performance", className="section-title"),
                html.Div([
                    html.Div([
                        html.H3("En Opération Commerciale"),
                        html.Div(id="prediction-op", children="En attente...", className="prediction-value"),
                    ], className="prediction-box"),
                    html.Div([
                        html.H3("Hors Opération Commerciale"),
                        html.Div(id="prediction-no-op", children="En attente...", className="prediction-value"),
                    ], className="prediction-box"),
                ], className="predictions-container"),
            ], className="main-section"),
            
            html.Div([
                html.H2("Historique des Campagnes", className="section-title"),
                html.Div(id="campaign-history", className="campaign-history"),
            ], className="main-section"),
            
            html.Div([
                html.H2("Historique des Performances", className="section-title"),
                dcc.Graph(id="performance-graph"),
            ], className="main-section"),
        ], className="dashboard-content"),
    ], className="main-container"),
])

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
        # Vérifier que les valeurs sont non-nulles
        if not model_op or not product_id:
            return "Erreur: Modèle ou produit non sélectionné", "Erreur: Modèle ou produit non sélectionné"
        
        try:
            # Vérifiez que launch_date n'est pas None
            if launch_date is None:
                raise ValueError("La date de lancement est requise.")
        
            # Préfixer les valeurs du modèle pour correspondre aux clés du dictionnaire
            model_op_key = f"{model_op}_op"
            model_no_op_key = f"{model_op}_hors_op"

            # Vérifiez que les clés existent dans le dictionnaire models
            if model_op_key not in models or model_no_op_key not in models:
                return "Erreur: Modèle sélectionné inconnu", "Erreur: Modèle sélectionné inconnu"
            
            # Récupération de la catégorie pour le product_id sélectionné
            category_for_sku = get_category_for_sku(df, product_id)

            # Transformation des données d'entrée en DataFrame
            input_data = setup.load_data_model(
                df,
                product_id,
                avg_price,
                indice_avg_price,
                impression_gs,
                on_front,
                launch_date
            )

            # Prédictions
            prediction_op = models[model_op_key].predict(input_data)[0]
            prediction_no_op = models[model_no_op_key].predict(input_data)[0]

            return f"{prediction_op:.0f} nouveaux clients", f"{prediction_no_op:.0f} nouveaux clients"
        
        except Exception as e:
            return f"Erreur: {str(e)}", f"Erreur: {str(e)}"

    # Message par défaut lorsque le bouton n'a pas encore été cliqué
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
            # Historique des campagnes
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

            # Graphique de performance
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

                # Ajouter les prédictions si les valeurs sont disponibles
                if model_op and launch_date:
                    try:
                        # Préfixer les valeurs du modèle pour correspondre aux clés du dictionnaire
                        model_op_key = f"{model_op}_op"
                        model_no_op_key = f"{model_op}_hors_op"

                        # Vérifiez que les clés existent dans le dictionnaire models
                        if model_op_key in models and model_no_op_key in models:
                            # Transformation des données d'entrée en DataFrame
                            input_data = setup.load_data_model(
                                df,
                                product_id,
                                avg_price,
                                indice_avg_price,
                                impression_gs,
                                on_front,
                                launch_date
                            )

                            # Prédictions
                            prediction_op = models[model_op_key].predict(input_data)[0]
                            prediction_no_op = models[model_no_op_key].predict(input_data)[0]

                            # Ajouter les prédictions au graphique
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

                # Mise en page du graphique
                fig.update_layout(
                    barmode='group',
                    title='Performance du produit au fil du temps',
                    xaxis_title='Date',
                    yaxis_title='Nombre de clients',
                    xaxis_tickformat='%d<br>%B',
                    legend_title_text='Type de clients',
                    margin=dict(l=40, r=20, t=40, b=20),
                    height=200
                )
            else:
                fig = go.Figure()

            return campaign_history, fig
        
        except Exception as e:
            return ["Erreur dans l'historique des campagnes"], go.Figure()

    # Message par défaut lorsque le produit n'est pas sélectionné
    return ["En attente..."], go.Figure()


if __name__ == '__main__':
    app.run_server(debug=False)
