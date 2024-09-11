import pandas as pd

def load_data_empty():
    data = {
        'avg_price': [0], 
        'indice_avg_price': [0], 
        'impression_gs': [0], 
        'on_front': [0], 
        'day_of_week_number': [0], 
        'week_in_month': [0], 
        'Category_1_ANIMALERIE': [0], 
        'Category_1_AU-QUOTIDIEN': [0], 
        'Category_1_AUTO': [0], 
        'Category_1_BAGAGES': [0], 
        'Category_1_BRICOLAGE': [0], 
        'Category_1_CORNER': [0], 
        'Category_1_ELECTROMENAGER': [0], 
        'Category_1_HIGH-TECH': [0], 
        'Category_1_INFORMATIQUE': [0], 
        'Category_1_JARDIN': [0], 
        'Category_1_JEUX-PC-VIDEO-CONSOLE': [0], 
        'Category_1_JUNIORS': [0], 
        'Category_1_MAISON': [0], 
        'Category_1_MEUBLE': [0], 
        'Category_1_PRÊT-À-PORTER': [0], 
        'Category_1_PUÉRICULTURE': [0], 
        'Category_1_SPORT': [0], 
        'Category_1_TELEPHONIE': [0], 
        'Category_1_TV SON': [0], 
        'Category_1_VIN-CHAMPAGNE': [0]
    }

    # Créer le DataFrame à partir du dictionnaire

    df_vide = pd.DataFrame(data)
    '''
    sku_to_find = "WHIOWFC2C26X"
    date = '05/11/1993'
    prix_moyen = 298.35
    indice_prix = 93
    impression = 120000
    on_front = 1
    '''
    '''
    _sku_to_find,
                    _date,
                    _prix_moyen,
                    _indice_prix,
                    _impression,
                    on_front
    '''
    '''
    sku_to_find = _sku_to_find
    date = _date
    prix_moyen = _prix_moyen
    indice_prix = _indice_prix
    impression = _impression
    on_front = on_front
    '''
    return df_vide

def get_category_for_sku(df, _sku_to_find):
    category = df.loc[df['product_id'] == _sku_to_find, 'Category_1'].values
    if len(category) > 0:
        return category[0]
    else:
        return "SKU non trouvé"

def encode_category(data, category):
    # Créer une copie du DataFrame pour éviter de modifier l'original
    encoded_data = data.copy()
    
    # Mettre 1 dans la colonne correspondant à la catégorie spécifiée et 0 dans les autres colonnes de catégorie
    for col in encoded_data.columns:
        if col.startswith('Category_1_'):
            if col == 'Category_1_' + category:
                encoded_data[col] = 1
            else:
                encoded_data[col] = 0
    
    return encoded_data

# Supposons que tu as déjà un DataFrame df avec une colonne 'date_col' contenant les dates
def infos_date(_date):
    date = pd.to_datetime(_date, format='%Y-%m-%d')
    jour_semaine = date.dayofweek
    semaine_mois = (date.day - 1) // 7 + 1
    return jour_semaine, semaine_mois


def load_data_model(
            _df,
            _sku_to_find,
            _prix_moyen,
            _indice_prix,
            _impression,
            on_front,
            _date
            ):
    df_vide = load_data_empty()
    # Utilisation de la fonction avec votre DataFrame df1 et le SKU spécifié
    category_for_sku = get_category_for_sku(_df,_sku_to_find)
    # Utilisation de la fonction avec votre DataFrame X_train et la catégorie spécifiée 'AUTO'    
    df_plus_cat = encode_category(df_vide, category_for_sku)
    # Remplacer la colonne 'day_of_week_number' par le numéro du jour de la semaine pour une date donnée
    jour, semaine = infos_date(_date)
    df_plus_cat['day_of_week_number'] = jour
    df_plus_cat['week_in_month'] = semaine
    df_plus_cat['avg_price'] = [_prix_moyen] * len(df_plus_cat)  # Assign a single value to each row
    df_plus_cat['indice_avg_price'] = [_indice_prix] * len(df_plus_cat)  # Assign a single value to each row
    df_plus_cat['impression_gs'] = [_impression] * len(df_plus_cat)  # Assign a single value to each row
    df_plus_cat['on_front'] = [on_front] * len(df_plus_cat)
    return df_plus_cat
