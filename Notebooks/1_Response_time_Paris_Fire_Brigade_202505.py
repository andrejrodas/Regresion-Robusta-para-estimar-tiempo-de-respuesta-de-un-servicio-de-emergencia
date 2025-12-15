# %%
# <a href="https://colab.research.google.com/github/AnIsAsPe/Regresion-Robusta-para-estimar-tiempo-de-respuesta-de-un-servicio-de-emergencia/blob/main/Notebooks/1_Response_time_Paris_Fire_Brigade_202505.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
## Cargar bibiliotecas y funciones

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn import metrics
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler         # solamente para la comparación da la improtancia entre variables explicativas

from scipy.stats import kurtosis, skew

import matplotlib.pyplot as plt
import seaborn as sns

# %%
def correlacion(dataframe, grafica=True, matrix=True, tamaño =(6, 4)):
    '''
    Funcion para obtener la matriz de correlacion y visualizarla en mapa de calor.
    '''
    corr=dataframe.corr()
    if grafica==True:
      fig = plt.figure(figsize = tamaño)
      ax = sns.heatmap(corr,
                       vmin = -1,
                       vmax = 1,
                       center = 0,
                       cmap = "coolwarm",
                       annot = True,
                       fmt=".2f",
                       square = True)
      ax.set_xticklabels(ax.get_xticklabels(),
                         rotation = 45,
                         horizontalalignment ='right')
    if matrix==True:
      return corr

# %%
def describe_datos(df):
    """
    Función para describir un DataFrame de pandas.

    Devuelve:
    --------
    DataFrame
        Devuelve un DataFrame con la descripción de cada columna, incluyendo:
    (1) Tipo de columna, (2) Número de valores nulos, (3) Porcentaje de valores nulos
    (4) Número de valores únicos y (5) Valores únicos

    """

    unicos =[]
    for col in df:
        unicos.append(df[col].unique())
    unicos = pd.Series(unicos, index=df.columns)
    descripcion = pd.concat(
        [
            df.dtypes,
            df.isna().sum(),
            round(df.isna().sum()/len(df)*100, 1),
            df.nunique(),
            unicos
        ],
        axis=1
    )

    descripcion.columns = ['dtypes', 'null', '%null', 'nunique', 'unique']
    return descripcion

# %%
def round_lat_lon_columns(df, columns, digitos):
    """
    Redondea las columnas de latitud y longitud especificadas en un DataFrame.

    Parameters:
    df (pd.DataFrame): El DataFrame de entrada.
    columns (list): Una lista de nombres de columnas a redondear.
    digitos (int): El número de decimales a redondear

    Returns:
    pd.DataFrame: El DataFrame con las columnas redondeadas.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].round(digitos)
    return df

# %%
def haversine(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia Haversine entre dos puntos en la tierra.

    Parameteros:
    lat1 (pd.Series or float): Latitud de punto inicial en grados.
    lon1 (pd.Series or float): Longitud del punto inicial en grados.
    lat2 (pd.Series or float): Latitud del punto final en grados.
    lon2 (pd.Series or float): Longitud del punto final en grados.

    Devuelve:
    pd.Series or float: Haversine distance(s) in kilometers.
    """
    R = 6371  # Radius of Earth in kilometers

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance.round(4)

# %%
# Function to plot categorical data with relative frequency
def plot_categorical_relative_frequency(df, column, fig_size=(15,5)):
    counts = df[column].value_counts()
    plt.figure(figsize=fig_size)
    # Orden basado en la frecuencia
    order = counts.index
    counts.plot(kind='bar')
    plt.title(f'Categorías de {column} ')
    plt.xlabel(column)
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.show()

def plot_categorical_boxplot(df, column, fig_size=(15,5)):
    counts = df[column].value_counts()
    plt.figure(figsize=fig_size)
    # Get the order of categories based on frequency from the previous step
    order = df[column].value_counts().index.tolist()
    sns.boxplot(x=column, y='delta departure-presentation', data=df, showfliers=False, order=order)
    plt.title(f'Delta Departure-Presentation por {column} ')
    plt.xlabel(column)
    plt.ylabel('Delta Departure-Presentation')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.show()

# %% [markdown]
# Lectura datos y selección de características

# %% [markdown]
__Cargar etiquetas__

# %%
# Variables respuesta
Ys = pd.read_csv('drive/My Drive/Datos/ParisFireBrigade/y_train.csv',
                     index_col=[0], sep=',',)
print(Ys.shape)
Ys.head(2)

# %%
# Se puede verificar si la última columna es la suma de las otras dos
(Ys.iloc[:,0] + Ys.iloc[:,1] == Ys.iloc[:,2]).sum()==len(Ys)

# %% [markdown]
Nos vamos a concentrar en el tiempo desde la salida del vehículo hasta la presentación en el lugar del siniestro

# %% [markdown]
__Cargar características__

# %%
# Conjunto de caracteristicas (X)
X = pd.read_csv('drive/My Drive/Datos/ParisFireBrigade/x_train.csv',
                sep=',', index_col=[0], parse_dates=['selection time'] )
print(X.shape)
X.head(2)

# %%
describe_datos(X)

# %% [markdown]
La cantidad de intervenciones en los datos no es igual a la cantidad de vehículos, De manera que existen intervenciones que son atendidas por más de un vehiculo de emergencia

# %%
# Vamos a redondear a 4 decimales latitud y longitud (11 mts de aproximación) lo que es suficiente para ubicaciones especificasya que un GPS da coordadas demasiado precisas
# 6 decimales son excesivos (la precisión es de 11 cm), vamos a quedarnos con una precisión

lat_long_cols = ['latitude before departure', 'longitude before departure',
                 'latitude intervention', 'longitude intervention']

X = round_lat_lon_columns(X, lat_long_cols, 4 )


# %%
X.loc[5105452, 'OSRM response']

# %%
X['departed from its rescue center'].value_counts()

# %%
X['status preceding selection'].value_counts()

# %%
mask_in_rescue_center = X['departed from its rescue center']==1

display('Rescue Centers Unicos',
        X[mask_in_rescue_center]['rescue center'].nunique(),
        X[mask_in_rescue_center]['latitude before departure'].nunique(),
        X[mask_in_rescue_center]['longitude before departure'].nunique())

# %% [markdown]
No todos los veículos que salieron de un 'rescue center', salieron del que oficialemnte les correspondía, por tanto no se usará el 'campo rescue center' por ser más una variable administrativa que operacional.

# %%
# Borrar columnas que no utilizaremos

col_borrar = [
    'intervention',  #identificador de la intervención

    # Redundantes
    'date key sélection',
    'time key sélection',         # incluidas en 'selection time'
    'emergency vehicle',          # 749 categorias, resumidas en 75 'emergency vehicle type'
    'alert reason',               # 126 categorías, resumidas en 9 'alert reason category'
    'status preceding selection', # igual a 'departed from its rescue center'

    # Rutas
    'OSRM response',                              # ruta estimada
    'GPS tracks departure-presentation',          # posiciones gps en la ruta
    'GPS tracks datetime departure-presentation', # tiempo de medición pesisiones gps en la ruta

    # Variable no relaccionadas con delta departure-presentation
    'rescue center',               # id, variable administrativa más que operacional
    'delta status preceding selection-selection',
    'delta position gps previous departure-departure'
            ]
X = X.drop(columns= col_borrar)

X.info()

# %% [markdown]
__Unir caracteristicas con étiquetas__

# %%
df = pd.concat([X,  Ys['delta departure-presentation']], axis=1)
print(df.shape)

# %% [markdown]
# <h2> Idetificación del tipo de variable de cada columna
# 
# Comparamos el tipo de cada columna con la documentación de los parámetros de entrada en [la documentación provista por el Challenge](https://paris-fire-brigade.github.io/data-challenge/challenge.html)

# %%
categorical_cols = [
    'alert reason category','location of the event','emergency vehicle type',
    ]
booleanas_cols=['intervention on public roads', 'departed from its rescue center']
df[booleanas_cols]=df[booleanas_cols].astype('bool')
df[categorical_cols] = df[categorical_cols].astype('object')
describe_datos(df)

# %% [markdown]
# Creación de nuevas variables

# %% [markdown]
## A partir de 'selection time'

# %%
df['hour'] = (df['selection time'].dt.hour +           \
              df['selection time'].dt.minute / 60 +    \
              df['selection time'].dt.second / 3600)
df['day_of_week'] = df['selection time'].dt.day_name()
df['weekend'] = df['selection time'].dt.dayofweek > 4
df['month'] = df['selection time'].dt.month_name()

# %% [markdown]
Rush hour: variable binaria para identificar eventos en horas pico

# %%
df['rush hour'] = False

# Poner como indice 'selection time'
df = df.reset_index()
df = df.set_index('selection time')

# identificar el indice de los registros de eventos en hora pico
pico_am = df.index.indexer_between_time('6:30','9:30')
pico_pm = df.index.indexer_between_time('16:00','19:00')
horas_pico_index= np.concatenate((pico_am, pico_pm))

# cambiar el valor de 'rush hour' a uno cuando sea horas pico
df.iloc[list(horas_pico_index),-1] = True

# Regresar a 0 el valor para los días sabados y domingos
df['rush hour']= np.where(df['weekend']==True, False, df['rush hour'])

# Regresar emerency vehicle selection como indice
df = df.reset_index()
df = df.set_index('emergency vehicle selection')

# %%
France_hollidays = {
    '02/01/2018': "New Year's Day",
    '02/04/2018': "Easter Monday",
    '01/05/2018': "Labor Day",
    '08/05/2018': "WWII Victory Day",
    '10/05/2018': "Ascension Day",
    '21/05/2018': "Whit Monday",
    '14/07/2018': "Bastille Day",
    '15/08/2018': "Assumtion of Mary",
    '01/11/2018': "All' Saints' Day",
    '11/11/2018': "Armistice Day",
    '25/12/1973': "Christmas Day",
}
# https://www.timeanddate.com/holidays/france/2018?hol=9

# %%
# prompt: # prompt: create column in df based in 'selection time' column and the dictionary France_hollidays

df['is_holiday'] = df['selection time'].dt.strftime('%d/%m/%Y').isin(France_hollidays.keys())

# %%
df[['selection time', 'hour', 'day_of_week', 'month', 'rush hour','weekend', 'is_holiday']].head()

# %% [markdown]
## A partir de longitud, latitud y distancia OSMR
# 
# Primero calculamos la Distancia Haversine  partir de latitud y longitud de salida y llegada del vehiculo de emergencia.
# Despues anlizaremos la relación con la distancia OSMR
# 

# %%
import folium

# Creamos un mapa centrado en el promedio de las coordenadas
map_center = [df['latitude intervention'].mean(), df['longitude intervention'].mean()]
m = folium.Map(location=map_center, zoom_start=10)

# Para una muestra de 1000 registros marcamos en el mapa el punto de salida,
# de la intervención y los unimos con una linea
for _, row in df.sample(min(1000, len(df))).iterrows():
    # Localización de la emergencia
    folium.CircleMarker(
        location=[row['latitude intervention'], row['longitude intervention']],
        radius=1,
        color='blue',
        fill=True,
        fill_color='blue'
    ).add_to(m)

    # Punto de salida
    folium.CircleMarker(
        location=[row['latitude before departure'], row['longitude before departure']],
        radius=1,
        color='red',
        fill=True,
        fill_color='red'
    ).add_to(m)
    # Linea que conecta el punto de salida con el de la intervención
    folium.PolyLine(
        locations=[[row['latitude before departure'], row['longitude before departure']],
                   [row['latitude intervention'], row['longitude intervention']]],
        color='black',
        weight=1
    ).add_to(m)


# Display the map
m

# %%
df['haversine_distance_km'] = haversine(
    df['latitude intervention'],
    df['longitude intervention'],
    df['latitude before departure'],
    df['longitude before departure']
)

df.haversine_distance_km.describe()

# %%
plt.figure(figsize=(4, 3))
df.haversine_distance_km.hist(bins=50)
plt.xlabel('Haversine_distance (km)')
plt.ylabel('Frecuencia')
plt.show()

# %%
plt.figure(figsize=(4, 4))
plt.scatter(df['haversine_distance_km'], df['OSRM estimated distance']/1000, alpha=0.5)
plt.title('Haversine Distance vs. OSRM Estimated Distance')
plt.xlabel('Haversine Distance (km)')
plt.ylabel('OSRM Estimated Distance (Km)')
plt.grid(True)
plt.show()

correlation = df['haversine_distance_km'].corr(df['OSRM estimated distance'])
print(f"Correlación entre Haversine Distance y OSRM Estimated Distance: {correlation:.4f}")

# %%
# Calculamos la relación entre la distancia Real y a Vuelo
# Esta diferencia nos proporciona información sobre la complejidad y conectividad del camino.

# Es necesario unificar las unidades.

road_complexity_factor =  (df['OSRM estimated distance'] / 1000) / df['haversine_distance_km']
road_complexity_factor.describe()

# %%

def crea_row_complexity_category(df):
    """
    1. Calcula el Ratio de Complejidad (Distancia Real / Distancia Vuelo).
    2. Maneja divisiones por cero.
    3. Categoriza en Bajo/Medio/Alto para el modelo.
    """
    df = df.copy()

    # --- PASO 1: CÁLCULO DEL RATIO ---
    # Sumamos un epsilon (1e-6) al denominador para evitar divisiones por cero si origen == destino
    road_complexity_factor = (
        (df['OSRM estimated distance'] / 1000) /
        (df['haversine_distance_km'] + 1e-6)
    )

    # --- PASO 2: DEFINICIÓN DE REGLAS ---
    condiciones = [
        road_complexity_factor < 1.3,
        road_complexity_factor < 1.8
    ]

    etiquetas = ['Low_Complexity', 'Medium_Complexity']

    # --- PASO 3: CATEGORIZACIÓN ---
    # Si no cumple ninguna condición anterior (es decir, es >= 1.8), le pone 'High_Complexity'
    df['row_complexity_category'] = np.select(condiciones, etiquetas, default='High_Complexity')

    return df

# %%
df = crea_row_complexity_category(df)
df['row_complexity_category'].value_counts()

# %% [markdown]
## Puntos de salida
# Variable categorica que identifica las coordenadas de salida.

# %%
# Redondeamos latitud y longitud de salida a 4 dígitos (aproxima el lugar con un error de 110 mts)
lat_round = X['latitude before departure'].round(4)
lon_round = X['longitude before departure'].round(4)

# starting_point_aux = (lat_round.astype(str) + '_' + lon_round.astype(str))
# df['starting_point'] = np.where(df['departed from its rescue center'], starting_point_aux, 'Other')

df['starting_point']  = (lat_round.astype(str) + '_' + lon_round.astype(str))
df['starting_point'].value_counts()

# %%
df.drop(
    columns=['latitude intervention','longitude intervention',
             'latitude before departure','longitude before departure',
             'selection time'],
    inplace=True
)

# %% [markdown]
# Manejo de Valores Nulos

# %%
describe_datos(df)

# %%
df['location of the event'].value_counts(dropna=False)/len(df)*100

# %% [markdown]
# 
# **Alternativas**
# - Borrar esos renglones,
# - borrar esa variable,
# - imputar valores:
#    - ¿remplazar por ceros?
#    - ¿reemplazar por una medida de tendencia central?
#    - ¿a parir del conocimiento que nos proporcionan las demás variables?

# %%
moda = df['location of the event'].mode().values[0]
df['location of the event'] = df['location of the event'].infer_objects(copy=False).fillna(moda)
# regresar a tipo objeto, para distinguirla como variable categórica
df['location of the event'] = df['location of the event'].astype('int').astype('object')

# %%
df.isna().sum()

# %% [markdown]
# Exploración y selección de variables

# %% [markdown]
## Variables booleanas

# %%
df.info()

# %%
booleanas_cols = df.select_dtypes(include='bool').columns.tolist()
booleanas_cols

# %%
for column in booleanas_cols:
    fig, axes = plt.subplots(1, 2, figsize=(6, 3)) # Create a figure with 1 row and 2 columns

    # Plot relative frequency
    counts = df[column].value_counts()
    counts.plot(kind='bar', ax=axes[0])
    # axes[0].set_title(f'Categorías de {column} ')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Frecuencia')
    axes[0].tick_params(axis='x', rotation=45)


    # Plot boxplot
    order = counts.index.tolist() # Use the same order as the frequency plot
    sns.boxplot(x=column, y='delta departure-presentation', data=df, showfliers=False, order=order, ax=axes[1])
    # axes[1].set_title(f'Delta Departure-Presentation por {column} ')
    axes[1].set_xlabel(column)
    axes[1].set_ylabel('Delta Departure-Presentation')
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

# %% [markdown]
## Numéricas
# 
# Identificar y eliminar variables:
# 
# 1. **No relevantes**: Análisis de correlación de las explicativas(X) con la etiqueta (y)
# 
# 
# 2. **Redundantes**: Análisis de correlación entre las variables explicativas (X)

# %%
numericas_cols = list(df.select_dtypes(include=['int64', 'float64']).columns)
numericas_cols

# %%
g = sns.PairGrid(df[numericas_cols].sample(frac=.25), height= 1.8)
g.map(sns.scatterplot, alpha=0.3)
plt.show()

# %% [markdown]
# <h2> ¿Hay variables que no son relevantes para la predicción de la variable respuesta?

# %%
corr_y = df[numericas_cols].corr()['delta departure-presentation']
corr_y.sort_values()

# %%
no_relevantes = corr_y.loc[corr_y.abs()<0.1].index.to_list()
no_relevantes

# %% [markdown]
# <h2> ¿Hay variables redundantes entre las variables explicativas?

# %%
# correlacion de las características numéricas en el conjunto X
corr_matrix = correlacion(df[numericas_cols].drop('delta departure-presentation', axis=1))

# %%
#Identificar variables redundantes

# Triangulo superior de la matriz de correlación en números absolutos
celda_sobre_diagonal_ppal = np.triu(np.ones_like(corr_matrix), 1).astype(bool)
triangulo_sup = corr_matrix.where(celda_sobre_diagonal_ppal).abs()


# Encontrar las columnas donde la correlación es 1
redundantes = [column for column in triangulo_sup.columns
          if any(triangulo_sup[column] >= .7)]
redundantes

# %%
df = df.drop(redundantes + no_relevantes, axis=1)

# %%
df.columns

# %% [markdown]
##  Variables categóricas

# %%
categoricas_cols = df.select_dtypes(include='object').columns
df[categoricas_cols].nunique()

# %%
df['month'].unique()  # falta septiembre

# %%
categoricas_cols[:-1]  # excluye starting_point para gráfica

# %%
for col in categoricas_cols[:-1]:
    counts = df[col].value_counts()
    order = counts.index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5)) # Create a figure with 1 row and 2 columns

    # Plot relative frequency
    counts.plot(kind='bar', ax=axes[0])
    axes[0].set_title(f'Categorías de {col} ')
    axes[0].set_xlabel(col)
    axes[0].set_ylabel('Frecuencia')
    axes[0].tick_params(axis='x', rotation=45)

    # Plot boxplot
    sns.boxplot(x=col, y='delta departure-presentation', data=df, showfliers=False, order=order, ax=axes[1])
    axes[1].set_title(f'Delta Departure-Presentation por {col} ')
    axes[1].set_xlabel(col)
    axes[1].set_ylabel('Delta Departure-Presentation')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

# %% [markdown]
¿Vale la pena considerar todas las categorias para el one hot encodig?

# %%
len(df)*0.001

# %%
min_frec = len(df)*0.001

for col in categoricas_cols:
  counts = df[col].value_counts()
  cat_frecuentes = counts[counts >= min_frec]
  print(f"{len(cat_frecuentes):2} de {df[col].nunique():3} categorias cumplen con la frec minima en '{col}'")

# %%
encoder = OneHotEncoder(handle_unknown='ignore',
                        sparse_output=False,
                        min_frequency=.001,   # las categorías que cumplan la condición serán agurpadas en un solo grupo de categorías infrecuentes.
                        drop='first')
encoder.fit(df[categoricas_cols]).set_output(transform = 'pandas')
he = encoder.transform(df[categoricas_cols])
len(he.columns)

# %%
df.shape, he.shape

# %%
df = df.drop(columns=categoricas_cols)
df = pd.concat([df, he], axis=1)
#Llevar la etiqueta hasta la ultima posición
df = df[[c for c in df if c != 'delta departure-presentation'] + ['delta departure-presentation']]
df.shape

# %% [markdown]
# Modelo de Regresión

# %%
y = df['delta departure-presentation']
X = df.drop([ 'delta departure-presentation'], axis=1)

# %%
# Línea base (error medio absoluto)
ema_base = np.mean(np.absolute(y - y.mean()))
print(f"""
El error de predecir el promedio del tiempo de respuesta,
sin usar las caracteristicas explicativas (X) es {ema_base:.2f}
      """)

# %% [markdown]
## Regresión Lineal

# %% [markdown]
Evaluaremos al modelo de regresión líneal asegurandonos que el resultado sea robusto

# %%
from sklearn.model_selection import KFold, cross_validate, GridSearchCV

cv = KFold(n_splits=10, shuffle=True, random_state=42)
model = LinearRegression()
cv_results = cross_validate(
    estimator=model,
    X=X,
    y=y,
    cv=cv,
    return_train_score=True,
    return_estimator=True,
    scoring='neg_mean_absolute_error',  # el algoritmo de optimización que usa sklearn maximisa este valor.
)

# %%
# cambiamos el signo para obtener MAE
df_scores = pd.DataFrame({'train_score': -cv_results['train_score'],
                          'test_score': -cv_results['test_score']
                          })
df_scores

# %%
print("\nMean cross-validation MAE:", df_scores['test_score'].mean())
print("Standard deviation of cross-validation MAE:", df_scores['test_score'].std())

# %% [markdown]
# <h2> Análisis de errores

# %% [markdown]
# Tenemos dos tipos de errores:
# 
# 
#  1) Cuando la predicción es más baja que el valor real  $ ~~~~ y > \hat{y} $
# 
#  2) Cuando la predicción es más alta que el valor real $ ~~~~ y < \hat{y} $
# 

# %%
errores = []
for i, (train_index, test_index) in enumerate(cv.split(X, y)):
    X_test_fold = X.iloc[test_index]
    y_test_fold = y.iloc[test_index]

    model = cv_results['estimator'][i]
    y_pred_fold = model.predict(X_test_fold)
    errores_fold = (y_test_fold - y_pred_fold)
    errores.extend( errores_fold)

errores = np.array(errores)

len(errores), len (y)  # ahora tenemos un error de predicción para cada y

# %%
media = np.mean(errores)
des_est = np.std(errores)
errores_teoricos = np.random.normal(media, des_est, 219337)


fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey='all')
for dat, subplot in zip((errores_teoricos, errores), ax.flatten()):
    sns.histplot(x=dat, ax=subplot, kde=True )

plt.show()

# %%
print(f'kurtosis: {kurtosis(errores)}')

# %%
print(f'kurtosis: {kurtosis(errores_teoricos)}')

# %% [markdown]
# <font size="+1"><b> Notas para producción:</b></font>
# 
# Después de evaluar el modelo de regresión lineal, si quisieramos ponerlo en producción  ¿qué tendríamos que hacer si tenemos 10 modelos entrenados para cada una de las particiones?

# %% [markdown]
## Regresión Robusta

# %% [markdown]
## Búsqueda de hiperparámetros
# <p align="center">
# <img src="https://drive.google.com/uc?id=1Aly6q5OLZ6_blCHKJuKwxHQCQ2CKyFli
# " width="576" height="404">
# </p>
# 
# 
# 
# 
# 
# 

# %%
%%time
cv_grid = KFold(n_splits=5, shuffle=True, random_state=42)

model_hr = HuberRegressor(max_iter=1000,  alpha=0,  warm_start=True,
                          fit_intercept=False, tol=1e-05)

model_grid_search = GridSearchCV(estimator=model_hr,
                           param_grid={
                               'epsilon':[1.25, 1.35, 1.5]  # Se recomienda un rango entre 1 y 2
                           },
                           scoring='neg_mean_absolute_error',
                           cv= cv_grid,   # estrategia de separación del conjunto de datos
                           verbose=3,
                              , # Al final entrena el modelo con los mejores parametros en todos los datos
                           )
cv = KFold(n_splits=3, shuffle=True, random_state=6)
cv_results = cross_validate(
    estimator=model_grid_search,
    X=X,
    y=y,
    cv=cv,
    #return_train_score=True,
    return_estimator=True,
    scoring='neg_mean_absolute_error',  # el algoritmo de optimización que usa sklearn maximisa este valor.
)



# %%
for cv_fold, estimator_in_fold in enumerate(cv_results["estimator"]):
    print(
        f"Best hyperparameters for fold #{cv_fold + 1}:\n"
        f"{estimator_in_fold.best_params_}"
    )

# %%
df_scores = pd.DataFrame({'test_score': -cv_results['test_score']
                          })
df_scores

# %%
print("Mean cross-validation MAE:", df_scores['test_score'].mean())
print("Standard deviation of cross-validation MAE:", df_scores['test_score'].std())

# %%
errores = []
for i, (train_index, test_index) in enumerate(cv.split(X, y)):
    X_test_fold = X.iloc[test_index]
    y_test_fold = y.iloc[test_index]

    model = cv_results['estimator'][i]
    y_pred_fold = model.predict(X_test_fold)
    errores_fold = (y_test_fold - y_pred_fold)
    errores.extend( errores_fold)

errores = np.array(errores)

len(errores), len (y)  # ahora tenemos un error de predicción para cada y

# %%
media = np.mean(errores)
des_est = np.std(errores)
errores_teoricos = np.random.normal(media, des_est, 219337)


fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey='all')
for dat, subplot in zip((errores_teoricos, errores), ax.flatten()):
    sns.histplot(x=dat, ax=subplot, kde=True )

plt.show()

# %%
print(f'kurtosis: {kurtosis(errores)}')

# %%
# Para producción utilizaremos el mejor modelo con los hiperparametros optimizados
# entrenado con todos los datos etiquetados disponibles.

modelo_final = HuberRegressor(alpha=0, epsilon=1.35, fit_intercept=False, max_iter=1000,
               warm_start=True).fit(X, y)

# %%
coeficientes_df = pd.DataFrame({'Característica': X.columns, 'Coeficiente': modelo_final.coef_})
coeficientes_df.sort_values(by='Coeficiente', ascending=False, inplace=True)
coeficientes_df

# %% [markdown]
# Regresión Polinomial

# %%
X_estandarizados = X.copy()
columnas = X.columns

scaler = StandardScaler()

X_estandarizados[columnas] = scaler.fit_transform(X[columnas])
X_estandarizados.describe()

# %%
# Eliminar columnas relacionadas con 'location of the event' para disminuir la dimensionalidad


cols_out = [col for col in X.columns if col.startswith('starting_point') or    \
            col.startswith('location of the event')]
cols_to_keep = [col for col in X.columns if col not in cols_out]
len(cols_to_keep)

# %%
from sklearn.preprocessing import PolynomialFeatures

polynomial_features = PolynomialFeatures(degree=2, interaction_only=True).set_output(transform='pandas')
X_poly= polynomial_features.fit_transform(X_estandarizados[cols_to_keep])
X_poly.shape

# %%
Xpoly_train, Xpoly_test, y_train, y_test = train_test_split(X_poly, y,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    random_state=261)

# %%
reg_poli = LinearRegression().fit(Xpoly_train, y_train)

y_pred = reg_poli.predict(Xpoly_test)

error_medio_absoluto = np.mean(np.absolute(y_test - y_pred ))
error_medio_absoluto

# %% [markdown]
# Referencias
# 
# * García, S., Luengo, J., & Herrera, F. (2015). Data Preprocessing in Data Mining. Intelligent Systems Reference Library. doi:10.1007/978-3-319-10247-4
# 
# 
# * Hawkins, D. M. (1980). Identification of Outliers. Springer Netherlands. https://doi.org/10.1007/978-94-015-3994-4
# 
# 
# * McDonald, A. (2021, septiembre 8). Using the missing Python library to Identify and Visualise Missing Data Prior to Machine Learning. Medium. https://towardsdatascience.com/using-the-missingno-python-library-to-identify-and-visualise-missing-data-prior-to-machine-learning-34c8c5b5f009
# 
# * Chandrashekar, G., & Sahin, F. (2014). A survey on feature selection methods. Computers & Electrical Engineering, 40(1), 16-28. https://doi.org/10.1016/j.compeleceng.2013.11.024
# 
# 
# .
# 
# 


