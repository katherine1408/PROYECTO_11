
# # OilyGian

# * Trabajas en la compañía de extracción de petróleo OilyGiant. Tu tarea es encontrar los mejores lugares donde abrir 200 pozos nuevos de petróleo.  
# * Tienes datos sobre muestras de crudo de tres regiones. Ya se conocen los parámetros de cada pozo petrolero de la región. Crea un modelo que ayude a elegir la región con el mayor margen de beneficio.

# ## Análisis exploratorio de datos (Python):

# ### Inicialización:

# In[1]:


# Importamos las librerías necesarias :
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import bootstrap




# ### Cargar Datos:

# In[2]:


# Cargar los datos:

geo_0 = pd.read_csv('geo_data_0.csv')
geo_1 = pd.read_csv('geo_data_1.csv')
geo_2 = pd.read_csv('geo_data_2.csv')


# In[3]:


# Verificamos la estructura y las primeras filas de cada archivo:

display(geo_0.head())
display(geo_1.head())
display(geo_2.head())


# ### Estudiar los datos que contienen:

# In[4]:


geo_0.info()
geo_1.info()
geo_2.info()


# In[5]:


geo_0.describe()


# In[6]:


geo_1.describe()


# In[7]:


geo_2.describe()


# ## Preparar los datos:

# ### Revisión de datos duplicados:

# In[8]:


geo_0.duplicated().sum()


# In[9]:


geo_1.duplicated().sum()


# In[10]:


geo_2.duplicated().sum()


# ### Revisión de datos nulos:

# In[11]:


geo_0.isna().sum()


# In[12]:


geo_1.isna().sum()


# In[13]:


geo_2.isna().sum()


# In[14]:


state = np.random.RandomState(54321)




# ## Analisis de Datos:

# ### Segmentación de datos:

# Train-Test Split: Utilicé la función train_test_split() para dividir los datos en conjuntos de entrenamiento y prueba, con el 75% de los datos para el entrenamiento y el 25% para la prueba

# In[15]:


# Función para preparar los datos y dividirlos en conjunto de entrenamiento y validación:

def preparar_datos (data):
    features = data[['f0', 'f1', 'f2']]
    target = data['product']
    return train_test_split(features, target, test_size=0.25, random_state=state)


# ### Entrenamiento (Modelo de Regresión):

# In[16]:


# Función para entrenar el modelo de regresión lineal y predecir:

def entrenar_modelo(features_train, features_valid, target_train, target_valid):
    model = LinearRegression()
    model.fit(features_train, target_train)
    predictions = model.predict(features_valid)
    mse = mean_squared_error(target_valid, predictions)  # Sin el argumento 'squared'
    rmse = np.sqrt(mse)  # Calcular la raíz cuadrada manualmente
    return predictions, rmse


# ### Evaluación de Modelo:

# In[17]:


# Función para calcular el volumen promedio de reservas:

def evaluar_modelo(target_valid, predictions):
    promedio_reservas = predictions.mean()
    return promedio_reservas


# ### Ejecución de Modelo para cada región:

# In[18]:


# Ejecutar modelo para cada región:

for i, data in enumerate([geo_0, geo_1, geo_2]):
    features_train, features_valid, target_train, target_valid = preparar_datos(data)
    predictions, rmse = entrenar_modelo(features_train, features_valid, target_train, target_valid)
    promedio_reservas = evaluar_modelo(target_valid, predictions)
    
    #print(f"Región {i}:")
    #print(f" - RMSE: {rmse}")
    #print(f" - Promedio de reservas: {promedio_reservas}")
    print(f"Región {i}: RMSE = {rmse:.2f}, Volumen medio de reservas predicho = {promedio_reservas:.2f}")


# ### Analisis: (Por cada región)

# 
# * Región 0
# 
# RMSE = 37.52: Este valor de RMSE es alto, lo que indica que el modelo tiene las predicciones no muy precisas, lo que puede agregar incertidumbre al cálculo de beneficios potenciales.
# Volumen medio de reservas = 92.53: Este volumen está por debajo del umbral mínimo de rentabilidad (111.1 unidades), lo que implica un riesgo de inversión.
# Aunque el volumen medio de reservas es relativamente alto en comparación con la Región 1, su imprecisión en la predicción (alto RMSE) y el volumen bajo respecto al umbral hacen que esta región no sea muy atractiva.  
# 
# * Región 1
# 
# RMSE = 0.89: Este es un valor extremadamente bajo, lo cual indica una excelente precisión del modelo. La baja variabilidad sugiere que el modelo predice las reservas con mucha confianza en esta región.
# Volumen medio de reservas = 69.26: Este volumen es considerablemente inferior al umbral de 111.1 unidades, lo que indica que,  los pozos en esta región generarían menos de los ingresos necesarios para evitar pérdidas.
# Aunque la precisión del modelo es excelente, el volumen de reservas es muy bajo, lo que implica que los pozos en esta región probablemente no sean rentables.  
# 
# * Región 2
# 
# RMSE = 40.00: Este RMSE es el más alto entre las tres regiones, lo cual significa que las predicciones tienen una gran dispersión respecto a los valores reales. Esto sugiere un alto nivel de incertidumbre en la predicción del volumen de reservas.
# Volumen medio de reservas = 95.17: Aunque es el más alto entre las tres regiones, este volumen todavía se encuentra por debajo del umbral de rentabilidad. Sin embargo, está más cerca del umbral de 111.1 unidades en comparación con las otras regiones.
# Aunque la precisión es baja, la Región 2 tiene el volumen medio de reservas más alto entre las tres.



# ## Determinación de las ganancias :

# ###  Almacenar los valores necesarios:

# In[19]:


# Parámetros financieros:

presupuesto = 100000000
pozos_a_seleccionar = 200
ingreso_por_unidad = 4500  # en miles de barriles
min_unidades_sin_perdidas = 111.1  # Equivalente a $500,000


# In[20]:


# Cálculo de beneficio por región:

def calcular_ganancia(predicciones):
    mejores_pozos = np.sort(predicciones)[-pozos_a_seleccionar:]
    ganancia_total = mejores_pozos.sum() * ingreso_por_unidad
    return ganancia_total


# In[21]:


# Calcular y comparar ganancias potenciales:

ganancias_regiones = []

for i, data in enumerate([geo_0, geo_1, geo_2]):
    _, features_valid, _, target_valid = preparar_datos(data)
    predictions, _ = entrenar_modelo(features_train, features_valid, target_train, target_valid)
    ganancia = calcular_ganancia(predictions)
    ganancias_regiones.append((ganancia, i))


# In[22]:


# Selección de la región más rentable
mejor_region = max(ganancias_regiones, key=lambda x: x[0])
print(f"La mejor región es la Región {mejor_region[1]} con una ganancia estimada de {mejor_region[0]:,.2f} USD")


# In[23]:


# Función de bootstrapping para evaluar el riesgo de pérdidas:

def bootstrap_analisis(predicciones, n_simulaciones=1000):
    bootstrap_res = bootstrap((predicciones,), np.mean, confidence_level=0.95, n_resamples=n_simulaciones, method='basic')
    beneficio_promedio = bootstrap_res.confidence_interval[0] * ingreso_por_unidad
    perdida_probabilidad = (predicciones < 111.1).mean()
    return beneficio_promedio, perdida_probabilidad


# In[24]:


# Aplicación de análisis de riesgo para cada región
for i, data in enumerate([geo_0, geo_1, geo_2]):
    _, features_valid, _, target_valid = preparar_datos(data)
    predictions, _ = entrenar_modelo(features_train, features_valid, target_train, target_valid)
    beneficio_promedio, perdida_probabilidad = bootstrap_analisis(predictions)
    print(f"Región {i}:")
    print(f" - Beneficio Promedio: {beneficio_promedio}")
    print(f" - Probabilidad de pérdida: {perdida_probabilidad:.2%}")



