# 🛢️ Sprint 11 – Selección Óptima de Pozos Petroleros con Machine Learning (OilyGiant)

## 📌 Descripción del Proyecto

Este proyecto aplica machine learning en un contexto de negocio realista. Trabajamos con **datos geológicos sintéticos** de tres regiones para determinar cuál es la mejor zona para abrir **200 nuevos pozos petroleros**. 

Como analista de datos en **OilyGiant**, tu objetivo es crear un modelo de regresión lineal para estimar la producción de cada pozo y luego usar técnicas estadísticas como **bootstrapping** para cuantificar el **riesgo** y la **rentabilidad esperada** de invertir en cada región.

## 🎯 Propósito

- Entrenar un modelo de regresión lineal para predecir reservas petroleras.
- Seleccionar los 200 pozos más prometedores en cada región.
- Calcular el beneficio estimado con un límite presupuestario de $100 millones.
- Evaluar el **riesgo de pérdidas** mediante **bootstrapping (1,000 simulaciones)**.
- Elegir la región con el **mayor beneficio promedio** y riesgo **menor al 2.5%**.

## 📁 Datasets utilizados

- `geo_data_0.csv`
- `geo_data_1.csv`
- `geo_data_2.csv`

Columnas:

- `id`: identificador único del pozo
- `f0`, `f1`, `f2`: características geológicas
- `product`: volumen de reservas en miles de barriles

## 🧰 Funcionalidades del Proyecto

### 🧪 Fase 1: Modelado

- División 75% entrenamiento / 25% validación
- Entrenamiento con **regresión lineal** en cada región
- Cálculo de métricas: RMSE y volumen medio predicho

### 💰 Fase 2: Cálculo de beneficio

- Ingreso por unidad: $4,500
- Selección de los 200 pozos con predicción más alta
- Cálculo de ganancia neta por región (presupuesto: $100M)

### 📉 Fase 3: Evaluación de riesgo

- **Bootstrapping** con 1000 simulaciones por región
- Cálculo de:
  - Beneficio promedio
  - Intervalo de confianza del 95%
  - Riesgo de pérdidas (p < 2.5%)

## 🛠️ Herramientas utilizadas

- Python  
- pandas  
- scikit-learn (`LinearRegression`)  
- numpy  
- matplotlib / seaborn  

---

📌 Proyecto desarrollado como parte del Sprint 11 del programa de Ciencia de Datos en **TripleTen**.
