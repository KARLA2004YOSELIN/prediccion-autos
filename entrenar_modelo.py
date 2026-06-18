# entrenar_modelo.py
# Entrena el modelo final (RandomForest) con las características seleccionadas
# en el notebook Regresion_Processes2.ipynb y lo exporta con joblib como .pkl

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder

# 1. Cargar el dataset
df = pd.read_csv('processes2.csv', index_col=0)

# 2. No hay valores nulos en este dataset (verificado en el notebook), no se trata nada

# 3. Codificar la columna de texto 'name' (marca) a numérico con OrdinalEncoder
#    Se entrena el encoder solo con esta columna para poder reutilizarlo en producción
oe_name = OrdinalEncoder(dtype=int)
df['name'] = oe_name.fit_transform(df[['name']])

# Guardamos el orden de las categorías para usarlo en el formulario / app.py
print("Marcas codificadas (orden del OrdinalEncoder):")
print(list(oe_name.categories_[0]))

# 4. Selección de características (las mismas que en el notebook: importancia > 0.01)
caracteristicas_sel = ['max_power (in bph)', 'year', 'Engine (CC)', 'km_driven', 'Mileage', 'name']

X = df[caracteristicas_sel]
y = df['selling_price']

# 5. Entrenar el modelo final con el 100% de los datos (para producción)
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X, y)

# 6. Exportar el modelo entrenado y el encoder de marcas con joblib
joblib.dump(modelo, 'modelo_processes2.pkl')
joblib.dump(oe_name, 'encoder_name.pkl')

print("\nModelo y encoder exportados correctamente:")
print(" - modelo_processes2.pkl")
print(" - encoder_name.pkl")
