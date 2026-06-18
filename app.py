from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado y el encoder de marcas
modelo = joblib.load('modelo_processes2.pkl')
encoder_name = joblib.load('encoder_name.pkl')
app.logger.debug('Modelo y encoder cargados correctamente.')

# Lista de marcas en el mismo orden que usó el OrdinalEncoder al entrenar
MARCAS = list(encoder_name.categories_[0])


@app.route('/')
def home():
    return render_template('formulario.html', marcas=MARCAS)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        marca = request.form['marca']
        year = float(request.form['year'])
        km_driven = float(request.form['km_driven'])
        max_power = float(request.form['max_power'])
        mileage = float(request.form['mileage'])
        engine_cc = float(request.form['engine_cc'])

        # Codificar la marca con el mismo encoder usado en el entrenamiento
        marca_df = pd.DataFrame([[marca]], columns=['name'])
        marca_codificada = encoder_name.transform(marca_df)[0][0]

        # Crear un DataFrame con los datos, en el mismo orden de columnas
        # usado al entrenar el modelo: ['max_power (in bph)', 'year',
        # 'Engine (CC)', 'km_driven', 'Mileage', 'name']
        data_df = pd.DataFrame(
            [[max_power, year, engine_cc, km_driven, mileage, marca_codificada]],
            columns=['max_power (in bph)', 'year', 'Engine (CC)', 'km_driven', 'Mileage', 'name']
        )
        app.logger.debug(f'DataFrame creado: {data_df}')

        # Realizar la predicción
        prediction = modelo.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')

        # Devolver la predicción como respuesta JSON
        return jsonify({'precio_estimado': round(float(prediction[0]), 2)})

    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
