from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

# Cargar modelo
modelo = joblib.load('mejor_modelo_entrenado.pkl')

# Diccionario de tasas por carrera
from tasas_ingreso import TASA_INGRESO_DICT

app = Flask(__name__)

@app.route('/')
def index():
    carreras = sorted(TASA_INGRESO_DICT.keys())  # Lista ordenada
    return render_template('formulario.html', carreras=carreras)

@app.route('/predecir', methods=['POST'])
def predecir():
    carrera = request.form['CARRERA'].upper()
    tasa = TASA_INGRESO_DICT.get(carrera, 0.0)

    datos = {
        'CARRERA': [carrera],
        'SEXO': [request.form['SEXO']],
        'COLEGIO_PROCENDENCIA': [request.form['COLEGIO_PROCENDENCIA']],
        'AÑO_PERIODO': [request.form['AÑO_PERIODO']],
        'PUNTAJE': [int(request.form['PUNTAJE'])],
        'EDAD': [int(request.form['EDAD'])],
        'NRO_POSTULACION': [int(request.form['NRO_POSTULACION'])],
        'TASA_INGRESO': [tasa]
    }

    df = pd.DataFrame(datos)
    pred = modelo.predict(df)[0]
    proba = modelo.predict_proba(df)[0].max()

    resultado = "INGRESAS" if pred == 1 else "NO INGRESAS"

    return f'''
        <h2>Resultado: {resultado}</h2>
        <p>Confianza: {round(proba * 100, 2)}%</p>
        <form action="/" method="get">
            <button type="submit">Volver a predecir</button>
        </form>
    '''

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
