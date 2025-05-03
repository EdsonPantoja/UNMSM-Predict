from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Cargar modelo
modelo = joblib.load('mejor_modelo_entrenado.pkl')

# Diccionario de tasas por carrera
TASA_INGRESO_DICT = {
    'ARQUITECTURA': 0.1,
    'INGENIERÍA': 0.2,
    'MEDICINA': 0.05,
    'DERECHO': 0.15
    # Puedes agregar más carreras y sus tasas aquí
}

@app.route('/')
def index():
    return render_template('formulario.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    carrera = request.form['CARRERA'].upper()
    tasa = TASA_INGRESO_DICT.get(carrera, 0.0)  # valor por defecto 0.0 si no está en el dict

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

    return f'''
        <h2>Resultado: {pred}</h2>
        <p>Confianza: {round(proba * 100, 2)}%</p>
        <p><b>Tasa de ingreso aplicada:</b> {tasa}</p>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
