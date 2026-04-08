import streamlit as st
import pandas as pd
from joblib import load

# Configuramos la página y aspecto general
st.set_page_config(page_title="Predictor de Clientes Bancarios", page_icon="🏦", layout="wide")
st.title("Herramienta de Predicción de Suscripciones")
st.info("Sistema de Machine Learning para clasificar si un cliente contratará el producto bancario.")

@st.cache_resource
def inicializar_modelo():
    return load("modelo_produccion.joblib")

try:
    diccionario_modelo = inicializar_modelo()
except Exception as error_carga:
    st.error(f"Hubo un problema al cargar el archivo del modelo: {error_carga}")
    st.stop()

# Extraemos los datos del paquete
modelo_pipeline = diccionario_modelo["pipeline"]
info_variables = diccionario_modelo["feature_metadata"]
clases_modelo = diccionario_modelo.get("classes_", [])

st.write("---")
st.subheader("📋 Introduce el perfil del nuevo cliente:")

# Formulario 
with st.form("formulario_cliente"):
    entradas_usuario = {}
    
    st.markdown("**Datos Cuantitativos (Numéricos)**")
    columnas_num = st.columns(3) 
    indice_num = 0
    
    for nombre_var, detalles in info_variables.items():
        if detalles["type"] == "numerical":
            with columnas_num[indice_num % 3]:
                valor_medio = float(detalles.get("median", 0.0))
                entradas_usuario[nombre_var] = st.number_input(
                    label=nombre_var.capitalize(), # Ponemos la primera letra en mayúscula
                    min_value=float(detalles.get("min", -1e9)),
                    max_value=float(detalles.get("max", 1e9)),
                    value=valor_medio,
                    step=1.0 if valor_medio.is_integer() else 0.1
                )
            indice_num += 1
            
    st.markdown("<br>**Datos Cualitativos (Categóricos)**", unsafe_allow_html=True)
    columnas_cat = st.columns(3)
    indice_cat = 0
    
    for nombre_var, detalles in info_variables.items():
        if detalles["type"] == "categorical":
            with columnas_cat[indice_cat % 3]:
                opciones = detalles.get("options", [])
                entradas_usuario[nombre_var] = st.selectbox(
                    label=nombre_var.capitalize(),
                    options=opciones,
                    index=0 if opciones else None
                )
            indice_cat += 1
            
    st.write("") # Espacio en blanco
    
    # Botón principal
    boton_enviar = st.form_submit_button("Lanzar Predicción", type="primary", use_container_width=True)

if boton_enviar:
    df_nuevo_cliente = pd.DataFrame([entradas_usuario])
    
    try:
        # Hacemos la predicción principal
        prediccion_final = modelo_pipeline.predict(df_nuevo_cliente)[0]
        
        # Mostramos un mensaje con el resultado
        if str(prediccion_final).lower() in ["yes", "1", "sí", "si"]:
            st.success(f"🎯 **Resultado:** El cliente **SÍ** parece propenso a suscribirse (Clase predicha: {prediccion_final})")
        else:
            st.warning(f"🛑 **Resultado:** El cliente **NO** parece propenso a suscribirse (Clase predicha: {prediccion_final})")
        
        try:
            probabilidades = modelo_pipeline.predict_proba(df_nuevo_cliente)[0]
            st.markdown("#### Nivel de Confianza:")
            
            cols_metricas = st.columns(len(clases_modelo))
            for idx, (clase, prob) in enumerate(zip(clases_modelo, probabilidades)):
                cols_metricas[idx].metric(label=f"Clase '{clase}'", value=f"{prob*100:.1f}%")
        except Exception:
            # Por si el modelo dice tener 'predict_proba' pero falla internamente
            st.info("💡 El modelo configurado ofrece clasificación directa.")
            
    except Exception as error_pred:
        st.error(f"Algo falló al intentar realizar la predicción: {error_pred}")