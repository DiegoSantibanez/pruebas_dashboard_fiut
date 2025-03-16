import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from sqlalchemy import create_engine

# Crear conexión usando SQLAlchemy
engine = create_engine('mariadb+mariadbconnector://testfiut:utem1234@localhost/mysql')

# Configuración de la página
st.set_page_config(
    page_title="Exploración datos FIUT",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
         'About': """
         # PROYECTO FIU UTEM 
         Dashboard creado por el equipo de integración de datos 
         - Diego Santibañez, dsantibanezo@utem.cl
         - Esteban Gomez, egomez@utem.cl
         - Hugo Osses, hosses@utem.cl
         """
        #'About': "# PROYECTO FIU UTEM"
    }
)





def main():
    # Aplicar estilo CSS personalizado para centrar imágenes en columnas
    st.markdown("""
    <style>
        /* Centrar contenido en las columnas */
        div[data-testid="column"] {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Columna izquierda para una imagen (con ruta corregida)
    with col1:
        st.image("imagenes/Ministerio de Ciencias color.png", width=150)

    # Columna derecha para otra imagen (con ruta corregida)
    with col2:
        st.image("imagenes/Isologo FIU UTEM color.png", width=400)

    st.title("Proyecto FIUT 2024 UTEM")

    df=pd.read_csv('data/estructura_archivos.csv')




# Ejecutar la aplicación
if __name__ == "__main__":
    main()