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

# Función para cargar los datos
@st.cache_data
def cargar_datos(ruta='data/estructura_archivos.csv'):
    """Carga los datos del archivo CSV o genera un DataFrame vacío si no existe"""
    try:
        df = pd.read_csv(ruta)
        return df
    except FileNotFoundError:
        st.error(f"Archivo {ruta} no encontrado. Por favor ejecuta primero el script de generación.")
        return pd.DataFrame()

# Función para procesar y limpiar los datos
def procesar_datos(df):
    """Procesa y limpia los datos para el análisis"""
    if df.empty:
        return df
    
    # Filtrar solo archivos (no directorios)
    df = df[df['tipo'] == 'Archivo'].copy()
    
    # Eliminar filas con extensión vacía
    df = df[df['extension'] != ''].copy()
    
    # Eliminar archivos .ipynb
    df = df[df['extension'] != '.ipynb'].copy()
    
    # Extraer dimensión de la ruta
    dims = []
    for ruta in df['ruta_relativa']:
        dim_encontrada = False
        for i in range(1, 8):
            dim_str = f"Dimensión {i}"
            if dim_str in ruta:
                dims.append(dim_str)
                dim_encontrada = True
                break
        if not dim_encontrada:
            dims.append('Sin clasificación')
    
    df['dimensiones'] = dims
    
    # Verificar columnas institucional/territorial
    if 'institucional' not in df.columns or 'territorial' not in df.columns:
        inst = []
        terr = []
        for ruta in df['ruta_relativa']:
            partes = ruta.split('\\')
            inst.append(partes[0] == 'Institucional')
            terr.append(partes[0] == 'Territorial')
        
        df['institucional'] = inst
        df['territorial'] = terr
    
    return df.reset_index(drop=True)

# Función para crear gráfico de barras institucional vs territorial
def crear_grafico_institucional_territorial(df):
    conteo = {
        'Institucional': df['institucional'].sum(),
        'Territorial': df['territorial'].sum()
    }
    
    fig = go.Figure([
        go.Bar(
            x=list(conteo.keys()),
            y=list(conteo.values()),
            marker_color=['#0A5C99', '#FEC109'],  # Cambiado a azul oscuro y amarillo
            text=list(conteo.values()),
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Distribución de Archivos por Categoría',
        yaxis_title='Número de Archivos',
        template='plotly_white',
        height=400
    )
    
    return fig

# Función para crear gráfico de distribución de extensiones
def crear_grafico_extensiones(df, filtro=None):
    # Aplicar filtro si es necesario
    if filtro == 'institucional':
        df_temp = df[df['institucional'] == True]
        titulo = 'Distribución de Tipos de Archivos - Institucional'
    elif filtro == 'territorial':
        df_temp = df[df['territorial'] == True]
        titulo = 'Distribución de Tipos de Archivos - Territorial'
    else:
        df_temp = df
        titulo = 'Distribución de Tipos de Archivos - Global'
    
    # Contar extensiones
    conteo_extensiones = df_temp['extension'].value_counts().reset_index()
    conteo_extensiones.columns = ['extension', 'conteo']
    
    # Calcular porcentaje
    total = conteo_extensiones['conteo'].sum()
    conteo_extensiones['porcentaje'] = (conteo_extensiones['conteo'] / total * 100).round(1)
    
    # Clasificar como "pequeña" si es menor al threshold
    threshold = 5
    conteo_extensiones['tamaño'] = ['pequeña' if p < threshold else 'normal' for p in conteo_extensiones['porcentaje']]
    
    # Crear gráfico con la nueva paleta de colores
    fig = px.pie(
        conteo_extensiones, 
        values='conteo', 
        names='extension',
        title=titulo,
        hole=0.3,
        color_discrete_sequence=['#0A5C99', '#1E88E5', '#FEC109', '#FC9F0B']  # Nueva paleta personalizada
    )
    
    # Configurar texto
    fig.update_traces(
        textposition=["outside" if t == "pequeña" else "inside" for t in conteo_extensiones['tamaño']],
        textinfo="percent+label",
        textfont_size=12,
        pull=[0.05 if t == "pequeña" else 0 for t in conteo_extensiones['tamaño']]
    )
    
    # Diseño
    fig.update_layout(
        template='plotly_white', 
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

# Función para crear gráfico de distribución por dimensiones
def crear_grafico_dimensiones(df, filtro=None):
    # Aplicar filtro si es necesario
    if filtro == 'institucional':
        df_temp = df[df['institucional'] == True]
        titulo = 'Distribución por Dimensiones - Institucional'
    elif filtro == 'territorial':
        df_temp = df[df['territorial'] == True]
        titulo = 'Distribución por Dimensiones - Territorial'
    else:
        df_temp = df
        titulo = 'Distribución por Dimensiones - Global'
    
    # Filtrar solo dimensiones clasificadas
    df_temp = df_temp[df_temp['dimensiones'] != 'Sin clasificación'].copy()
    
    # Si no hay datos, devolver mensaje de error
    if df_temp.empty:
        return None
    
    # Contar dimensiones
    conteo_dimensiones = df_temp['dimensiones'].value_counts().reset_index()
    conteo_dimensiones.columns = ['dimension', 'conteo']
    
    # Ordenar por nombre de dimensión
    conteo_dimensiones = conteo_dimensiones.sort_values('dimension')
    
    # Crear gráfico con la nueva paleta de colores
    fig = px.pie(
        conteo_dimensiones, 
        values='conteo', 
        names='dimension',
        title=titulo,
        hole=0.3,
        color_discrete_sequence=['#0A5C99', '#1E88E5', '#FEC109', '#FC9F0B']  # Nueva paleta personalizada
    )
    
    # Configurar texto
    fig.update_traces(
        textposition='auto',
        textinfo="percent+label",
        textfont_size=12
    )
    
    # Diseño
    fig.update_layout(
        template='plotly_white', 
        height=500
    )
    
    return fig

# Función para crear gráfico comparativo de extensiones por categoría
def crear_grafico_comparativo_extensiones(df):
    # Obtener top 5 extensiones
    top_ext = df['extension'].value_counts().head(5).index.tolist()
    
    # Filtrar dataframe
    df_inst = df[df['institucional'] == True]
    df_terr = df[df['territorial'] == True]
    
    # Contar extensiones por categoría
    ext_inst = df_inst[df_inst['extension'].isin(top_ext)]['extension'].value_counts()
    ext_terr = df_terr[df_terr['extension'].isin(top_ext)]['extension'].value_counts()
    
    # Completar valores faltantes con ceros
    for ext in top_ext:
        if ext not in ext_inst:
            ext_inst[ext] = 0
        if ext not in ext_terr:
            ext_terr[ext] = 0
    
    # Ordenar por el total
    total_ext = ext_inst + ext_terr
    orden = total_ext.sort_values(ascending=False).index
    
    # Crear figura con subplots
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=("Tipos de archivos Institucionales", "Tipos de archivos Territoriales"),
                        specs=[[{"type": "pie"}, {"type": "pie"}]])
    
    # Añadir gráficos de pastel con nuevos colores
    fig.add_trace(
        go.Pie(
            labels=orden,
            values=[ext_inst[ext] for ext in orden],
            name="Institucional",
            hole=0.4,
            marker=dict(colors=['#0A5C99', '#1E88E5', '#FEC109', '#FC9F0B'])  # Nueva paleta personalizada
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Pie(
            labels=orden,
            values=[ext_terr[ext] for ext in orden],
            name="Territorial",
            hole=0.4,
            marker=dict(colors=['#0A5C99', '#1E88E5', '#FEC109', '#FC9F0B'])  # Nueva paleta personalizada
        ),
        row=1, col=2
    )
    
    # Actualizar diseño
    fig.update_layout(
        title_text="Comparación de Tipos de Archivos por Categoría",
        height=500,
        template="plotly_white"
    )
    
    return fig

# Función para crear heatmap de extensiones por dimensión
def crear_heatmap_extension_dimension(df):
    # Obtener top 6 extensiones
    top_ext = df['extension'].value_counts().head(6).index.tolist()
    
    # Filtrar dataframe
    df_filt = df[(df['extension'].isin(top_ext)) & (df['dimensiones'] != 'Sin clasificación')]
    
    if df_filt.empty:
        return None
    
    # Crear tabla pivote
    pivot = pd.pivot_table(
        df_filt,
        values='nombre',
        index='extension',
        columns='dimensiones',
        aggfunc='count',
        fill_value=0
    )
    
    # Crear heatmap con la paleta personalizada
    # Para heatmaps es mejor usar una escala de un solo color, así que usamos azules
    fig = px.imshow(
        pivot,
        labels=dict(x="Dimensión", y="Extensión", color="Cantidad"),
        x=pivot.columns,
        y=pivot.index,
        color_continuous_scale=[[0, '#E3F2FD'], [0.5, '#1E88E5'], [1, '#0A5C99']],  # Escala de azules de la paleta
        title='Distribución de Tipos de Archivos por Dimensión'
    )
    
    # Añadir valores en las celdas
    annotations = []
    for i, ext in enumerate(pivot.index):
        for j, dim in enumerate(pivot.columns):
            annotations.append(dict(
                x=dim, y=ext,
                text=str(pivot.loc[ext, dim]),
                showarrow=False,
                font=dict(color='white' if pivot.loc[ext, dim] > pivot.values.max()/2 else 'black')
            ))
    
    fig.update_layout(annotations=annotations, height=450)
    
    return fig

# Función para crear gráfico de métodos de obtención
def crear_grafico_metodos_obtencion():
    
    dfh=pd.read_excel('data/DataLake_registro_FIUT_UTEM.xlsx')
    dfh['METODO'].value_counts()

    dfhh={
        'nombres':[], 
        'conteo':[]
    }
    for i,j in dfh['METODO'].value_counts().items():
        dfhh['nombres'].append(i)
        dfhh['conteo'].append(j)
    dfhh=pd.DataFrame(dfhh)

    dfhh['nombres'][0]= 'Web Scrapping'
    dfhh['nombres'][1]= 'Universidad'
    dfhh['nombres'][2]= 'Descargados'
    
    fig=px.pie(
    dfhh, 
    values='conteo', 
    names='nombres', 
    title='Distrubución métodos de obtención de los archivos',
    color_discrete_sequence=['#0A5C99', '#1E88E5', '#FEC109'],  # Nueva paleta personalizada
    hole=0.3,  # Para hacer un gráfico de dona
    )

    # Configurar texto con posiciones adaptativas
    fig.update_traces(
        textposition='auto',  # 'auto' ajusta la posición automáticamente
        textinfo='percent+label',  # Muestra porcentaje y etiqueta
        textfont_size=12,  # Tamaño de texto más grande
        rotation=270
    )

    # Mejorar el diseño
    fig.update_layout(
        template='presentation', 
        height=400,
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.8,  # Posición de la leyenda
            xanchor="center",
            font=dict(size=12)
        ),
        margin=dict(l=20, r=20, t=60, b=20),  # Márgenes reducidos
        uniformtext_minsize=10,  # Tamaño mínimo de texto
        uniformtext_mode='hide'  # Ocultar texto si no hay espacio
    )
    
    return fig



# Función para crear gráfico de estado de indicadores con opciones seleccionables
def crear_grafico_estados_interactivo(df):
    """
    Crea un gráfico circular interactivo con opciones seleccionables para visualizar
    los estados de los indicadores por origen o categoría.
    
    Args:
        df: DataFrame con los datos de los indicadores
    """
    st.subheader("Análisis de Estados de Indicadores")
    
    # Crear columnas para los controles
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Selector de origen
        origen_option = st.radio(
            "Filtrar por origen:",
            ["Todos", "Institucional", "Territorial"],
            index=0
        )
        
        # Selector de agrupación
        agrupar_por = st.radio(
            "Agrupar por:",
            ["Estado", "Dimensión", "Origen"],
            index=0
        )
    
    # Filtrar datos según la selección
    if origen_option == "Institucional":
        df_filtrado = df[df['Origen'] == 'Institucional']
        titulo_origen = "Institucional"
    elif origen_option == "Territorial":
        df_filtrado = df[df['Origen'] == 'Territorial']
        titulo_origen = "Territorial"
    else:
        df_filtrado = df
        titulo_origen = "Global"
    
    # Agrupar datos según la selección
    if agrupar_por == "Estado":
        conteo = df_filtrado['Estado'].value_counts().reset_index()
        conteo.columns = ['categoria', 'conteo']
        titulo = f'Distribución por Estado - {titulo_origen}'
    elif agrupar_por == "Dimensión":
        # Extraer solo el nombre de la dimensión (sin el número)
        df_filtrado['Dimension_Simple'] = df_filtrado['Dimension'].apply(
            lambda x: x.split(':')[0] if ':' in x else x
        )
        conteo = df_filtrado['Dimension_Simple'].value_counts().reset_index()
        conteo.columns = ['categoria', 'conteo']
        titulo = f'Distribución por Dimensión - {titulo_origen}'
    else:  # Origen
        conteo = df_filtrado['Origen'].value_counts().reset_index()
        conteo.columns = ['categoria', 'conteo']
        titulo = f'Distribución por Origen - {titulo_origen}'
    
    # Ordenar por categoría (excepto para Estado que tiene un orden específico)
    if agrupar_por != "Estado":
        conteo = conteo.sort_values('categoria')
    else:
        # Orden personalizado para estados: PENDIENTE, EN PROCESO, LISTO
        orden_estados = {"PENDIENTE": 1, "EN PROCESO": 2, "LISTO": 3}
        conteo['orden'] = conteo['categoria'].map(orden_estados)
        conteo = conteo.sort_values('orden')
        conteo = conteo.drop('orden', axis=1)
    
    # Mostrar resumen numérico
    with col1:
        st.markdown(f"### Resumen")
        total = conteo['conteo'].sum()
        for i, row in conteo.iterrows():
            porcentaje = round(row['conteo'] / total * 100, 1)
            st.markdown(f"**{row['categoria']}**: {row['conteo']} ({porcentaje}%)")
    
    # Crear gráfico con la paleta personalizada
    with col2:
        # Paleta de colores según el tipo de agrupación
        if agrupar_por == "Estado":
            # Paleta específica para estados
            color_map = {
                "PENDIENTE": "#FEC109",  # Amarillo
                "EN PROCESO": "#1E88E5",  # Azul medio
                "LISTO": "#0A5C99"       # Azul oscuro
            }
            colors = [color_map.get(cat, "#FC9F0B") for cat in conteo['categoria']]
        else:
            # Usar la paleta general para otras agrupaciones
            colors = ['#0A5C99', '#1E88E5', '#FEC109', '#FC9F0B', '#4CAF50', '#9C27B0', '#FF5722']
            # Repetir colores si hay más categorías que colores
            colors = colors * (len(conteo) // len(colors) + 1)
            colors = colors[:len(conteo)]
        
        fig = px.pie(
            conteo, 
            values='conteo', 
            names='categoria',
            title=titulo,
            hole=0.3,
            color_discrete_sequence=colors
        )
        
        # Configurar texto
        fig.update_traces(
            textposition='auto',
            textinfo="percent+label",
            textfont_size=12
        )
        
        # Diseño
        fig.update_layout(
            template='plotly_white', 
            height=450
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar datos adicionales o interpretación
    if agrupar_por == "Estado":
        st.markdown("""
        <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; margin-top:10px;">
        <h4>Interpretación de Estados</h4>
        <ul>
            <li><strong>PENDIENTE:</strong> Indicadores que aún no han iniciado su implementación</li>
            <li><strong>EN PROCESO:</strong> Indicadores que están actualmente en fase de implementación</li>
            <li><strong>LISTO:</strong> Indicadores que han sido completados</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Mostrar tabla de datos filtrados
    with st.expander("Ver datos detallados"):
        st.dataframe(
            df_filtrado[['ID', 'Dimension', 'Estado', 'Origen']], 
            use_container_width=True,
            hide_index=True
        )


# Cargar datos de indicadores
@st.cache_data
def cargar_indicadores(ruta='data/porcentajes avances.csv'):
    try:
        df = pd.read_csv(ruta, sep=';')
        # Renombrar columnas para mayor claridad
        df = df.rename(columns={
            'ID': 'ID',
            'Dimension': 'Dimension',
            'Estado': 'Estado',
            'Origen': 'Origen'
        })
        return df
    except FileNotFoundError:
        st.error(f"Archivo {ruta} no encontrado.")
        return pd.DataFrame()

# Usar la función en tu aplicación
df_indicadores = cargar_indicadores()


# Función para crear y mostrar el treemap de dimensiones e indicadores
def mostrar_treemap_dimensiones():
    """
    Crea y muestra un treemap interactivo que visualiza las dimensiones e indicadores
    tanto institucionales como territoriales con números de indicador y texto más grande.
    """
    st.subheader("Treemap de dimensiones e indicadores")
    
    # Verificar archivos disponibles y mostrar información de depuración
    archivos_disp = [f for f in os.listdir('data') if f.endswith('.csv')]
    
    if not any(f.lower() in ['institucional.csv', 'territorial.csv'] for f in archivos_disp):
        st.error("No se encontraron los archivos necesarios: Institucional.csv y territorial.csv")
        st.info(f"Archivos CSV disponibles: {', '.join(archivos_disp)}")
        st.info(f"Directorio actual: {os.getcwd()}")
        return
    
    # Función para cargar datos con mejor manejo de errores
    def cargar_csv_seguro(nombre_archivo):
        try:
            # Intenta diferentes codificaciones
            for encoding in ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']:
                try:
                    ruta_completa = os.path.join(os.getcwd(), nombre_archivo)
                    df = pd.read_csv(ruta_completa, encoding=encoding)
                    return df
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    st.warning(f"Error al cargar {nombre_archivo} con {encoding}: {str(e)}")
                    continue
            
            # Si ninguna codificación funcionó
            st.error(f"No se pudo cargar {nombre_archivo} con ninguna codificación.")
            return None
        except Exception as e:
            st.error(f"Error inesperado al cargar {nombre_archivo}: {str(e)}")
            return None
    
    # Cargar los dataframes
    institucional_df = cargar_csv_seguro('data/Institucional.csv')
    territorial_df = cargar_csv_seguro('data/territorial.csv')
    
    # Verificar si se cargaron los datos
    if institucional_df is None or territorial_df is None:
        st.error("No se pudieron cargar uno o ambos archivos CSV.")
        return
    
    # Crear datos simulados si los archivos no contienen la estructura esperada
    if 'Dimension' not in institucional_df.columns or 'Indicador' not in institucional_df.columns:
        st.warning("Los archivos CSV no tienen el formato esperado. Usando datos simulados.")
        
        # Crear dataframes simulados basados en el CSV de porcentajes avances
        df_indicadores = cargar_indicadores()
        
        # Verificar si se cargó el archivo de indicadores
        if df_indicadores.empty:
            st.error("No se pudieron cargar los datos de indicadores.")
            return
        
        # Crear dataframes simulados
        institucional_df = df_indicadores[df_indicadores['Origen'] == 'Institucional'].copy()
        institucional_df['Dimension'] = institucional_df['Dimension']
        institucional_df['Indicador'] = institucional_df['ID'] + ": " + institucional_df['Estado']
        
        territorial_df = df_indicadores[df_indicadores['Origen'] == 'Territorial'].copy()
        territorial_df['Dimension'] = territorial_df['Dimension']
        territorial_df['Indicador'] = territorial_df['ID'] + ": " + territorial_df['Estado']
    
    # Convertir "Indicadores" a "Indicador" para uniformidad
    if 'Indicadores' in territorial_df.columns and 'Indicador' not in territorial_df.columns:
        territorial_df = territorial_df.rename(columns={'Indicadores': 'Indicador'})
    
    # # Acortar el texto de los indicadores para mejor visualización en el treemap
    # def acortar_texto(texto, max_longitud=60):
    #     if isinstance(texto, str) and len(texto) > max_longitud:
    #         return texto[:max_longitud] + "..."
    #     return texto
    
    # # Crear versiones cortas de los indicadores para el treemap
    # institucional_df['Indicador_Corto'] = institucional_df['Indicador'].apply(acortar_texto)
    # territorial_df['Indicador_Corto'] = territorial_df['Indicador'].apply(acortar_texto)
    
    # Agregar números de indicador (I_1, I_2, etc. para institucionales y T_1, T_2, etc. para territoriales)
    # Crear una nueva columna con números de índice
    institucional_df = institucional_df.reset_index(drop=True)
    territorial_df = territorial_df.reset_index(drop=True)
    
    # Agregar números usando enumerate para evitar problemas con índices
    institucional_df['Indicador_Numerado'] = [f"I_{i+1}: {ind}" for i, ind in enumerate(institucional_df['Indicador'])]
    territorial_df['Indicador_Numerado'] = [f"T_{i+1}: {ind}" for i, ind in enumerate(territorial_df['Indicador'])]
    
    # Preparar los datos
    institucional_df['Valor'] = 10
    institucional_df['Categoria'] = 'Institucional'
    territorial_df['Valor'] = 10
    territorial_df['Categoria'] = 'Territorial'
    
    # Combinar ambos dataframes
    df_combined = pd.concat([institucional_df, territorial_df], ignore_index=True)
    
    # Verificar que tenemos las columnas necesarias
    columnas_requeridas = ['Categoria', 'Dimension', 'Indicador_Numerado', 'Valor']
    columnas_faltantes = [col for col in columnas_requeridas if col not in df_combined.columns]
    
    if columnas_faltantes:
        st.error(f"Faltan columnas requeridas: {', '.join(columnas_faltantes)}")
        st.write("Columnas disponibles:", df_combined.columns.tolist())
        return
    
    # Crear un treemap con la paleta de colores personalizada
    try:
        fig = px.treemap(
            df_combined,
            path=['Categoria', 'Dimension', 'Indicador_Numerado'],
            values='Valor',
            color='Categoria',  # Colorear por categoría
            color_discrete_map={
                'Institucional': '#0A5C99',
                'Territorial': '#FEC109'
            }
        )
        
        # Actualizar trazas para que el texto sea más grande
        fig.update_traces(
            textfont=dict(size=24),  # Aumentar tamaño de fuente significativamente
            texttemplate='%{label}',
            hovertemplate='<b>%{label}</b><br>Categoría: %{root}<br>Dimensión: %{parent}'
        )
        
        # Ajustar los márgenes y altura
        fig.update_layout(
            margin=dict(t=50, l=25, r=25, b=25),
            height=900,  # Aumentar altura para mejor visualización
            template='plotly_white'
        )
        
        # Mostrar el treemap
        st.plotly_chart(fig, use_container_width=True)
        
        # Generar leyenda adicional para los números de indicadores
        st.subheader("Leyenda detallada de indicadores")
        
        # Crear pestañas para las categorías
        tab1, tab2 = st.tabs(["Indicadores Institucionales", "Indicadores Territoriales"])
        
        with tab1:
            # Mostrar indicadores institucionales en una tabla ordenada
            st.markdown("### Indicadores Institucionales")
            for idx, row in institucional_df.iterrows():
                st.markdown(f"**I_{idx+1}:** {row['Indicador']}")
                
        with tab2:
            # Mostrar indicadores territoriales en una tabla ordenada
            st.markdown("### Indicadores Territoriales")
            for idx, row in territorial_df.iterrows():
                st.markdown(f"**T_{idx+1}:** {row['Indicador']}")
        
    except Exception as e:
        st.error(f"Error al crear el treemap: {str(e)}")
        st.write("Estructura de los datos:", df_combined.head())


# Función para cargar y mostrar la tabla de comunas
def mostrar_tabla_comunas():
    """
    Carga y muestra una tabla con información de las comunas de la Región Metropolitana.
    """
    st.subheader("Comunas del proyecto - Región Metropolitana")

    querycomunas="""select cpt.nombre_comuna as 'Nombre comuna', cpt.nombre_provincia as 'Nombre provincia', cr.nombre as 'Nombre region' from fiut.comunas_provincias_territorio cpt
    join fiut.chile_regiones cr on cr.nombre='Metropolitana de Santiago';"""
    
    # Cargar el dataframe
    df_comunas = pd.read_sql(querycomunas, engine)
    # df_comunas = pd.read_csv('data/Comunas.csv')

    
    if not df_comunas.empty:
        # Mostrar tabla sin el índice
        st.dataframe(
            df_comunas,
            use_container_width=True,
            hide_index=True
        )
        
        # # Información adicional
        # col1, col2 = st.columns(2)
        
        # with col1:
        #     # Contar comunas por provincia
        #     comunas_por_provincia = df_comunas['Provincia'].value_counts().reset_index()
        #     comunas_por_provincia.columns = ['Provincia', 'Cantidad de Comunas']
            
        #     st.write("Distribución por Provincia:")
        #     st.dataframe(
        #         comunas_por_provincia,
        #         use_container_width=True,
        #         hide_index=True
        #     )
        
        # with col2:
        #     st.markdown("""
        #     <div style="background-color:#f0f2f6; padding:15px; border-radius:10px;">
        #     <h4>Acerca de las comunas prioritarias</h4>
        #     <p>Estas 22 comunas han sido identificadas como prioritarias para el proyecto CINET, 
        #     enfocado en el desarrollo de centros interdisciplinarios en nuevas economías y 
        #     tecnologías.</p>
        #     <p>La selección abarca comunas de 5 provincias diferentes de la Región Metropolitana, 
        #     con especial énfasis en comunas pertenecientes a la provincia de Santiago.</p>
        #     </div>
        #     """, unsafe_allow_html=True)
    else:
        st.warning("No se pudo cargar la información de comunas.")

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
    
    # Cargar y procesar datos
    df = cargar_datos()
    df = procesar_datos(df)
    
    if df.empty:
        st.warning("No hay datos disponibles para analizar.")
        return
    
    # Contador total de archivos
    total_archivos = len(df)
    # Usar la función en tu aplicación
    st.markdown(f"### Levantamiento de un diagnóstico integral del territorio local y de las capacidades institucionales UTEM para la creación de un Centro Interdisciplinario en nuevas economías y tecnologías, orientado al desarrollo de localidades prioritarias de la Región Metropolitana. (CINET)")
    # Cargar datos de indicadores para las métricas de completitud
    # Cargar datos de indicadores para las métricas de completitud
    df_indicadores = cargar_indicadores()

    # Calcular porcentajes de completitud y conteos por estado
    if not df_indicadores.empty:
        # Calcular para Institucional
        df_inst = df_indicadores[df_indicadores['Origen'] == 'Institucional']
        total_inst = len(df_inst)
        completados_inst = len(df_inst[df_inst['Estado'] == 'LISTO'])
        en_proceso_inst = len(df_inst[df_inst['Estado'] == 'EN PROCESO'])
        pendientes_inst = len(df_inst[df_inst['Estado'] == 'PENDIENTE'])
        porc_completitud_inst = completados_inst / total_inst * 100
        porc_proceso_inst = en_proceso_inst / total_inst * 100
        porc_pendientes_inst = pendientes_inst / total_inst * 100
        
        # Calcular para Territorial
        df_terr = df_indicadores[df_indicadores['Origen'] == 'Territorial']
        total_terr = len(df_terr)
        completados_terr = len(df_terr[df_terr['Estado'] == 'LISTO'])
        en_proceso_terr = len(df_terr[df_terr['Estado'] == 'EN PROCESO'])
        pendientes_terr = len(df_terr[df_terr['Estado'] == 'PENDIENTE'])
        porc_completitud_terr = completados_terr / total_terr * 100
        porc_proceso_terr = en_proceso_terr / total_terr * 100
        porc_pendientes_terr = pendientes_terr / total_terr * 100
        
        # Calcular global
        total_global = len(df_indicadores)
        completados_global = len(df_indicadores[df_indicadores['Estado'] == 'LISTO'])
        en_proceso_global = len(df_indicadores[df_indicadores['Estado'] == 'EN PROCESO'])
        pendientes_global = len(df_indicadores[df_indicadores['Estado'] == 'PENDIENTE'])
        porc_completitud_global = completados_global / total_global * 100
        porc_proceso_global = en_proceso_global / total_global * 100
        porc_pendientes_global = pendientes_global / total_global * 100

    # Métricas principales con tres columnas
    col1, col2, col3 = st.columns(3)

    with col1:
        if not df_indicadores.empty:
            # Usar None como delta para no mostrar la flecha
            st.metric(
                "Indicadores Institucionales", 
                f"{porc_completitud_inst:.1f}% Completados", 
                f"Total: {total_inst}",
                delta_color="off"  # Desactivar el color del delta
            )
            # Añadir conteo detallado por estado con porcentajes
            st.markdown(f"""
            <div style="padding-left:10px;">
                <span style="color:#0A5C99;font-weight:bold;">✓ Listos:</span> {completados_inst} ({porc_completitud_inst:.1f}%)<br>
                <span style="color:#1E88E5;font-weight:bold;">⟳ En Proceso:</span> {en_proceso_inst} ({porc_proceso_inst:.1f}%)<br>
                <span style="color:#FEC109;font-weight:bold;">⏱ Pendientes:</span> {pendientes_inst} ({porc_pendientes_inst:.1f}%)
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric("Indicadores Institucionales", "Sin datos", "")

    with col2:
        if not df_indicadores.empty:
            st.metric(
                "Indicadores Territoriales", 
                f"{porc_completitud_terr:.1f}% Completados", 
                f"Total: {total_terr}",
                delta_color="off"  # Desactivar el color del delta
            )
            # Añadir conteo detallado por estado con porcentajes
            st.markdown(f"""
            <div style="padding-left:10px;">
                <span style="color:#0A5C99;font-weight:bold;">✓ Listos:</span> {completados_terr} ({porc_completitud_terr:.1f}%)<br>
                <span style="color:#1E88E5;font-weight:bold;">⟳ En Proceso:</span> {en_proceso_terr} ({porc_proceso_terr:.1f}%)<br>
                <span style="color:#FEC109;font-weight:bold;">⏱ Pendientes:</span> {pendientes_terr} ({porc_pendientes_terr:.1f}%)
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric("Indicadores Territoriales", "Sin datos", "")

    with col3:
        if not df_indicadores.empty:
            st.metric(
                "Avance General", 
                f"{porc_completitud_global:.1f}% Completado", 
                f"Total: {total_global} Indicadores",
                delta_color="off"  # Desactivar el color del delta
            )
            # Añadir conteo detallado por estado con porcentajes
            st.markdown(f"""
            <div style="padding-left:10px;">
                <span style="color:#0A5C99;font-weight:bold;">✓ Listos:</span> {completados_global} ({porc_completitud_global:.1f}%)<br>
                <span style="color:#1E88E5;font-weight:bold;">⟳ En Proceso:</span> {en_proceso_global} ({porc_proceso_global:.1f}%)<br>
                <span style="color:#FEC109;font-weight:bold;">⏱ Pendientes:</span> {pendientes_global} ({porc_pendientes_global:.1f}%)
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric("Avance General", "Sin datos", "")
    
    # Pestañas para diferentes análisis
    tab1, tab2, tab3, tab4, tab5, tab6, tab7= st.tabs([
        "Vista General", 
        "Análisis por Dimensiones",
        "Análisis de Estado Indicadores",
        "Insights Adicionales",
        "Mapa Geográfico",
        "Mapa Sedes",
        "Análisis de Archivos"
    ])


    # NOTA: Añadir treemap a la vista general 
    # NOTA: Los graficos de vista general pasan a análisis por Tipo
    # TAB 1: Vista General
    with tab1:
        st.markdown("""
        <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; margin-top:20px;">
         <h3>Descripción </h3>
        <p>El proyecto busca potenciar la investigación aplicada y la innovación en la Universidad Tecnológica Metropolitana mediante un diagnóstico integral del territorio y de sus capacidades institucionales, identificando fortalezas y brechas en gestión, infraestructura, oferta académica y colaboración; a partir de este análisis, se plantea la creación de un centro interdisciplinario que impulse la transferencia tecnológica y establezca alianzas estratégicas entre la academia, la industria y el sector público, contribuyendo al desarrollo sostenible y competitivo de la Región Metropolitana.</p>
        <h3>Objetivo del Fondo de Financiamiento Estructural de I+D+i (FIU) Territorial: </h3>
        <p>Potenciar la contribución de universidades con acreditación entre 3 y 5 años al desarrollo territorial y los procesos de  descentralización, mediante el financiamiento de capacidades mínimas de I+D+i, incluyendo su respectiva gestión y gobernanza institucional.</p>
        </div>
        """, unsafe_allow_html=True)
        
        mostrar_tabla_comunas()

        mostrar_treemap_dimensiones()
        

    with tab2:
        st.header("Análisis por Dimensiones")
    
        # Mantén el código existente aquí...
        col1, col2 = st.columns(2)
        
        with col1:
            # Selector para filtrar dimensiones
            filtro_dim = st.radio(
                "Seleccionar categoría para dimensiones:",
                ["Global", "Institucional", "Territorial"],
                horizontal=True
            )
            filter_dim = None if filtro_dim == "Global" else filtro_dim.lower()
            
            # Gráfico de dimensiones
            grafico_dim = crear_grafico_dimensiones(df, filter_dim)
            if grafico_dim:
                st.plotly_chart(grafico_dim, use_container_width=True, key=f"dim_{filtro_dim}_chart")       
            else:
                st.warning(f"No hay datos suficientes para mostrar dimensiones en la categoría {filtro_dim}")
        
        with col2:
            st.markdown("""
            <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; margin-top:35px;">
            <h4>¿Qué son las dimensiones?</h4>
            <p>Las dimensiones representan áreas funcionales o temáticas dentro de las categorías principales.
            Cada dimensión agrupa información relacionada con un aspecto específico de la gestión institucional
            o territorial, facilitando la organización y recuperación de la información.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Cargar el CSV de nombres de dimensiones
            nombres_dimensiones = pd.read_csv("data/nombres_dimensiones.csv")
            # Crear un diccionario para mapear id a nombre
            dict_dimensiones = dict(zip(nombres_dimensiones['id_dim'], nombres_dimensiones['nombre_dim']))

            # Mostrar estadísticas por dimensión
            st.subheader("Estadísticas por Dimensión")

            # Filtrar según selección
            if filter_dim == 'institucional':
                df_stat = df[df['institucional'] == True]
            elif filter_dim == 'territorial':
                df_stat = df[df['territorial'] == True]
            else:
                df_stat = df
                
            # Calcular estadísticas de dimensiones sin "Sin clasificación"
            df_dims = df_stat[df_stat['dimensiones'] != 'Sin clasificación']

            if not df_dims.empty:
                dim_stats = df_dims['dimensiones'].value_counts()
                
                # Crear DataFrame para las estadísticas
                data = []
                for dim in dim_stats.index:
                    # Extraer el número de dimensión
                    if isinstance(dim, str) and dim.startswith('Dimensión '):
                        dim_num = int(dim.replace('Dimensión ', ''))
                    else:
                        dim_num = int(dim) if str(dim).isdigit() else 0
                    
                    # Obtener el nombre completo
                    nombre_completo = dict_dimensiones.get(dim_num, "Sin nombre")
                    
                    data.append({
                        'Número': dim_num,
                        'Dimensión': dim, 
                        'Nombre Dimensión': nombre_completo,
                        'Total Archivos': dim_stats[dim],
                        'Porcentaje': round(dim_stats[dim] / dim_stats.sum() * 100, 1)
                    })
                
                # Crear DataFrame y ordenar por número de dimensión
                dim_df = pd.DataFrame(data)
                dim_df = dim_df.sort_values('Número')
                
                # Mostrar el DataFrame sin el índice y sin la columna de número
                st.dataframe(
                    dim_df[['Dimensión', 'Nombre Dimensión', 'Total Archivos', 'Porcentaje']], 
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No hay datos de dimensiones disponibles para esta selección.")
        
        # Heatmap de extensiones por dimensión
        st.subheader("Relación entre Tipos de Archivos y Dimensiones")
        
        heatmap = crear_heatmap_extension_dimension(df)
        if heatmap:
            st.plotly_chart(heatmap, use_container_width=True, key="heatmap_chart")
        else:
            st.warning("No hay suficientes datos para crear el mapa de calor.")
        
        st.markdown("""
        <div style="background-color:#f0f2f6; padding:15px; border-radius:10px;">
        <h4>¿Qué nos muestra este mapa de calor?</h4>
        <p>Este mapa de calor muestra la concentración de diferentes tipos de archivos en cada dimensión, 
        permitiendo identificar:</p>
        <ul>
            <li>Qué formatos son más utilizados en cada dimensión</li>
            <li>Posibles patrones de uso específicos por área temática</li>
            <li>Dimensiones con mayor diversidad o especialización en formatos</li>
        </ul>
        <p>Esta información puede ser útil para entender mejor los flujos de trabajo y necesidades de 
        información en diferentes áreas de la organización.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # NOTA: Añadir metodología de trabajo
    # TAB 4: Insights Adicionales
    with tab3:
        # Cargar datos de indicadores
        df_indicadores = cargar_indicadores()
        
        # Si los datos se cargaron correctamente, mostrar el gráfico interactivo
        if not df_indicadores.empty:
            crear_grafico_estados_interactivo(df_indicadores)
        else:
            st.warning("No se pudieron cargar los datos de indicadores.")
    with tab4:
        st.header("Insights Adicionales")
        
        # Método de obtención (ejemplo)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Métodos de Obtención de Archivos")
            st.plotly_chart(crear_grafico_metodos_obtencion(), use_container_width=True, key="metodos_obtencion_chart")
        
        with col2:
            st.markdown("""
            <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; margin-top:35px;">
            <h4>Fuentes de información</h4>
            <p>Los archivos del Data Lake provienen de diferentes fuentes, lo que influye en su formato, 
            estructura y calidad. Las principales fuentes son:</p>
            <ul>
                <li><strong>Web Scraping:</strong> Datos extraídos automáticamente de sitios web</li>
                <li><strong>Universidad:</strong> Documentos generados internamente por la institución</li>
                <li><strong>Descargados:</strong> Archivos obtenidos de fuentes externas como portales oficiales</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Análisis de tamaño de archivos
        st.subheader("Tamaño de Archivos por Extensión")
        
        # Función para convertir tamaño a KB
        def extraer_tamano_kb(tam_str):
            try:
                if isinstance(tam_str, str):
                    partes = tam_str.split()
                    valor = float(partes[0])
                    unidad = partes[1]
                    
                    if unidad == 'B':
                        return valor / 1024
                    elif unidad == 'KB':
                        return valor
                    elif unidad == 'MB':
                        return valor * 1024
                    elif unidad == 'GB':
                        return valor * 1024 * 1024
                    else:
                        return 0
                else:
                    return 0
            except:
                return 0
        
        # Calcular tamaño en KB
        df['tamano_kb'] = df['tamano'].apply(extraer_tamano_kb)
        
        # Agrupar por extensión
        tamano_por_ext = df.groupby('extension')['tamano_kb'].agg(['mean', 'sum', 'count']).reset_index()
        tamano_por_ext.columns = ['Extensión', 'Tamaño Promedio (KB)', 'Tamaño Total (KB)', 'Cantidad']
        tamano_por_ext = tamano_por_ext.sort_values('Tamaño Total (KB)', ascending=False).head(10)
        
        # Redondear valores
        tamano_por_ext['Tamaño Promedio (KB)'] = tamano_por_ext['Tamaño Promedio (KB)'].round(2)
        tamano_por_ext['Tamaño Total (KB)'] = tamano_por_ext['Tamaño Total (KB)'].round(2)
        
        # Mostrar tabla
        st.dataframe(tamano_por_ext, use_container_width=True)
        
        st.markdown("""
        <div style="background-color:#f0f2f6; padding:15px; border-radius:10px;">
        <h4>Conclusiones generales</h4>
        <p>El análisis del Data Lake revela patrones importantes sobre cómo se almacena y organiza la 
        información en la organización:</p>
        <ul>
            <li>La mayor parte de los archivos son de tipo <strong>hoja de cálculo</strong>, indicando un 
            enfoque en análisis de datos cuantitativos</li>
            <li>Existe una diferencia notable entre la cantidad de archivos <strong>institucionales</strong> 
            versus <strong>territoriales</strong></li>
            <li>Cada dimensión muestra preferencias específicas por ciertos formatos, reflejando sus 
            necesidades particulares</li>
        </ul>
        <p>Esta información puede utilizarse para optimizar la gestión documental, mejorar los procesos 
        de captura de datos y facilitar el acceso a la información relevante.</p>
        </div>
        """, unsafe_allow_html=True)

    # NOTA: hablar del territorio 
    # TAB 5: Mapa Geográfico
    with tab5:
        st.header("Mapa de la Región Metropolitana")
        
        # Puedes ajustar el tamaño del mapa según necesites
        mapa_height = 600
        
        # Función para leer el archivo HTML
        def cargar_html_mapa(ruta_html):
            try:
                with open(ruta_html, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                return html_content
            except FileNotFoundError:
                st.error(f"No se encontró el archivo HTML del mapa en: {ruta_html}")
                return None
        
        # Ruta a tu archivo HTML (ajusta según donde esté guardado)
        ruta_mapa = "mapas_html/mapa_rm_final.html"
        
        # Cargar y mostrar el mapa
        html_mapa = cargar_html_mapa(ruta_mapa)
        if html_mapa:
            st.markdown("Este mapa muestra las diferentes provincias y comunas de la Región Metropolitana.")
            components.html(html_mapa, height=mapa_height)
        else:
            st.warning("No se pudo cargar el mapa. Verifica la ruta del archivo HTML.")
            
        # Agregar contexto sobre el mapa
        st.markdown("""
        <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; margin-top:20px;">
        <h4>Acerca del mapa</h4>
        <p>Este mapa interactivo muestra la distribución territorial de la Región Metropolitana de Santiago, 
        con sus diferentes provincias identificadas por colores:</p>
        <ul>
            <li><strong>Santiago:</strong> Zona central y de mayor densidad de población</li>
            <li><strong>Cordillera:</strong> Zona este, limítrofe con la cordillera de los Andes</li>
            <li><strong>Chacabuco:</strong> Zona norte de la región</li>
            <li><strong>Maipo:</strong> Zona sur</li>
            <li><strong>Melipilla:</strong> Zona suroeste</li>
            <li><strong>Talagante:</strong> Zona oeste</li>
        </ul>
        <p>Puedes interactuar con el mapa para ver información detallada de cada comuna.</p>
        </div>
        """, unsafe_allow_html=True)

    with tab6:
        st.header("Mapa de la Región Metropolitana")
        
        # Puedes ajustar el tamaño del mapa según necesites
        mapa_height = 600
        
        # Función para leer el archivo HTML
        def cargar_html_mapa(ruta_html):
            try:
                with open(ruta_html, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                return html_content
            except FileNotFoundError:
                st.error(f"No se encontró el archivo HTML del mapa en: {ruta_html}")
                return None
        
        # Ruta a tu archivo HTML (ajusta según donde esté guardado)
        ruta_mapa = "mapas_html/mapa_sedes_utem.html"
        
        # Cargar y mostrar el mapa
        html_mapa = cargar_html_mapa(ruta_mapa)
        if html_mapa:
            st.markdown("Este mapa muestra las diferentes provincias y comunas de la Región Metropolitana.")
            components.html(html_mapa, height=mapa_height)
        else:
            st.warning("No se pudo cargar el mapa. Verifica la ruta del archivo HTML.")

    

    with tab7:
        
        st.header("Análisis archivos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(crear_grafico_institucional_territorial(df), use_container_width=True, key="inst_terr_chart")
            
            st.markdown("""
            <div style="background-color:#f0f2f6; padding:15px; border-radius:10px;">
            <h4>¿Qué nos muestra este gráfico?</h4>
            <p>Este gráfico muestra la distribución de archivos entre las categorías <strong>Institucional</strong> 
            y <strong>Territorial</strong>, permitiendo identificar rápidamente el balance entre estos dos tipos 
            de información en el Data Lake.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.plotly_chart(crear_grafico_extensiones(df), use_container_width=True, key="ext_general_chart")
            
            st.markdown("""
            <div style="background-color:#f0f2f6; padding:15px; border-radius:10px;">
            <h4>Tipos de archivos en el Data Lake</h4>
            <p>La distribución de tipos de archivos nos permite entender qué formatos predominan en el repositorio,
            lo que refleja los tipos de datos y documentos más utilizados en la organización.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Gráfico comparativo de extensiones por categoría
        st.header("Comparación de Tipos de Archivos por Categoría")
        st.plotly_chart(crear_grafico_comparativo_extensiones(df), use_container_width=True, key="ext_comp_chart")
        
        st.markdown("""
        <div style="background-color:#f0f2f6; padding:15px; border-radius:10px;">
        <h4>Diferencias entre categorías</h4>
        <p>Esta comparación permite identificar si existen patrones o preferencias diferentes en el uso de formatos 
        de archivos entre las áreas institucionales y territoriales. Esto puede reflejar diferentes necesidades
        o flujos de trabajo específicos para cada categoría.</p>
        </div>
        """, unsafe_allow_html=True)


        st.header("Análisis Detallado por Tipo de Archivo")
        
        # Selector para filtrar por categoría
        filtro_cat = st.radio(
            "Seleccionar categoría:",
            ["Global", "Institucional", "Territorial"],
            horizontal=True
        )
        filtro = None if filtro_cat == "Global" else filtro_cat.lower()
        
        # Gráfico de extensiones filtrado
        st.plotly_chart(crear_grafico_extensiones(df, filtro), use_container_width=True, key=f"ext_{filtro_cat}_chart")
        
        # Mostrar top extensiones con estadísticas
        st.subheader(f"Top 5 Extensiones - {filtro_cat}")
        
        # Filtrar según selección
        if filtro == 'institucional':
            df_temp = df[df['institucional'] == True]
        elif filtro == 'territorial':
            df_temp = df[df['territorial'] == True]
        else:
            df_temp = df
            
        # Calcular estadísticas
        top_ext = df_temp['extension'].value_counts().head(5)
        top_ext_df = pd.DataFrame({
            'Extensión': top_ext.index,
            'Cantidad': top_ext.values,
            'Porcentaje': (top_ext.values / len(df_temp) * 100).round(1)
        })
        
        # Mostrar tabla
        st.dataframe(top_ext_df, use_container_width=True)
        
        st.markdown("""
        <div style="background-color:#f0f2f6; padding:15px; border-radius:10px;">
        <h4>Interpretación de los tipos de archivos</h4>
        <p>Los diferentes tipos de archivos tienen propósitos específicos:</p>
        <ul>
            <li><strong>.xlsx/.xls:</strong> Hojas de cálculo para análisis de datos, registros y reportes cuantitativos</li>
            <li><strong>.pdf:</strong> Documentos formales, informes finales, documentación oficial</li>
            <li><strong>.docx/.doc:</strong> Documentos de texto, informes en proceso, documentación detallada</li>
            <li><strong>.pptx/.ppt:</strong> Presentaciones para reuniones y exposiciones</li>
            <li><strong>.csv:</strong> Datos estructurados para análisis y procesamiento</li>
        </ul>
        <p>La predominancia de ciertos formatos puede indicar el enfoque principal del trabajo en cada área.</p>
        </div>
        """, unsafe_allow_html=True)

# Ejecutar la aplicación
if __name__ == "__main__":
    main()
