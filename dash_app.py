import dash
from dash import dcc, html, dash_table, Input, Output, State, callback
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from sqlalchemy import create_engine
import base64
from io import BytesIO
from PIL import Image

# Crear conexión usando SQLAlchemy
engine = create_engine('mariadb+mariadbconnector://testfiut:utem1234@localhost/mysql')

# Inicializar la aplicación Dash
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="Exploración datos FIUT",
    external_stylesheets=[
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"
    ]
)
server = app.server

# Estilos personalizados
colors = {
    "background": "#FFFFFF",
    "text": "#333333",
    "blue": "#0A5C99",
    "medium_blue": "#1E88E5",
    "yellow": "#FEC109",
    "light_yellow": "#FC9F0B",
    "gray_bg": "#f0f2f6"
}

# Función para cargar los datos
def cargar_datos(ruta='data/estructura_archivos.csv'):
    """Carga los datos del archivo CSV o genera un DataFrame vacío si no existe"""
    try:
        df = pd.read_csv(ruta)
        return df
    except FileNotFoundError:
        print(f"Archivo {ruta} no encontrado. Por favor ejecuta primero el script de generación.")
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
            marker_color=[colors["blue"], colors["yellow"]],
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
        color_discrete_sequence=[colors["blue"], colors["medium_blue"], colors["yellow"], colors["light_yellow"]]
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
    
    # Si no hay datos, devolver un gráfico vacío
    if df_temp.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No hay datos suficientes para mostrar dimensiones en esta categoría",
            height=500
        )
        return fig
    
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
        color_discrete_sequence=[colors["blue"], colors["medium_blue"], colors["yellow"], colors["light_yellow"]]
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
            marker=dict(colors=[colors["blue"], colors["medium_blue"], colors["yellow"], colors["light_yellow"]])
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Pie(
            labels=orden,
            values=[ext_terr[ext] for ext in orden],
            name="Territorial",
            hole=0.4,
            marker=dict(colors=[colors["blue"], colors["medium_blue"], colors["yellow"], colors["light_yellow"]])
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
        fig = go.Figure()
        fig.update_layout(
            title="No hay suficientes datos para crear el mapa de calor",
            height=450
        )
        return fig
    
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
    fig = px.imshow(
        pivot,
        labels=dict(x="Dimensión", y="Extensión", color="Cantidad"),
        x=pivot.columns,
        y=pivot.index,
        color_continuous_scale=[[0, '#E3F2FD'], [0.5, colors["medium_blue"]], [1, colors["blue"]]],
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
    try:
        dfh = pd.read_excel('data/DataLake_registro_FIUT_UTEM.xlsx')
        dfhh = {
            'nombres': [], 
            'conteo': []
        }
        for i, j in dfh['METODO'].value_counts().items():
            dfhh['nombres'].append(i)
            dfhh['conteo'].append(j)
        dfhh = pd.DataFrame(dfhh)

        # Renombrar métodos para mejor claridad
        if len(dfhh) >= 3:
            dfhh['nombres'][0] = 'Web Scrapping'
            dfhh['nombres'][1] = 'Universidad'
            dfhh['nombres'][2] = 'Descargados'
        
        fig = px.pie(
            dfhh, 
            values='conteo', 
            names='nombres', 
            title='Distribución métodos de obtención de los archivos',
            color_discrete_sequence=[colors["blue"], colors["medium_blue"], colors["yellow"]],
            hole=0.3,
        )

        # Configurar texto con posiciones adaptativas
        fig.update_traces(
            textposition='auto',
            textinfo='percent+label',
            textfont_size=12,
            rotation=270
        )

        # Mejorar el diseño
        fig.update_layout(
            template='presentation', 
            height=400,
            legend=dict(
                orientation="v",
                yanchor="bottom",
                y=0.8,
                xanchor="center",
                font=dict(size=12)
            ),
            margin=dict(l=20, r=20, t=60, b=20),
            uniformtext_minsize=10,
            uniformtext_mode='hide'
        )
        
        return fig
    except Exception as e:
        print(f"Error al crear gráfico de métodos de obtención: {str(e)}")
        fig = go.Figure()
        fig.update_layout(
            title="No se pudieron cargar los datos de métodos de obtención",
            height=400
        )
        return fig

# Función para crear gráfico de estados interactivo
def crear_grafico_estados(df_indicadores, origen="Todos", agrupar_por="Estado"):
    """
    Crea un gráfico circular con los estados de los indicadores.
    
    Args:
        df_indicadores: DataFrame con los datos de los indicadores
        origen: Filtro de origen ("Todos", "Institucional", "Territorial")
        agrupar_por: Tipo de agrupación ("Estado", "Dimensión", "Origen")
    """
    # Filtrar datos según la selección
    if origen == "Institucional":
        df_filtrado = df_indicadores[df_indicadores['Origen'] == 'Institucional']
        titulo_origen = "Institucional"
    elif origen == "Territorial":
        df_filtrado = df_indicadores[df_indicadores['Origen'] == 'Territorial']
        titulo_origen = "Territorial"
    else:
        df_filtrado = df_indicadores
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
    
    # Crear gráfico según el tipo de agrupación
    if agrupar_por == "Estado":
        # Paleta específica para estados
        color_map = {
            "PENDIENTE": colors["yellow"],
            "EN PROCESO": colors["medium_blue"],
            "LISTO": colors["blue"]
        }
        colors_list = [color_map.get(cat, colors["light_yellow"]) for cat in conteo['categoria']]
    else:
        # Usar la paleta general para otras agrupaciones
        palette = [colors["blue"], colors["medium_blue"], colors["yellow"], colors["light_yellow"], "#4CAF50", "#9C27B0", "#FF5722"]
        # Repetir colores si hay más categorías que colores
        colors_list = (palette * (len(conteo) // len(palette) + 1))[:len(conteo)]
    
    fig = px.pie(
        conteo, 
        values='conteo', 
        names='categoria',
        title=titulo,
        hole=0.3,
        color_discrete_sequence=colors_list
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
    
    return fig, conteo, df_filtrado

# Cargar datos para análisis de tamaño de archivos
def analisis_tamano_archivos(df):
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
    
    return tamano_por_ext

# Cargar datos de indicadores
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
        print(f"Archivo {ruta} no encontrado.")
        return pd.DataFrame()

# Función para cargar datos de las comunas
def cargar_datos_comunas():
    try:
        # Intentar cargar desde la base de datos
        query = """select cpt.nombre_comuna as 'Nombre comuna', cpt.nombre_provincia as 'Nombre provincia', 
                cr.nombre as 'Nombre region' from fiut.comunas_provincias_territorio cpt
                join fiut.chile_regiones cr on cr.nombre='Metropolitana de Santiago';"""
        df_comunas = pd.read_sql(query, engine)
        return df_comunas
    except:
        # Si falla, intentar cargar desde CSV
        try:
            df_comunas = pd.read_csv('data/Comunas.csv')
            return df_comunas
        except:
            # Si ambos fallan, retornar un DataFrame vacío
            return pd.DataFrame()

# Función para leer archivos HTML
def leer_html(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error leyendo HTML: {str(e)}")
        return None

# Preprocesar datos
df = cargar_datos()
df = procesar_datos(df)
df_indicadores = cargar_indicadores()

# Calcular estadísticas de indicadores
if not df_indicadores.empty:
    # Calcular para Institucional
    df_inst = df_indicadores[df_indicadores['Origen'] == 'Institucional']
    total_inst = len(df_inst)
    completados_inst = len(df_inst[df_inst['Estado'] == 'LISTO'])
    en_proceso_inst = len(df_inst[df_inst['Estado'] == 'EN PROCESO'])
    pendientes_inst = len(df_inst[df_inst['Estado'] == 'PENDIENTE'])
    porc_completitud_inst = completados_inst / total_inst * 100 if total_inst > 0 else 0
    porc_proceso_inst = en_proceso_inst / total_inst * 100 if total_inst > 0 else 0
    porc_pendientes_inst = pendientes_inst / total_inst * 100 if total_inst > 0 else 0
    
    # Calcular para Territorial
    df_terr = df_indicadores[df_indicadores['Origen'] == 'Territorial']
    total_terr = len(df_terr)
    completados_terr = len(df_terr[df_terr['Estado'] == 'LISTO'])
    en_proceso_terr = len(df_terr[df_terr['Estado'] == 'EN PROCESO'])
    pendientes_terr = len(df_terr[df_terr['Estado'] == 'PENDIENTE'])
    porc_completitud_terr = completados_terr / total_terr * 100 if total_terr > 0 else 0
    porc_proceso_terr = en_proceso_terr / total_terr * 100 if total_terr > 0 else 0
    porc_pendientes_terr = pendientes_terr / total_terr * 100 if total_terr > 0 else 0
    
    # Calcular global
    total_global = len(df_indicadores)
    completados_global = len(df_indicadores[df_indicadores['Estado'] == 'LISTO'])
    en_proceso_global = len(df_indicadores[df_indicadores['Estado'] == 'EN PROCESO'])
    pendientes_global = len(df_indicadores[df_indicadores['Estado'] == 'PENDIENTE'])
    porc_completitud_global = completados_global / total_global * 100 if total_global > 0 else 0
    porc_proceso_global = en_proceso_global / total_global * 100 if total_global > 0 else 0
    porc_pendientes_global = pendientes_global / total_global * 100 if total_global > 0 else 0
else:
    total_inst = total_terr = total_global = 0
    porc_completitud_inst = porc_completitud_terr = porc_completitud_global = 0
    completados_inst = completados_terr = completados_global = 0
    en_proceso_inst = en_proceso_terr = en_proceso_global = 0
    pendientes_inst = pendientes_terr = pendientes_global = 0
    porc_proceso_inst = porc_proceso_terr = porc_proceso_global = 0
    porc_pendientes_inst = porc_pendientes_terr = porc_pendientes_global = 0

# Cargar imágenes
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        print(f"Error al cargar imagen {image_path}: {str(e)}")
        return ""

# Rutas de imágenes
try:
    img_ministerio = encode_image("imagenes/Ministerio de Ciencias color.png")
    img_fiut = encode_image("imagenes/Isologo FIU UTEM color.png")
except:
    img_ministerio = ""
    img_fiut = ""

# Layout de la aplicación con tabs
app.layout = html.Div([
    # Header con logos
    html.Div([
        html.Div([
            html.Img(src=img_ministerio, style={"height": "80px"})
        ], style={"width": "20%", "display": "inline-block", "textAlign": "center"}),
        
        html.Div([
            html.H1("Proyecto FIUT 2024 UTEM", style={"textAlign": "center"})
        ], style={"width": "60%", "display": "inline-block"}),
        
        html.Div([
            html.Img(src=img_fiut, style={"height": "100px"})
        ], style={"width": "20%", "display": "inline-block", "textAlign": "center"})
    ], style={"display": "flex", "alignItems": "center", "padding": "20px"}),
    
    # Descripción del proyecto
    html.Div([
        html.H3("Levantamiento de un diagnóstico integral del territorio local y de las capacidades institucionales UTEM para la creación de un Centro Interdisciplinario en nuevas economías y tecnologías, orientado al desarrollo de localidades prioritarias de la Región Metropolitana. (CINET)"),
        
        # Métricas principales (3 columnas)
        html.Div([
            # Institucional
            html.Div([
                html.Div([
                    html.H4("Indicadores Institucionales"),
                    html.H2(f"{porc_completitud_inst:.1f}% Completados"),
                    html.P(f"Total: {total_inst}")
                ], style={"textAlign": "center", "padding": "10px", "backgroundColor": "white", "borderRadius": "5px", "boxShadow": "0 4px 8px 0 rgba(0,0,0,0.2)"}),
                
                html.Div([
                    html.P([
                        html.Span("✓ Listos: ", style={"color": colors["blue"], "fontWeight": "bold"}),
                        f"{completados_inst} ({porc_completitud_inst:.1f}%)"
                    ]),
                    html.P([
                        html.Span("⟳ En Proceso: ", style={"color": colors["medium_blue"], "fontWeight": "bold"}),
                        f"{en_proceso_inst} ({porc_proceso_inst:.1f}%)"
                    ]),
                    html.P([
                        html.Span("⏱ Pendientes: ", style={"color": colors["yellow"], "fontWeight": "bold"}),
                        f"{pendientes_inst} ({porc_pendientes_inst:.1f}%)"
                    ])
                ], style={"paddingLeft": "10px"})
            ], className="four columns"),
            
            # Territorial
            html.Div([
                html.Div([
                    html.H4("Indicadores Territoriales"),
                    html.H2(f"{porc_completitud_terr:.1f}% Completados"),
                    html.P(f"Total: {total_terr}")
                ], style={"textAlign": "center", "padding": "10px", "backgroundColor": "white", "borderRadius": "5px", "boxShadow": "0 4px 8px 0 rgba(0,0,0,0.2)"}),
                
                html.Div([
                    html.P([
                        html.Span("✓ Listos: ", style={"color": colors["blue"], "fontWeight": "bold"}),
                        f"{completados_terr} ({porc_completitud_terr:.1f}%)"
                    ]),
                    html.P([
                        html.Span("⟳ En Proceso: ", style={"color": colors["medium_blue"], "fontWeight": "bold"}),
                        f"{en_proceso_terr} ({porc_proceso_terr:.1f}%)"
                    ]),
                    html.P([
                        html.Span("⏱ Pendientes: ", style={"color": colors["yellow"], "fontWeight": "bold"}),
                        f"{pendientes_terr} ({porc_pendientes_terr:.1f}%)"
                    ])
                ], style={"paddingLeft": "10px"})
            ], className="four columns"),
            
            # Global
            html.Div([
                html.Div([
                    html.H4("Avance General"),
                    html.H2(f"{porc_completitud_global:.1f}% Completado"),
                    html.P(f"Total: {total_global} Indicadores")
                ], style={"textAlign": "center", "padding": "10px", "backgroundColor": "white", "borderRadius": "5px", "boxShadow": "0 4px 8px 0 rgba(0,0,0,0.2)"}),
                
                html.Div([
                    html.P([
                        html.Span("✓ Listos: ", style={"color": colors["blue"], "fontWeight": "bold"}),
                        f"{completados_global} ({porc_completitud_global:.1f}%)"
                    ]),
                    html.P([
                        html.Span("⟳ En Proceso: ", style={"color": colors["medium_blue"], "fontWeight": "bold"}),
                        f"{en_proceso_global} ({porc_proceso_global:.1f}%)"
                    ]),
                    html.P([
                        html.Span("⏱ Pendientes: ", style={"color": colors["yellow"], "fontWeight": "bold"}),
                        f"{pendientes_global} ({porc_pendientes_global:.1f}%)"
                    ])
                ], style={"paddingLeft": "10px"})
            ], className="four columns"),
        ], className="row", style={"marginBottom": "20px"}),
    ], style={"padding": "20px", "backgroundColor": "#f9f9f9", "marginBottom": "20px", "borderRadius": "5px"}),
    
    # Pestañas para diferentes análisis
    dcc.Tabs(id="main-tabs", value="tab-general", children=[
        # Tab 1: Vista General
        dcc.Tab(label="Vista General", value="tab-general", children=[
            html.Div([
                # Descripción del proyecto
                html.Div([
                    html.H3("Descripción"),
                    html.P("El proyecto busca potenciar la investigación aplicada y la innovación en la Universidad Tecnológica Metropolitana mediante un diagnóstico integral del territorio y de sus capacidades institucionales, identificando fortalezas y brechas en gestión, infraestructura, oferta académica y colaboración; a partir de este análisis, se plantea la creación de un centro interdisciplinario que impulse la transferencia tecnológica y establezca alianzas estratégicas entre la academia, la industria y el sector público, contribuyendo al desarrollo sostenible y competitivo de la Región Metropolitana."),
                    
                    html.H3("Objetivo del Fondo de Financiamiento Estructural de I+D+i (FIU) Territorial:"),
                    html.P("Potenciar la contribución de universidades con acreditación entre 3 y 5 años al desarrollo territorial y los procesos de descentralización, mediante el financiamiento de capacidades mínimas de I+D+i, incluyendo su respectiva gestión y gobernanza institucional.")
                ], style={"backgroundColor": colors["gray_bg"], "padding": "15px", "borderRadius": "10px", "marginTop": "20px", "marginBottom": "20px"}),
                
                # Comunas del proyecto
                html.Div([
                    html.H3("Comunas del proyecto - Región Metropolitana"),
                    html.Div(id="tabla-comunas")
                ]),
                
                # Treemap de dimensiones e indicadores
                html.Div([
                    html.H3("Treemap de dimensiones e indicadores"),
                    html.Div(id="treemap-container")
                ])
            ], style={"padding": "15px"})
        ]),
        
        # Tab 2: Análisis por Dimensiones
        dcc.Tab(label="Análisis por Dimensiones", value="tab-dimensions", children=[
            html.Div([
                html.H3("Análisis por Dimensiones"),
                
                html.Div([
                    # Columna 1: Gráfico y selector
                    html.Div([
                        html.Label("Seleccionar categoría para dimensiones:"),
                        dcc.RadioItems(
                            id="filtro-dimensiones",
                            options=[
                                {"label": "Global", "value": "global"},
                                {"label": "Institucional", "value": "institucional"},
                                {"label": "Territorial", "value": "territorial"}
                            ],
                            value="global",
                            labelStyle={"display": "inline-block", "marginRight": "20px"}
                        ),
                        dcc.Graph(id="grafico-dimensiones")
                    ], className="six columns"),
                    
                    # Columna 2: Info y estadísticas
                    html.Div([
                        html.Div([
                            html.H4("¿Qué son las dimensiones?"),
                            html.P("Las dimensiones representan áreas funcionales o temáticas dentro de las categorías principales. Cada dimensión agrupa información relacionada con un aspecto específico de la gestión institucional o territorial, facilitando la organización y recuperación de la información.")
                        ], style={"backgroundColor": colors["gray_bg"], "padding": "15px", "borderRadius": "10px", "marginTop": "35px"}),
                        
                        html.H4("Estadísticas por Dimensión"),
                        html.Div(id="tabla-dimensiones")
                    ], className="six columns"),
                ], className="row"),
                
                # Heatmap
                html.Div([
                    html.H3("Relación entre Tipos de Archivos y Dimensiones"),
                    dcc.Graph(id="heatmap-extension-dimension"),
                    
                    html.Div([
                        html.H4("¿Qué nos muestra este mapa de calor?"),
                        html.P("Este mapa de calor muestra la concentración de diferentes tipos de archivos en cada dimensión, permitiendo identificar:"),
                        html.Ul([
                            html.Li("Qué formatos son más utilizados en cada dimensión"),
                            html.Li("Posibles patrones de uso específicos por área temática"),
                            html.Li("Dimensiones con mayor diversidad o especialización en formatos")
                        ]),
                        html.P("Esta información puede ser útil para entender mejor los flujos de trabajo y necesidades de información en diferentes áreas de la organización.")
                    ], style={"backgroundColor": colors["gray_bg"], "padding": "15px", "borderRadius": "10px"})
                ])
            ], style={"padding": "15px"})
        ]),
        
        # Tab 3: Análisis de Estado Indicadores
        dcc.Tab(label="Análisis de Estado Indicadores", value="tab-indicators", children=[
            html.Div([
                html.H3("Análisis de Estados de Indicadores"),
                
                html.Div([
                    # Columna 1: Controles
                    html.Div([
                        html.Label("Filtrar por origen:"),
                        dcc.RadioItems(
                            id="filtro-origen-estado",
                            options=[
                                {"label": "Todos", "value": "Todos"},
                                {"label": "Institucional", "value": "Institucional"},
                                {"label": "Territorial", "value": "Territorial"}
                            ],
                            value="Todos",
                            labelStyle={"display": "block", "marginBottom": "10px"}
                        ),
                        
                        html.Label("Agrupar por:"),
                        dcc.RadioItems(
                            id="agrupar-por-estado",
                            options=[
                                {"label": "Estado", "value": "Estado"},
                                {"label": "Dimensión", "value": "Dimensión"},
                                {"label": "Origen", "value": "Origen"}
                            ],
                            value="Estado",
                            labelStyle={"display": "block", "marginBottom": "10px"}
                        ),
                        
                        html.H4("Resumen"),
                        html.Div(id="resumen-estados")
                    ], className="three columns"),
                    
                    # Columna 2: Gráfico
                    html.Div([
                        dcc.Graph(id="grafico-estados")
                    ], className="nine columns"),
                ], className="row"),
                
                # Interpretación de Estados
                html.Div([
                    html.H4("Interpretación de Estados"),
                    html.Ul([
                        html.Li([html.Strong("PENDIENTE:"), " Indicadores que aún no han iniciado su implementación"]),
                        html.Li([html.Strong("EN PROCESO:"), " Indicadores que están actualmente en fase de implementación"]),
                        html.Li([html.Strong("LISTO:"), " Indicadores que han sido completados"])
                    ])
                ], style={"backgroundColor": colors["gray_bg"], "padding": "15px", "borderRadius": "10px", "marginTop": "10px"}),
                
                # Tabla de datos detallados
                html.Details([
                    html.Summary("Ver datos detallados"),
                    html.Div(id="tabla-datos-detallados")
                ], style={"marginTop": "15px"})
            ], style={"padding": "15px"})
        ]),
        
        # Tab 4: Insights Adicionales
        dcc.Tab(label="Insights Adicionales", value="tab-insights", children=[
            html.Div([
                html.H3("Insights Adicionales"),
                
                html.Div([
                    # Columna 1: Métodos obtención
                    html.Div([
                        html.H4("Métodos de Obtención de Archivos"),
                        dcc.Graph(id="grafico-metodos-obtencion")
                    ], className="six columns"),
                    
                    # Columna 2: Info
                    html.Div([
                        html.Div([
                            html.H4("Fuentes de información"),
                            html.P("Los archivos del Data Lake provienen de diferentes fuentes, lo que influye en su formato, estructura y calidad. Las principales fuentes son:"),
                            html.Ul([
                                html.Li([html.Strong("Web Scraping:"), " Datos extraídos automáticamente de sitios web"]),
                                html.Li([html.Strong("Universidad:"), " Documentos generados internamente por la institución"]),
                                html.Li([html.Strong("Descargados:"), " Archivos obtenidos de fuentes externas como portales oficiales"])
                            ])
                        ], style={"backgroundColor": colors["gray_bg"], "padding": "15px", "borderRadius": "10px", "marginTop": "35px"})
                    ], className="six columns"),
                ], className="row"),
                
                # Análisis de tamaño
                html.Div([
                    html.H4("Tamaño de Archivos por Extensión"),
                    html.Div(id="tabla-tamano-archivos")
                ]),
                
                # Conclusiones
                html.Div([
                    html.H4("Conclusiones generales"),
                    html.P("El análisis del Data Lake revela patrones importantes sobre cómo se almacena y organiza la información en la organización:"),
                    html.Ul([
                        html.Li(["La mayor parte de los archivos son de tipo ", html.Strong("hoja de cálculo"), ", indicando un enfoque en análisis de datos cuantitativos"]),
                        html.Li(["Existe una diferencia notable entre la cantidad de archivos ", html.Strong("institucionales"), " versus ", html.Strong("territoriales")]),
                        html.Li("Cada dimensión muestra preferencias específicas por ciertos formatos, reflejando sus necesidades particulares")
                    ]),
                    html.P("Esta información puede utilizarse para optimizar la gestión documental, mejorar los procesos de captura de datos y facilitar el acceso a la información relevante.")
                ], style={"backgroundColor": colors["gray_bg"], "padding": "15px", "borderRadius": "10px"})
            ], style={"padding": "15px"})
        ]),
        
        # Tab 5: Mapa Geográfico
        dcc.Tab(label="Mapa Geográfico", value="tab-map", children=[
            html.Div([
                html.H3("Mapa de la Región Metropolitana"),
                html.P("Este mapa muestra las diferentes provincias y comunas de la Región Metropolitana."),
                
                # Contenedor para el mapa HTML
                html.Iframe(
                    id="mapa-geografico",
                    src="/assets/mapa_rm_final.html",
                    style={"width": "100%", "height": "600px", "border": "none"}
                ),
                
                # Información adicional
                html.Div([
                    html.H4("Acerca del mapa"),
                    html.P("Este mapa interactivo muestra la distribución territorial de la Región Metropolitana de Santiago, con sus diferentes provincias identificadas por colores:"),
                    html.Ul([
                        html.Li([html.Strong("Santiago:"), " Zona central y de mayor densidad de población"]),
                        html.Li([html.Strong("Cordillera:"), " Zona este, limítrofe con la cordillera de los Andes"]),
                        html.Li([html.Strong("Chacabuco:"), " Zona norte de la región"]),
                        html.Li([html.Strong("Maipo:"), " Zona sur"]),
                        html.Li([html.Strong("Melipilla:"), " Zona suroeste"]),
                        html.Li([html.Strong("Talagante:"), " Zona oeste"])
                    ]),
                    html.P("Puedes interactuar con el mapa para ver información detallada de cada comuna.")
                ], style={"backgroundColor": colors["gray_bg"], "padding": "15px", "borderRadius": "10px", "marginTop": "20px"})
            ], style={"padding": "15px"})
        ]),
        
        # Tab 6: Mapa Sedes
        dcc.Tab(label="Mapa Sedes", value="tab-sedes", children=[
            html.Div([
                html.H3("Mapa de Sedes UTEM"),
                
                # Contenedor para el mapa HTML
                html.Iframe(
                    id="mapa-sedes",
                    src="/assets/mapa_sedes_utem.html",
                    style={"width": "100%", "height": "600px", "border": "none"}
                )
            ], style={"padding": "15px"})
        ]),
        
        # Tab 7: Análisis de Archivos
        dcc.Tab(label="Análisis de Archivos", value="tab-files", children=[
            html.Div([
                html.H3("Análisis archivos"),
                
                html.Div([
                    # Columna 1: Institucional vs Territorial
                    html.Div([
                        dcc.Graph(id="grafico-inst-terr"),
                        
                        html.Div([
                            html.H4("¿Qué nos muestra este gráfico?"),
                            html.P("Este gráfico muestra la distribución de archivos entre las categorías Institucional y Territorial, permitiendo identificar rápidamente el balance entre estos dos tipos de información en el Data Lake.")
                        ], style={"backgroundColor": colors["gray_bg"], "padding": "15px", "borderRadius": "10px"})
                    ], className="six columns"),
                    
                    # Columna 2: Distribución extensiones
                    html.Div([
                        dcc.Graph(id="grafico-extensiones-general"),
                        
                        html.Div([
                            html.H4("Tipos de archivos en el Data Lake"),
                            html.P("La distribución de tipos de archivos nos permite entender qué formatos predominan en el repositorio, lo que refleja los tipos de datos y documentos más utilizados en la organización.")
                        ], style={"backgroundColor": colors["gray_bg"], "padding": "15px", "borderRadius": "10px"})
                    ], className="six columns"),
                ], className="row"),
                
                # Comparación extensiones
                html.Div([
                    html.H3("Comparación de Tipos de Archivos por Categoría"),
                    dcc.Graph(id="grafico-comparativo-extensiones"),
                    
                    html.Div([
                        html.H4("Diferencias entre categorías"),
                        html.P("Esta comparación permite identificar si existen patrones o preferencias diferentes en el uso de formatos de archivos entre las áreas institucionales y territoriales. Esto puede reflejar diferentes necesidades o flujos de trabajo específicos para cada categoría.")
                    ], style={"backgroundColor": colors["gray_bg"], "padding": "15px", "borderRadius": "10px"})
                ]),
                
                # Análisis detallado por categoría
                html.Div([
                    html.H3("Análisis Detallado por Tipo de Archivo"),
                    
                    html.Label("Seleccionar categoría:"),
                    dcc.RadioItems(
                        id="filtro-categoria-archivos",
                        options=[
                            {"label": "Global", "value": "global"},
                            {"label": "Institucional", "value": "institucional"},
                            {"label": "Territorial", "value": "territorial"}
                        ],
                        value="global",
                        labelStyle={"display": "inline-block", "marginRight": "20px"}
                    ),
                    
                    dcc.Graph(id="grafico-extensiones-filtrado"),
                    
                    html.H4(id="titulo-top-extensiones"),
                    html.Div(id="tabla-top-extensiones"),
                    
                    html.Div([
                        html.H4("Interpretación de los tipos de archivos"),
                        html.P("Los diferentes tipos de archivos tienen propósitos específicos:"),
                        html.Ul([
                            html.Li([html.Strong(".xlsx/.xls:"), " Hojas de cálculo para análisis de datos, registros y reportes cuantitativos"]),
                            html.Li([html.Strong(".pdf:"), " Documentos formales, informes finales, documentación oficial"]),
                            html.Li([html.Strong(".docx/.doc:"), " Documentos de texto, informes en proceso, documentación detallada"]),
                            html.Li([html.Strong(".pptx/.ppt:"), " Presentaciones para reuniones y exposiciones"]),
                            html.Li([html.Strong(".csv:"), " Datos estructurados para análisis y procesamiento"])
                        ]),
                        html.P("La predominancia de ciertos formatos puede indicar el enfoque principal del trabajo en cada área.")
                    ], style={"backgroundColor": colors["gray_bg"], "padding": "15px", "borderRadius": "10px"})
                ])
            ], style={"padding": "15px"})
        ])
    ])
], style={"fontFamily": "Arial, sans-serif", "margin": "0 auto", "maxWidth": "1200px"})

# Callback para cargar tabla de comunas
@app.callback(
    Output('tabla-comunas', 'children'),
    Input('main-tabs', 'value')
)
def actualizar_tabla_comunas(tab):
    if tab == 'tab-general':
        df_comunas = cargar_datos_comunas()
        if not df_comunas.empty:
            return dash_table.DataTable(
                data=df_comunas.to_dict('records'),
                columns=[{"name": col, "id": col} for col in df_comunas.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={
                    'backgroundColor': colors["blue"],
                    'color': 'white',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ]
            )
        else:
            return html.Div("No se pudo cargar la información de comunas.")
    return dash.no_update

# Callback para cargar gráfico de dimensiones
@app.callback(
    Output('grafico-dimensiones', 'figure'),
    Input('filtro-dimensiones', 'value')
)
def actualizar_grafico_dimensiones(filtro):
    filtro_real = None if filtro == 'global' else filtro
    return crear_grafico_dimensiones(df, filtro_real)

# Callback para cargar tabla de dimensiones
@app.callback(
    Output('tabla-dimensiones', 'children'),
    Input('filtro-dimensiones', 'value')
)
def actualizar_tabla_dimensiones(filtro):
    # Filtrar según selección
    filtro_real = None if filtro == 'global' else filtro
    if filtro_real == 'institucional':
        df_stat = df[df['institucional'] == True]
    elif filtro_real == 'territorial':
        df_stat = df[df['territorial'] == True]
    else:
        df_stat = df
        
    # Calcular estadísticas de dimensiones sin "Sin clasificación"
    df_dims = df_stat[df_stat['dimensiones'] != 'Sin clasificación']

    if not df_dims.empty:
        dim_stats = df_dims['dimensiones'].value_counts()
        
        # Cargar nombres de dimensiones
        try:
            nombres_dimensiones = pd.read_csv("data/nombres_dimensiones.csv")
            # Crear un diccionario para mapear id a nombre
            dict_dimensiones = dict(zip(nombres_dimensiones['id_dim'], nombres_dimensiones['nombre_dim']))
        except:
            dict_dimensiones = {}
        
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
                'Dimensión': dim, 
                'Nombre Dimensión': nombre_completo,
                'Total Archivos': dim_stats[dim],
                'Porcentaje': round(dim_stats[dim] / dim_stats.sum() * 100, 1)
            })
        
        # Crear DataFrame y ordenar por dimensión
        dim_df = pd.DataFrame(data)
        dim_df = dim_df.sort_values('Dimensión')
        
        # Mostrar la tabla
        return dash_table.DataTable(
            data=dim_df.to_dict('records'),
            columns=[{"name": col, "id": col} for col in dim_df.columns if col != 'Número'],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '8px'},
            style_header={
                'backgroundColor': colors["blue"],
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )
    else:
        return html.Div("No hay datos de dimensiones disponibles para esta selección.")

# Callback para cargar heatmap
@app.callback(
    Output('heatmap-extension-dimension', 'figure'),
    Input('main-tabs', 'value')
)
def actualizar_heatmap(tab):
    if tab == 'tab-dimensions':
        return crear_heatmap_extension_dimension(df)
    return dash.no_update

# Callback para actualizar gráfico de estados
@app.callback(
    [Output('grafico-estados', 'figure'),
     Output('resumen-estados', 'children'),
     Output('tabla-datos-detallados', 'children')],
    [Input('filtro-origen-estado', 'value'),
     Input('agrupar-por-estado', 'value')]
)

# Callback para actualizar gráfico de estados
@app.callback(
    [Output('grafico-estados', 'figure'),
     Output('resumen-estados', 'children'),
     Output('tabla-datos-detallados', 'children')],
    [Input('filtro-origen-estado', 'value'),
     Input('agrupar-por-estado', 'value')]
)
def actualizar_grafico_estados(origen, agrupar_por):
    if not df_indicadores.empty:
        # Crear el gráfico y obtener datos de resumen
        fig, conteo, df_filtrado = crear_grafico_estados(df_indicadores, origen, agrupar_por)
        
        # Crear resumen
        total = conteo['conteo'].sum()
        resumen = []
        for i, row in conteo.iterrows():
            porcentaje = round(row['conteo'] / total * 100, 1)
            resumen.append(html.P(f"{row['categoria']}: {row['conteo']} ({porcentaje}%)"))
        
        # Crear tabla de datos detallados
        tabla = dash_table.DataTable(
            data=df_filtrado[['ID', 'Dimension', 'Estado', 'Origen']].to_dict('records'),
            columns=[{"name": col, "id": col} for col in ['ID', 'Dimension', 'Estado', 'Origen']],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '8px'},
            style_header={
                'backgroundColor': colors["blue"],
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )
        
        return fig, resumen, tabla
    
    # Si no hay datos, devolver valores por defecto
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="No se pudieron cargar los datos de indicadores",
        height=450
    )
    return empty_fig, html.P("No hay datos disponibles"), html.Div("No hay datos disponibles")

# Callback para gráfico de métodos de obtención
@app.callback(
    Output('grafico-metodos-obtencion', 'figure'),
    Input('main-tabs', 'value')
)
def actualizar_grafico_metodos(tab):
    if tab == 'tab-insights':
        return crear_grafico_metodos_obtencion()
    return dash.no_update

# Callback para tabla de tamaño de archivos
@app.callback(
    Output('tabla-tamano-archivos', 'children'),
    Input('main-tabs', 'value')
)
def actualizar_tabla_tamanos(tab):
    if tab == 'tab-insights':
        tamano_por_ext = analisis_tamano_archivos(df)
        return dash_table.DataTable(
            data=tamano_por_ext.to_dict('records'),
            columns=[{"name": col, "id": col} for col in tamano_por_ext.columns],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '8px'},
            style_header={
                'backgroundColor': colors["blue"],
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )
    return dash.no_update

# Callbacks para la pestaña de análisis de archivos
@app.callback(
    Output('grafico-inst-terr', 'figure'),
    Input('main-tabs', 'value')
)
def actualizar_grafico_inst_terr(tab):
    if tab == 'tab-files':
        return crear_grafico_institucional_territorial(df)
    return dash.no_update

@app.callback(
    Output('grafico-extensiones-general', 'figure'),
    Input('main-tabs', 'value')
)
def actualizar_grafico_extensiones_general(tab):
    if tab == 'tab-files':
        return crear_grafico_extensiones(df)
    return dash.no_update

@app.callback(
    Output('grafico-comparativo-extensiones', 'figure'),
    Input('main-tabs', 'value')
)
def actualizar_grafico_comparativo_extensiones(tab):
    if tab == 'tab-files':
        return crear_grafico_comparativo_extensiones(df)
    return dash.no_update

@app.callback(
    [Output('grafico-extensiones-filtrado', 'figure'),
     Output('titulo-top-extensiones', 'children'),
     Output('tabla-top-extensiones', 'children')],
    [Input('filtro-categoria-archivos', 'value')]
)
def actualizar_analisis_extensiones(filtro):
    filtro_real = None if filtro == 'global' else filtro
    
    # Filtrar según selección
    if filtro_real == 'institucional':
        df_temp = df[df['institucional'] == True]
        titulo = "Institucional"
    elif filtro_real == 'territorial':
        df_temp = df[df['territorial'] == True]
        titulo = "Territorial"
    else:
        df_temp = df
        titulo = "Global"
    
    # Crear gráfico filtrado
    fig = crear_grafico_extensiones(df, filtro_real)
    
    # Calcular top 5 extensiones
    top_ext = df_temp['extension'].value_counts().head(5)
    top_ext_df = pd.DataFrame({
        'Extensión': top_ext.index,
        'Cantidad': top_ext.values,
        'Porcentaje': (top_ext.values / len(df_temp) * 100).round(1)
    })
    
    # Crear tabla
    tabla = dash_table.DataTable(
        data=top_ext_df.to_dict('records'),
        columns=[{"name": col, "id": col} for col in top_ext_df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '8px'},
        style_header={
            'backgroundColor': colors["blue"],
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ]
    )
    
    return fig, f"Top 5 Extensiones - {titulo}", tabla

# Callback para cargar el iframe del mapa
@app.callback(
    Output('mapa-geografico', 'src'),
    Input('main-tabs', 'value')
)
def actualizar_mapa_geografico(tab):
    if tab == 'tab-map':
        # Verificar si el archivo existe primero
        if os.path.exists("assets/mapa_rm_final.html"):
            return "/assets/mapa_rm_final.html"
        # Intentar cargar desde la ruta original en el código streamlit
        elif os.path.exists("mapas_html/mapa_rm_final.html"):
            # Copiar el archivo a assets si es necesario
            import shutil
            os.makedirs("assets", exist_ok=True)
            shutil.copy("mapas_html/mapa_rm_final.html", "assets/mapa_rm_final.html")
            return "/assets/mapa_rm_final.html"
        else:
            return "about:blank"
    return dash.no_update

# Callback para cargar el iframe del mapa de sedes
@app.callback(
    Output('mapa-sedes', 'src'),
    Input('main-tabs', 'value')
)
def actualizar_mapa_sedes(tab):
    if tab == 'tab-sedes':
        # Verificar si el archivo existe primero
        if os.path.exists("assets/mapa_sedes_utem.html"):
            return "/assets/mapa_sedes_utem.html"
        # Intentar cargar desde la ruta original en el código streamlit
        elif os.path.exists("mapas_html/mapa_sedes_utem.html"):
            # Copiar el archivo a assets si es necesario
            import shutil
            os.makedirs("assets", exist_ok=True)
            shutil.copy("mapas_html/mapa_sedes_utem.html", "assets/mapa_sedes_utem.html")
            return "/assets/mapa_sedes_utem.html"
        else:
            return "about:blank"
    return dash.no_update

# Callback para mostrar treemap de dimensiones
@app.callback(
    Output('treemap-container', 'children'),
    Input('main-tabs', 'value')
)
def actualizar_treemap(tab):
    if tab == 'tab-general':
        try:
            # Verificar archivos disponibles
            archivos_disp = [f for f in os.listdir('data') if f.endswith('.csv')]
            
            if not any(f.lower() in ['institucional.csv', 'territorial.csv'] for f in archivos_disp):
                return html.Div([
                    html.P("No se encontraron los archivos necesarios: Institucional.csv y territorial.csv", 
                          style={"color": "red"}),
                    html.P(f"Archivos CSV disponibles: {', '.join(archivos_disp)}"),
                    html.P(f"Directorio actual: {os.getcwd()}")
                ])
            
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
                            print(f"Error al cargar {nombre_archivo} con {encoding}: {str(e)}")
                            continue
                    
                    # Si ninguna codificación funcionó
                    return None
                except Exception as e:
                    print(f"Error inesperado al cargar {nombre_archivo}: {str(e)}")
                    return None
            
            # Cargar los dataframes
            institucional_df = cargar_csv_seguro('data/Institucional.csv')
            territorial_df = cargar_csv_seguro('data/territorial.csv')
            
            # Verificar si se cargaron los datos
            if institucional_df is None or territorial_df is None:
                return html.Div("No se pudieron cargar uno o ambos archivos CSV.", style={"color": "red"})
            
            # Usar datos de indicadores si los archivos no tienen la estructura esperada
            if 'Dimension' not in institucional_df.columns or 'Indicador' not in institucional_df.columns:
                # Verificar si se cargó el archivo de indicadores
                if df_indicadores.empty:
                    return html.Div("No se pudieron cargar los datos de indicadores.", style={"color": "red"})
                
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
            
            # Agregar números de indicador
            institucional_df = institucional_df.reset_index(drop=True)
            territorial_df = territorial_df.reset_index(drop=True)
            
            institucional_df['Indicador_Numerado'] = [f"I_{i+1}: {ind}" for i, ind in enumerate(institucional_df['Indicador'])]
            territorial_df['Indicador_Numerado'] = [f"T_{i+1}: {ind}" for i, ind in enumerate(territorial_df['Indicador'])]
            
            # Preparar los datos
            institucional_df['Valor'] = 10
            institucional_df['Categoria'] = 'Institucional'
            territorial_df['Valor'] = 10
            territorial_df['Categoria'] = 'Territorial'
            
            # Combinar ambos dataframes
            df_combined = pd.concat([institucional_df, territorial_df], ignore_index=True)
            
            # Verificar columnas necesarias
            columnas_requeridas = ['Categoria', 'Dimension', 'Indicador_Numerado', 'Valor']
            columnas_faltantes = [col for col in columnas_requeridas if col not in df_combined.columns]
            
            if columnas_faltantes:
                return html.Div([
                    html.P(f"Faltan columnas requeridas: {', '.join(columnas_faltantes)}", style={"color": "red"}),
                    html.P(f"Columnas disponibles: {', '.join(df_combined.columns.tolist())}")
                ])
            
            # Crear treemap
            fig = px.treemap(
                df_combined,
                path=['Categoria', 'Dimension', 'Indicador_Numerado'],
                values='Valor',
                color='Categoria',
                color_discrete_map={
                    'Institucional': colors["blue"],
                    'Territorial': colors["yellow"]
                }
            )
            
            # Actualizar diseño
            fig.update_traces(
                textfont=dict(size=14),
                texttemplate='%{label}',
                hovertemplate='<b>%{label}</b><br>Categoría: %{root}<br>Dimensión: %{parent}'
            )
            
            fig.update_layout(
                margin=dict(t=50, l=25, r=25, b=25),
                height=800,
                template='plotly_white'
            )
            
            # Crear leyenda detallada en tabs
            return html.Div([
                dcc.Graph(figure=fig),
                
                html.H4("Leyenda detallada de indicadores"),
                dcc.Tabs([
                    dcc.Tab(label="Indicadores Institucionales", children=[
                        html.Div([
                            html.P([html.Strong(f"I_{i+1}: "), html.Span(row['Indicador'])]) 
                            for i, row in institucional_df.iterrows()
                        ], style={"padding": "15px"})
                    ]),
                    dcc.Tab(label="Indicadores Territoriales", children=[
                        html.Div([
                            html.P([html.Strong(f"T_{i+1}: "), html.Span(row['Indicador'])]) 
                            for i, row in territorial_df.iterrows()
                        ], style={"padding": "15px"})
                    ])
                ])
            ])
            
        except Exception as e:
            return html.Div([
                html.P("Error al crear el treemap:", style={"color": "red"}),
                html.P(str(e))
            ])
    
    return dash.no_update

# Correr la aplicación
if __name__ == '__main__':
    # Asegurarse que la carpeta assets exista para servir archivos estáticos
    os.makedirs('assets', exist_ok=True)
    
    # Copiar archivos de mapas si existen
    import shutil
    if os.path.exists("mapas_html"):
        for map_file in ["mapa_rm_final.html", "mapa_sedes_utem.html"]:
            if os.path.exists(f"mapas_html/{map_file}"):
                try:
                    shutil.copy(f"mapas_html/{map_file}", f"assets/{map_file}")
                except Exception as e:
                    print(f"No se pudo copiar {map_file}: {str(e)}")
    
    app.run_server(debug=True, port=8050)