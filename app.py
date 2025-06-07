import os
import streamlit as st
import pandas as pd
import pyodbc
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# Descargar recursos de NLTK (solo la primera vez)
nltk.download('stopwords')
nltk.download('vader_lexicon')

# --------------------------------
# 1. Conexión a SQL Server usando variables de entorno
# --------------------------------
@st.cache_resource
def get_connection():
    server   = os.getenv("DB_SERVER")
    database = os.getenv("DB_NAME")
    user     = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    if not all([server, database, user, password]):
        st.error("🔴 Faltan variables de entorno para la conexión SQL Server. Revisa los Secrets.")
        st.stop()

    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={user};"
        f"PWD={password}"
    )
    return pyodbc.connect(conn_str)

# --------------------------------
# 2. Cargar y procesar Posts
# --------------------------------
@st.cache_data
def load_posts():
    conn = get_connection()
    query = "SELECT * FROM Posts"
    df = pd.read_sql(query, conn)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    if 'Reacciones' not in df.columns:
        df['Reacciones'] = 0
    return df[['Post_ID', 'Fecha', 'Autor', 'Mensaje', 'Reacciones']].copy()

# --------------------------------
# 3. Cargar y procesar Comments
# --------------------------------
@st.cache_data
def load_comments():
    conn = get_connection()
    query = "SELECT * FROM Comments"
    df_c = pd.read_sql(query, conn)
    df_c['Fecha'] = pd.to_datetime(df_c['Fecha'])
    if 'Reacciones' not in df_c.columns:
        df_c['Reacciones'] = 0
    return df_c[['Comment_ID', 'Post_ID', 'Autor', 'Mensaje', 'Fecha', 'Reacciones']].copy()

# --------------------------------
# 4. Cargar DataFrames
# --------------------------------
df_posts = load_posts()
df_comments = load_comments()

# --------------------------------
# 5. Función para categorizar Posts con diccionario ampliado
# --------------------------------
@st.cache_data
def categorize_posts(df):
    categoria_dict = {
        "Política": [
            "gobierno", "elección", "parlamento", "ley", "ministro",
            "presidente", "alcalde", "congreso", "diputado", "senador",
            "votación", "campaña", "coalición", "partido", "oposición",
            "constitución"
        ],
        "Economía": [
            "economía", "inflación", "PIB", "mercado", "impuestos",
            "índice", "inversión", "banco", "dólar", "bolsa",
            "desempleo", "subsidio", "exportación", "importación",
            "crisis económica", "finanzas"
        ],
        "Salud": [
            "hospital", "vacuna", "pandemia", "covid", "salud",
            "paciente", "enfermero", "doctor", "ministerio de salud",
            "enfermedad", "vacunación", "epidemia", "emergencia sanitaria",
            "consultorio", "medicamento"
        ],
        "Educación": [
            "escuela", "universidad", "colegio", "profesor", "alumno",
            "matrícula", "prueba", "examen", "currículo", "docente",
            "taller", "bachillerato", "educación superior", "pedagogía"
        ],
        "Seguridad": [
            "delito", "policía", "robo", "homicidio", "seguridad ciudadana",
            "justicia", "tribunal", "carabinero", "prisión", "recluso",
            "vigilancia", "crimen", "violencia", "menor infractor",
            "accidente"
        ],
        "Medio Ambiente": [
            "contaminación", "reciclaje", "cambio climático", "deforestación",
            "biodiversidad", "remanente forestal", "vertedero", "energías renovables",
            "agua", "basura", "aire", "sustentabilidad", "biocombustible"
        ],
        "Tecnología": [
            "tecnología", "software", "hardware", "inteligencia artificial",
            "innovación", "start-up", "aplicación", "redes sociales",
            "ciberseguridad", "blockchain", "criptomoneda", "internet",
            "transformación digital"
        ],
        "Cultura": [
            "museo", "exposición", "teatro", "cine", "libro", "arte",
            "concierto", "festival", "danza", "pintura", "escultura",
            "fotografía", "música", "teatro comunitario", "patrimonio"
        ],
        "Deportes": [
            "fútbol", "tenis", "olímpico", "messi", "torneo", "jugador",
            "selección", "liga", "partido", "entrenador", "maratón",
            "juegos", "deporte femenino", "estadio"
        ],
        "Internacional": [
            "onu", "relaciones exteriores", "conflicto", "guerra",
            "tratado", "acuerdo", "embargo", "diplomático", "visita oficial",
            "sánchez", "biden", "putin", "xi jinping", "ue", "otan"
        ],
        "Sociedad": [
            "comunidad", "vecino", "manifestación", "protesta",
            "derechos humanos", "inmigración", "desigualdad", "pobreza",
            "migrante", "niñez", "juventud", "embarazo adolescente",
            "movilidad social"
        ],
        "Otros": []
    }

    def asignar_categoria(texto):
        if not isinstance(texto, str):
            return "Otros"
        texto_min = texto.lower()
        for cat, keywords in categoria_dict.items():
            for kw in keywords:
                pattern = rf"\b{re.escape(kw)}\b"
                if re.search(pattern, texto_min):
                    return cat
        return "Otros"

    df['Categoria'] = df['Mensaje'].apply(asignar_categoria)
    return df

df_posts = categorize_posts(df_posts)

# 5.1 Asignar categoría a Comments (basado en categoría del Post padre)
df_comments = df_comments.merge(df_posts[['Post_ID', 'Categoria']], on='Post_ID', how='left')
df_comments['Categoria'] = df_comments['Categoria'].fillna("Otros")

# --------------------------------
# 6. Sidebar: selección de dataset, categoría, rango de fechas, n-gramas y opciones
# --------------------------------
st.sidebar.header("Filtros y Opciones")

# 6.1. Selector de dataset
dataset_option = st.sidebar.selectbox("Seleccionar dataset", ["Posts", "Comments"])

# 6.2. Selector de categoría
if dataset_option == "Posts":
    categorias = ["Todas"] + sorted(df_posts["Categoria"].unique().tolist())
else:
    categorias = ["Todas"] + sorted(df_comments["Categoria"].unique().tolist())
selected_category = st.sidebar.selectbox("Categoría", categorias)

# 6.3. Slider de rango de fechas
if dataset_option == "Posts":
    min_fecha = df_posts['Fecha'].min().date()
    max_fecha = df_posts['Fecha'].max().date()
else:
    min_fecha = df_comments['Fecha'].min().date()
    max_fecha = df_comments['Fecha'].max().date()
fecha_inicio, fecha_fin = st.sidebar.slider(
    "Rango de fechas",
    min_value=min_fecha,
    max_value=max_fecha,
    value=(min_fecha, max_fecha),
    format="YYYY-MM-DD"
)

# 6.4. Selector de tipo de n-grama (ahora incluye cuatrigramas)
ngram_option = st.sidebar.selectbox("Tipo de n-grama", ["Unigramas", "Bigramas", "Trigramas", "Cuatrigramas"])
if ngram_option == "Unigramas":
    ngram_range = (1, 1)
elif ngram_option == "Bigramas":
    ngram_range = (2, 2)
elif ngram_option == "Trigramas":
    ngram_range = (3, 3)
else:  # Cuatrigramas
    ngram_range = (4, 4)

# 6.5. Checkbox para usar métricas de Likes
use_likes = st.sidebar.checkbox("Mostrar métrica de Likes (sumatoria)", value=False)

# 6.6. Input para búsqueda de palabras clave en Posts
post_search = st.sidebar.text_input("Buscar en Posts", "")

# --------------------------------
# 7. Filtrado según selección para Posts o Comments
# --------------------------------
if dataset_option == "Posts":
    df = df_posts.copy()
else:
    df = df_comments.copy()

# Filtrar por categoría
if selected_category != "Todas":
    df = df[df["Categoria"] == selected_category]

# Filtrar por fechas
mask = (df['Fecha'].dt.date >= fecha_inicio) & (df['Fecha'].dt.date <= fecha_fin)
df = df.loc[mask]

# Filtrar por palabra clave en Posts si aplica
if dataset_option == "Posts" and post_search.strip() != "":
    df = df[df['Mensaje'].str.contains(post_search, case=False, na=False)]

# --------------------------------
# 8. KPI Dashboard
# --------------------------------
st.title("Análisis Avanzado de N-gramas, Sentimiento y Temas con KPIs")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Registros", f"{len(df)}")
avg_reacts = df["Reacciones"].mean() if "Reacciones" in df.columns else 0
col2.metric("Reacciones Promedio", f"{avg_reacts:.2f}")

if dataset_option == "Comments":
    sia = SentimentIntensityAnalyzer()
    df['compound_tmp'] = df['Mensaje'].fillna("").apply(lambda t: sia.polarity_scores(str(t))['compound'])
    avg_sent = df['compound_tmp'].mean()
    if avg_sent < -0.1:
        status = "🔴 Negativo"
    elif avg_sent > 0.1:
        status = "🟢 Positivo"
    else:
        status = "🟠 Neutral"
    col3.metric("Sentimiento Promedio", f"{avg_sent:.2f}", status)
else:
    col3.metric("Sentimiento Promedio", "N/A")

num_cat = df['Categoria'].nunique() if 'Categoria' in df.columns else 0
col4.metric("Categorías Activas", f"{num_cat}")

# --------------------------------
# 9. Si buscamos en Posts, mostrar n-grams de Comments relacionados
# --------------------------------
if dataset_option == "Posts" and post_search.strip() != "":
    st.header("N-gramas de Comentarios Relacionados a los Posts Encontrados")
    # Obtener IDs de posts filtrados que contienen la búsqueda
    post_ids = df['Post_ID'].unique().tolist()
    # Filtrar comentarios cuyos Post_ID estén en esa lista
    df_related_comments = df_comments[df_comments['Post_ID'].isin(post_ids)].copy()
    if not df_related_comments.empty:
        textos_rel = df_related_comments['Mensaje'].dropna().astype(str)
        base_stop = list(stopwords.words("spanish"))
        extra_stop = [
            "http", "https", "www", "com", "org", "net", "ftp",
            "https://", "http://", "://", "url", "rt"
        ]
        spanish_stopwords = base_stop + extra_stop
        @st.cache_data
        def gen_ngrams_relat(text_series: pd.Series, ngram_range=(1, 1)):
            vectorizer = CountVectorizer(
                ngram_range=ngram_range,
                stop_words=spanish_stopwords
            )
            X = vectorizer.fit_transform(text_series)
            ngrams = vectorizer.get_feature_names_out()
            counts = X.sum(axis=0).A1
            return pd.DataFrame({"ngram": ngrams, "count": counts}).sort_values(by="count", ascending=False)

        df_ngrams_rel = gen_ngrams_relat(textos_rel, ngram_range=ngram_range)
        st.subheader(f"Top N-gramas en Comentarios relacionados ({ngram_option})")
        st.dataframe(df_ngrams_rel.head(50), use_container_width=True)
    else:
        st.write("No hay comentarios relacionados con los posts encontrados.")

# --------------------------------
# 10. Detección de Outliers por Reacciones
# --------------------------------
st.header("Detección de Outliers por Reacciones")

if len(df) > 0 and "Reacciones" in df.columns:
    mean_reacts = df["Reacciones"].mean()
    std_reacts = df["Reacciones"].std()
    threshold = mean_reacts + 2 * std_reacts
    df_outliers = df[df["Reacciones"] > threshold].copy()
    st.markdown(f"**Umbral de outlier (media + 2·desviación): {threshold:.2f}**")
    if not df_outliers.empty:
        id_col = "Post_ID" if dataset_option == "Posts" else "Comment_ID"
        st.dataframe(
            df_outliers[[id_col, "Autor", "Fecha", "Reacciones", "Mensaje"]]
            .sort_values(by="Reacciones", ascending=False),
            use_container_width=True
        )
    else:
        st.write("No se encontraron registros fuera del umbral de outliers.")
else:
    st.write("No hay datos suficientes para detectar outliers.")

# --------------------------------
# 11. Generación de N-gramas y cálculo de likes
# --------------------------------
st.header(f"Top N-gramas ({ngram_option})")

# Filtramos los textos nulos para que el conteo de n-gramas sea correcto
textos = df["Mensaje"].dropna().astype(str)

base_stop = list(stopwords.words("spanish"))
extra_stop = [
    "http", "https", "www", "com", "org", "net", "ftp",
    "https://", "http://", "://", "url", "rt"
]
spanish_stopwords = base_stop + extra_stop

@st.cache_data
def generar_ngrams(text_series: pd.Series, ngram_range=(1, 1), likes=None):
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        stop_words=spanish_stopwords
    )
    X = vectorizer.fit_transform(text_series)
    ngrams = vectorizer.get_feature_names_out()
    counts = X.sum(axis=0).A1
    df_ngrams = pd.DataFrame({"ngram": ngrams, "count": counts})
    if likes is not None:
        likes_array = likes.values
        likes_sum = X.T.dot(likes_array)
        df_ngrams["likes_sum"] = likes_sum
    return df_ngrams.sort_values(by="count", ascending=False)

# Alinear el vector de reacciones con los textos utilizados
likes_param = df.loc[textos.index, "Reacciones"] if use_likes else None
df_ngrams = generar_ngrams(textos, ngram_range=ngram_range, likes=likes_param)

if use_likes:
    st.markdown("*Ordenado por frecuencia de aparición*")
    st.dataframe(df_ngrams[["ngram", "count", "likes_sum"]].head(50), use_container_width=True)
    st.markdown("**Para ordenar por métricas de likes, haz clic en la cabecera 'likes_sum' en la tabla.**")
else:
    st.dataframe(df_ngrams[["ngram", "count"]].head(50), use_container_width=True)

# --------------------------------
# 12. Tendencias temporales de N-grama (opción en sidebar)
# --------------------------------
show_trend = st.sidebar.checkbox("Mostrar tendencias temporales de un N-grama", value=False)
if show_trend:
    ngram_for_trend = st.sidebar.text_input("Introduce un N-grama para su tendencia (exacto)", "")
    if ngram_for_trend.strip() != "":
        st.subheader(f"Tendencia Temporal para el N-grama: '{ngram_for_trend}'")
        df_temp = df.copy()
        df_temp['texto_lower'] = df_temp['Mensaje'].str.lower().fillna('')
        pattern = rf"\b{re.escape(ngram_for_trend.lower())}\b"
        df_temp['match_ngram'] = df_temp['texto_lower'].apply(lambda t: bool(re.search(pattern, t)))
        trend_series = df_temp.groupby(df_temp['Fecha'].dt.date)['match_ngram'].sum()
        fecha_index = pd.date_range(start=fecha_inicio, end=fecha_fin)
        trend_series = trend_series.reindex(fecha_index.date, fill_value=0)
        st.line_chart(trend_series)

# --------------------------------
# 13. Word Cloud (opción en sidebar)
# --------------------------------
show_wordcloud = st.sidebar.checkbox("Mostrar Word Cloud", value=False)
if show_wordcloud:
    st.subheader("Word Cloud del Conjunto Filtrado")
    combined_text = " ".join(textos)
    wc = WordCloud(
        width=800, height=400,
        background_color="white",
        stopwords=set(spanish_stopwords)
    ).generate(combined_text)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc)

# --------------------------------
# 14. Topic Modeling LDA (opción en sidebar)
# --------------------------------
show_topics = st.sidebar.checkbox("Mostrar Topic Modeling", value=False)
if show_topics:
    num_topics = st.sidebar.slider("Número de tópicos LDA", min_value=2, max_value=10, value=3)
    st.subheader(f"Topic Modeling con LDA ({num_topics} tópicos)")
    tf_vectorizer = CountVectorizer(
        stop_words=spanish_stopwords,
        max_df=0.95,
        min_df=2
    )
    tf_matrix = tf_vectorizer.fit_transform(textos)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda.fit(tf_matrix)
    words = tf_vectorizer.get_feature_names_out()
    topics = {}
    for idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-10:][::-1]
        top_words = [words[i] for i in top_indices]
        topics[f"Tópico {idx+1}"] = ", ".join(top_words)
    df_topics = pd.DataFrame.from_dict(topics, orient='index', columns=['Palabras Clave'])
    st.write(df_topics)

# --------------------------------
# 15. Correlación Sentimiento vs Reacciones (solo Comments)
# --------------------------------
show_corr = st.sidebar.checkbox("Mostrar correlación Sentimiento vs Likes (Comments)", value=False)
if dataset_option == "Comments" and show_corr:
    st.subheader("Correlación Sentimiento vs Reacciones (Comments)")
    sia = SentimentIntensityAnalyzer()
    df_corr = df.copy()
    df_corr['compound'] = df_corr['Mensaje'].fillna("").apply(lambda t: sia.polarity_scores(str(t))['compound'])
    fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
    ax_corr.scatter(df_corr['compound'], df_corr['Reacciones'], alpha=0.5)
    ax_corr.set_xlabel("Sentimiento (compound)")
    ax_corr.set_ylabel("Número de Reacciones")
    ax_corr.set_title("Sentimiento vs Reacciones")
    st.pyplot(fig_corr)

# --------------------------------
# 16. Indicaciones finales
# --------------------------------
st.markdown("""
- Ahora puedes analizar cuatrigramas además de uni-, bi- y trigramas.
- El KPI Dashboard muestra al inicio la visión general.
- La búsqueda en Posts despliega n-gramas de comentarios relacionados.
- Se incorpora detección de outliers por reacciones.
- En caso de querer ver tendencias de un n-grama, activa la casilla "Mostrar tendencias temporales de un N-grama".
- Para Word Cloud, activa "Mostrar Word Cloud".
- Para Topic Modeling, activa "Mostrar Topic Modeling" y elige el número de tópicos.
- Para correlación Sentimiento vs Reacciones en Comments, activa "Mostrar correlación Sentimiento vs Likes (Comments)".
- Recarga la app si se añaden nuevos datos a la base para actualizar rangos y métricas.
""")


