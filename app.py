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
from pysentimiento import create_analyzer

nltk.download('stopwords')
nltk.download('vader_lexicon')

@st.cache_resource
def get_senti_es():
    return create_analyzer(task="sentiment", lang="es")
sent_analyzer = get_senti_es()

@st.cache_resource
def get_emotion_es():
    return create_analyzer(task="emotion", lang="es")
emotion_analyzer = get_emotion_es()

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

@st.cache_data
def load_posts():
    conn = get_connection()
    query = "SELECT * FROM Posts"
    df = pd.read_sql(query, conn)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Post_ID'] = df['Post_ID'].astype(str)
    if 'Reacciones' not in df.columns:
        df['Reacciones'] = 0
    return df[['Post_ID', 'Fecha', 'Autor', 'Mensaje', 'Reacciones']].copy()

@st.cache_data
def load_comments():
    conn = get_connection()
    query = "SELECT * FROM Comments"
    df_c = pd.read_sql(query, conn)
    df_c['Fecha'] = pd.to_datetime(df_c['Fecha'])
    df_c['Post_ID'] = df_c['Post_ID'].astype(str)
    if 'Reacciones' not in df_c.columns:
        df_c['Reacciones'] = 0
    return df_c[['Comment_ID', 'Post_ID', 'Autor', 'Mensaje', 'Fecha', 'Reacciones']].copy()

df_posts = load_posts()
df_comments = load_comments()

@st.cache_data
def categorize_posts(df):
    categoria_dict = {
        "Política": [
            "gobierno", "elección", "parlamento", "ley", "ministro",
            "presidente", "alcalde", "congreso", "diputado", "senador",
            "votación", "campaña", "coalición", "partido", "oposición",
            "constitución","Gonzalo Winter","Gabriel Boric","cosejal","alcalde","luis malla","gobernador","jorge diaz",
            "republicanos","liberales","partido","toha" 
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
            "accidente","cerro chuño","delicuentes","delicuentes","ladron"
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
df_comments = df_comments.merge(df_posts[['Post_ID', 'Categoria']], on='Post_ID', how='left')
df_comments['Categoria'] = df_comments['Categoria'].fillna("Otros")

st.sidebar.header("Filtros y Opciones")
dataset_option = st.sidebar.selectbox("Seleccionar dataset", ["Posts", "Comments"])
if dataset_option == "Posts":
    categorias = ["Todas"] + sorted(df_posts["Categoria"].unique().tolist())
else:
    categorias = ["Todas"] + sorted(df_comments["Categoria"].unique().tolist())
selected_category = st.sidebar.selectbox("Categoría", categorias)

fixed_start = datetime.date(2025, 3, 1)
max_fecha = datetime.date.today()
fecha_inicio, fecha_fin = st.sidebar.slider(
    "Rango de fechas",
    min_value=fixed_start,
    max_value=max_fecha,
    value=(fixed_start, max_fecha),
    format="YYYY-MM-DD"
)
ngram_option = st.sidebar.selectbox(
    "Tipo de n-grama",
    ["Unigramas", "Bigramas", "Trigramas", "Cuatrigramas", "Pentagramas"]
)
ngram_range = {
    "Unigramas": (1,1),
    "Bigramas": (2,2),
    "Trigramas": (3,3),
    "Cuatrigramas": (4,4),
    "Pentagramas": (5,5)
}[ngram_option]
use_likes = st.sidebar.checkbox("Mostrar métrica de Likes (sumatoria)", value=False)
post_search = st.sidebar.text_input("Buscar en Posts", "")

if dataset_option == "Posts":
    df = df_posts.copy()
else:
    df = df_comments.copy()
if selected_category != "Todas":
    df = df[df["Categoria"] == selected_category]
mask = (df['Fecha'].dt.date >= fecha_inicio) & (df['Fecha'].dt.date <= fecha_fin)
df = df.loc[mask]
if dataset_option == "Posts" and post_search.strip() != "":
    df = df[df['Mensaje'].str.contains(post_search, case=False, na=False)]

# --- PREVIEW DEL MENSAJE PARA EL MULTISELECT Y TABLA ---
if dataset_option == "Posts" and not df.empty:
    df['Preview'] = df['Mensaje'].str.slice(0, 30).fillna('') + '...'

st.title("Análisis Avanzado de N-gramas, Sentimiento, Emociones y Temas con KPIs")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Registros", f"{len(df)}")
avg_reacts = df["Reacciones"].mean() if "Reacciones" in df.columns else 0
col2.metric("Reacciones Promedio", f"{avg_reacts:.2f}")
if dataset_option == "Comments":
    sia = SentimentIntensityAnalyzer()
    df['compound_tmp'] = df['Mensaje'].fillna("").apply(
        lambda t: sia.polarity_scores(str(t))['compound']
    )
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

# Buscador por ID de Post (siempre visible en modo Posts)
if dataset_option == "Posts":
    st.subheader("Buscar Post por ID")
    post_id_input = st.text_input("Introduce Post ID", key="postid_input")
    if post_id_input.strip() != "":
        pid_str = post_id_input.strip()
        df_found = df_posts[df_posts['Post_ID'] == pid_str]
        if not df_found.empty:
            post_row = df_found.iloc[0]
            st.markdown(f"**Post encontrado:**")
            st.markdown(f"- ID: {post_row['Post_ID']}")
            st.markdown(f"- Autor: {post_row['Autor']}")
            st.markdown(f"- Fecha: {post_row['Fecha'].date()}")
            st.markdown(f"- Mensaje: {post_row['Mensaje']}")
            st.markdown(f"- Reacciones: {post_row['Reacciones']}")
            st.markdown(f"- Categoría: {post_row['Categoria']}")
            ejemplo_url = f"https://facebook.com/{post_row['Post_ID']}"
            st.markdown(f"[Ir al Post]({ejemplo_url})", unsafe_allow_html=True)
            df_coms_post = df_comments[df_comments['Post_ID'] == pid_str]
            if not df_coms_post.empty:
                with st.expander("Ver comentarios relacionados a este Post"):
                    st.dataframe(
                        df_coms_post[['Autor', 'Mensaje', 'Fecha', 'Reacciones']],
                        use_container_width=True
                    )
        else:
            st.info("No se encontró ningún post con ese ID en la base de datos.")

# Selección de posts desde multiselect y tabla (si hay búsqueda de posts)
if dataset_option == "Posts" and post_search.strip() != "" and not df.empty:
    st.subheader("Selecciona los Posts que quieres analizar (puedes buscar por texto):")
    # --- OPCIONES CON PREVIEW ---
    multiselect_options = [
        (row["Post_ID"], f"{row['Preview']} (ID: {row['Post_ID']})")
        for idx, row in df.iterrows()
    ]
    default_options = [opt[0] for opt in multiselect_options]
    selected_post_ids = st.multiselect(
        "Selecciona los Post_ID para el análisis (puedes buscar por texto):",
        options=[x[0] for x in multiselect_options],
        default=default_options,
        format_func=lambda x: dict(multiselect_options).get(x, str(x)),
        help="Elige los posts para analizar. El texto mostrado es un resumen del mensaje."
    )
    st.dataframe(
        df[["Post_ID", "Preview", "Autor", "Fecha", "Reacciones", "Categoria"]].sort_values("Fecha", ascending=False),
        use_container_width=True
    )

    # --- ANALISIS SOLO PARA LOS POST SELECCIONADOS ---
    if len(selected_post_ids) == 0:
        st.info("Selecciona al menos un Post en la lista para mostrar el análisis de comentarios y emociones.")
    else:
        st.header("N‐gramas, Sentimiento y Emociones de Comentarios Relacionados")
        post_ids = selected_post_ids
        df_related_comments = df_comments[df_comments['Post_ID'].isin(post_ids)].copy()
        if not df_related_comments.empty:
            df_related_comments['Mensaje_Limpio'] = (
                df_related_comments['Mensaje']
                .astype(str)
                .str.replace(r"http\S+|www\S+|https\S+", "", regex=True)
                .str.replace(r"[^\w\sáéíóúñ]", "", regex=True)
                .str.strip()
            )
            with st.spinner("Analizando sentimiento de comentarios relacionados..."):
                df_related_comments['Sentimiento'] = df_related_comments['Mensaje_Limpio'].apply(
                    lambda txt: sent_analyzer.predict(txt).output if txt else "none"
                )
            with st.spinner("Analizando emociones de comentarios relacionados..."):
                df_related_comments['Emocion'] = df_related_comments['Mensaje_Limpio'].apply(
                    lambda txt: emotion_analyzer.predict(txt).output if txt else "none"
                )
            sentiment_validos = ['POS', 'NEU', 'NEG']
            df_valid = df_related_comments[df_related_comments['Sentimiento'].isin(sentiment_validos)]
            # --- Sentimiento ---
            senti_counts = df_valid['Sentimiento'].value_counts(normalize=True)
            st.info("**Distribución de Sentimiento en los comentarios relacionados:**")
            for s, name in zip(['POS', 'NEU', 'NEG'], ['Positivo', 'Neutro', 'Negativo']):
                pct = 100 * senti_counts.get(s, 0)
                st.write(f"{name}: {pct:.1f}%")
            senti_counts_abs = df_valid['Sentimiento'].value_counts()
            labels = ['Positivo', 'Neutro', 'Negativo']
            keys = ['POS', 'NEU', 'NEG']
            values = [senti_counts_abs.get(k, 0) for k in keys]
            colors = ['#3ad29f', '#b6bbc4', '#f34f4f']
            fig, ax = plt.subplots()
            wedges, texts, autotexts = ax.pie(
                values,
                labels=labels,
                autopct=lambda pct: ('%1.1f%%' % pct) if pct > 0 else '',
                startangle=90,
                colors=colors,
                wedgeprops=dict(width=0.45, edgecolor='white')
            )
            plt.setp(texts, size=14, weight="bold")
            plt.setp(autotexts, size=13)
            ax.set_title('Distribución de Sentimiento en Comentarios', fontsize=18, weight='bold', pad=25)
            ax.axis('equal')
            fig.patch.set_facecolor('white')
            st.pyplot(fig)
            # --- Emociones ---
            emociones_validas = ['joy', 'anger', 'fear', 'sadness', 'surprise', 'others']
            emociones_nombres = {
                'joy': 'Alegría',
                'anger': 'Enojo',
                'fear': 'Miedo',
                'sadness': 'Tristeza',
                'surprise': 'Sorpresa',
                'others': 'Otras'
            }
            colores_em = ['#F9E79F', '#E74C3C', '#5DADE2', '#566573', '#F7DC6F', '#BFC9CA']
            emocion_counts = df_valid['Emocion'].value_counts()
            labels_em = [emociones_nombres.get(e, e) for e in emociones_validas]
            values_em = [emocion_counts.get(e, 0) for e in emociones_validas]
            labels_plot = [label if val > 0 else '' for label, val in zip(labels_em, values_em)]
            fig_em, ax_em = plt.subplots(figsize=(6,6))
            wedges, texts, autotexts = ax_em.pie(
                values_em,
                labels=labels_plot,
                autopct=lambda pct: ('%1.1f%%' % pct) if pct > 0 else '',
                startangle=90,
                colors=colores_em,
                wedgeprops=dict(width=0.45, edgecolor='white')
            )
            plt.setp(texts, size=14, weight="bold")
            plt.setp(autotexts, size=13)
            ax_em.set_title('Distribución de Emociones en Comentarios', fontsize=18, weight='bold', pad=25)
            ax_em.axis('equal')
            fig_em.patch.set_facecolor('white')
            st.pyplot(fig_em)
            # N-gramas y tabla como antes...
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
            df_ngrams_rel = gen_ngrams_relat(df_valid['Mensaje_Limpio'].dropna().astype(str), ngram_range=ngram_range)
            st.subheader(f"Top N‐gramas en Comentarios relacionados ({ngram_option})")
            st.dataframe(df_ngrams_rel.head(50), use_container_width=True)
            with st.expander("Ver comentarios relacionados y su calificación de sentimiento y emoción"):
                st.dataframe(
                    df_valid[['Autor', 'Mensaje', 'Sentimiento', 'Emocion', 'Fecha', 'Reacciones']],
                    use_container_width=True
                )
        else:
            st.write("No hay comentarios relacionados con los posts seleccionados.")

# Detección de outliers por reacciones
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

# N-gramas generales
st.header(f"Top N‐gramas ({ngram_option})")
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
likes_param = df["Reacciones"] if use_likes else None
df_ngrams = generar_ngrams(textos, ngram_range=ngram_range, likes=likes_param)
if use_likes:
    st.markdown("*Ordenado por frecuencia de aparición*")
    st.dataframe(df_ngrams[["ngram", "count", "likes_sum"]].head(50), use_container_width=True)
    st.markdown("**Para ordenar por métricas de likes, haz clic en la cabecera 'likes_sum' en la tabla.**")
else:
    st.dataframe(df_ngrams[["ngram", "count"]].head(50), use_container_width=True)

# Tendencias temporales
show_trend = st.sidebar.checkbox("Mostrar tendencias temporales de un N‐grama", value=False)
if show_trend:
    ngram_for_trend = st.sidebar.text_input("Introduce un N‐grama para su tendencia (exacto)", "")
    if ngram_for_trend.strip() != "":
        st.subheader(f"Tendencia Temporal para el N‐grama: '{ngram_for_trend}'")
        df_temp = df.copy()
        df_temp['texto_lower'] = df_temp['Mensaje'].str.lower().fillna('')
        pattern = rf"\b{re.escape(ngram_for_trend.lower())}\b"
        df_temp['match_ngram'] = df_temp['texto_lower'].apply(lambda t: bool(re.search(pattern, t)))
        trend_series = df_temp.groupby(df_temp['Fecha'].dt.date)['match_ngram'].sum()
        fecha_index = pd.date_range(start=fecha_inicio, end=fecha_fin)
        trend_series = trend_series.reindex(fecha_index.date, fill_value=0)
        st.line_chart(trend_series)

# Word cloud
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

# Topic modeling LDA
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

# Correlación sentimiento vs reacciones
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

st.markdown("""
- Puedes analizar Pentagramas además de uni-, bi-, tri- y cuatrigramas.
- El KPI Dashboard muestra al inicio la visión general.
- La búsqueda en Posts despliega tabla resumen, n-gramas, sentimiento y **emociones** de comentarios relacionados (con gráficos y tabla).
- Buscador por ID de Post con enlace y detalles.
- Se incorpora detección de outliers por reacciones.
- Para tendencias, wordcloud, topic modeling y correlación sentimiento/likes usa las opciones de la barra lateral.
- Recarga la app si se añaden nuevos datos para actualizar rangos y métricas.
""")
