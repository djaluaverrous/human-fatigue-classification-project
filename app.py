# ================================================================
# app.py
# Human Fatigue Prediction — Streamlit App
# ================================================================
# Cara menjalankan:
#   cd human_fatigue_project
#   streamlit run app.py
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# PAGE CONFIG — HARUS PALING ATAS
# ================================================================

st.set_page_config(
    page_title="Human Fatigue AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# CUSTOM CSS — UI MODERN
# ================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── SIDEBAR BACKGROUND ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
}

[data-testid="stSidebar"] > div:first-child {
    background: transparent !important;
}

/* ── SEMUA BUTTON DI SIDEBAR ── */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: rgba(255,255,255,0.55) !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    text-align: left !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    margin-bottom: 4px !important;
    box-shadow: none !important;
}

/* ── HOVER STATE ── */
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #ffffff !important;
    box-shadow: none !important;
}

/* ── ACTIVE / PRIMARY BUTTON (halaman aktif) ── */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: none !important;
    color: #ffffff !important;
    box-shadow: 0 4px 15px rgba(102,126,234,0.4) !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #7c8ff0, #8a5cb8) !important;
    color: #ffffff !important;
}

/* ── FOCUS STATE (hilangkan outline biru default) ── */
[data-testid="stSidebar"] .stButton > button:focus {
    box-shadow: none !important;
    outline: none !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: rgba(255,255,255,0.55) !important;
    background: transparent !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:focus {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: none !important;
    color: #ffffff !important;
}

/* ── SEMBUNYIKAN ELEMEN DEFAULT STREAMLIT ── */
#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}
header    {visibility: hidden;}

/* ── METRIC CARDS ── */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(102,126,234,0.3);
    margin-bottom: 1rem;
}
.metric-card h3 {
    color: rgba(255,255,255,0.8) !important;
    font-size: 13px;
    font-weight: 500;
    margin: 0 0 8px 0;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.metric-card h2 {
    color: #ffffff !important;
    font-size: 28px;
    font-weight: 700;
    margin: 0;
}

/* ── RESULT CARDS ── */
.result-high     { background: linear-gradient(135deg, #ff416c, #ff4b2b); }
.result-moderate { background: linear-gradient(135deg, #f7971e, #ffd200); }
.result-low      { background: linear-gradient(135deg, #11998e, #38ef7d); }

/* ── INSIGHT BOX ── */
.insight-box {
    background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
    border-left: 4px solid #667eea;
}

/* ── RECOMMENDATION ITEM ── */
.recom-item {
    background: white;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 8px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-left: 4px solid #667eea;
    font-size: 14px;
}

/* ── SECTION HEADER ── */
.section-header {
    font-size: 24px;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 1rem;
    padding-bottom: 8px;
    border-bottom: 3px solid #667eea;
}
</style>
""", unsafe_allow_html=True)


# ================================================================
# LOAD MODEL & DATA
# ================================================================

@st.cache_resource
def load_model():
    model         = joblib.load('model/fatigue_model.pkl')
    label_encoder = joblib.load('model/label_encoder.pkl')
    columns       = joblib.load('model/columns.pkl')
    return model, label_encoder, columns

@st.cache_data
def load_data():
    df = pd.read_csv('data/human_fatigue.csv')
    df = df.drop_duplicates()
    if 'System_Recommendation' in df.columns:
        df = df.drop(columns=['System_Recommendation'])
    return df

@st.cache_data
def load_accuracy():
    try:
        data = joblib.load('model/model_accuracy.pkl')
        print("DEBUG accuracy data:", data)  # ← debug
        
        result = {}
        for k, v in data.items():
            # Handle semua kemungkinan format nilai
            if isinstance(v, dict):
                # Kalau tersimpan sebagai dict
                result[k] = float(v.get('accuracy', 0)) * 100
            elif isinstance(v, float) and v <= 1.0:
                # Kalau tersimpan sebagai 0.85 (bukan 85.0)
                result[k] = float(v) * 100
            else:
                # Kalau sudah dalam format 85.0
                result[k] = float(v)
        return result
        
    except FileNotFoundError:
        st.warning("⚠️ model_accuracy.pkl tidak ditemukan. "
                   "Jalankan python train_model.py dulu.")
        return {
            'Random Forest'      : 0.0,
            'Decision Tree'      : 0.0,
            'Logistic Regression': 0.0
        }
    except Exception as e:
        st.error(f"Error load accuracy: {e}")
        return {
            'Random Forest'      : 0.0,
            'Decision Tree'      : 0.0,
            'Logistic Regression': 0.0
        }

try:
    model, label_encoder, columns = load_model()
    df = load_data()
    accuracy_data = load_accuracy()
    
    model_loaded = True

except Exception as e:
    model_loaded = False
    st.error(f"❌ Error loading model: {e}")
    st.info("💡 Jalankan terlebih dahulu: python train_model.py")
    st.stop()

# ================================================================
# SIDEBAR NAVIGATION
# ================================================================

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 24px 0;'>
        <div style='font-size: 48px;'>🧠</div>
        <div style='font-size: 18px; font-weight: 700; 
                    color: white; margin-top: 8px;'>
            Fatigue AI
        </div>
        <div style='font-size: 12px; color: rgba(255,255,255,0.5);'>
            Human Fatigue Prediction System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Navigation menggunakan radio button yang sudah di-style
    pages = {
        "🏠  Home"             : "Home",
        "📊  EDA Dashboard"    : "EDA",
        "🤖  Prediction"       : "Prediction",
        "📁  Batch Prediction" : "Batch",
        "📈  Model Performance": "Performance",
        "ℹ️  About"            : "About"
    }

    if 'page' not in st.session_state:
        st.session_state.page = "Home"

    for label, value in pages.items():
        is_active = st.session_state.page == value
        if st.button(
            label,
            key=f"nav_{value}",
            use_container_width=True,
            type="primary" if is_active else "secondary"
        ):
            st.session_state.page = value
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; font-size:11px; 
                color:rgba(255,255,255,0.3); padding: 8px;'>
        Built with ❤️ using Streamlit<br>
        Random Forest Classifier
    </div>
    """, unsafe_allow_html=True)

page = st.session_state.page


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def get_recommendation(level, hours_awake, sleep_hours,
                        stress, caffeine, error_rate):
    """Generate AI recommendation berdasarkan prediksi & input"""

    recs = []

    if level == "High":
        recs = [
            ("🛑", "Hentikan aktivitas berat sekarang",
             "Tubuh Anda membutuhkan istirahat segera."),
            ("😴", "Tidur siang 20–30 menit",
             "Power nap terbukti memulihkan konsentrasi hingga 34%."),
            ("🚫", "Hindari multitasking",
             f"Anda sudah melakukan banyak task switches hari ini."),
            ("🧘", "Lakukan teknik pernapasan",
             "4-7-8 breathing: hirup 4 detik, tahan 7, hembuskan 8."),
            ("🌙", "Tidur lebih awal malam ini",
             f"Anda hanya tidur {sleep_hours} jam tadi malam. Target minimal 7 jam."),
        ]
        if caffeine >= 3:
            recs.append(("☕", "Kurangi konsumsi kafein",
                         f"Anda sudah {caffeine} cangkir hari ini. Batas aman 2–3 cangkir."))
        if stress >= 7:
            recs.append(("💆", "Kelola stres segera",
                         f"Level stres Anda {stress}/10. Coba meditasi atau jalan kaki."))

    elif level == "Moderate":
        recs = [
            ("⏸️", "Ambil istirahat singkat setiap 45 menit",
             "Teknik Pomodoro: 25 menit kerja, 5 menit istirahat."),
            ("💧", "Minum air yang cukup",
             "Dehidrasi ringan dapat menurunkan konsentrasi hingga 20%."),
            ("🎵", "Dengarkan musik instrumental",
             "Musik tanpa lirik terbukti meningkatkan fokus."),
            ("🍎", "Makan camilan sehat",
             "Kacang, buah, atau dark chocolate membantu energi stabil."),
        ]
        if sleep_hours < 6:
            recs.append(("😴", "Prioritaskan tidur malam ini",
                         f"Tidur {sleep_hours} jam kurang dari ideal. Targetkan 7–8 jam."))
        if hours_awake >= 10:
            recs.append(("⏰", "Batasi jam kerja hari ini",
                         f"Anda sudah terjaga {hours_awake} jam. Pertimbangkan istirahat."))

    else:  # Low
        recs = [
            ("✅", "Kondisi Anda sangat baik!",
             "Pertahankan keseimbangan kerja dan istirahat yang ada."),
            ("🏃", "Waktu yang baik untuk aktivitas produktif",
             "Manfaatkan energi optimal ini untuk tugas prioritas."),
            ("🔄", "Jaga konsistensi pola tidur",
             "Tidur dan bangun di jam yang sama setiap hari."),
            ("🥗", "Lanjutkan pola makan sehat",
             "Nutrisi yang baik mendukung performa kognitif optimal."),
        ]

    return recs

def fatigue_emoji(level):
    return {"High": "🔴", "Moderate": "🟡", "Low": "🟢"}.get(level, "⚪")

def fatigue_color(level):
    return {
        "High"    : "#ff416c",
        "Moderate": "#f7971e",
        "Low"     : "#11998e"
    }.get(level, "#667eea")


# ================================================================
# PAGE 1 — HOME
# ================================================================

if page == "Home":

    # Header
    st.markdown("""
    <div style='padding: 32px 0 16px 0;'>
        <h1 style='font-size: 36px; font-weight: 800; 
                   color: #1a1a2e; margin: 0;'>
            🧠 Human Fatigue Prediction System
        </h1>
        <p style='color: #718096; font-size: 16px; margin-top: 8px;'>
            AI-powered monitoring system menggunakan Random Forest Classifier
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Metric Cards Row 1
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Total Data</h3>
            <h2>{df.shape[0]:,}</h2>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Total Fitur</h3>
            <h2>{len(columns)}</h2>
        </div>""", unsafe_allow_html=True)

    with col3:
        rf_acc = accuracy_data.get('Random Forest', 0.0)
        display_acc = rf_acc if rf_acc > 1 else rf_acc * 100
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Model Accuracy</h3>
            <h2>{rf_acc:.1f}%</h2>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Algorithm</h3>
            <h2 style='font-size:18px;'>Random Forest</h2>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset Preview & Distribusi
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("<div class='section-header'>📋 Dataset Preview</div>",
                    unsafe_allow_html=True)
        st.dataframe(
            df.head(10),
            use_container_width=True,
            height=300
        )

    with col_right:
        st.markdown("<div class='section-header'>📊 Distribusi Target</div>",
                    unsafe_allow_html=True)
        dist = df['Fatigue_Level'].value_counts().reset_index()
        dist.columns = ['Fatigue_Level', 'Count']
        order_map = {'Low': 0, 'Moderate': 1, 'High': 2}
        dist['order'] = dist['Fatigue_Level'].map(order_map)
        dist = dist.sort_values('order')

        fig = px.pie(
            dist,
            values='Count',
            names='Fatigue_Level',
            color='Fatigue_Level',
            color_discrete_map={
                'Low'     : '#11998e',
                'Moderate': '#f7971e',
                'High'    : '#ff416c'
            },
            hole=0.55
        )
        fig.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=280,
            showlegend=True,
            legend=dict(orientation='h', y=-0.1)
        )
        fig.update_traces(textposition='outside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    # Key Insights
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>💡 Key Insights Dataset</div>",
                unsafe_allow_html=True)

    i1, i2, i3 = st.columns(3)

    with i1:
        avg_sleep = df['Sleep_Hours_Last_Night'].mean()
        st.markdown(f"""
        <div class='insight-box'>
            😴 <strong>Rata-rata Jam Tidur</strong><br>
            <span style='font-size:22px; font-weight:700; color:#667eea;'>
                {avg_sleep:.1f} jam
            </span><br>
            <small style='color:#718096;'>
                per malam dari seluruh dataset
            </small>
        </div>""", unsafe_allow_html=True)

    with i2:
        avg_stress = df['Stress_Level_1_10'].mean()
        st.markdown(f"""
        <div class='insight-box'>
            😰 <strong>Rata-rata Stress Level</strong><br>
            <span style='font-size:22px; font-weight:700; color:#667eea;'>
                {avg_stress:.1f} / 10
            </span><br>
            <small style='color:#718096;'>
                skala 1–10 dari seluruh dataset
            </small>
        </div>""", unsafe_allow_html=True)

    with i3:
        high_pct = (df['Fatigue_Level'] == 'High').mean() * 100
        st.markdown(f"""
        <div class='insight-box'>
            🔴 <strong>High Fatigue Rate</strong><br>
            <span style='font-size:22px; font-weight:700; color:#ff416c;'>
                {high_pct:.1f}%
            </span><br>
            <small style='color:#718096;'>
                dari total populasi dataset
            </small>
        </div>""", unsafe_allow_html=True)


# ================================================================
# PAGE 2 — EDA DASHBOARD
# ================================================================

elif page == "EDA":

    st.markdown("<h2 class='section-header'>📊 Exploratory Data Analysis</h2>",
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Distribusi", "🔥 Korelasi",
        "📦 Boxplot", "🔵 Scatter"
    ])

    palette = {
        'Low'     : '#11998e',
        'Moderate': '#f7971e',
        'High'    : '#ff416c'
    }

    with tab1:
        st.subheader("Distribusi Fatigue Level")
        col1, col2 = st.columns(2)

        with col1:
            dist = df['Fatigue_Level'].value_counts().reset_index()
            dist.columns = ['Level', 'Count']
            fig = px.bar(
                dist, x='Level', y='Count',
                color='Level',
                color_discrete_map=palette,
                text='Count',
                title='Jumlah per Fatigue Level'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.histogram(
                df, x='Hours_Awake',
                color='Fatigue_Level',
                color_discrete_map=palette,
                nbins=20,
                title='Distribusi Hours Awake per Fatigue Level',
                barmode='overlay',
                opacity=0.75
            )
            fig2.update_layout(height=380)
            st.plotly_chart(fig2, use_container_width=True)

        # Insight
        st.markdown("""
        <div class='insight-box'>
            💡 <strong>Insight:</strong> Semakin lama seseorang terjaga (Hours Awake tinggi),
            semakin besar kemungkinan mengalami High Fatigue. 
            Pola ini konsisten di seluruh dataset.
        </div>""", unsafe_allow_html=True)

    with tab2:
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=np.number)
        corr = numeric_df.corr()

        fig = px.imshow(
            corr,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title='Korelasi antar Fitur Numerik',
            zmin=-1, zmax=1
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Top korelasi
        st.markdown("**🔗 Korelasi Tertinggi dengan Fitur Lain:**")
        corr_pairs = (corr.abs()
                      .unstack()
                      .sort_values(ascending=False)
                      .drop_duplicates())
        corr_pairs = corr_pairs[corr_pairs < 1.0].head(5)
        for (f1, f2), val in corr_pairs.items():
            st.markdown(f"""
            <div class='insight-box'>
                🔗 <strong>{f1}</strong> ↔ <strong>{f2}</strong> :
                korelasi = <strong>{val:.3f}</strong>
            </div>""", unsafe_allow_html=True)

    with tab3:
        st.subheader("Distribusi Fitur per Fatigue Level")

        feature_select = st.selectbox(
            "Pilih Fitur:",
            options=[
                'Hours_Awake', 'Sleep_Hours_Last_Night',
                'Stress_Level_1_10', 'Cognitive_Load_Score',
                'Error_Rate', 'Decisions_Made',
                'Task_Switches', 'Caffeine_Intake_Cups'
            ]
        )

        fig = px.box(
            df,
            x='Fatigue_Level',
            y=feature_select,
            color='Fatigue_Level',
            color_discrete_map=palette,
            category_orders={'Fatigue_Level': ['Low', 'Moderate', 'High']},
            title=f'{feature_select} vs Fatigue Level',
            points='outliers'
        )
        fig.update_layout(showlegend=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

        # Statistik per group
        st.markdown("**📊 Statistik per Fatigue Level:**")
        stats = (df.groupby('Fatigue_Level')[feature_select]
                 .agg(['mean', 'median', 'std'])
                 .round(2))
        st.dataframe(stats, use_container_width=True)

    with tab4:
        st.subheader("Scatter Plot Analysis")

        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X-axis:", options=[
                'Sleep_Hours_Last_Night', 'Hours_Awake',
                'Stress_Level_1_10', 'Cognitive_Load_Score'
            ])
        with col2:
            y_axis = st.selectbox("Y-axis:", options=[
                'Stress_Level_1_10', 'Cognitive_Load_Score',
                'Error_Rate', 'Decisions_Made'
            ])

        fig = px.scatter(
            df,
            x=x_axis, y=y_axis,
            color='Fatigue_Level',
            color_discrete_map=palette,
            size='Hours_Awake',
            hover_data=['Hours_Awake', 'Sleep_Hours_Last_Night'],
            category_orders={'Fatigue_Level': ['Low', 'Moderate', 'High']},
            title=f'{x_axis} vs {y_axis}',
            opacity=0.7
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)


# ================================================================
# PAGE 3 — PREDICTION
# ================================================================

elif page == "Prediction":

    st.markdown("<h2 class='section-header'>🤖 Fatigue Level Prediction</h2>",
                unsafe_allow_html=True)
    st.markdown("Masukkan kondisi Anda saat ini untuk memprediksi tingkat kelelahan.")
    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**⏰ Aktivitas**")
            hours_awake = st.slider(
                "Jam Terjaga (Hours Awake)", 1, 24, 8,
                help="Berapa jam Anda sudah terjaga hari ini?"
            )
            decisions_made = st.slider(
                "Jumlah Keputusan (Decisions Made)", 0, 200, 50,
                help="Berapa banyak keputusan yang sudah Anda buat hari ini?"
            )
            task_switches = st.slider(
                "Pergantian Tugas (Task Switches)", 0, 50, 10,
                help="Berapa kali Anda berpindah tugas hari ini?"
            )

        with col2:
            st.markdown("**😴 Tidur & Waktu**")
            sleep_hours = st.slider(
                "Jam Tidur Tadi Malam", 0.0, 12.0, 7.0, step=0.5,
                help="Berapa jam Anda tidur tadi malam?"
            )
            avg_decision_time = st.slider(
                "Rata-rata Waktu Keputusan (detik)", 1.0, 120.0, 30.0, step=0.5,
                help="Rata-rata waktu yang dibutuhkan untuk membuat satu keputusan"
            )
            time_of_day = st.selectbox(
                "Waktu Saat Ini",
                ["Morning", "Afternoon", "Evening", "Night"],
                help="Pilih waktu saat ini"
            )

        with col3:
            st.markdown("**😰 Kondisi Mental**")
            stress = st.slider(
                "Level Stres (1–10)", 1, 10, 5,
                help="1 = sangat santai, 10 = sangat stres"
            )
            caffeine = st.slider(
                "Konsumsi Kafein (cangkir)", 0, 10, 2,
                help="Berapa cangkir kopi/teh hari ini?"
            )
            error_rate = st.slider(
                "Error Rate (0.0–1.0)", 0.0, 1.0, 0.1, step=0.01,
                help="Seberapa sering Anda membuat kesalahan hari ini?"
            )
            cognitive_load = st.slider(
                "Cognitive Load Score", 1, 100, 50,
                help="Seberapa berat beban pikiran Anda? (1=ringan, 100=sangat berat)"
            )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "🔮 Prediksi Sekarang",
            use_container_width=True,
            type="primary"
        )

    # ── HASIL PREDIKSI ──────────────────────────────────────────
    if submitted:

        input_data = pd.DataFrame([{
            'Hours_Awake'           : hours_awake,
            'Decisions_Made'        : decisions_made,
            'Task_Switches'         : task_switches,
            'Avg_Decision_Time_sec' : avg_decision_time,
            'Sleep_Hours_Last_Night': sleep_hours,
            'Time_of_Day'           : time_of_day,
            'Caffeine_Intake_Cups'  : caffeine,
            'Stress_Level_1_10'     : stress,
            'Error_Rate'            : error_rate,
            'Cognitive_Load_Score'  : cognitive_load
        }])

        prediction  = model.predict(input_data)
        probability = model.predict_proba(input_data)[0]
        pred_label  = label_encoder.inverse_transform(prediction)[0]
        confidence  = probability.max() * 100
        classes     = label_encoder.classes_

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 📊 Hasil Prediksi")

        # Result Card
        col_res, col_prob = st.columns([1, 2])

        with col_res:
            css_class = {
                "High"    : "result-high",
                "Moderate": "result-moderate",
                "Low"     : "result-low"
            }.get(pred_label, "result-low")

            emoji = fatigue_emoji(pred_label)

            st.markdown(f"""
            <div class='{css_class}' style='border-radius:16px; padding:32px;
                         text-align:center; color:white;'>
                <div style='font-size:56px;'>{emoji}</div>
                <div style='font-size:14px; opacity:0.85; 
                            text-transform:uppercase; letter-spacing:2px;
                            margin: 8px 0;'>Fatigue Level</div>
                <div style='font-size:36px; font-weight:800;'>
                    {pred_label}
                </div>
                <div style='font-size:14px; opacity:0.85; margin-top:8px;'>
                    Confidence: {confidence:.1f}%
                </div>
            </div>""", unsafe_allow_html=True)

        with col_prob:
            st.markdown("**Probabilitas per Kelas:**")
            prob_df = pd.DataFrame({
                'Fatigue Level': classes,
                'Probabilitas' : probability
            }).sort_values('Probabilitas', ascending=True)

            fig = px.bar(
                prob_df,
                x='Probabilitas', y='Fatigue Level',
                orientation='h',
                color='Fatigue Level',
                color_discrete_map={
                    'Low'     : '#11998e',
                    'Moderate': '#f7971e',
                    'High'    : '#ff416c'
                },
                text=prob_df['Probabilitas'].apply(lambda x: f'{x*100:.1f}%')
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                showlegend=False, height=220,
                xaxis=dict(range=[0, 1.15]),
                margin=dict(t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── AI RECOMMENDATION ───────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 🧠 AI Recommendation")

        recs = get_recommendation(
            pred_label, hours_awake, sleep_hours,
            stress, caffeine, error_rate
        )

        col_r1, col_r2 = st.columns(2)
        for i, (icon, title, desc) in enumerate(recs):
            col = col_r1 if i % 2 == 0 else col_r2
            with col:
                st.markdown(f"""
                <div class='recom-item'>
                    <strong>{icon} {title}</strong><br>
                    <small style='color:#718096;'>{desc}</small>
                </div>""", unsafe_allow_html=True)


# ================================================================
# PAGE 4 — BATCH PREDICTION
# ================================================================

elif page == "Batch":

    st.markdown("<h2 class='section-header'>📁 Batch Prediction</h2>",
                unsafe_allow_html=True)
    st.markdown("Upload file CSV untuk prediksi massal sekaligus.")

    # Template download
    st.markdown("**📥 Download Template CSV:**")
    template = pd.DataFrame([{
        'Hours_Awake'           : 8,
        'Decisions_Made'        : 50,
        'Task_Switches'         : 10,
        'Avg_Decision_Time_sec' : 30.0,
        'Sleep_Hours_Last_Night': 7.0,
        'Time_of_Day'           : 'Morning',
        'Caffeine_Intake_Cups'  : 2,
        'Stress_Level_1_10'     : 5,
        'Error_Rate'            : 0.1,
        'Cognitive_Load_Score'  : 50
    }])

    csv_template = template.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Template",
        data=csv_template,
        file_name="fatigue_template.csv",
        mime="text/csv"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload CSV Anda di sini:",
        type=['csv'],
        help="Pastikan format kolom sesuai template di atas"
    )

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"✅ File berhasil diupload: {df_upload.shape[0]} baris")

            with st.expander("👀 Preview Data Upload"):
                st.dataframe(df_upload.head(), use_container_width=True)

            # Prediksi
            with st.spinner("🔮 Memproses prediksi..."):
                preds    = model.predict(df_upload)
                proba    = model.predict_proba(df_upload)
                labels   = label_encoder.inverse_transform(preds)
                confidence_list = proba.max(axis=1) * 100

                df_result = df_upload.copy()
                df_result['Predicted_Fatigue'] = labels
                df_result['Confidence_%']      = confidence_list.round(1)

            st.markdown("### 📊 Hasil Prediksi")

            # Summary
            col1, col2, col3 = st.columns(3)
            vc = pd.Series(labels).value_counts()
            with col1:
                st.metric("🔴 High Fatigue",
                          f"{vc.get('High', 0)} orang",
                          f"{vc.get('High', 0)/len(labels)*100:.1f}%")
            with col2:
                st.metric("🟡 Moderate Fatigue",
                          f"{vc.get('Moderate', 0)} orang",
                          f"{vc.get('Moderate', 0)/len(labels)*100:.1f}%")
            with col3:
                st.metric("🟢 Low Fatigue",
                          f"{vc.get('Low', 0)} orang",
                          f"{vc.get('Low', 0)/len(labels)*100:.1f}%")

            st.dataframe(
                df_result.style.applymap(
                    lambda v: (
                        'background-color: #ffcccc' if v == 'High'
                        else 'background-color: #fff3cc' if v == 'Moderate'
                        else 'background-color: #ccffee' if v == 'Low'
                        else ''
                    ),
                    subset=['Predicted_Fatigue']
                ),
                use_container_width=True,
                height=400
            )

            # Download hasil
            csv_result = df_result.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Hasil Prediksi",
                data=csv_result,
                file_name="fatigue_prediction_result.csv",
                mime="text/csv",
                type="primary"
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.info("💡 Pastikan kolom CSV sesuai dengan template yang disediakan.")


# ================================================================
# PAGE 5 — MODEL PERFORMANCE
# ================================================================

elif page == "Performance":

    st.markdown("<h2 class='section-header'>📈 Model Performance</h2>",
                unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # HITUNG ULANG ACCURACY LANGSUNG DARI MODEL
    # Tidak bergantung pada pkl file
    # ─────────────────────────────────────────────
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import classification_report

    @st.cache_data
    def compute_all_metrics():
        df_m = load_data()
        if 'Decision_Fatigue_Score' in df_m.columns:
            df_m = df_m.drop(columns=['Decision_Fatigue_Score'])

        X_m = df_m.drop(columns=['Fatigue_Level'])
        y_m = label_encoder.transform(df_m['Fatigue_Level'])

        numeric_features = [
            'Hours_Awake', 'Decisions_Made', 'Task_Switches',
            'Avg_Decision_Time_sec', 'Sleep_Hours_Last_Night',
            'Caffeine_Intake_Cups', 'Stress_Level_1_10',
            'Error_Rate', 'Cognitive_Load_Score'
        ]
        categorical_features = ['Time_of_Day']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore',
                                     sparse_output=False))
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
            X_m, y_m,
            test_size=0.2,
            random_state=42,
            stratify=y_m
        )

        classifiers = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=10,
                min_samples_split=5, min_samples_leaf=2,
                class_weight='balanced', random_state=42),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=8, min_samples_split=5,
                class_weight='balanced', random_state=42),
            'Logistic Regression': LogisticRegression(
                max_iter=1000, class_weight='balanced',
                random_state=42)
        }

        metrics = {}
        for name, clf in classifiers.items():
            pipe = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', clf)
            ])
            pipe.fit(X_train_m, y_train_m)
            y_pred_m = pipe.predict(X_test_m)
            acc      = accuracy_score(y_test_m, y_pred_m)
            report   = classification_report(
                y_test_m, y_pred_m,
                target_names=label_encoder.classes_,
                output_dict=True,
                zero_division=0
            )
            metrics[name] = {
                'accuracy' : round(acc * 100, 2),
                'report'   : report,
                'y_pred'   : y_pred_m,
                'y_test'   : y_test_m
            }
        return metrics

    # Tampilkan spinner saat hitung
    with st.spinner("⏳ Menghitung performa model... (1–2 menit)"):
        all_metrics = compute_all_metrics()

    # ─────────────────────────────────────────────
    # SECTION 1 — ACCURACY COMPARISON
    # ─────────────────────────────────────────────
    st.markdown("### 🏆 Perbandingan Akurasi 3 Model")

    acc_df = pd.DataFrame([
        {'Model': k, 'Accuracy (%)': v['accuracy']}
        for k, v in all_metrics.items()
    ]).sort_values('Accuracy (%)', ascending=False)

    col1, col2 = st.columns([3, 2])

    with col1:
        colors_model = {
            'Random Forest'      : '#667eea',
            'Decision Tree'      : '#764ba2',
            'Logistic Regression': '#f093fb'
        }
        acc_df['Color'] = acc_df['Model'].map(colors_model)

        fig_acc = px.bar(
            acc_df,
            x='Model',
            y='Accuracy (%)',
            color='Model',
            color_discrete_map=colors_model,
            text='Accuracy (%)',
            title='Akurasi 3 Model ML'
        )
        fig_acc.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        fig_acc.update_layout(
            showlegend=False,
            height=380,
            yaxis=dict(range=[0, 115]),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        for _, row in acc_df.iterrows():
            is_best = row['Model'] == acc_df.iloc[0]['Model']
            bg = "linear-gradient(135deg, #667eea, #764ba2)" \
                 if is_best else \
                 "linear-gradient(135deg, #a0aec0, #718096)"
            crown = "👑 " if is_best else ""
            st.markdown(f"""
            <div class='metric-card' 
                 style='background:{bg}; margin-bottom:12px;'>
                <h3>{crown}{row['Model']}</h3>
                <h2>{row['Accuracy (%)']:.1f}%</h2>
            </div>""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # SECTION 2 — CLASSIFICATION REPORT
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Classification Report Detail")

    model_tabs = st.tabs([
        "🌲 Random Forest",
        "🌿 Decision Tree",
        "📉 Logistic Regression"
    ])

    for i, (mname, tab) in enumerate(
        zip(['Random Forest', 'Decision Tree', 'Logistic Regression'],
            model_tabs)
    ):
        with tab:
            report = all_metrics[mname]['report']
            acc    = all_metrics[mname]['accuracy']

            # Metric row
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Accuracy",
                       f"{acc:.1f}%")
            mc2.metric("Precision (avg)",
                       f"{report['weighted avg']['precision']*100:.1f}%")
            mc3.metric("Recall (avg)",
                       f"{report['weighted avg']['recall']*100:.1f}%")
            mc4.metric("F1-Score (avg)",
                       f"{report['weighted avg']['f1-score']*100:.1f}%")

            # Per-class breakdown
            st.markdown("**Per-Class Performance:**")
            class_data = []
            for cls in label_encoder.classes_:
                if cls in report:
                    class_data.append({
                        'Class'    : cls,
                        'Precision': round(
                            report[cls]['precision'] * 100, 1),
                        'Recall'   : round(
                            report[cls]['recall'] * 100, 1),
                        'F1-Score' : round(
                            report[cls]['f1-score'] * 100, 1),
                        'Support'  : int(report[cls]['support'])
                    })

            class_df = pd.DataFrame(class_data)

            fig_cls = px.bar(
                class_df.melt(
                    id_vars='Class',
                    value_vars=['Precision', 'Recall', 'F1-Score'],
                    var_name='Metric',
                    value_name='Score (%)'
                ),
                x='Class', y='Score (%)',
                color='Metric',
                barmode='group',
                color_discrete_sequence=['#667eea', '#11998e', '#f7971e'],
                title=f'Per-Class Metrics — {mname}',
                category_orders={
                    'Class': ['Low', 'Moderate', 'High']}
            )
            fig_cls.update_layout(
                height=320,
                yaxis=dict(range=[0, 115])
            )
            st.plotly_chart(fig_cls, use_container_width=True)

    # ─────────────────────────────────────────────
    # SECTION 3 — CONFUSION MATRIX
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Confusion Matrix")

    cm_tabs = st.tabs([
        "🌲 Random Forest",
        "🌿 Decision Tree",
        "📉 Logistic Regression"
    ])

    for mname, tab in zip(
        ['Random Forest', 'Decision Tree', 'Logistic Regression'],
        cm_tabs
    ):
        with tab:
            y_t  = all_metrics[mname]['y_test']
            y_p  = all_metrics[mname]['y_pred']
            cm   = confusion_matrix(y_t, y_p)
            classes = label_encoder.classes_

            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual",
                            color="Count"),
                x=classes, y=classes,
                text_auto=True,
                color_continuous_scale='Blues',
                title=f'Confusion Matrix — {mname}'
            )
            fig_cm.update_layout(height=400)
            st.plotly_chart(fig_cm, use_container_width=True)

    # ─────────────────────────────────────────────
    # SECTION 4 — FEATURE IMPORTANCE
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🌟 Feature Importance — Random Forest")

    rf_clf  = model.named_steps['classifier']
    preproc = model.named_steps['preprocessor']

    numeric_features = [
        'Hours_Awake', 'Decisions_Made', 'Task_Switches',
        'Avg_Decision_Time_sec', 'Sleep_Hours_Last_Night',
        'Caffeine_Intake_Cups', 'Stress_Level_1_10',
        'Error_Rate', 'Cognitive_Load_Score'
    ]
    ohe_cols = (preproc
                .named_transformers_['cat']
                .named_steps['onehot']
                .get_feature_names_out(['Time_of_Day']))

    feature_names = numeric_features + list(ohe_cols)
    importances   = rf_clf.feature_importances_

    feat_df = pd.DataFrame({
        'Feature'   : feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)

    fig_fi = px.bar(
        feat_df,
        x='Importance', y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='viridis',
        title='Feature Importance Score'
    )
    fig_fi.update_layout(
        height=420,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # Top 3 insight
    top3 = feat_df.sort_values(
        'Importance', ascending=False).head(3)
    st.markdown("**🔑 Top 3 Fitur Terpenting:**")
    for _, row in top3.iterrows():
        st.markdown(f"""
        <div class='insight-box'>
            ⭐ <strong>{row['Feature']}</strong> —
            importance: <strong>{row['Importance']:.4f}</strong>
        </div>""", unsafe_allow_html=True)

# ================================================================
# PAGE 6 — ABOUT
# ================================================================

elif page == "About":

    st.markdown("<h2 class='section-header'>ℹ️ About Project</h2>",
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
                    border-radius: 16px; padding: 28px;'>
            <h3 style='color:#1a1a2e; margin-top:0;'>
                🧠 Human Fatigue Prediction System
            </h3>
            <p style='color:#4a5568;'>
                Sistem AI yang memprediksi tingkat kelelahan manusia
                menggunakan Machine Learning berbasis data aktivitas,
                tidur, stres, dan faktor kognitif lainnya.
            </p>
            <h4 style='color:#1a1a2e;'>🎯 Tujuan Project</h4>
            <p style='color:#4a5568;'>
                Membantu individu dan organisasi mendeteksi kelelahan
                lebih awal sehingga dapat mengambil tindakan pencegahan
                sebelum produktivitas menurun atau terjadi kecelakaan kerja.
            </p>
            <h4 style='color:#1a1a2e;'>🔬 Metodologi</h4>
            <p style='color:#4a5568;'>
                Random Forest dipilih sebagai model utama karena
                kemampuannya menangani data tabular, tidak sensitif
                terhadap outlier, dan memberikan feature importance
                yang interpretatif.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        tech_stack = [
            ("🐍", "Python 3.10+"),
            ("📊", "Streamlit"),
            ("🤖", "Scikit-Learn"),
            ("📈", "Plotly"),
            ("🐼", "Pandas & NumPy"),
            ("💾", "Joblib"),
        ]
        st.markdown("**🛠️ Tech Stack:**")
        for icon, tech in tech_stack:
            st.markdown(f"""
            <div class='recom-item'>
                {icon} {tech}
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; color:#718096; font-size:13px;
                padding: 16px; border-top: 1px solid #e2e8f0;'>
        Built with ❤️ | Human Fatigue Prediction System | 2025
    </div>""", unsafe_allow_html=True)