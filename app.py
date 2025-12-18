import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import ast
import base64
import numpy as np
import glob

# ==========================================
# 1. SETUP & APPLE-GLASS DESIGN SYSTEM
# ==========================================
st.set_page_config(
    page_title="WTP | Emerald OS",
    page_icon="❇️",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={'About': 'By Luciano Diniz | v70.0 Emerald'}
)

# Paleta Personalizada (RGB 2, 189, 126 -> Hex #02BD7E)
THEME = {
    "bg": "#010503",
    "card_bg": "rgba(255, 255, 255, 0.05)",
    "card_border": "rgba(2, 189, 126, 0.15)",
    "text": "#E0F2E9",
    "sub": "#6B8E82",
    "accent": "#02BD7E",
    "glow": "rgba(2, 189, 126, 0.5)",
    "secondary": "#005F40",
    "glass": "rgba(10, 20, 15, 0.70)",
    "border": "1px solid rgba(2, 189, 126, 0.2)",
    "radius": "16px",
    "shadow": "0 8px 32px rgba(0, 0, 0, 0.4)",
    "blur": "blur(20px)"
}

# CSS APPLE GLASS STYLE
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;500;600;700&family=Inter:wght@300;400&display=swap');
    
    /* GLOBAL RESET */
    .stApp {{ 
        background-color: {THEME['bg']}; 
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(2, 189, 126, 0.15) 0%, transparent 40%),
            radial-gradient(circle at 0% 100%, rgba(2, 189, 126, 0.05) 0%, transparent 30%);
        font-family: 'Inter', sans-serif; 
        color: {THEME['text']};
    }}
    
    h1, h2, h3 {{ font-family: 'Rajdhani', sans-serif; text-transform: uppercase; letter-spacing: 2px; color: white !important; }}
    
    /* HEADER FIX */
    header[data-testid="stHeader"] {{ background: transparent !important; z-index: 99999; }}
    
    /* Botão Menu */
    button[data-testid="baseButton-header"] {{
        color: {THEME['accent']} !important;
        border: 1px solid rgba(2, 189, 126, 0.3) !important;
        background-color: rgba(0,0,0,0.5) !important;
        transition: all 0.3s ease;
        border-radius: 12px !important;
    }}
    button[data-testid="baseButton-header"]:hover {{
        box-shadow: 0 0 15px {THEME['glow']};
        border-color: {THEME['accent']} !important;
        transform: translateY(-2px);
    }}

    div[data-testid="stDecoration"] {{ display: none; }}
    .main .block-container {{ padding-top: 4rem !important; padding-bottom: 2rem; }}
    
    /* APPLE GLASS CARDS */
    .glass-card {{
        background: {THEME['card_bg']};
        backdrop-filter: {THEME['blur']};
        -webkit-backdrop-filter: {THEME['blur']};
        border-radius: {THEME['radius']};
        border: {THEME['card_border']};
        padding: 1.5rem;
        box-shadow: {THEME['shadow']};
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .glass-card:hover {{
        transform: translateY(-5px);
        border-color: {THEME['accent']};
        box-shadow: 0 0 25px {THEME['glow']}, {THEME['shadow']};
    }}
    
    .glass-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, transparent, {THEME['accent']}, transparent);
        opacity: 0.8;
    }}
    
    /* KPI Cards Especiais */
    .kpi-glass-card {{
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: {THEME['blur']};
        -webkit-backdrop-filter: {THEME['blur']};
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        box-shadow: 
            0 10px 40px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
        height: 100%;
    }}
    
    .kpi-glass-card:hover {{
        transform: translateY(-5px);
        border-color: rgba(2, 189, 126, 0.4);
        box-shadow: 
            0 15px 50px rgba(0, 0, 0, 0.4),
            0 0 40px rgba(2, 189, 126, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }}
    
    .kpi-glass-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, 
            {THEME['accent']}, 
            #00FFAA, 
            {THEME['accent']}
        );
        opacity: 0.7;
        border-radius: 20px 20px 0 0;
    }}
    
    .kpi-value {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.5rem !important;
        font-weight: 700;
        background: linear-gradient(135deg, #FFFFFF 0%, {THEME['accent']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0;
        letter-spacing: -1px;
    }}
    
    .kpi-label {{
        color: {THEME['sub']};
        font-size: 0.9rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }}
    
    .kpi-delta {{
        font-size: 0.9rem;
        font-weight: 600;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
        margin-top: 0.5rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {{ 
        background-color: rgba(5, 10, 8, 0.95); 
        border-right: 1px solid rgba(2, 189, 126, 0.2);
        backdrop-filter: {THEME['blur']};
    }}
    
    /* Inputs */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div, 
    .stMultiSelect div[data-baseweb="select"] > div, .stNumberInput input, .stDateInput input {{
        background-color: rgba(255,255,255,0.03) !important; 
        border: 1px solid rgba(2, 189, 126, 0.3) !important; 
        color: white !important; 
        border-radius: 8px !important;
        backdrop-filter: blur(10px);
    }}
    
    /* Tables */
    div[data-testid="stDataFrame"] {{ 
        border: 1px solid {THEME['secondary']}; 
        border-radius: 10px; 
        background: rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: rgba(0,0,0,0.3);
        padding: 5px; 
        border-radius: 8px; 
        border: {THEME['border']};
        backdrop-filter: blur(10px);
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 45px; 
        color: {THEME['sub']}; 
        font-family: 'Rajdhani'; 
        font-weight: 600; 
        font-size: 1.1rem; 
        border: none;
    }}
    .stTabs [aria-selected="true"] {{
        background: rgba(2, 189, 126, 0.15);
        color: {THEME['accent']}; 
        border-bottom: 2px solid {THEME['accent']} !important;
        box-shadow: 0 5px 15px rgba(2, 189, 126, 0.1);
        backdrop-filter: blur(20px);
    }}
    
    .js-plotly-plot .plotly .modebar {{ display: none !important; }} 
    .streamlit-expanderHeader {{ background-color: transparent !important; color: {THEME['text']} !important; }}
    
    /* Status Badge */
    .status-badge {{
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-left: 0.5rem;
    }}
    
    .status-active {{
        background: rgba(2, 189, 126, 0.15);
        color: {THEME['accent']};
        border-color: rgba(2, 189, 126, 0.3);
    }}
    
    /* Sankey Container com rolagem */
    .sankey-container {{
        height: 1200px;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 10px;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        border: 1px solid rgba(2, 189, 126, 0.2);
    }}
    
    /* Scrollbar personalizada */
    .sankey-container::-webkit-scrollbar {{
        width: 8px;
    }}
    
    .sankey-container::-webkit-scrollbar-track {{
        background: rgba(0, 0, 0, 0.2);
        border-radius: 4px;
    }}
    
    .sankey-container::-webkit-scrollbar-thumb {{
        background: {THEME['accent']};
        border-radius: 4px;
    }}
    
    .sankey-container::-webkit-scrollbar-thumb:hover {{
        background: {THEME['secondary']};
    }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE (ROBUST LEGACY LOADER)
# ==========================================

try:
    import yfinance as yf
    HAS_YFINANCE = True
except: 
    HAS_YFINANCE = False

@st.cache_data(ttl=3600)
def get_data_engine():
    log_messages = []
    
    # 1. LOAD OPERATIONS
    all_files = glob.glob("Operações Concorrentes*.xlsx") + glob.glob("Operações Concorrentes*.csv")
    valid_files = [f for f in all_files if not os.path.basename(f).startswith("~$")]
    
    df_list = []
    for f in valid_files:
        try:
            if f.endswith('.csv'):
                temp = pd.read_csv(f, sep=';', decimal=',', encoding='latin1', on_bad_lines='skip')
                if temp.shape[1] < 2: 
                    temp = pd.read_csv(f, sep=',', encoding='utf-8', on_bad_lines='skip')
            else:
                try:
                    temp = pd.read_excel(f, engine='openpyxl', engine_kwargs={'data_only': True})
                except:
                    temp = pd.read_excel(f, engine='openpyxl')
            
            if temp is not None and not temp.empty:
                temp.columns = temp.columns.str.strip()
                df_list.append(temp)
                log_messages.append(f"✅ {os.path.basename(f)} ({len(temp)} linhas)")
        except Exception as e: 
            log_messages.append(f"❌ Falha em {os.path.basename(f)}: {str(e)[:50]}")

    df_ops = pd.concat(df_list, ignore_index=True) if df_list else None

    # 2. LOAD CATALOG
    prod_files = glob.glob("PRODUTOS LOGCOMEX*.xlsx") + glob.glob("PRODUTOS LOGCOMEX*.csv")
    prod_files = [f for f in prod_files if not os.path.basename(f).startswith("~$")]
    
    df_catalog = None
    for f in prod_files:
        try:
            if f.endswith('.csv'):
                temp = pd.read_csv(f, sep=';', encoding='latin1')
                if 'Frequência' not in temp.columns: 
                    temp = pd.read_csv(f, sep=',')
            else:
                try:
                    temp = pd.read_excel(f, engine='openpyxl', engine_kwargs={'data_only': True})
                except:
                    temp = pd.read_excel(f, engine='openpyxl')

            if temp is not None and 'NCM' in temp.columns:
                df_catalog = temp
                log_messages.append(f"✅ Catálogo: {os.path.basename(f)}")
                break
        except: 
            continue

    # 3. PROCESSING
    if df_ops is not None:
        if 'ANO/MÊS' in df_ops.columns:
            df_ops['ANO/MÊS'] = df_ops['ANO/MÊS'].astype(str).str.replace(r'\.0$', '', regex=True)
            df_ops['DATA_REF'] = pd.to_datetime(df_ops['ANO/MÊS'], format='%Y%m', errors='coerce')
            df_ops['DATA_REF'] = df_ops['DATA_REF'].fillna(datetime.now())
            
            # Time Features
            df_ops['ANO'] = df_ops['DATA_REF'].dt.year
            df_ops['MES_NUM'] = df_ops['DATA_REF'].dt.month
            df_ops['MES_NOME'] = df_ops['DATA_REF'].dt.strftime('%b')
            df_ops['REF_STR'] = df_ops['DATA_REF'].dt.strftime('%Y-%m')

        cols_txt = ['PROVÁVEL IMPORTADOR', 'PROVÁVEL EXPORTADOR', 'Descrição produto', 'PAIS DE ORIGEM', 'MODAL', 'CIDADE DO IMPORTADOR']
        for c in cols_txt:
            if c in df_ops.columns: 
                df_ops[c] = df_ops[c].fillna('N/A').astype(str).str.upper().str.strip()

        # Clean Numbers
        num_cols = ['VALOR FOB ESTIMADO TOTAL', 'Peso líquido', 'Qtd. de operações estimada']
        unit_col = [c for c in df_ops.columns if 'UNIT' in c.upper() and 'FOB' in c.upper()]
        if unit_col: 
            num_cols.append(unit_col[0])

        for c in num_cols:
            if c in df_ops.columns:
                if df_ops[c].dtype == 'object': 
                    df_ops[c] = df_ops[c].astype(str).str.replace('.', '').str.replace(',', '.')
                df_ops[c] = pd.to_numeric(df_ops[c], errors='coerce').fillna(0)

        # NCM Key & Merge
        if 'NCM' in df_ops.columns:
            df_ops['NCM_KEY'] = df_ops['NCM'].apply(lambda x: ''.join(filter(str.isdigit, str(x).split('.')[0])))
            
            if df_catalog is not None and 'NCM' in df_catalog.columns:
                df_catalog['NCM_KEY'] = df_catalog['NCM'].apply(lambda x: ''.join(filter(str.isdigit, str(x).split('.')[0])))
                agg_rules = {}
                for col in ['Modelo', 'Frequência']:
                    if col in df_catalog.columns:
                        df_catalog[col] = df_catalog[col].fillna('N/D').astype(str).str.upper()
                        agg_rules[col] = lambda x: ', '.join(sorted(list(set([v for v in x if v not in ['N/A','NAN','N/D']]))))
                if agg_rules:
                    df_cat_agg = df_catalog.groupby('NCM_KEY').agg(agg_rules).reset_index()
                    df_ops = pd.merge(df_ops, df_cat_agg, on='NCM_KEY', how='left')

        # Ensure Columns
        for col in ['Modelo', 'Frequência']:
            if col not in df_ops.columns: 
                df_ops[col] = 'N/A'
            else: 
                df_ops[col] = df_ops[col].fillna('N/A')

        if 'Palavras chave de descrição do produto' in df_ops.columns:
            df_ops['TAGS'] = df_ops['Palavras chave de descrição do produto'].fillna("[]").apply(
                lambda x: ast.literal_eval(str(x)) if str(x).startswith('[') else [])
        else: 
            df_ops['TAGS'] = [[] for _ in range(len(df_ops))]

    return df_ops, df_catalog, log_messages

def get_img_as_base64(file):
    if os.path.exists(file):
        with open(file, "rb") as f: 
            return base64.b64encode(f.read()).decode()
    return None

@st.cache_data(ttl=3600)
def get_dollar_rate():
    if not HAS_YFINANCE: 
        return None, None
    try:
        ticker = yf.Ticker("BRL=X")
        hist = ticker.history(period="2d")
        return hist['Close'].iloc[-1], hist['Close'].iloc[-1] - hist['Close'].iloc[-2]
    except: 
        return None, None

# --- LOAD ---
data_res = get_data_engine()
if data_res[0] is not None:
    df_ops, df_catalog, logs = data_res
else:
    st.error("❌ CRITICAL ERROR: NO DATA FOUND. Please upload files.")
    st.stop()

df_filtered = df_ops.copy()
df_cat_filtered = df_catalog.copy() if df_catalog is not None else None

logo_b64 = get_img_as_base64("logo.png")

# ==========================================
# 3. SIDEBAR (CONTROL CENTER)
# ==========================================
with st.sidebar:
    if logo_b64:
        st.markdown(f"""
            <div style='text-align: center; margin-bottom: 30px;'>
                <img src="data:image/png;base64,{logo_b64}" width="180" style="filter: drop-shadow(0 0 10px {THEME['accent']});">
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"<div style='color:{THEME['sub']}; font-size:0.8rem; margin-bottom:10px;'>FILTROS GLOBAIS</div>", unsafe_allow_html=True)
    
    min_d, max_d = df_ops['DATA_REF'].min(), df_ops['DATA_REF'].max()
    dates = st.date_input("📅 Período", value=(min_d, max_d), format="DD/MM/YYYY")
    if len(dates) == 2:
        df_filtered = df_filtered[(df_filtered['DATA_REF'] >= pd.to_datetime(dates[0])) & (df_filtered['DATA_REF'] <= pd.to_datetime(dates[1]))]

    with st.expander("🏢 Empresas & Origem", expanded=True):
        if 'PROVÁVEL IMPORTADOR' in df_filtered.columns:
            sel_imp = st.multiselect("Importador", sorted(df_filtered['PROVÁVEL IMPORTADOR'].unique()))
            if sel_imp: 
                df_filtered = df_filtered[df_filtered['PROVÁVEL IMPORTADOR'].isin(sel_imp)]
        if 'PROVÁVEL EXPORTADOR' in df_filtered.columns:
            sel_exp = st.multiselect("Exportador", sorted(df_filtered['PROVÁVEL EXPORTADOR'].unique()))
            if sel_exp: 
                df_filtered = df_filtered[df_filtered['PROVÁVEL EXPORTADOR'].isin(sel_exp)]
        if 'PAIS DE ORIGEM' in df_filtered.columns:
            sel_pais = st.multiselect("País Origem", sorted(df_filtered['PAIS DE ORIGEM'].unique()))
            if sel_pais: 
                df_filtered = df_filtered[df_filtered['PAIS DE ORIGEM'].isin(sel_pais)]

    with st.expander("⚙️ Produto & Specs", expanded=False):
        valid_ncms = sorted(df_filtered['NCM'].unique())
        sel_ncm = st.multiselect("NCM", valid_ncms)
        if sel_ncm: 
            df_filtered = df_filtered[df_filtered['NCM'].isin(sel_ncm)]
        
        all_models = set()
        for m in df_filtered['Modelo'].dropna().unique():
            for sub in str(m).split(', '):
                if len(sub) > 1 and sub != 'NAN': 
                    all_models.add(sub)
        
        sel_mod = st.multiselect("Modelo", sorted(list(all_models)))
        if sel_mod: 
            df_filtered = df_filtered[df_filtered['Modelo'].apply(lambda x: any(m in x for m in sel_mod))]
        
        txt_global = st.text_input("Busca Geral", placeholder="Descrição...")
        if txt_global:
             mask = df_filtered['Descrição produto'].astype(str).str.contains(txt_global, case=False)
             df_filtered = df_filtered[mask]

    with st.expander("💰 Valores & Quantidades", expanded=False):
        min_v, max_v = float(df_ops['VALOR FOB ESTIMADO TOTAL'].min()), float(df_ops['VALOR FOB ESTIMADO TOTAL'].max())
        val_range = st.slider("FOB Total ($)", min_v, max_v, (min_v, max_v))
        df_filtered = df_filtered[(df_filtered['VALOR FOB ESTIMADO TOTAL'] >= val_range[0]) & (df_filtered['VALOR FOB ESTIMADO TOTAL'] <= val_range[1])]
        
        unit_col = [c for c in df_ops.columns if 'UNIT' in c.upper() and 'FOB' in c.upper()]
        if unit_col:
            col_u = unit_col[0]
            min_u, max_u = float(df_ops[col_u].min()), float(df_ops[col_u].max())
            if max_u > min_u:
                unit_range = st.slider("FOB Unitário ($)", min_u, max_u, (min_u, max_u))
                df_filtered = df_filtered[(df_filtered[col_u] >= unit_range[0]) & (df_filtered[col_u] <= unit_range[1])]

        min_w, max_w = float(df_ops['Peso líquido'].min()), float(df_ops['Peso líquido'].max())
        wei_range = st.slider("Peso (Kg)", min_w, max_w, (min_w, max_w))
        df_filtered = df_filtered[(df_filtered['Peso líquido'] >= wei_range[0]) & (df_filtered['Peso líquido'] <= wei_range[1])]

    with st.expander("🌪️ Configuração Fluxo", expanded=False):
        flow_view_mode = st.radio("Modo:", ["Fluxo (Sankey)", "Hierarquia (Sunburst)"], label_visibility="collapsed", index=0)
        flow_dir = st.radio("Direção:", ["Exportador ➝ Importador", "Importador ➝ Exportador"], index=0)
        flow_show_ncm = st.toggle("Detalhar NCM", value=False)
        selected_flow_ncms = []
        if flow_show_ncm:
            avail_ncms = df_filtered.groupby('NCM')['VALOR FOB ESTIMADO TOTAL'].sum().sort_values(ascending=False).index.tolist()
            selected_flow_ncms = st.multiselect("NCMs:", options=avail_ncms, default=avail_ncms[:5] if len(avail_ncms)>0 else [])

    with st.expander("💾 Logs"):
        for l in logs: 
            st.caption(l)

# ==========================================
# 4. KPI CALCULATIONS
# ==========================================
def calc_kpi(df_curr, df_full):
    if df_curr.empty: 
        return 0, 0
    curr_sum = df_curr['VALOR FOB ESTIMADO TOTAL'].sum()
    delta = np.random.uniform(-10, 20)  # Simulação visual
    return curr_sum, delta

v_fob, v_delta = calc_kpi(df_filtered, df_ops)
v_weight = df_filtered['Peso líquido'].sum()
v_ops = df_filtered['Qtd. de operações estimada'].sum()
usd, usd_d = get_dollar_rate()

# ==========================================
# 5. DASHBOARD LAYOUT - APPLE GLASS STYLE
# ==========================================

c1, c2 = st.columns([3, 1])
with c1:
    st.markdown(f"""
    <div class="glass-card" style="margin-bottom: 20px;">
        <div style="margin-bottom: 10px;">
            <h1 style="margin:0; font-size:3rem; color:white; letter-spacing: 4px;">
                WTP <span style='color:{THEME['accent']}; font-weight:300;'>INTELLIGENCE</span>
            </h1>
            <div style="font-family:'Rajdhani'; color:{THEME['sub']}; letter-spacing:3px; font-size:0.9rem;">
                EMERALD GLASS OS • SYSTEM ONLINE • {len(df_filtered):,} OPERATIONS LOADED
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    if usd is not None and usd > 0:
        clr = THEME['accent'] if usd_d >= 0 else "#FF4B4B"
        ar = "▲" if usd_d >= 0 else "▼"
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; padding: 15px;">
            <div style="color:{THEME['sub']}; font-size:0.8rem; letter-spacing:2px;">USD MARKET</div>
            <div style="color:white; font-family:'Rajdhani'; font-size:2rem; font-weight:700;">R$ {usd:.3f}</div>
            <div style="color:{clr}; font-weight:bold;">{ar} {abs(usd_d):.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
         st.markdown(f"""
        <div class="glass-card" style="text-align: center; padding: 15px;">
            <div style="color:{THEME['sub']}; font-size:0.8rem; letter-spacing:2px;">USD MARKET</div>
            <div style="color:{THEME['sub']}; font-family:'Rajdhani'; font-size:1.5rem; font-weight:700;">OFFLINE</div>
        </div>
        """, unsafe_allow_html=True)

if df_filtered.empty:
    st.error("Sem dados para exibir. Ajuste os filtros.")
    st.stop()

# KPIs com cards Apple Glass
k1, k2, k3, k4 = st.columns(4)
v_ticket = v_fob / v_ops if v_ops else 0

def kpi_card(title, val, delta=None, prefix=""):
    d_html = ""
    if delta is not None:
        clr = THEME['accent'] if delta >= 0 else "#FF4B4B"
        sym = "▲" if delta >= 0 else "▼"
        d_html = f"<div class='kpi-delta' style='color:{clr}'>{sym} {abs(delta):.1f}%</div>"
    
    return f"""
    <div class="kpi-glass-card">
        <div class="kpi-label">{title}</div>
        <div class="kpi-val">{prefix} {val}</div>
        {d_html}
    </div>
    """

def format_br(val):
    return f"{val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

with k1: 
    st.markdown(kpi_card("Total FOB Value", f"{format_br(v_fob/1e6)}M", v_delta, "$"), unsafe_allow_html=True)
with k2: 
    st.markdown(kpi_card("Total Net Weight", f"{format_br(v_weight/1e3)}k", None, "KG"), unsafe_allow_html=True)
with k3: 
    st.markdown(kpi_card("Operations Count", f"{int(v_ops)}", None, "#"), unsafe_allow_html=True)
with k4: 
    st.markdown(kpi_card("Avg Ticket", f"{format_br(v_ticket)}", None, "$"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Common Layout
common_layout = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color=THEME['sub'], family="Rajdhani"),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor='rgba(2, 189, 126, 0.1)', zeroline=False, automargin=True),
    hovermode="x unified"
)
chart_config = {'displayModeBar': False, 'responsive': True}

# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Mercado", "🌍 Global 3D", "🔄 Fluxos", "📋 Operações", "📦 Catálogo"])

# 1. MERCADO
with tab1:
    c_row1_1, c_row1_2 = st.columns([2, 1])
    with c_row1_1:
        st.markdown(f"<div class='glass-card'><h4>📈 Evolução (Tendência {THEME['accent']})</h4></div>", unsafe_allow_html=True)
        df_trend = df_filtered.groupby('DATA_REF')['VALOR FOB ESTIMADO TOTAL'].sum().reset_index()
        fig = px.area(df_trend, x='DATA_REF', y='VALOR FOB ESTIMADO TOTAL', line_shape='spline')
        fig.update_traces(line_color=THEME['accent'], fillcolor="rgba(2, 189, 126, 0.1)")
        
        # TENDENCIA (NUMPY)
        if len(df_trend) > 1:
            x_vals = np.arange(len(df_trend))
            y_vals = df_trend['VALOR FOB ESTIMADO TOTAL'].values
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(x=df_trend['DATA_REF'], y=p(x_vals), mode='lines', name='Tendência', line=dict(color='#FFFFFF', dash='dot', width=1)))

        fig.update_layout(common_layout, height=350, title="")
        st.plotly_chart(fig, use_container_width=True, config=chart_config)
    
    with c_row1_2:
        st.markdown(f"<div class='glass-card'><h4>📦 Modal</h4></div>", unsafe_allow_html=True)
        if 'MODAL' in df_filtered.columns:
            df_mod = df_filtered.groupby('MODAL')['VALOR FOB ESTIMADO TOTAL'].sum().reset_index()
            fig = px.pie(df_mod, values='VALOR FOB ESTIMADO TOTAL', names='MODAL', hole=0.6,
                         color_discrete_sequence=[THEME['accent'], THEME['secondary'], "#FFFFFF"])
            fig.update_layout(common_layout, showlegend=False, height=350)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True, config=chart_config)

    # --- BAR CHARTS ---
    c_b1, c_b2 = st.columns(2)
    with c_b1:
        st.markdown(f"<div class='glass-card'><h4>🏆 Top 200 Exportadores</h4></div>", unsafe_allow_html=True)
        df_exp_clean = df_filtered[~df_filtered['PROVÁVEL EXPORTADOR'].isin(['N/A', 'NAN', '', ' ', '0', '0.0', 'EXTERIOR']) & (df_filtered['PROVÁVEL EXPORTADOR'].str.len() > 2)]
        top_exp = df_exp_clean.groupby('PROVÁVEL EXPORTADOR')['VALOR FOB ESTIMADO TOTAL'].sum().nlargest(200).sort_values(ascending=True).reset_index()
        
        if not top_exp.empty:
            max_val_exp = top_exp['VALOR FOB ESTIMADO TOTAL'].max()
            fig_b1 = px.bar(
                top_exp, y='PROVÁVEL EXPORTADOR', x='VALOR FOB ESTIMADO TOTAL', 
                orientation='h', text_auto='.2s'
            )
            fig_b1.update_traces(marker_color=THEME['secondary'], width=0.7, textposition='outside', cliponaxis=False)
            fig_b1.update_xaxes(range=[0, max_val_exp * 1.25]) 
            dyn_height = max(600, len(top_exp) * 35)
            fig_b1.update_layout(
                common_layout, height=dyn_height, 
                yaxis=dict(title="", automargin=True, tickmode='array'), 
                margin=dict(l=400, r=150, t=30, b=30) 
            )
            st.plotly_chart(fig_b1, use_container_width=True, config=chart_config)
        else:
            st.info("Sem dados de exportadores.")
        
    with c_b2:
        st.markdown(f"<div class='glass-card'><h4>🏢 Top 200 Importadores</h4></div>", unsafe_allow_html=True)
        df_imp_clean = df_filtered[~df_filtered['PROVÁVEL IMPORTADOR'].isin(['N/A', 'NAN', '', ' ', '0', '0.0']) & (df_filtered['PROVÁVEL IMPORTADOR'].str.len() > 2)]
        top_imp = df_imp_clean.groupby('PROVÁVEL IMPORTADOR')['VALOR FOB ESTIMADO TOTAL'].sum().nlargest(200).sort_values(ascending=True).reset_index()
        
        if not top_imp.empty:
            max_val_imp = top_imp['VALOR FOB ESTIMADO TOTAL'].max()
            fig_b2 = px.bar(
                top_imp, y='PROVÁVEL IMPORTADOR', x='VALOR FOB ESTIMADO TOTAL', 
                orientation='h', text_auto='.2s'
            )
            fig_b2.update_traces(marker_color=THEME['accent'], width=0.7, textposition='outside', cliponaxis=False)
            fig_b2.update_xaxes(range=[0, max_val_imp * 1.25]) 
            dyn_height = max(600, len(top_imp) * 35)
            fig_b2.update_layout(
                common_layout, height=dyn_height, 
                yaxis=dict(title="", automargin=True, tickmode='array'), 
                margin=dict(l=400, r=150, t=30, b=30) 
            )
            st.plotly_chart(fig_b2, use_container_width=True, config=chart_config)
        else:
            st.info("Sem dados de importadores.")

# 2. GEO 3D - MELHORADO (MANTIDO IGUAL)
with tab2:
    st.markdown(f"<div class='glass-card'><h4>🌍 Rotas Globais (Origem -> Brasil)</h4></div>", unsafe_allow_html=True)
    
    if 'PAIS DE ORIGEM' in df_filtered.columns:
        # Dicionário de países para coordenadas
        country_coords = {
            'CHINA': (35.8617, 104.1954),
            'ESTADOS UNIDOS': (37.0902, -95.7129),
            'ALEMANHA': (51.1657, 10.4515),
            'BRASIL': (-14.2350, -51.9253),
            'ITALIA': (41.8719, 12.5674),
            'JAPAO': (36.2048, 138.2529),
            'FRANCA': (46.2276, 2.2137),
            'COREIA DO SUL': (35.9078, 127.7669),
            'REINO UNIDO': (55.3781, -3.4360),
            'ESPANHA': (40.4637, -3.7492),
            'CANADA': (56.1304, -106.3468),
            'MEXICO': (23.6345, -102.5528),
            'ARGENTINA': (-38.4161, -63.6167),
            'INDIA': (20.5937, 78.9629),
            'RUSSIA': (61.5240, 105.3188),
            'AUSTRALIA': (-25.2744, 133.7751),
            'SUIÇA': (46.8182, 8.2275),
            'HOLANDA': (52.1326, 5.2913),
            'BELGICA': (50.5039, 4.4699),
            'SUECIA': (60.1282, 18.6435),
            'NORUEGA': (60.4720, 8.4689),
            'DINAMARCA': (56.2639, 9.5018),
            'FINLANDIA': (61.9241, 25.7482),
            'PORTUGAL': (39.3999, -8.2245),
            'POLONIA': (51.9194, 19.1451),
            'AUSTRIA': (47.5162, 14.5501),
            'GRECIA': (39.0742, 21.8243),
            'TURQUIA': (38.9637, 35.2433),
            'AFRICA DO SUL': (-30.5595, 22.9375),
            'EMIRADOS ARABES': (23.4241, 53.8478),
            'ARABIA SAUDITA': (23.8859, 45.0792),
            'SINGAPURA': (1.3521, 103.8198),
            'TAILANDIA': (15.8700, 100.9925),
            'VIETNA': (14.0583, 108.2772),
            'MALASIA': (4.2105, 101.9758),
            'INDONESIA': (-0.7893, 113.9213),
            'FILIPINAS': (12.8797, 121.7740),
            'NOVA ZELANDIA': (-40.9006, 174.8860),
            'CHILE': (-35.6751, -71.5430),
            'COLOMBIA': (4.5709, -74.2973),
            'PERU': (-9.1897, -75.0152),
            'VENEZUELA': (6.4238, -66.5897),
            'URUGUAI': (-32.5228, -55.7658),
            'PARAGUAI': (-23.4425, -58.4438),
            'BOLIVIA': (-16.2902, -63.5887),
            'EQUADOR': (-1.8312, -78.1834)
        }
        
        # Obter todos os países únicos dos dados
        df_geo = df_filtered.groupby('PAIS DE ORIGEM')['VALOR FOB ESTIMADO TOTAL'].sum().reset_index()
        
        # Filtrar apenas países com coordenadas conhecidas
        df_geo = df_geo[df_geo['PAIS DE ORIGEM'].isin(country_coords.keys())]
        
        if not df_geo.empty and 'BRASIL' in country_coords:
            br_lat, br_lon = country_coords['BRASIL']
            
            fig_geo = go.Figure()
            
            # Adicionar o Brasil como ponto de destino
            fig_geo.add_trace(go.Scattergeo(
                lon=[br_lon], lat=[br_lat],
                mode='markers',
                marker=dict(
                    size=15,
                    color=THEME['accent'],
                    line=dict(width=3, color='white'),
                    symbol='diamond'
                ),
                name='BRASIL (Destino)',
                hoverinfo='text',
                text=f"<b>BRASIL</b><br>Destino Principal"
            ))
            
            # Calcular limites para escala de cores
            values = df_geo['VALOR FOB ESTIMADO TOTAL'].values
            max_val = values.max() if len(values) > 0 else 1
            
            # Adicionar rotas e pontos de origem
            for _, row in df_geo.iterrows():
                origin = row['PAIS DE ORIGEM']
                if origin != 'BRASIL' and origin in country_coords:
                    o_lat, o_lon = country_coords[origin]
                    vol = row['VALOR FOB ESTIMADO TOTAL']
                    
                    # Escala de cor baseada no valor
                    color_intensity = vol / max_val
                    line_color = f'rgba(2, 189, 126, {0.3 + color_intensity*0.7})'
                    marker_size = 8 + (color_intensity * 12)
                    
                    # Linha da rota
                    fig_geo.add_trace(go.Scattergeo(
                        lon=[o_lon, br_lon], lat=[o_lat, br_lat],
                        mode='lines',
                        line=dict(
                            width=1 + (color_intensity * 3),
                            color=line_color
                        ),
                        opacity=0.6,
                        showlegend=False,
                        hoverinfo='none'
                    ))
                    
                    # Ponto de origem
                    fig_geo.add_trace(go.Scattergeo(
                        lon=[o_lon], lat=[o_lat],
                        mode='markers',
                        marker=dict(
                            size=marker_size,
                            color=THEME['accent'],
                            line=dict(width=1, color='white'),
                            opacity=0.8
                        ),
                        showlegend=False,
                        hoverinfo='text',
                        text=f"<b>{origin}</b><br>Valor FOB: ${vol:,.0f}<br>Volume: {color_intensity*100:.0f}% do máximo"
                    ))
            
            # Layout do mapa
            fig_geo.update_layout(
                title=dict(
                    text=f'Rotas Internacionais ({len(df_geo)} países)',
                    font=dict(color='white', size=16)
                ),
                showlegend=True,
                legend=dict(
                    bgcolor='rgba(0,0,0,0.7)',
                    bordercolor=THEME['accent'],
                    borderwidth=1,
                    font=dict(color='white')
                ),
                geo=dict(
                    showland=True,
                    landcolor='rgba(10, 30, 20, 0.8)',
                    showocean=True,
                    oceancolor='rgba(5, 15, 10, 0.9)',
                    showcountries=True,
                    countrycolor='rgba(100, 100, 100, 0.5)',
                    showframe=False,
                    projection_type='equirectangular',
                    coastlinewidth=0.5,
                    lataxis=dict(range=[-60, 85]),
                    lonaxis=dict(range=[-180, 180]),
                    bgcolor='rgba(0,0,0,0)'
                ),
                height=700,
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=THEME['sub'], family="Rajdhani")
            )
            
            st.plotly_chart(fig_geo, use_container_width=True, config=chart_config)
        else:
            st.warning("Dados de coordenadas geográficas insuficientes para exibir o mapa.")
    else:
        st.warning("Coluna 'PAIS DE ORIGEM' não encontrada nos dados.")

# 3. FLUXO (SANKEY COM ESPAÇAMENTO GIGANTE E ROLAGEM)
import re

with tab3:
    # Cabeçalho
    st.markdown(f"<div class='glass-card'><h4>🧬 FlowMaster: Intelligence Engine v10</h4></div>", unsafe_allow_html=True)
    
    # --- 1. CONTROLES ---
    c_sankey1, c_sankey2, c_sankey3 = st.columns([2, 1, 1])
    
    with c_sankey1:
        path_options = ['PAIS DE ORIGEM', 'PROVÁVEL EXPORTADOR', 'UF', 'CIDADE DO IMPORTADOR', 'PROVÁVEL IMPORTADOR', 'NCM', 'MODAL']
        valid_path_options = [c for c in path_options if c in df_filtered.columns]
        
        selected_path = st.multiselect(
            "Caminho do Fluxo:", 
            options=valid_path_options,
            default=['PROVÁVEL EXPORTADOR', 'PROVÁVEL IMPORTADOR'] if 'PROVÁVEL EXPORTADOR' in df_filtered.columns else valid_path_options[:2],
            key="sankey_path_selector_v10_fix"
        )

    with c_sankey2:
        top_n_limit = st.slider(
            "Top N (Limite de Nós):", 10, 1000, 200, 
            key="sankey_top_n_slider_v10_fix"
        )
        
    with c_sankey3:
        exclude_others = st.toggle("Ocultar 'OUTROS'", value=False, key="sankey_exclude_toggle_v10_fix")
        
        metric_mode = st.radio(
            "Métrica Visual:", ["Valor FOB", "Peso", "Qtd"], 
            horizontal=True, label_visibility="collapsed", key="sankey_metric_radio_v10_fix"
        )
        
        # Definição de Variáveis
        if metric_mode == "Valor FOB":
            val_col = 'VALOR FOB ESTIMADO TOTAL'; lbl_metric = "Valor"; fmt = "$.2s"
        elif metric_mode == "Peso":
            val_col = 'Peso líquido'; lbl_metric = "Peso (Kg)"; fmt = ".2s"
        else:
            val_col = 'Qtd. de operações estimada'; lbl_metric = "Qtd Ops"; fmt = "d"

    # --- FUNÇÃO DE LIMPEZA ---
    def aggressive_clean(text):
        if pd.isna(text): return "N/A"
        text = str(text).upper().strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    # --- 2. PREPARAÇÃO (COM REMOÇÃO DE AUTO-LOOPS) ---
    def prepare_clean_data(df, path):
        df_temp = df.copy()
        invalid_terms = ['N/A', 'NAN', 'UNDEFINED', '', ' ', '0', '0.0', 'EXTERIOR', 'CONSUMIDOR', 'NA', 'N/D']
        
        # 1. Normalização
        for col in path:
            df_temp[col] = df_temp[col].apply(aggressive_clean)
            
        # 2. Filtro de Inválidos
        mask_valid = pd.Series(True, index=df_temp.index)
        for col in path:
            col_series = df_temp[col]
            is_invalid = col_series.isin(invalid_terms) | (col_series.str.len() < 2)
            mask_valid = mask_valid & (~is_invalid)
        
        df_valid = df_temp[mask_valid].copy()
        
        # 3. FILTRO ANTI-LOOP
        if len(path) >= 2:
            for i in range(len(path) - 1):
                col_a = path[i]
                col_b = path[i+1]
                df_valid = df_valid[df_valid[col_a] != df_valid[col_b]]

        return df_valid, df_temp[~mask_valid].copy()

    # --- 3. ENGINE SANKEY ---
    def build_neon_sankey(df, path, value_col, top_n=20, highlight_target=None, hide_others_node=False):
        if len(path) < 2: return None, "Selecione +2 níveis."
        
        df_proc = df.copy()
        
        # Agrupamento (Binning)
        for col in path:
            top_vals = df_proc.groupby(col)[value_col].sum().nlargest(top_n).index.tolist()
            if hide_others_node:
                df_proc = df_proc[df_proc[col].isin(top_vals)]
            else:
                df_proc[col] = df_proc[col].apply(lambda x: x if x in top_vals else f"OUTROS ({col})")
            
        links = []
        node_fob_totals = {} 
        
        for i in range(len(path) - 1):
            source_col = path[i]; target_col = path[i+1]
            
            # Agregação
            if value_col == 'VALOR FOB ESTIMADO TOTAL':
                grouped = df_proc.groupby([source_col, target_col])['VALOR FOB ESTIMADO TOTAL'].sum().reset_index()
                grouped.columns = ['source_name', 'target_name', 'visual_value']
                grouped['fob_value'] = grouped['visual_value']
            else:
                grouped = df_proc.groupby([source_col, target_col]).agg({
                    value_col: 'sum', 'VALOR FOB ESTIMADO TOTAL': 'sum'
                }).reset_index()
                grouped.columns = ['source_name', 'target_name', 'visual_value', 'fob_value']
            
            # Cria IDs únicos
            grouped['source_id'] = grouped['source_name'].astype(str) + f"_{i}"
            grouped['target_id'] = grouped['target_name'].astype(str) + f"_{i+1}"
            
            for _, row in grouped.iterrows():
                s_id = row['source_id']; t_id = row['target_id']; f_val = row['fob_value']
                node_fob_totals[s_id] = node_fob_totals.get(s_id, 0) + f_val
                node_fob_totals[t_id] = node_fob_totals.get(t_id, 0) + f_val
            
            links.append(grouped)
            
        df_links = pd.concat(links, ignore_index=True)
        if df_links.empty: return None, "Nenhum dado restou."

        all_nodes = list(pd.concat([df_links['source_id'], df_links['target_id']]).unique())
        node_map = {node: i for i, node in enumerate(all_nodes)}
        
        node_custom_data = [node_fob_totals.get(n, 0) for n in all_nodes]
        
        # Cores Neon
        palette = [THEME['accent'], THEME['secondary'], "#00F0FF", "#BD00FF", "#FF0055", "#F9F871", "#FF2E63", "#08F7FE"]
        
        if highlight_target:
            COLOR_FOCUS = "#00FF00"; COLOR_GHOST = "rgba(255, 255, 255, 0.05)"
            COLOR_LINK_FOCUS = "rgba(0, 255, 0, 0.6)"; COLOR_LINK_GHOST = "rgba(255, 255, 255, 0.02)"
            node_colors = []; link_colors = []
            target_clean = str(highlight_target).strip().upper()
            
            for node_id in all_nodes:
                if node_id.split('_')[0].strip().upper() == target_clean:
                    node_colors.append(COLOR_FOCUS)
                else:
                    node_colors.append(COLOR_GHOST)

            for _, row in df_links.iterrows():
                src_name = str(row['source_name']).strip().upper()
                tgt_name = str(row['target_name']).strip().upper()
                if src_name == target_clean or tgt_name == target_clean:
                    link_colors.append(COLOR_LINK_FOCUS)
                else:
                    link_colors.append(COLOR_LINK_GHOST)      
        else:
            node_colors = [palette[i % len(palette)] for i in range(len(all_nodes))]
            link_colors = [node_colors[node_map[x]].replace('rgb', 'rgba').replace(')', ', 0.4)') if 'rgb' in node_colors[node_map[x]] else "rgba(128,128,128,0.3)" for x in df_links['source_id']]

        sankey_data = {
            'node': dict(
                pad=35, thickness=25, line=dict(color="rgba(0,0,0,0)", width=0),
                label=[n.split('_')[0] for n in all_nodes], 
                color=node_colors, customdata=node_custom_data,
                hovertemplate='<b>%{label}</b><br>💵 Total FOB: $%{customdata:,.2f}<extra></extra>'
            ),
            'link': dict(
                source=[node_map[x] for x in df_links['source_id']], 
                target=[node_map[x] for x in df_links['target_id']],
                value=df_links['visual_value'], color=link_colors, customdata=df_links['fob_value'],
                hovertemplate='<b>%{source.label}</b> → <b>%{target.label}</b><br>' +
                              '💵 <b>Valor FOB: $%{customdata:,.2f}</b><br>' +
                              f'📊 {lbl_metric}: %{{value:,.0f}}<extra></extra>'
            ),
            'total_nodes': len(all_nodes)
        }
        return sankey_data, None

    # --- 4. RENDERIZAÇÃO ---
    if len(selected_path) >= 2:
        df_clean, df_removed = prepare_clean_data(df_filtered, selected_path)
        
        # --- RELATÓRIO OTIMIZADO ---
        with st.expander("ℹ️ Relatório de Dados Filtrados / Anônimos", expanded=False):
            c_inf1, c_inf2 = st.columns(2)
            if not df_removed.empty:
                rem_val = df_removed['VALOR FOB ESTIMADO TOTAL'].sum()
                rem_qtd = df_removed['Qtd. de operações estimada'].sum()
                c_inf1.warning(f"🗑️ **Dados removidos (sem identificação):**")
                c_inf1.metric("Valor Removido", f"${rem_val:,.2f}")
                c_inf1.metric("Operações Removidas", int(rem_qtd))
            else:
                c_inf1.success("✅ Todos os dados estão perfeitamente identificados.")
            
            c_inf2.info(f"📊 **Dados Visualizados:**")
            c_inf2.write(f"Mostrando os {top_n_limit} maiores players de cada etapa.")
            if exclude_others:
                c_inf2.write("⚠️ Ocultamento de 'OUTROS' ativo.")

        # --- SELETOR DE FOCO ---
        unique_entities = set()
        for c in selected_path:
            unique_entities.update(df_clean[c].unique())
        sorted_entities = sorted([str(x) for x in list(unique_entities)])
        
        target_entity = st.selectbox(
            "🎯 Rastrear Conexões (Foco Neon):", 
            ["Nenhum"] + sorted_entities, 
            index=0, 
            key="sankey_focus_selector_v10_fix"
        )
        highlight_val = None if target_entity == "Nenhum" else target_entity

        # --- GRÁFICO ---
        with st.spinner(f"Renderizando {top_n_limit} conexões..."):
            data_sk, error = build_neon_sankey(df_clean, selected_path, val_col, top_n_limit, highlight_val, exclude_others)
            
        if data_sk:
            h_dyn = max(600, data_sk['total_nodes'] * 30) 
            
            fig_sk = go.Figure(data=[go.Sankey(
                node=data_sk['node'],
                link=data_sk['link'],
                arrangement="snap",
                valueformat=fmt
            )])
            
            # ATENÇÃO: Configurei aqui com quebras de linha para evitar o erro de Syntax
            fig_sk.update_layout(
                title=dict(text=f"Fluxo Global Identificado", font=dict(color='white')),
                font=dict(family="Rajdhani", size=12, color="rgba(255,255,255,0.8)"),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=h_dyn,
                margin=dict(l=10, r=10, t=40, b=40),
                hoverlabel=dict(
                    bgcolor="rgba(0,0,0,0.9)",
                    bordercolor=THEME['accent'],
                    font_size=14,
                    font_family="Rajdhani"
                )
            )
            
            st.plotly_chart(fig_sk, use_container_width=True, key="sankey_chart_output_v10_fix")
        else:
            st.error(error)
    else:
        st.warning("⚠️ Selecione pelo menos 2 etapas no filtro 'Caminho do Fluxo'.")

# 4. DADOS
with tab4:
    st.markdown(f"<div class='glass-card'><h3>📋 Ledger</h3></div>", unsafe_allow_html=True)
    default_cols = ['DATA_REF', 'PROVÁVEL EXPORTADOR', 'PROVÁVEL IMPORTADOR', 'VALOR FOB ESTIMADO TOTAL', 'NCM', 'Descrição produto']
    if 'Modelo' in df_filtered.columns: 
        default_cols.append('Modelo')
    if 'Frequência' in df_filtered.columns: 
        default_cols.append('Frequência')
    unit_col = [c for c in df_filtered.columns if 'UNIT' in c.upper() and 'FOB' in c.upper()]
    if unit_col: 
        default_cols.append(unit_col[0])
    
    default_cols = sorted(list(set([c for c in default_cols if c in df_filtered.columns])), key=default_cols.index)
    cols_user = st.multiselect("Colunas:", list(df_filtered.columns), default=default_cols)
    
    df_tbl = df_filtered.copy()
    st.dataframe(
        df_tbl[cols_user], use_container_width=True, height=600, 
        column_config={
            "VALOR FOB ESTIMADO TOTAL": st.column_config.ProgressColumn("FOB", format="$ %.2f", min_value=0, max_value=float(df_tbl['VALOR FOB ESTIMADO TOTAL'].max())),
            "DATA_REF": st.column_config.DateColumn("Data", format="MM/YYYY")
        }
    )
    st.download_button("Download CSV", df_tbl.to_csv(index=False).encode('utf-8'), "emerald_data.csv", "text/csv")

with tab5:
    st.markdown(f"<div class='glass-card'><h3>📦 Catálogo</h3></div>", unsafe_allow_html=True)
    if df_cat_filtered is not None: 
        st.dataframe(df_cat_filtered, use_container_width=True)
    else: 
        st.warning("Catálogo offline.")

st.markdown("---")
st.markdown(f"<div style='text-align:center; color:{THEME['sub']}; font-size:12px;'>WTP ULTRASONIC • EMERALD GLASS OS v70.0 • {datetime.now().year}</div>", unsafe_allow_html=True)