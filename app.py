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
import re
import io
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ==========================================
# 1. SETUP & PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="WTP Comex | Enterprise Intelligence",
    page_icon="❇️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': 'WTP Comex Intelligence OS • v85.0 Enterprise • Lead: Luciano Diniz'
    }
)

# ==========================================
# 2. DESIGN SYSTEM & REFINED OBSIDIAN-SLATE GLASS
# ==========================================
THEME = {
    "bg": "#0B0F17",
    "surface": "rgba(17, 24, 39, 0.82)",
    "surface_card": "rgba(20, 29, 46, 0.65)",
    "border": "rgba(255, 255, 255, 0.08)",
    "border_accent": "rgba(16, 185, 129, 0.28)",
    "text_main": "#F8FAFC",
    "text_muted": "#94A3B8",
    "accent_primary": "#10B981",    # Emerald
    "accent_mint": "#34D399",       # Mint
    "accent_cyan": "#06B6D4",       # Tech Cyan
    "accent_blue": "#3B82F6",       # Blue
    "warning": "#F59E0B",           # Amber
    "danger": "#EF4444",            # Coral Red
    "radius_sm": "8px",
    "radius_md": "14px",
    "radius_lg": "18px",
    "shadow_sm": "0 4px 14px rgba(0, 0, 0, 0.35)",
    "shadow_lg": "0 10px 30px rgba(0, 0, 0, 0.55)"
}

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    /* GLOBAL RESET & TYPOGRAPHY */
    .stApp {{ 
        background-color: {THEME['bg']}; 
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(16, 185, 129, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 90% 90%, rgba(6, 182, 212, 0.04) 0%, transparent 40%),
            radial-gradient(circle at 10% 85%, rgba(59, 130, 246, 0.03) 0%, transparent 35%);
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif; 
        color: {THEME['text_main']};
    }}
    
    /* HEADINGS */
    h1, h2, h3, h4, h5 {{ 
        font-family: 'Plus Jakarta Sans', sans-serif !important; 
        font-weight: 700 !important; 
        color: #FFFFFF !important; 
        letter-spacing: -0.3px !important;
    }}
    
    header[data-testid="stHeader"] {{ background: transparent !important; z-index: 999; }}
    div[data-testid="stDecoration"] {{ display: none; }}
    .main .block-container {{ padding-top: 1.5rem !important; padding-bottom: 3rem !important; max-width: 100% !important; }}
    
    /* CARDS - CLEAN MODERN GLASS */
    .glass-card {{
        background: {THEME['surface_card']};
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border-radius: {THEME['radius_md']};
        border: 1px solid {THEME['border']};
        padding: 1.25rem 1.4rem;
        box-shadow: {THEME['shadow_sm']};
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
        position: relative;
        overflow: hidden;
        margin-bottom: 1rem;
    }}
    
    .glass-card:hover {{
        border-color: {THEME['border_accent']};
        box-shadow: {THEME['shadow_lg']}, 0 0 20px rgba(16, 185, 129, 0.08);
    }}
    
    /* HERO HEADER */
    .hero-container {{
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(17, 24, 39, 0.7) 100%);
        border: 1px solid {THEME['border_accent']};
        border-radius: {THEME['radius_lg']};
        padding: 1.3rem 1.6rem;
        margin-bottom: 1.2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }}
    
    .hero-title {{
        font-size: clamp(1.6rem, 2.8vw, 2.4rem);
        font-weight: 800;
        margin: 0;
        line-height: 1.1;
        letter-spacing: -0.5px;
    }}
    
    .hero-subtitle {{
        font-size: 0.85rem;
        color: {THEME['text_muted']};
        margin-top: 5px;
        font-weight: 500;
        letter-spacing: 0.3px;
    }}
    
    /* KPI HERO METRIC CARDS */
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 0.85rem;
        margin-bottom: 1.3rem;
    }}
    
    .kpi-card {{
        background: rgba(17, 24, 39, 0.75);
        backdrop-filter: blur(16px);
        border-radius: {THEME['radius_md']};
        border: 1px solid {THEME['border']};
        padding: 1.1rem 1.25rem;
        box-shadow: {THEME['shadow_sm']};
        position: relative;
        min-width: 0;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    
    .kpi-card:hover {{
        border-color: {THEME['accent_primary']};
        transform: translateY(-2px);
    }}
    
    .kpi-card-top {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.35rem;
    }}
    
    .kpi-label {{
        font-size: 0.74rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: {THEME['text_muted']};
        white-space: normal;
        word-break: break-word;
        line-height: 1.25;
    }}
    
    .kpi-value {{
        font-family: 'JetBrains Mono', 'Plus Jakarta Sans', monospace;
        font-size: clamp(1.2rem, 1.55vw, 1.7rem);
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.2;
        letter-spacing: -0.4px;
        font-variant-numeric: tabular-nums;
        margin: 0.2rem 0;
        white-space: normal;
        word-break: break-word;
    }}
    
    .kpi-footer {{
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.78rem;
        color: {THEME['text_muted']};
        margin-top: 0.35rem;
        white-space: normal;
        word-break: break-word;
        line-height: 1.3;
    }}
    .kpi-footer span {{
        white-space: normal;
        word-break: break-word;
    }}
    
    .kpi-tag {{
        display: inline-flex;
        align-items: center;
        padding: 0.12rem 0.5rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }}
    
    .tag-pos {{ background: rgba(16, 185, 129, 0.18); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }}
    .tag-neg {{ background: rgba(239, 68, 68, 0.18); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.3); }}
    .tag-neutral {{ background: rgba(148, 163, 184, 0.12); color: #CBD5E1; }}
    
    /* STATUS PILL */
    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0.32rem 0.8rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }}
    
    .status-dot {{
        width: 7px; height: 7px;
        border-radius: 50%;
        background-color: {THEME['accent_primary']};
        box-shadow: 0 0 8px {THEME['accent_primary']};
    }}
    
    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {{
        background-color: #080C14 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }}
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] {{
        background: rgba(17, 24, 39, 0.6);
        padding: 4px;
        border-radius: 12px;
        border: 1px solid {THEME['border']};
        gap: 4px;
        backdrop-filter: blur(12px);
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 40px;
        color: {THEME['text_muted']};
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 600;
        font-size: 0.92rem;
        border-radius: 8px;
        padding: 0 1rem;
        border: none;
        transition: all 0.2s ease;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: rgba(16, 185, 129, 0.18) !important;
        color: #FFFFFF !important;
        border-bottom: 2px solid {THEME['accent_primary']} !important;
        box-shadow: 0 2px 10px rgba(16, 185, 129, 0.15);
    }}
    
    /* INPUTS & CONTROLS */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div, 
    .stMultiSelect div[data-baseweb="select"] > div, .stNumberInput input, .stDateInput input {{
        background-color: rgba(17, 24, 39, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        font-size: 0.88rem;
    }}
    
    .stTextInput input:focus, .stSelectbox div[data-baseweb="select"]:focus-within {{
        border-color: {THEME['accent_primary']} !important;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.25) !important;
    }}
    
    /* STREAMLIT BUTTONS */
    div.stButton > button {{
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.3) 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(16, 185, 129, 0.4) !important;
        border-radius: 9px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
    }}
    div.stButton > button:hover {{
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.35) 0%, rgba(5, 150, 105, 0.5) 100%) !important;
        border-color: {THEME['accent_primary']} !important;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.25) !important;
        transform: translateY(-1px);
    }}
    
    /* DATAFRAMES & TABLES */
    div[data-testid="stDataFrame"] {{
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        background: rgba(11, 15, 23, 0.6);
        overflow: hidden;
    }}
    
    /* PLOTLY CONTAINER */
    .js-plotly-plot .plotly .modebar {{ display: none !important; }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. AUTHENTICATION & ACCESS CONTROL
# ==========================================
def check_password():
    DEFAULT_PASS = "WTP_C0m3x"
    try:
        SENHA_CORRETA = st.secrets.get("PASSWORD", DEFAULT_PASS)
    except Exception:
        SENHA_CORRETA = DEFAULT_PASS
    
    if st.session_state.get("authenticated", False):
        return True

    def validate():
        if st.session_state.get("pwd_input") == SENHA_CORRETA:
            st.session_state["authenticated"] = True
            st.session_state["login_error"] = False
        else:
            st.session_state["login_error"] = True

    st.markdown("<br><br>", unsafe_allow_html=True)
    c_center = st.columns([1, 1.1, 1])[1]
    with c_center:
        st.markdown(
            f"<div class='glass-card' style='text-align: center; padding: 2.5rem 2rem;'>"
            f"<div style='font-size: 2.8rem; margin-bottom: 0.6rem;'>🔐</div>"
            f"<h2 style='margin: 0; font-size: 2rem;'>WTP COMEX</h2>"
            f"<div style='color: {THEME['accent_primary']}; font-weight: 600; font-size: 0.9rem; margin-bottom: 1.5rem; letter-spacing: 0.5px;'>ENTERPRISE INTELLIGENCE TERMINAL</div>"
            f"<p style='color: {THEME['text_muted']}; font-size: 0.85rem; margin-bottom: 1.5rem;'>Ambiente corporativo restrito. Digite a credencial para continuar.</p>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        st.text_input("Senha de Acesso:", type="password", key="pwd_input", on_change=validate, placeholder="Digite a senha...")
        if st.button("Entrar no Terminal", use_container_width=True, on_click=validate):
            pass
        
        if st.session_state.get("login_error", False):
            st.error("⛔ Senha incorreta. Tente novamente.")
            
    return False

if not check_password():
    st.stop()

# ==========================================
# 4. PRECISE DECIMAL & CURRENCY FORMATTERS
# ==========================================
def fmt_moeda_br(val, compact=False, moeda='USD'):
    """
    Formata moeda com padrão brasileiro impecável:
    R$ 1.234.567,89 ou $ 1.234.567,89 (com separador de milhar e 2 decimais).
    """
    if pd.isna(val) or val is None:
        return ("R$ 0,00" if moeda == 'BRL' else "$ 0,00")
    v = float(val)
    prefix = ("R$ " if moeda == 'BRL' else "$ ")
    
    if compact:
        if abs(v) >= 1e9:
            s = f"{v/1e9:,.2f} B".replace(",", "X").replace(".", ",").replace("X", ".")
            return prefix + s
        elif abs(v) >= 1e6:
            s = f"{v/1e6:,.2f} M".replace(",", "X").replace(".", ",").replace("X", ".")
            return prefix + s
        elif abs(v) >= 1e3:
            s = f"{v/1e3:,.2f} k".replace(",", "X").replace(".", ",").replace("X", ".")
            return prefix + s
            
    s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return prefix + s

def fmt_peso_br(val, compact=False):
    """Formata peso líquido em Kg ou Toneladas com separador de milhar."""
    if pd.isna(val) or val is None:
        return "0,00 kg"
    v = float(val)
    if compact and abs(v) >= 1e3:
        s = f"{v/1e3:,.2f} t".replace(",", "X").replace(".", ",").replace("X", ".")
        return s
    s = f"{v:,.2f} kg".replace(",", "X").replace(".", ",").replace("X", ".")
    return s

def fmt_int_br(val):
    """Formata número inteiro com separador de milhar padrão brasileiro."""
    if pd.isna(val) or val is None:
        return "0"
    return f"{int(val):,}".replace(",", ".")

def fmt_pct_br(val):
    """Formata percentual com 2 casas decimais."""
    if pd.isna(val) or val is None:
        return "0,00%"
    return f"{float(val):,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")

def clean_num(val):
    """Higienização robusta de números de strings do Excel."""
    if pd.isna(val) or val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace('$', '').replace('R$', '').replace('USD', '').replace(' ', '')
    if ',' in s and '.' in s:
        s = s.replace('.', '').replace(',', '.')
    elif ',' in s:
        s = s.replace(',', '.')
    try:
        return float(s)
    except:
        return 0.0

def truncate_text(text, max_len=32):
    """Evita corte abrupto de texto em eixos de gráficos."""
    s = str(text).strip()
    if len(s) > max_len:
        return s[:max_len-3] + "..."
    return s

def get_img_as_base64(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

# Dicionário NCM oficial
NCM_DESCRIPTIONS = {
    '85159000': 'Partes de Máquinas/Aparelhos de Soldar (Ultrassom/Laser)',
    '85158090': 'Outras Máquinas e Aparelhos de Soldar (Ultrassom/Fricção)',
    '84798999': 'Outras Máquinas Mecânicas com Função Própria',
    '85371020': 'Controladores Programáveis (CLP/Automação Industrial)',
    '85389010': 'Partes p/ Quadros, Consoles e Painéis Elétricos',
    '40169300': 'Juntas, Gaxetas e Vedações de Borracha Vulcanizada',
    '84819090': 'Partes de Válvulas, Torneiras e Reguladores',
    '84688010': 'Máquinas e Aparelhos para Soldar a Gás',
    '84688090': 'Outros Aparelhos e Máquinas de Soldar Mecânicos'
}

# ==========================================
# 5. DATA ENGINE CONSOLIDATOR
# ==========================================
@st.cache_data(ttl=3600)
def load_all_comex_data():
    log_messages = []
    
    # 1. Carrega todas as planilhas de operações
    ops_patterns = glob.glob("Operações Concorrentes*.xlsx") + glob.glob("Operações Concorrentes*.csv")
    ops_files = sorted([f for f in ops_patterns if not os.path.basename(f).startswith("~$")])
    
    dfs_ops = []
    for f in ops_files:
        try:
            if f.endswith('.csv'):
                temp = pd.read_csv(f, sep=';', decimal=',', encoding='latin1', on_bad_lines='skip')
                if temp.shape[1] < 3:
                    temp = pd.read_csv(f, sep=',', encoding='utf-8', on_bad_lines='skip')
            else:
                temp = pd.read_excel(f, engine='openpyxl')
                
            if temp is not None and not temp.empty:
                temp.columns = temp.columns.str.strip()
                temp['ARQUIVO_ORIGEM'] = os.path.basename(f)
                dfs_ops.append(temp)
                log_messages.append(f"✅ Operações: {os.path.basename(f)} ({len(temp):,} linhas)")
        except Exception as e:
            log_messages.append(f"❌ Erro em {os.path.basename(f)}: {str(e)[:45]}")
            
    df_ops = pd.concat(dfs_ops, ignore_index=True) if dfs_ops else None
    
    # 2. Carrega catálogos de produtos
    cat_patterns = glob.glob("PRODUTOS LOGCOMEX*.xlsx") + glob.glob("PRODUTOS LOGCOMEX*.csv")
    cat_files = sorted([f for f in cat_patterns if not os.path.basename(f).startswith("~$")])
    
    dfs_cat = []
    for f in cat_files:
        try:
            if f.endswith('.csv'):
                temp = pd.read_csv(f, sep=';', encoding='latin1')
            else:
                temp = pd.read_excel(f, engine='openpyxl')
            if temp is not None and not temp.empty:
                temp.columns = temp.columns.str.strip()
                dfs_cat.append(temp)
                log_messages.append(f"✅ Catálogo: {os.path.basename(f)} ({len(temp):,} itens)")
        except Exception as e:
            log_messages.append(f"❌ Erro catálogo {os.path.basename(f)}: {str(e)[:45]}")
            
    df_cat = pd.concat(dfs_cat, ignore_index=True) if dfs_cat else None
    
    if df_ops is None or df_ops.empty:
        return None, None, log_messages
        
    # 3. Processamento de Datas
    if 'ANO/MÊS' in df_ops.columns:
        df_ops['ANO_MES_RAW'] = df_ops['ANO/MÊS'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        df_ops['DATA_REF'] = pd.to_datetime(df_ops['ANO_MES_RAW'], format='%Y%m', errors='coerce')
        df_ops['DATA_REF'] = df_ops['DATA_REF'].fillna(datetime.now())
        df_ops['ANO'] = df_ops['DATA_REF'].dt.year
        df_ops['MES_NUM'] = df_ops['DATA_REF'].dt.month
        df_ops['MES_NOME'] = df_ops['DATA_REF'].dt.strftime('%b')
        df_ops['MES_ANO'] = df_ops['DATA_REF'].dt.strftime('%m/%Y')
        df_ops['PERIODO_ORDEM'] = df_ops['DATA_REF'].dt.strftime('%Y-%m')
        df_ops['TRIMESTRE'] = df_ops['DATA_REF'].apply(lambda d: f"{d.year}-T{((d.month-1)//3)+1}")
    else:
        now = datetime.now()
        df_ops['DATA_REF'] = now
        df_ops['ANO'] = now.year
        df_ops['MES_NUM'] = now.month
        df_ops['MES_NOME'] = now.strftime('%b')
        df_ops['MES_ANO'] = now.strftime('%m/%Y')
        df_ops['PERIODO_ORDEM'] = now.strftime('%Y-%m')
        df_ops['TRIMESTRE'] = f"{now.year}-T1"

    # 4. Normalização de Textos
    text_cols = [
        'PROVÁVEL IMPORTADOR', 'PROVÁVEL EXPORTADOR', 'Descrição produto',
        'PAIS DE ORIGEM', 'MODAL', 'CIDADE DO IMPORTADOR', 'UF IMPORTADOR',
        'Provável Incoterm', 'URF de Entrada'
    ]
    for c in text_cols:
        if c in df_ops.columns:
            df_ops[c] = df_ops[c].fillna('N/A').astype(str).str.upper().str.strip()
        else:
            df_ops[c] = 'N/A'

    # 5. Higienização Numérica
    df_ops['FOB_TOTAL'] = df_ops['VALOR FOB ESTIMADO TOTAL'].apply(clean_num) if 'VALOR FOB ESTIMADO TOTAL' in df_ops.columns else 0.0
    df_ops['CIF_TOTAL'] = df_ops['VALOR CIF TOTAL'].apply(clean_num) if 'VALOR CIF TOTAL' in df_ops.columns else df_ops['FOB_TOTAL']
    df_ops['FRETE_TOTAL'] = df_ops['Valor Frete total'].apply(clean_num) if 'Valor Frete total' in df_ops.columns else 0.0
    df_ops['SEGURO_TOTAL'] = df_ops['Valor Seguro total'].apply(clean_num) if 'Valor Seguro total' in df_ops.columns else 0.0
    df_ops['PESO_LIQUIDO'] = df_ops['Peso líquido'].apply(clean_num) if 'Peso líquido' in df_ops.columns else 0.0
    df_ops['QTD_OPS'] = df_ops['Qtd. de operações estimada'].apply(clean_num) if 'Qtd. de operações estimada' in df_ops.columns else 1.0
    df_ops['QTD_OPS'] = df_ops['QTD_OPS'].replace(0, 1.0)
    
    df_ops['PRECO_KG'] = np.where(df_ops['PESO_LIQUIDO'] > 0, df_ops['FOB_TOTAL'] / df_ops['PESO_LIQUIDO'], 0.0)
    df_ops['TICKET_MEDIO'] = df_ops['FOB_TOTAL'] / df_ops['QTD_OPS']

    # 6. Formatação NCM
    if 'NCM' in df_ops.columns:
        df_ops['NCM_KEY'] = df_ops['NCM'].astype(str).apply(lambda x: ''.join(filter(str.isdigit, str(x).split('.')[0])))
        
        def format_ncm(k):
            k = k.zfill(8)
            if len(k) == 8:
                return f"{k[:4]}.{k[4:6]}.{k[6:]}"
            return k
            
        df_ops['NCM_FORMATADO'] = df_ops['NCM_KEY'].apply(format_ncm)
        df_ops['NCM_DESC'] = df_ops['NCM_KEY'].apply(lambda k: NCM_DESCRIPTIONS.get(k, 'Outros Equipamentos Industriais'))
        df_ops['NCM_LABEL'] = df_ops['NCM_FORMATADO'] + " - " + df_ops['NCM_DESC']
    else:
        df_ops['NCM_KEY'] = '00000000'
        df_ops['NCM_FORMATADO'] = '0000.00.00'
        df_ops['NCM_LABEL'] = 'N/A'

    # 7. Cruzamento com Catálogo WTP / Logcomex
    if df_cat is not None and not df_cat.empty and 'NCM' in df_cat.columns:
        df_cat['NCM_KEY'] = df_cat['NCM'].astype(str).apply(lambda x: ''.join(filter(str.isdigit, str(x).split('.')[0])))
        agg_rules = {}
        for col in ['Modelo', 'Marca', 'Frequência']:
            if col in df_cat.columns:
                df_cat[col] = df_cat[col].fillna('N/D').astype(str).str.upper().str.strip()
                agg_rules[col] = lambda x: ', '.join(sorted(list(set([v for v in x if v not in ['N/A', 'NAN', 'N/D', 'NOT INFORMED']]))))[:200]
                
        if agg_rules:
            df_cat_agg = df_cat.groupby('NCM_KEY').agg(agg_rules).reset_index()
            df_ops = pd.merge(df_ops, df_cat_agg, on='NCM_KEY', how='left')

    for col in ['Modelo', 'Marca', 'Frequência']:
        if col not in df_ops.columns:
            df_ops[col] = 'N/D'
        else:
            df_ops[col] = df_ops[col].fillna('N/D')

    return df_ops, df_cat, log_messages

@st.cache_data(ttl=1800)
def fetch_usd_exchange_rate():
    try:
        import yfinance as yf
        ticker = yf.Ticker("BRL=X")
        hist = ticker.history(period="2d", timeout=4)
        if len(hist) >= 2:
            close = float(hist['Close'].iloc[-1])
            prev = float(hist['Close'].iloc[-2])
            delta = close - prev
            pct = (delta / prev) * 100
            return close, delta, pct
        elif len(hist) == 1:
            return float(hist['Close'].iloc[-1]), 0.0, 0.0
    except:
        pass
    return 5.60, 0.0, 0.0

# CARREGAMENTO DA BASE CONSOLIDADA
df_ops_raw, df_cat_raw, load_logs = load_all_comex_data()
if df_ops_raw is None or df_ops_raw.empty:
    st.error("❌ Nenhuma planilha de dados encontrada em `C:\\wtpcomex`. Verifique os arquivos.")
    st.stop()

# ==========================================
# 6. SIDEBAR CONTROLS & MODERN FILTERS
# ==========================================
with st.sidebar:
    logo_b64 = get_img_as_base64("logo.png")
    if logo_b64:
        st.markdown(
            f"<div style='text-align: center; margin-bottom: 15px;'>"
            f"<img src='data:image/png;base64,{logo_b64}' width='180' style='filter: drop-shadow(0 4px 10px rgba(16, 185, 129, 0.3));'>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='text-align: center; margin-bottom: 15px;'>"
            f"<h3 style='color:{THEME['accent_primary']}; margin:0;'>WTP COMEX</h3>"
            f"<div style='font-size:0.75rem; color:{THEME['text_muted']};'>INTELLIGENCE TERMINAL</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown(
        f"<div style='text-align: center; margin-bottom: 15px;'>"
        f"<span class='status-pill'><span class='status-dot'></span><span>{len(df_ops_raw):,} OPERAÇÕES ANALISADAS</span></span>"
        f"</div>",
        unsafe_allow_html=True
    )

    # 1. MOEDA DE VISUALIZAÇÃO (USD vs BRL)
    st.markdown(f"<div style='font-size:0.75rem; font-weight:700; color:{THEME['accent_primary']}; margin-bottom:4px;'>💵 MOEDA DE VISUALIZAÇÃO</div>", unsafe_allow_html=True)
    moeda_select = st.radio("Moeda:", ["USD ($)", "BRL (R$)"], horizontal=True, label_visibility="collapsed")
    is_brl = (moeda_select == "BRL (R$)")
    moeda_code = "BRL" if is_brl else "USD"

    # 2. BUSCA RÁPIDA GLOBAL
    st.markdown(f"<div style='font-size:0.75rem; font-weight:700; color:{THEME['accent_primary']}; margin-top:12px; margin-bottom:4px;'>🔍 BUSCA RÁPIDA UNIVERSAL</div>", unsafe_allow_html=True)
    search_query = st.text_input("Busca Universal", placeholder="Empresa, NCM, Produto, Porto, Cidade...", label_visibility="collapsed")

    # 3. FILTRO DE DATAS COM PRESETS
    st.markdown(f"<div style='font-size:0.75rem; font-weight:700; color:{THEME['accent_primary']}; margin-top:14px; margin-bottom:4px;'>📅 PERÍODO TEMPORAL</div>", unsafe_allow_html=True)
    preset_periodo = st.radio(
        "Período:",
        ["Todo o Histórico", "Ano 2026 (YTD)", "Últimos 12 Meses", "Últimos 24 Meses", "Personalizado"],
        index=0, label_visibility="collapsed"
    )
    
    min_date = df_ops_raw['DATA_REF'].min().date()
    max_date = df_ops_raw['DATA_REF'].max().date()
    
    if preset_periodo == "Todo o Histórico":
        sel_dates = (min_date, max_date)
    elif preset_periodo == "Ano 2026 (YTD)":
        start_2026 = max(min_date, datetime(2026, 1, 1).date())
        sel_dates = (start_2026, max_date)
    elif preset_periodo == "Últimos 12 Meses":
        start_12m = max(min_date, max_date - timedelta(days=365))
        sel_dates = (start_12m, max_date)
    elif preset_periodo == "Últimos 24 Meses":
        start_24m = max(min_date, max_date - timedelta(days=730))
        sel_dates = (start_24m, max_date)
    else:
        sel_dates = st.date_input("Intervalo de Datas:", value=(min_date, max_date), min_value=min_date, max_value=max_date, format="DD/MM/YYYY")

    # FILTRAGEM
    df_filtered = df_ops_raw.copy()
    
    if isinstance(sel_dates, (tuple, list)) and len(sel_dates) == 2:
        df_filtered = df_filtered[
            (df_filtered['DATA_REF'].dt.date >= sel_dates[0]) & 
            (df_filtered['DATA_REF'].dt.date <= sel_dates[1])
        ]

    # 4. EMPRESAS (IMPORTADORES & EXPORTADORES)
    with st.expander("🏢 Concorrentes & Parceiros", expanded=True):
        imp_counts = df_filtered['PROVÁVEL IMPORTADOR'].value_counts()
        imp_options = [f"{idx} ({count})" for idx, count in imp_counts.items() if idx not in ['N/A', '']]
        imp_map = {f"{idx} ({count})": idx for idx, count in imp_counts.items()}
        
        sel_imp_labels = st.multiselect("Importador (Brasil):", imp_options, placeholder="Selecione importadores...")
        sel_imp = [imp_map[x] for x in sel_imp_labels]
        if sel_imp:
            df_filtered = df_filtered[df_filtered['PROVÁVEL IMPORTADOR'].isin(sel_imp)]

        exp_counts = df_filtered['PROVÁVEL EXPORTADOR'].value_counts()
        exp_options = [f"{idx} ({count})" for idx, count in exp_counts.items() if idx not in ['N/A', '', 'EXTERIOR']]
        exp_map = {f"{idx} ({count})": idx for idx, count in exp_counts.items()}
        
        sel_exp_labels = st.multiselect("Fabricante / Exportador:", exp_options, placeholder="Selecione fabricantes...")
        sel_exp = [exp_map[x] for x in sel_exp_labels]
        if sel_exp:
            df_filtered = df_filtered[df_filtered['PROVÁVEL EXPORTADOR'].isin(sel_exp)]

    # 5. NCM & PRODUTOS
    with st.expander("📦 Classificação Fiscal (NCM) & Specs", expanded=False):
        ncm_counts = df_filtered['NCM_LABEL'].value_counts()
        sel_ncms = st.multiselect("NCM:", options=ncm_counts.index.tolist(), placeholder="Filtrar NCMs...")
        if sel_ncms:
            df_filtered = df_filtered[df_filtered['NCM_LABEL'].isin(sel_ncms)]

        unique_brands = sorted([b for b in df_filtered['Marca'].dropna().unique() if b not in ['N/D', 'NAN', '']])
        if unique_brands:
            sel_brands = st.multiselect("Marca Técnica:", unique_brands, placeholder="Marcas...")
            if sel_brands:
                df_filtered = df_filtered[df_filtered['Marca'].isin(sel_brands)]

    # 6. LOGÍSTICA & ADUANA
    with st.expander("🚢 Logística, Modais & Portos", expanded=False):
        paises = sorted([p for p in df_filtered['PAIS DE ORIGEM'].unique() if p not in ['N/A', '']])
        sel_paises = st.multiselect("País de Origem:", paises, placeholder="Países...")
        if sel_paises:
            df_filtered = df_filtered[df_filtered['PAIS DE ORIGEM'].isin(sel_paises)]

        modais = sorted([m for m in df_filtered['MODAL'].unique() if m not in ['N/A', '']])
        sel_modais = st.multiselect("Modal de Transporte:", modais, placeholder="Modais...")
        if sel_modais:
            df_filtered = df_filtered[df_filtered['MODAL'].isin(sel_modais)]

        urfs = sorted([u for u in df_filtered['URF de Entrada'].unique() if u not in ['N/A', '']])
        sel_urfs = st.multiselect("URF de Entrada (Porto/Aeroporto):", urfs, placeholder="Recintos alfandegados...")
        if sel_urfs:
            df_filtered = df_filtered[df_filtered['URF de Entrada'].isin(sel_urfs)]

        cidades = sorted([c for c in df_filtered['CIDADE DO IMPORTADOR'].unique() if c not in ['N/A', '']])
        sel_cidades = st.multiselect("Cidade do Comprador:", cidades, placeholder="Cidades...")
        if sel_cidades:
            df_filtered = df_filtered[df_filtered['CIDADE DO IMPORTADOR'].isin(sel_cidades)]

    # 7. BUSCA UNIVERSAL APLICADA
    if search_query:
        q_up = search_query.upper().strip()
        mask_search = (
            df_filtered['PROVÁVEL IMPORTADOR'].str.contains(q_up, na=False) |
            df_filtered['PROVÁVEL EXPORTADOR'].str.contains(q_up, na=False) |
            df_filtered['NCM_LABEL'].str.contains(q_up, na=False) |
            df_filtered['Descrição produto'].str.contains(q_up, na=False) |
            df_filtered['CIDADE DO IMPORTADOR'].str.contains(q_up, na=False) |
            df_filtered['PAIS DE ORIGEM'].str.contains(q_up, na=False) |
            df_filtered['URF de Entrada'].str.contains(q_up, na=False)
        )
        df_filtered = df_filtered[mask_search]

    # 8. GOVERNANÇA & DESDUPLICAÇÃO
    with st.expander("⚙️ Governança & Opções", expanded=False):
        dedup_toggle = st.toggle("⚡ Desduplicar Transações Sobrepostas", value=True, help="Remove registros com mesma chave entre downloads de 36 meses.")
        hide_generics = st.toggle("🚫 Ocultar Genéricos ('EXTERIOR', 'N/A')", value=False)
        
        if dedup_toggle:
            sub_k = ['ANO_MES_RAW', 'PROVÁVEL IMPORTADOR', 'PROVÁVEL EXPORTADOR', 'NCM_KEY', 'FOB_TOTAL', 'PESO_LIQUIDO']
            sub_k = [c for c in sub_k if c in df_filtered.columns]
            df_filtered = df_filtered.drop_duplicates(subset=sub_k)
            
        if hide_generics:
            df_filtered = df_filtered[~df_filtered['PROVÁVEL IMPORTADOR'].isin(['N/A', 'NAN', 'UNDEFINED', ''])]
            df_filtered = df_filtered[~df_filtered['PROVÁVEL EXPORTADOR'].isin(['N/A', 'NAN', 'EXTERIOR', ''])]

        st.caption("Logs do Motor de Dados:")
        for l in load_logs:
            st.caption(l)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔒 Sair do Terminal", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

# ==========================================
# 7. CÂMBIO & CÁLCULO DE MÉTRICAS MULTI-MOEDA
# ==========================================
usd_rate, usd_diff, usd_pct = fetch_usd_exchange_rate()
fx_rate = usd_rate if is_brl else 1.0

# Multiplica valores pelo câmbio se visualizado em BRL
v_fob_raw = df_filtered['FOB_TOTAL'].sum()
v_cif_raw = df_filtered['CIF_TOTAL'].sum()
v_frete_raw = df_filtered['FRETE_TOTAL'].sum()

v_fob = v_fob_raw * fx_rate
v_cif = v_cif_raw * fx_rate
v_frete = v_frete_raw * fx_rate
v_peso = df_filtered['PESO_LIQUIDO'].sum()
v_ops = df_filtered['QTD_OPS'].sum()
v_linhas = len(df_filtered)

v_ticket = v_fob / v_ops if v_ops > 0 else 0.0
v_preco_kg = v_fob / v_peso if v_peso > 0 else 0.0
v_share_frete = (v_frete / v_fob * 100) if v_fob > 0 else 0.0

# Deltas reais homólogos
def compute_real_deltas(df_curr, df_full):
    if df_curr.empty or df_full.empty:
        return None, None
    c_min = df_curr['DATA_REF'].min()
    c_max = df_curr['DATA_REF'].max()
    dur = (c_max - c_min).days + 1
    
    p_max = c_min - timedelta(days=1)
    p_min = p_max - timedelta(days=dur)
    
    df_p = df_full[(df_full['DATA_REF'] >= p_min) & (df_full['DATA_REF'] <= p_max)]
    if df_p.empty:
        return None, None
        
    p_fob = df_p['FOB_TOTAL'].sum()
    p_peso = df_p['PESO_LIQUIDO'].sum()
    
    d_fob = ((df_curr['FOB_TOTAL'].sum() - p_fob) / p_fob * 100) if p_fob > 0 else 0.0
    d_peso = ((df_curr['PESO_LIQUIDO'].sum() - p_peso) / p_peso * 100) if p_peso > 0 else 0.0
    return d_fob, d_peso

delta_fob, delta_peso = compute_real_deltas(df_filtered, df_ops_raw)

# ==========================================
# 8. TOP HERO BANNER & TICKER
# ==========================================
h_col1, h_col2 = st.columns([3.2, 1.2])

with h_col1:
    data_ini = df_filtered['DATA_REF'].min().strftime('%d/%m/%Y')
    data_fim = df_filtered['DATA_REF'].max().strftime('%d/%m/%Y')
    st.markdown(
        f"<div class='hero-container'>"
        f"<div>"
        f"<h1 class='hero-title'>WTP <span style='color:{THEME['accent_primary']};'>COMEX</span> INTELLIGENCE</h1>"
        f"<div class='hero-subtitle'>TERMINAL ESPECIALISTA EM COMÉRCIO EXTERIOR & MAQUINÁRIO INDUSTRIAL • v85.0</div>"
        f"</div>"
        f"<div style='text-align: right;'>"
        f"<span class='status-pill'><span class='status-dot'></span><span>{v_linhas:,} OPERAÇÕES • BASE CONSOLIDADA</span></span>"
        f"<div style='font-size: 0.8rem; color: {THEME['text_muted']}; margin-top: 5px; font-family: monospace;'>"
        f"{data_ini} ➔ {data_fim}"
        f"</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True
    )

with h_col2:
    if usd_rate > 0:
        clr_fx = THEME['accent_mint'] if usd_diff >= 0 else THEME['danger']
        sym_fx = "▲" if usd_diff >= 0 else "▼"
        st.markdown(
            f"<div class='glass-card' style='text-align: center; padding: 1.15rem 1rem; margin-bottom: 1.2rem;'>"
            f"<div style='font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: {THEME['text_muted']};'>USD/BRL COMERCIAL</div>"
            f"<div style='font-family: monospace; font-size: 1.95rem; font-weight: 700; color: #FFFFFF; margin: 3px 0;'>R$ {usd_rate:.4f}</div>"
            f"<div style='font-size: 0.82rem; font-weight: 600; color: {clr_fx};'>{sym_fx} {abs(usd_diff):.4f} ({usd_pct:+.2f}%)</div>"
            f"</div>",
            unsafe_allow_html=True
        )

if df_filtered.empty:
    st.warning("⚠️ Nenhuma transação encontrada com os filtros informados. Redefina os filtros na barra lateral.")
    st.stop()

# ==========================================
# 9. KPI HERO ROW (PERFEITAMENTE FORMATADO)
# ==========================================
def render_kpi_card(label, main_val, subtext, delta=None):
    delta_markup = ""
    if delta is not None:
        tag_cls = "tag-pos" if delta >= 0 else "tag-neg"
        sym = "▲" if delta >= 0 else "▼"
        delta_markup = f"<span class='kpi-tag {tag_cls}'>{sym} {abs(delta):.1f}%</span>"
        
    return (
        f"<div class='kpi-card'>"
        f"<div class='kpi-card-top'><span class='kpi-label'>{label}</span>{delta_markup}</div>"
        f"<div class='kpi-value'>{main_val}</div>"
        f"<div class='kpi-footer'><span>{subtext}</span></div>"
        f"</div>"
    )

k_r1_1, k_r1_2, k_r1_3 = st.columns(3)
with k_r1_1:
    st.markdown(render_kpi_card(f"FOB TOTAL ({moeda_code})", fmt_moeda_br(v_fob, compact=True, moeda=moeda_code), f"Total: {fmt_moeda_br(v_fob, moeda=moeda_code)}", delta_fob), unsafe_allow_html=True)
with k_r1_2:
    st.markdown(render_kpi_card(f"CIF ESTIMADO ({moeda_code})", fmt_moeda_br(v_cif, compact=True, moeda=moeda_code), f"Total: {fmt_moeda_br(v_cif, moeda=moeda_code)}"), unsafe_allow_html=True)
with k_r1_3:
    st.markdown(render_kpi_card("PESO LÍQUIDO", fmt_peso_br(v_peso, compact=True), f"Total: {fmt_peso_br(v_peso)}", delta_peso), unsafe_allow_html=True)

st.markdown("<div style='margin-top: 0.65rem;'></div>", unsafe_allow_html=True)

k_r2_1, k_r2_2, k_r2_3 = st.columns(3)
with k_r2_1:
    st.markdown(render_kpi_card("OPERAÇÕES ESTIMADAS", fmt_int_br(v_ops), f"{v_linhas:,} transações registradas"), unsafe_allow_html=True)
with k_r2_2:
    st.markdown(render_kpi_card("TICKET MÉDIO", fmt_moeda_br(v_ticket, moeda=moeda_code), "Média Financeira por Operação"), unsafe_allow_html=True)
with k_r2_3:
    st.markdown(render_kpi_card(f"PREÇO MÉDIO / KG ({moeda_code})", fmt_moeda_br(v_preco_kg, moeda=moeda_code), f"Frete: {fmt_pct_br(v_share_frete)} do FOB"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Layout e tema padrão dos gráficos Plotly
common_layout = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color=THEME['text_muted'], family="Plus Jakarta Sans", size=12),
    margin=dict(l=15, r=15, t=35, b=20),
    xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color=THEME['text_muted'])),
    yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', zeroline=False, tickfont=dict(color=THEME['text_muted']), automargin=True),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="rgba(17, 24, 39, 0.95)", bordercolor=THEME['accent_primary'], font_size=12, font_family="Plus Jakarta Sans")
)
chart_config = {'displayModeBar': False, 'responsive': True}

# ==========================================
# 10. MÓDULOS ESPECIALIZADOS EM ABAS
# ==========================================
tabs = st.tabs([
    "📊 Visão Executiva",
    "🏢 Concorrência & Pareto",
    "🎯 Análise NCM & Preços",
    "🚢 Logística & Aduana",
    "🌍 Rotas & Globo 3D",
    "🔄 Cadeia & Fluxos",
    "📋 Ledger & Exportação",
    "📦 Catálogo Técnico"
])

# ----------------------------------------------------
# TAB 1: VISÃO EXECUTIVA
# ----------------------------------------------------
with tabs[0]:
    c_met1, c_met2 = st.columns([2.3, 1])
    
    with c_met1:
        st.markdown(f"<div class='glass-card'><h4>📈 Evolução Mensal & Tendência de Importações</h4></div>", unsafe_allow_html=True)
        
        df_monthly = df_filtered.groupby(['PERIODO_ORDEM', 'DATA_REF']).agg({
            'FOB_TOTAL': 'sum',
            'PESO_LIQUIDO': 'sum',
            'QTD_OPS': 'sum',
            'FRETE_TOTAL': 'sum'
        }).reset_index().sort_values('DATA_REF')
        
        if is_brl:
            df_monthly['VAL_SHOW'] = df_monthly['FOB_TOTAL'] * fx_rate
            label_val = "Valor FOB (R$)"
        else:
            df_monthly['VAL_SHOW'] = df_monthly['FOB_TOTAL']
            label_val = "Valor FOB ($)"
            
        fig_evol = px.area(
            df_monthly, x='DATA_REF', y='VAL_SHOW',
            labels={'DATA_REF': 'Data', 'VAL_SHOW': label_val},
            line_shape='spline'
        )
        fig_evol.update_traces(
            line_color=THEME['accent_primary'],
            fillcolor="rgba(16, 185, 129, 0.15)",
            hovertemplate="<b>%{x|%b/%Y}</b><br>Volume: " + ("R$ %{y:,.2f}" if is_brl else "$ %{y:,.2f}") + "<extra></extra>"
        )
        
        # Média móvel
        if len(df_monthly) >= 3:
            df_monthly['MA3'] = df_monthly['VAL_SHOW'].rolling(window=3, min_periods=1).mean()
            fig_evol.add_trace(go.Scatter(
                x=df_monthly['DATA_REF'], y=df_monthly['MA3'],
                mode='lines', name='Média Móvel (3M)',
                line=dict(color=THEME['accent_mint'], width=2, dash='dot')
            ))
            
        fig_evol.update_layout(common_layout, height=360)
        st.plotly_chart(fig_evol, use_container_width=True, config=chart_config)
        
    with c_met2:
        st.markdown(f"<div class='glass-card'><h4>📦 Modais de Transporte</h4></div>", unsafe_allow_html=True)
        df_modal = df_filtered.groupby('MODAL')['FOB_TOTAL'].sum().reset_index()
        fig_modal = px.pie(
            df_modal, values='FOB_TOTAL', names='MODAL', hole=0.62,
            color_discrete_sequence=[THEME['accent_primary'], THEME['accent_mint'], THEME['accent_cyan'], THEME['accent_blue']]
        )
        fig_modal.update_traces(
            textposition='inside', textinfo='percent+label',
            hovertemplate="<b>%{label}</b><br>Participação: %{percent}<extra></extra>"
        )
        fig_modal.update_layout(common_layout, showlegend=False, height=360)
        st.plotly_chart(fig_modal, use_container_width=True, config=chart_config)

    # SEÇÃO INCOTERMS E ESTADOS
    row2_1, row2_2 = st.columns(2)
    with row2_1:
        st.markdown(f"<div class='glass-card'><h4>📜 Incoterms Mais Praticados</h4></div>", unsafe_allow_html=True)
        df_inco = df_filtered[df_filtered['Provável Incoterm'] != 'N/A'].groupby('Provável Incoterm')['FOB_TOTAL'].sum().reset_index()
        if not df_inco.empty:
            df_inco['VAL_SHOW'] = df_inco['FOB_TOTAL'] * fx_rate
            fig_inco = px.bar(
                df_inco.sort_values('VAL_SHOW', ascending=True),
                x='VAL_SHOW', y='Provável Incoterm', orientation='h',
                color='VAL_SHOW', color_continuous_scale=[[0, '#064E3B'], [1, THEME['accent_primary']]]
            )
            max_inco = df_inco['VAL_SHOW'].max() if not df_inco.empty else 1.0
            fig_inco.update_xaxes(range=[0, max_inco * 1.32], automargin=True)
            fig_inco.update_traces(
                texttemplate=('%{x:,.2s}'), textposition='outside', cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>Total: " + ("R$ %{x:,.2f}" if is_brl else "$ %{x:,.2f}") + "<extra></extra>"
            )
            fig_inco.update_layout(common_layout, height=300, coloraxis_showscale=False, margin=dict(l=15, r=85, t=35, b=20))
            st.plotly_chart(fig_inco, use_container_width=True, config=chart_config)
            
    with row2_2:
        st.markdown(f"<div class='glass-card'><h4>📍 Estados de Destino no Brasil (UF)</h4></div>", unsafe_allow_html=True)
        df_uf = df_filtered[~df_filtered['UF IMPORTADOR'].isin(['N/A', '', 'EXTERIOR'])].groupby('UF IMPORTADOR')['FOB_TOTAL'].sum().nlargest(10).reset_index()
        if not df_uf.empty:
            df_uf['VAL_SHOW'] = df_uf['FOB_TOTAL'] * fx_rate
            fig_uf = px.bar(
                df_uf.sort_values('VAL_SHOW', ascending=True),
                x='VAL_SHOW', y='UF IMPORTADOR', orientation='h',
                color_discrete_sequence=[THEME['accent_primary']]
            )
            max_uf = df_uf['VAL_SHOW'].max() if not df_uf.empty else 1.0
            fig_uf.update_xaxes(range=[0, max_uf * 1.32], automargin=True)
            fig_uf.update_traces(
                texttemplate=('%{x:,.2s}'), textposition='outside', cliponaxis=False,
                hovertemplate="<b>UF: %{y}</b><br>Total: " + ("R$ %{x:,.2f}" if is_brl else "$ %{x:,.2f}") + "<extra></extra>"
            )
            fig_uf.update_layout(common_layout, height=300, margin=dict(l=15, r=85, t=35, b=20))
            st.plotly_chart(fig_uf, use_container_width=True, config=chart_config)

# ----------------------------------------------------
# TAB 2: CONCORRÊNCIA, MARKET SHARE & PARETO
# ----------------------------------------------------
with tabs[1]:
    t2_ctrl1, t2_ctrl2 = st.columns([2, 1])
    with t2_ctrl1:
        top_n = st.select_slider("Quantidade de Players no Ranking:", options=[10, 20, 30, 50, 100], value=20)
    with t2_ctrl2:
        view_type_rank = st.radio("Visualização:", ["Gráficos de Barras", "Treemap Hierárquico", "Curva ABC (Pareto)"], horizontal=True)

    df_imp_valid = df_filtered[~df_filtered['PROVÁVEL IMPORTADOR'].isin(['N/A', 'NAN', '', 'EXTERIOR']) & (df_filtered['PROVÁVEL IMPORTADOR'].str.len() > 2)]
    df_exp_valid = df_filtered[~df_filtered['PROVÁVEL EXPORTADOR'].isin(['N/A', 'NAN', '', 'EXTERIOR']) & (df_filtered['PROVÁVEL EXPORTADOR'].str.len() > 2)]
    
    if view_type_rank == "Curva ABC (Pareto)":
        st.markdown(f"<div class='glass-card'><h4>📊 Curva ABC / Pareto de Concentração de Mercado (Top Importadores)</h4></div>", unsafe_allow_html=True)
        top_pareto = df_imp_valid.groupby('PROVÁVEL IMPORTADOR')['FOB_TOTAL'].sum().nlargest(top_n).reset_index()
        top_pareto['VAL_SHOW'] = top_pareto['FOB_TOTAL'] * fx_rate
        total_p = top_pareto['VAL_SHOW'].sum()
        top_pareto['PCT_ACUM'] = (top_pareto['VAL_SHOW'].cumsum() / total_p) * 100
        top_pareto['NOME_CURTO'] = top_pareto['PROVÁVEL IMPORTADOR'].apply(lambda x: truncate_text(x, 28))
        
        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Bar(
            x=top_pareto['NOME_CURTO'], y=top_pareto['VAL_SHOW'],
            name='Volume Financeiro', marker_color=THEME['accent_primary'],
            hovertemplate="<b>%{x}</b><br>Valor: " + ("R$ %{y:,.2f}" if is_brl else "$ %{y:,.2f}") + "<extra></extra>"
        ))
        fig_pareto.add_trace(go.Scatter(
            x=top_pareto['NOME_CURTO'], y=top_pareto['PCT_ACUM'],
            name='% Acumulado', yaxis='y2', mode='lines+markers',
            line=dict(color=THEME['warning'], width=2.5),
            marker=dict(size=6, color=THEME['warning']),
            hovertemplate="<b>%{x}</b><br>Concentração Acumulada: %{y:.1f}%<extra></extra>"
        ))
        
        layout_p = common_layout.copy()
        layout_p['yaxis2'] = dict(
            overlaying='y', side='right', range=[0, 105], showgrid=False,
            tickfont=dict(color=THEME['warning']), ticksuffix="%"
        )
        layout_p['height'] = 450
        fig_pareto.update_layout(layout_p)
        st.plotly_chart(fig_pareto, use_container_width=True, config=chart_config)

    elif view_type_rank == "Treemap Hierárquico":
        st.markdown(f"<div class='glass-card'><h4>🌳 Treemap Hierárquico: País ➔ Fabricante ➔ Importador</h4></div>", unsafe_allow_html=True)
        df_tree = df_filtered[
            ~df_filtered['PAIS DE ORIGEM'].isin(['N/A', '']) &
            ~df_filtered['PROVÁVEL EXPORTADOR'].isin(['N/A', 'EXTERIOR', '']) &
            ~df_filtered['PROVÁVEL IMPORTADOR'].isin(['N/A', ''])
        ].copy()
        df_tree['VAL_SHOW'] = df_tree['FOB_TOTAL'] * fx_rate
        
        top_imp_tree = df_tree.groupby('PROVÁVEL IMPORTADOR')['VAL_SHOW'].sum().nlargest(25).index
        df_tree = df_tree[df_tree['PROVÁVEL IMPORTADOR'].isin(top_imp_tree)]
        
        if not df_tree.empty:
            fig_tree = px.treemap(
                df_tree, path=['PAIS DE ORIGEM', 'PROVÁVEL EXPORTADOR', 'PROVÁVEL IMPORTADOR'],
                values='VAL_SHOW', color='VAL_SHOW',
                color_continuous_scale=[[0, '#064E3B'], [1, THEME['accent_primary']]]
            )
            fig_tree.update_layout(common_layout, height=520, coloraxis_showscale=False)
            fig_tree.update_traces(hovertemplate="<b>%{label}</b><br>Total: " + ("R$ %{value:,.2f}" if is_brl else "$ %{value:,.2f}") + "<extra></extra>")
            st.plotly_chart(fig_tree, use_container_width=True, config=chart_config)
            
    else:
        # BARRAS HORIZONTAIS COM TRUNCAMENTO SEGURO
        r_c1, r_c2 = st.columns(2)
        with r_c1:
            st.markdown(f"<div class='glass-card'><h4>🏢 Top {top_n} Importadores (Brasil)</h4></div>", unsafe_allow_html=True)
            top_imp = df_imp_valid.groupby('PROVÁVEL IMPORTADOR').agg({'FOB_TOTAL': 'sum'}).nlargest(top_n, 'FOB_TOTAL').sort_values('FOB_TOTAL', ascending=True).reset_index()
            top_imp['VAL_SHOW'] = top_imp['FOB_TOTAL'] * fx_rate
            top_imp['LABEL_EIXO'] = top_imp['PROVÁVEL IMPORTADOR'].apply(lambda x: truncate_text(x, 30))
            
            fig_imp = px.bar(
                top_imp, y='LABEL_EIXO', x='VAL_SHOW', orientation='h',
                color_discrete_sequence=[THEME['accent_primary']],
                custom_data=['PROVÁVEL IMPORTADOR']
            )
            max_imp_val = top_imp['VAL_SHOW'].max() if not top_imp.empty else 1.0
            fig_imp.update_xaxes(range=[0, max_imp_val * 1.32], automargin=True)
            fig_imp.update_traces(
                texttemplate=('%{x:,.2s}'), textposition='outside', cliponaxis=False,
                hovertemplate="<b>%{customdata[0]}</b><br>Valor: " + ("R$ %{x:,.2f}" if is_brl else "$ %{x:,.2f}") + "<extra></extra>"
            )
            dyn_h = max(420, len(top_imp) * 30)
            fig_imp.update_layout(common_layout, height=dyn_h, yaxis=dict(title="", automargin=True), margin=dict(l=15, r=85, t=35, b=20))
            st.plotly_chart(fig_imp, use_container_width=True, config=chart_config)
            
        with r_c2:
            st.markdown(f"<div class='glass-card'><h4>🏭 Top {top_n} Fabricantes / Exportadores</h4></div>", unsafe_allow_html=True)
            top_exp = df_exp_valid.groupby('PROVÁVEL EXPORTADOR').agg({'FOB_TOTAL': 'sum'}).nlargest(top_n, 'FOB_TOTAL').sort_values('FOB_TOTAL', ascending=True).reset_index()
            top_exp['VAL_SHOW'] = top_exp['FOB_TOTAL'] * fx_rate
            top_exp['LABEL_EIXO'] = top_exp['PROVÁVEL EXPORTADOR'].apply(lambda x: truncate_text(x, 30))
            
            fig_exp = px.bar(
                top_exp, y='LABEL_EIXO', x='VAL_SHOW', orientation='h',
                color_discrete_sequence=[THEME['accent_mint']],
                custom_data=['PROVÁVEL EXPORTADOR']
            )
            max_exp_val = top_exp['VAL_SHOW'].max() if not top_exp.empty else 1.0
            fig_exp.update_xaxes(range=[0, max_exp_val * 1.32], automargin=True)
            fig_exp.update_traces(
                texttemplate=('%{x:,.2s}'), textposition='outside', cliponaxis=False,
                hovertemplate="<b>%{customdata[0]}</b><br>Valor: " + ("R$ %{x:,.2f}" if is_brl else "$ %{x:,.2f}") + "<extra></extra>"
            )
            dyn_h = max(420, len(top_exp) * 30)
            fig_exp.update_layout(common_layout, height=dyn_h, yaxis=dict(title="", automargin=True), margin=dict(l=15, r=85, t=35, b=20))
            st.plotly_chart(fig_exp, use_container_width=True, config=chart_config)

    # RAIO-X DE CONCORRENTE
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='glass-card'><h4>🔍 Raio-X Detalhado de Concorrente / Importador Específico</h4></div>", unsafe_allow_html=True)
    
    unique_players = sorted(df_imp_valid['PROVÁVEL IMPORTADOR'].unique())
    selected_player = st.selectbox("Escolha uma empresa para inspecionar parceiros, portos e ticket médio:", ["-- Selecione uma Empresa --"] + unique_players)
    
    if selected_player != "-- Selecione uma Empresa --":
        df_p = df_filtered[df_filtered['PROVÁVEL IMPORTADOR'] == selected_player]
        
        pf_fob = df_p['FOB_TOTAL'].sum() * fx_rate
        pf_peso = df_p['PESO_LIQUIDO'].sum()
        pf_ops = df_p['QTD_OPS'].sum()
        pf_pkg = pf_fob / pf_peso if pf_peso > 0 else 0.0
        
        pr1, pr2, pr3, pr4 = st.columns(4)
        with pr1: st.metric("FOB Total", fmt_moeda_br(pf_fob, moeda=moeda_code))
        with pr2: st.metric("Peso Líquido", fmt_peso_br(pf_peso))
        with pr3: st.metric("Qtd Operações", fmt_int_br(pf_ops))
        with pr4: st.metric(f"Preço / Kg ({moeda_code})", fmt_moeda_br(pf_pkg, moeda=moeda_code))
        
        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            st.markdown("**Principais Fabricantes Fornecedores:**")
            p_exps = df_p.groupby('PROVÁVEL EXPORTADOR')['FOB_TOTAL'].sum().reset_index().sort_values('FOB_TOTAL', ascending=False)
            p_exps['Valor FOB'] = (p_exps['FOB_TOTAL'] * fx_rate).apply(lambda v: fmt_moeda_br(v, moeda=moeda_code))
            st.dataframe(p_exps[['PROVÁVEL EXPORTADOR', 'Valor FOB']], use_container_width=True, height=220)
            
        with sub_c2:
            st.markdown("**NCMs Mais Importados:**")
            p_ncms = df_p.groupby('NCM_LABEL')['FOB_TOTAL'].sum().reset_index().sort_values('FOB_TOTAL', ascending=False)
            p_ncms['Valor FOB'] = (p_ncms['FOB_TOTAL'] * fx_rate).apply(lambda v: fmt_moeda_br(v, moeda=moeda_code))
            st.dataframe(p_ncms[['NCM_LABEL', 'Valor FOB']], use_container_width=True, height=220)

# ----------------------------------------------------
# TAB 3: ANÁLISE NCM & PREÇOS
# ----------------------------------------------------
with tabs[2]:
    st.markdown(f"<div class='glass-card'><h4>🎯 Dispersão de Preço Médio ($/Kg) vs. Volume (Margens e Outliers de Mercado)</h4></div>", unsafe_allow_html=True)
    
    df_scat = df_filtered[
        (df_filtered['PESO_LIQUIDO'] > 0) & 
        (df_filtered['PRECO_KG'] > 0) & 
        (df_filtered['PRECO_KG'] < 10000) &
        (~df_filtered['PROVÁVEL IMPORTADOR'].isin(['N/A', 'NAN', '']))
    ].copy()
    
    if not df_scat.empty:
        df_scat_agg = df_scat.groupby(['PROVÁVEL IMPORTADOR', 'NCM_LABEL']).agg({
            'FOB_TOTAL': 'sum', 'PESO_LIQUIDO': 'sum', 'QTD_OPS': 'sum'
        }).reset_index()
        df_scat_agg['PRECO_KG'] = (df_scat_agg['FOB_TOTAL'] * fx_rate) / df_scat_agg['PESO_LIQUIDO']
        df_scat_agg['FOB_SHOW'] = df_scat_agg['FOB_TOTAL'] * fx_rate
        
        df_scat_agg['FOB_STR'] = df_scat_agg['FOB_SHOW'].apply(lambda v: fmt_moeda_br(v, moeda=moeda_code))
        df_scat_agg['PRECO_STR'] = df_scat_agg['PRECO_KG'].apply(lambda v: fmt_moeda_br(v, moeda=moeda_code))
        df_scat_agg['PESO_STR'] = df_scat_agg['PESO_LIQUIDO'].apply(fmt_peso_br)
        
        fig_scatter = px.scatter(
            df_scat_agg,
            x='PESO_LIQUIDO', y='PRECO_KG',
            size='FOB_SHOW', color='NCM_LABEL',
            hover_name='PROVÁVEL IMPORTADOR',
            custom_data=['NCM_LABEL', 'PESO_STR', 'PRECO_STR', 'FOB_STR'],
            labels={'PESO_LIQUIDO': 'Volume (Kg)', 'PRECO_KG': f'Preço Médio ({moeda_code}/Kg)'},
            log_x=True, size_max=36
        )
        fig_scatter.update_traces(
            hovertemplate=(
                "<b>%{hovertext}</b><br><br>"
                "📦 <b>NCM:</b> %{customdata[0]}<br>"
                "⚖️ <b>Peso:</b> %{customdata[1]}<br>"
                "💵 <b>Preço Médio:</b> %{customdata[2]} / kg<br>"
                "💰 <b>FOB Total:</b> %{customdata[3]}<extra></extra>"
            )
        )
        layout_scat = common_layout.copy()
        layout_scat['height'] = 480
        layout_scat['legend'] = dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
            title=None,
            font=dict(size=11)
        )
        fig_scatter.update_layout(layout_scat)
        st.plotly_chart(fig_scatter, use_container_width=True, config=chart_config)

    # TABELA FORMATADA POR NCM (SEM NENHUM NÚMERO CRU)
    st.markdown(f"<div class='glass-card'><h4>📊 Tabela de Inteligência por NCM (Com Formatação Rigorosa de Casas Decimais)</h4></div>", unsafe_allow_html=True)
    
    df_ncm_t = df_filtered.groupby('NCM_LABEL').agg(
        Total_FOB=('FOB_TOTAL', 'sum'),
        Total_Peso=('PESO_LIQUIDO', 'sum'),
        Operacoes=('QTD_OPS', 'sum'),
        Preco_Min_Kg=('PRECO_KG', lambda x: x[x>0].min() if len(x[x>0])>0 else 0),
        Preco_Max_Kg=('PRECO_KG', 'max')
    ).reset_index().sort_values('Total_FOB', ascending=False)
    
    df_ncm_t['Preco_Medio_Kg'] = np.where(df_ncm_t['Total_Peso'] > 0, df_ncm_t['Total_FOB'] / df_ncm_t['Total_Peso'], 0.0)
    
    # Preparação para exibição perfeitamente formatada
    df_ncm_disp = pd.DataFrame()
    df_ncm_disp['Classificação Fiscal (NCM)'] = df_ncm_t['NCM_LABEL']
    df_ncm_disp[f'FOB Total ({moeda_code})'] = (df_ncm_t['Total_FOB'] * fx_rate).apply(lambda v: fmt_moeda_br(v, moeda=moeda_code))
    df_ncm_disp['Peso Total'] = df_ncm_t['Total_Peso'].apply(fmt_peso_br)
    df_ncm_disp['Operações'] = df_ncm_t['Operacoes'].apply(fmt_int_br)
    df_ncm_disp[f'Preço Médio / Kg ({moeda_code})'] = (df_ncm_t['Preco_Medio_Kg'] * fx_rate).apply(lambda v: fmt_moeda_br(v, moeda=moeda_code))
    df_ncm_disp[f'Mínimo / Kg ({moeda_code})'] = (df_ncm_t['Preco_Min_Kg'] * fx_rate).apply(lambda v: fmt_moeda_br(v, moeda=moeda_code))
    df_ncm_disp[f'Máximo / Kg ({moeda_code})'] = (df_ncm_t['Preco_Max_Kg'] * fx_rate).apply(lambda v: fmt_moeda_br(v, moeda=moeda_code))
    
    st.dataframe(df_ncm_disp, use_container_width=True, height=280)

# ----------------------------------------------------
# TAB 4: LOGÍSTICA & PORTOS
# ----------------------------------------------------
with tabs[3]:
    log1, log2 = st.columns(2)
    with log1:
        st.markdown(f"<div class='glass-card'><h4>🏛️ Recintos Alfandegados & Portos de Entrada no Brasil</h4></div>", unsafe_allow_html=True)
        df_urf = df_filtered[~df_filtered['URF de Entrada'].isin(['N/A', '', 'EXTERIOR'])].groupby('URF de Entrada').agg({'FOB_TOTAL': 'sum'}).reset_index().sort_values('FOB_TOTAL', ascending=True).tail(15)
        if not df_urf.empty:
            df_urf['VAL_SHOW'] = df_urf['FOB_TOTAL'] * fx_rate
            df_urf['URF_CURTA'] = df_urf['URF de Entrada'].apply(lambda x: truncate_text(x, 32))
            
            fig_urf = px.bar(
                df_urf, x='VAL_SHOW', y='URF_CURTA', orientation='h',
                color_discrete_sequence=[THEME['accent_primary']],
                custom_data=['URF de Entrada']
            )
            max_urf = df_urf['VAL_SHOW'].max() if not df_urf.empty else 1.0
            fig_urf.update_xaxes(range=[0, max_urf * 1.32], automargin=True)
            fig_urf.update_traces(
                texttemplate=('%{x:,.2s}'), textposition='outside', cliponaxis=False,
                hovertemplate="<b>%{customdata[0]}</b><br>Valor: " + ("R$ %{x:,.2f}" if is_brl else "$ %{x:,.2f}") + "<extra></extra>"
            )
            fig_urf.update_layout(common_layout, height=450, yaxis=dict(title="", automargin=True), margin=dict(l=15, r=85, t=35, b=20))
            st.plotly_chart(fig_urf, use_container_width=True, config=chart_config)
            
    with log2:
        st.markdown(f"<div class='glass-card'><h4>💸 Custo Logístico Internacional (Frete por Modal)</h4></div>", unsafe_allow_html=True)
        df_frete_m = df_filtered[df_filtered['FRETE_TOTAL'] > 0].groupby('MODAL').agg({
            'FRETE_TOTAL': 'sum', 'FOB_TOTAL': 'sum'
        }).reset_index()
        
        if not df_frete_m.empty:
            df_frete_m['VAL_SHOW'] = df_frete_m['FRETE_TOTAL'] * fx_rate
            max_frete = df_frete_m['VAL_SHOW'].max() if not df_frete_m.empty else 1.0
            fig_frete = px.bar(
                df_frete_m, x='MODAL', y='VAL_SHOW',
                color='MODAL', color_discrete_sequence=[THEME['accent_primary'], THEME['accent_mint'], THEME['accent_cyan']]
            )
            fig_frete.update_yaxes(range=[0, max_frete * 1.3], automargin=True)
            fig_frete.update_traces(
                texttemplate=('%{y:,.2s}'), textposition='outside', cliponaxis=False,
                hovertemplate="<b>Modal %{x}</b><br>Frete: " + ("R$ %{y:,.2f}" if is_brl else "$ %{y:,.2f}") + "<extra></extra>"
            )
            fig_frete.update_layout(common_layout, height=450, showlegend=False)
            st.plotly_chart(fig_frete, use_container_width=True, config=chart_config)

# ----------------------------------------------------
# TAB 5: ROTAS & GLOBO 3D
# ----------------------------------------------------
with tabs[4]:
    g_c1, g_c2 = st.columns([3, 1])
    with g_c2:
        map_style = st.radio("Projeção Global:", ["Globo 3D Esférico", "Mapa Planisfério 2D"], index=0)
        
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
        'SUIÇA': (46.8182, 8.2275),
        'HOLANDA': (52.1326, 5.2913),
        'BELGICA': (50.5039, 4.4699),
        'SUECIA': (60.1282, 18.6435),
        'INDIA': (20.5937, 78.9629),
        'TAIWAN': (23.6978, 120.9605),
        'POLONIA': (51.9194, 19.1451),
        'AUSTRIA': (47.5162, 14.5501)
    }
    
    df_geo = df_filtered.groupby('PAIS DE ORIGEM')['FOB_TOTAL'].sum().reset_index()
    df_geo = df_geo[df_geo['PAIS DE ORIGEM'].isin(country_coords.keys())]
    
    if not df_geo.empty and 'BRASIL' in country_coords:
        br_lat, br_lon = country_coords['BRASIL']
        fig_g = go.Figure()
        
        # Ponto Brasil
        fig_g.add_trace(go.Scattergeo(
            lon=[br_lon], lat=[br_lat], mode='markers+text',
            text=["BRASIL"], textposition="top center",
            marker=dict(size=14, color=THEME['accent_primary'], line=dict(width=2, color='white'), symbol='diamond'),
            name='Brasil'
        ))
        
        max_geo = df_geo['FOB_TOTAL'].max()
        for _, r in df_geo.iterrows():
            orig = r['PAIS DE ORIGEM']
            if orig != 'BRASIL' and orig in country_coords:
                o_lat, o_lon = country_coords[orig]
                v = r['FOB_TOTAL'] * fx_rate
                size_pt = 6 + (r['FOB_TOTAL'] / max_geo * 15)
                
                # Rota
                fig_g.add_trace(go.Scattergeo(
                    lon=[o_lon, br_lon], lat=[o_lat, br_lat], mode='lines',
                    line=dict(width=1.5 + (r['FOB_TOTAL'] / max_geo * 2.5), color="rgba(16, 185, 129, 0.45)"),
                    showlegend=False, hoverinfo='none'
                ))
                # Origem
                fig_g.add_trace(go.Scattergeo(
                    lon=[o_lon], lat=[o_lat], mode='markers',
                    marker=dict(size=size_pt, color=THEME['accent_mint'], line=dict(width=1, color='white')),
                    name=orig, hoverinfo='text',
                    text=f"<b>{orig}</b><br>Valor: " + fmt_moeda_br(v, moeda=moeda_code)
                ))
                
        proj = 'orthographic' if map_style == "Globo 3D Esférico" else 'natural earth'
        fig_g.update_layout(
            geo=dict(
                projection_type=proj,
                showland=True, landcolor='rgba(20, 29, 46, 0.9)',
                showocean=True, oceancolor='rgba(9, 13, 20, 0.95)',
                showcountries=True, countrycolor='rgba(255, 255, 255, 0.12)',
                bgcolor='rgba(0,0,0,0)'
            ),
            height=620, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_g, use_container_width=True, config=chart_config)

# ----------------------------------------------------
# TAB 6: CADEIA DE SUPRIMENTOS & FLUXOS
# ----------------------------------------------------
with tabs[5]:
    st.markdown(f"<div class='glass-card'><h4>🔄 Mapeamento Visual de Fluxos & Cadeia de Suprimentos Internacional</h4></div>", unsafe_allow_html=True)
    
    # Base limpa para modelagem dos fluxos
    df_flw_base = df_filtered[
        ~df_filtered['PAIS DE ORIGEM'].isin(['N/A', '', 'EXTERIOR']) &
        ~df_filtered['PROVÁVEL EXPORTADOR'].isin(['N/A', 'EXTERIOR', '', 'NAN']) &
        ~df_filtered['PROVÁVEL IMPORTADOR'].isin(['N/A', '', 'NAN'])
    ].copy()

    c_flw_h1, c_flw_h2, c_flw_h3 = st.columns([1.1, 1, 1])
    with c_flw_h1:
        flw_mode = st.radio("Modelo Visual:", ["Sankey Dinâmico Executivo", "Sunburst Hierárquico Multi-Nível"], horizontal=True)
    with c_flw_h2:
        top_stage_n = st.slider("Top N Conexões por Etapa:", min_value=3, max_value=15, value=7, help="Define a quantidade máxima de nós por coluna para manter o diagrama totalmente límpido, espaçoso e sem emaranhados.")
    with c_flw_h3:
        min_fob_opt = st.selectbox(
            "Filtro de Relevância (FOB Mínimo):",
            ["Todas as Operações", "> $ 10.000", "> $ 50.000", "> $ 100.000", "> $ 250.000"],
            index=0,
            help="Descarta micro-operações de baixo valor que causam ruído e cruzamento excessivo de linhas."
        )

    # Painel interativo de filtros por país, exportador e importador
    with st.expander("🎛️ Filtros Dedicados do Fluxo (Isolar Países, Fornecedores ou Compradores)", expanded=True):
        paises_rank = df_flw_base.groupby('PAIS DE ORIGEM')['FOB_TOTAL'].sum().sort_values(ascending=False).index.tolist()
        exps_rank = df_flw_base.groupby('PROVÁVEL EXPORTADOR')['FOB_TOTAL'].sum().sort_values(ascending=False).index.tolist()
        imps_rank = df_flw_base.groupby('PROVÁVEL IMPORTADOR')['FOB_TOTAL'].sum().sort_values(ascending=False).index.tolist()

        f_c1, f_c2, f_c3 = st.columns(3)
        with f_c1:
            sel_flw_paises = st.multiselect("🌍 Países de Origem:", options=paises_rank, default=[], placeholder="Todos ou selecione países específicos...", key="flw_filter_paises")
        with f_c2:
            sel_flw_exps = st.multiselect("🏭 Fabricantes / Exportadores:", options=exps_rank, default=[], placeholder="Todos ou selecione fornecedores específicos...", key="flw_filter_exps")
        with f_c3:
            sel_flw_imps = st.multiselect("🏢 Importadores no Brasil:", options=imps_rank, default=[], placeholder="Todos ou selecione compradores específicos...", key="flw_filter_imps")

        fc_sub1, fc_sub2 = st.columns([2, 1])
        with fc_sub1:
            agrupar_outros = st.checkbox("Agrupar fluxos secundários fora do Top N em 'OUTROS' (Preserva 100% do volume financeiro sem poluir o visual)", value=False)
        with fc_sub2:
            st.caption(f"Base disponível: {len(df_flw_base):,} operações para modelagem.")

    df_flw = df_flw_base.copy()

    # Aplicação de corte por valor mínimo
    if min_fob_opt == "> $ 10.000":
        df_flw = df_flw[df_flw['FOB_TOTAL'] >= 10000]
    elif min_fob_opt == "> $ 50.000":
        df_flw = df_flw[df_flw['FOB_TOTAL'] >= 50000]
    elif min_fob_opt == "> $ 100.000":
        df_flw = df_flw[df_flw['FOB_TOTAL'] >= 100000]
    elif min_fob_opt == "> $ 250.000":
        df_flw = df_flw[df_flw['FOB_TOTAL'] >= 250000]

    # Aplicação dos filtros multiselect
    if sel_flw_paises:
        df_flw = df_flw[df_flw['PAIS DE ORIGEM'].isin(sel_flw_paises)]
    if sel_flw_exps:
        df_flw = df_flw[df_flw['PROVÁVEL EXPORTADOR'].isin(sel_flw_exps)]
    if sel_flw_imps:
        df_flw = df_flw[df_flw['PROVÁVEL IMPORTADOR'].isin(sel_flw_imps)]

    if df_flw.empty:
        st.info("ℹ️ Nenhuma conexão encontrada para os filtros selecionados nesta aba.")
    else:
        # Define Top N por coluna para evitar aglomeração excessiva
        active_top_paises = df_flw.groupby('PAIS DE ORIGEM')['FOB_TOTAL'].sum().nlargest(top_stage_n if not sel_flw_paises else len(sel_flw_paises)).index.tolist()
        active_top_exps = df_flw.groupby('PROVÁVEL EXPORTADOR')['FOB_TOTAL'].sum().nlargest(top_stage_n if not sel_flw_exps else len(sel_flw_exps)).index.tolist()
        active_top_imps = df_flw.groupby('PROVÁVEL IMPORTADOR')['FOB_TOTAL'].sum().nlargest(top_stage_n if not sel_flw_imps else len(sel_flw_imps)).index.tolist()

        if agrupar_outros:
            df_flw_plot = df_flw.copy()
            df_flw_plot['PAIS DE ORIGEM'] = df_flw_plot['PAIS DE ORIGEM'].apply(lambda x: x if x in active_top_paises else 'OUTROS PAÍSES')
            df_flw_plot['PROVÁVEL EXPORTADOR'] = df_flw_plot['PROVÁVEL EXPORTADOR'].apply(lambda x: x if x in active_top_exps else 'OUTROS FORNECEDORES')
            df_flw_plot['PROVÁVEL IMPORTADOR'] = df_flw_plot['PROVÁVEL IMPORTADOR'].apply(lambda x: x if x in active_top_imps else 'OUTROS IMPORTADORES')
        else:
            df_flw_plot = df_flw[
                df_flw['PAIS DE ORIGEM'].isin(active_top_paises) &
                df_flw['PROVÁVEL EXPORTADOR'].isin(active_top_exps) &
                df_flw['PROVÁVEL IMPORTADOR'].isin(active_top_imps)
            ].copy()

        if df_flw_plot.empty:
            st.warning("⚠️ Os filtros combinados resultaram em conexões vazias. Experimente aumentar o Top N ou selecionar opções mais amplas.")
        elif flw_mode == "Sunburst Hierárquico Multi-Nível":
            df_flw_plot['VAL_SHOW'] = df_flw_plot['FOB_TOTAL'] * fx_rate
            fig_sun = px.sunburst(
                df_flw_plot, path=['PAIS DE ORIGEM', 'PROVÁVEL EXPORTADOR', 'PROVÁVEL IMPORTADOR'],
                values='VAL_SHOW', color='VAL_SHOW',
                color_continuous_scale=[[0, '#064E3B'], [0.5, '#059669'], [1, THEME['accent_primary']]]
            )
            fig_sun.update_layout(common_layout, height=650, coloraxis_showscale=False)
            fig_sun.update_traces(hovertemplate="<b>%{label}</b><br>Volume: " + ("R$ %{value:,.2f}" if is_brl else "$ %{value:,.2f}") + "<extra></extra>")
            st.plotly_chart(fig_sun, use_container_width=True, config=chart_config)
        else:
            # SANKEY EXECUTIVO
            sk_path = ['PAIS DE ORIGEM', 'PROVÁVEL EXPORTADOR', 'PROVÁVEL IMPORTADOR']
            node_keys = []
            node_labels = []
            node_colors = []
            node_map = {}

            # Paletas por estágio para diferenciar perfeitamente a origem, o fornecedor e o comprador
            stage_palettes = {
                'PAIS DE ORIGEM': ['#06B6D4', '#0EA5E9', '#38BDF8', '#0284C7', '#0891B2'],
                'PROVÁVEL EXPORTADOR': ['#10B981', '#059669', '#34D399', '#14B8A6', '#15803D'],
                'PROVÁVEL IMPORTADOR': ['#8B5CF6', '#7C3AED', '#A855F7', '#6366F1', '#4F46E5']
            }

            for stage_idx, col in enumerate(sk_path):
                col_totals = df_flw_plot.groupby(col)['FOB_TOTAL'].sum().sort_values(ascending=False)
                for i, (val, tot) in enumerate(col_totals.items()):
                    n_id = f"{col}:{val}"
                    if n_id not in node_map:
                        node_map[n_id] = len(node_keys)
                        node_keys.append(n_id)
                        tot_conv = tot * fx_rate
                        lbl = f"{truncate_text(val, 24)} ({fmt_moeda_br(tot_conv, compact=True, moeda=moeda_code)})"
                        node_labels.append(lbl)
                        
                        if 'OUTROS' in val:
                            node_colors.append('#64748B')
                        else:
                            palette = stage_palettes[col]
                            node_colors.append(palette[i % len(palette)])

            l_src, l_tgt, l_val, l_col = [], [], [], []
            for i in range(len(sk_path) - 1):
                c_a, c_b = sk_path[i], sk_path[i+1]
                grp = df_flw_plot.groupby([c_a, c_b])['FOB_TOTAL'].sum().reset_index()
                link_c = "rgba(6, 182, 212, 0.38)" if i == 0 else "rgba(139, 92, 246, 0.38)"
                
                for _, r in grp.iterrows():
                    val_conv = r['FOB_TOTAL'] * fx_rate
                    if val_conv > 0:
                        l_src.append(node_map[f"{c_a}:{r[c_a]}"])
                        l_tgt.append(node_map[f"{c_b}:{r[c_b]}"])
                        l_val.append(val_conv)
                        if 'OUTROS' in str(r[c_a]) or 'OUTROS' in str(r[c_b]):
                            l_col.append("rgba(100, 116, 139, 0.22)")
                        else:
                            l_col.append(link_c)

            fig_sk = go.Figure(data=[go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=30,
                    thickness=22,
                    line=dict(color="rgba(255, 255, 255, 0.15)", width=1),
                    label=node_labels,
                    color=node_colors,
                    hovertemplate="<b>%{label}</b><extra></extra>"
                ),
                link=dict(
                    source=l_src,
                    target=l_tgt,
                    value=l_val,
                    color=l_col,
                    hovertemplate="De: <b>%{source.label}</b><br>Para: <b>%{target.label}</b><br>Volume: " + ("R$ %{value:,.2f}" if is_brl else "$ %{value:,.2f}") + "<extra></extra>"
                )
            )])
            fig_sk.update_layout(
                common_layout,
                height=680,
                margin=dict(l=15, r=15, t=30, b=20)
            )
            st.plotly_chart(fig_sk, use_container_width=True, config=chart_config)

        # TABELA DE DETALHAMENTO DAS ROTAS DO FLUXO
        with st.expander("📑 Matriz Detalhada de Rotas e Cadeias de Fornecimento do Gráfico", expanded=False):
            df_flw_matrix = df_flw_plot.groupby(['PAIS DE ORIGEM', 'PROVÁVEL EXPORTADOR', 'PROVÁVEL IMPORTADOR']).agg({
                'FOB_TOTAL': 'sum',
                'PESO_LIQUIDO': 'sum',
                'QTD_OPS': 'sum'
            }).reset_index().sort_values('FOB_TOTAL', ascending=False)
            
            tot_matrix = df_flw_matrix['FOB_TOTAL'].sum()
            df_flw_matrix['% do Fluxo'] = ((df_flw_matrix['FOB_TOTAL'] / tot_matrix) * 100).apply(fmt_pct_br) if tot_matrix > 0 else "0,0%"
            df_flw_matrix[f'Valor FOB ({moeda_code})'] = (df_flw_matrix['FOB_TOTAL'] * fx_rate).apply(lambda v: fmt_moeda_br(v, moeda=moeda_code))
            df_flw_matrix['Peso Líquido'] = df_flw_matrix['PESO_LIQUIDO'].apply(fmt_peso_br)
            df_flw_matrix['Operações'] = df_flw_matrix['QTD_OPS'].apply(fmt_int_br)
            
            st.dataframe(
                df_flw_matrix[['PAIS DE ORIGEM', 'PROVÁVEL EXPORTADOR', 'PROVÁVEL IMPORTADOR', f'Valor FOB ({moeda_code})', '% do Fluxo', 'Peso Líquido', 'Operações']],
                use_container_width=True,
                height=340,
                hide_index=True
            )

# ----------------------------------------------------
# TAB 7: LEDGER & CENTRAL DE EXPORTAÇÃO
# ----------------------------------------------------
with tabs[6]:
    st.markdown(f"<div class='glass-card'><h4>📋 Central de Dados & Exportação Executiva (.xlsx Multi-Abas & .csv)</h4></div>", unsafe_allow_html=True)
    
    # GERADOR DE EXCEL MULTI-ABAS COM SUPORTE MONETÁRIO NUMÉRICO REAL
    def generate_excel_suite(df_in):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Resumo Executivo
            resumo_rows = [
                {'Indicador Executivo': 'Data de Geração do Relatório', 'Valor': datetime.now().strftime('%d/%m/%Y %H:%M')},
                {'Indicador Executivo': 'Cotação Dólar Comercial (USD/BRL)', 'Valor': round(usd_rate, 4)},
                {'Indicador Executivo': 'Valor FOB Total ($)', 'Valor': round(df_in['FOB_TOTAL'].sum(), 2)},
                {'Indicador Executivo': 'Valor CIF Estimado ($)', 'Valor': round(df_in['CIF_TOTAL'].sum(), 2)},
                {'Indicador Executivo': 'Frete Internacional Total ($)', 'Valor': round(df_in['FRETE_TOTAL'].sum(), 2)},
                {'Indicador Executivo': 'Peso Líquido Total (Kg)', 'Valor': round(df_in['PESO_LIQUIDO'].sum(), 2)},
                {'Indicador Executivo': 'Total de Operações', 'Valor': int(df_in['QTD_OPS'].sum())},
                {'Indicador Executivo': 'Ticket Médio ($)', 'Valor': round((df_in['FOB_TOTAL'].sum() / df_in['QTD_OPS'].sum()) if df_in['QTD_OPS'].sum()>0 else 0, 2)},
                {'Indicador Executivo': 'Preço Médio / Kg ($)', 'Valor': round((df_in['FOB_TOTAL'].sum() / df_in['PESO_LIQUIDO'].sum()) if df_in['PESO_LIQUIDO'].sum()>0 else 0, 2)}
            ]
            pd.DataFrame(resumo_rows).to_excel(writer, sheet_name='Resumo_Executivo', index=False)
            
            # Sheet 2: Top Importadores
            r_imp = df_in.groupby('PROVÁVEL IMPORTADOR').agg(
                FOB_Total=('FOB_TOTAL', 'sum'), Peso_Total=('PESO_LIQUIDO', 'sum'), Operacoes=('QTD_OPS', 'sum')
            ).reset_index().sort_values('FOB_Total', ascending=False)
            r_imp.head(100).to_excel(writer, sheet_name='Top_Importadores', index=False)
            
            # Sheet 3: Top Fabricantes
            r_exp = df_in.groupby('PROVÁVEL EXPORTADOR').agg(
                FOB_Total=('FOB_TOTAL', 'sum'), Peso_Total=('PESO_LIQUIDO', 'sum'), Operacoes=('QTD_OPS', 'sum')
            ).reset_index().sort_values('FOB_Total', ascending=False)
            r_exp.head(100).to_excel(writer, sheet_name='Top_Exportadores', index=False)
            
            # Sheet 4: NCM
            r_ncm = df_in.groupby('NCM_LABEL').agg(
                FOB_Total=('FOB_TOTAL', 'sum'), Peso_Total=('PESO_LIQUIDO', 'sum'), Operacoes=('QTD_OPS', 'sum')
            ).reset_index().sort_values('FOB_Total', ascending=False)
            r_ncm.to_excel(writer, sheet_name='Resumo_NCM', index=False)
            
            # Sheet 5: Base de Dados
            cols_exp = [
                'DATA_REF', 'PROVÁVEL IMPORTADOR', 'PROVÁVEL EXPORTADOR', 'PAIS DE ORIGEM',
                'NCM_FORMATADO', 'MODAL', 'Provável Incoterm', 'URF de Entrada',
                'FOB_TOTAL', 'CIF_TOTAL', 'FRETE_TOTAL', 'PESO_LIQUIDO', 'QTD_OPS', 'Descrição produto'
            ]
            cols_exp = [c for c in cols_exp if c in df_in.columns]
            df_in[cols_exp].head(10000).to_excel(writer, sheet_name='Operacoes_Filtradas', index=False)
            
        wb = openpyxl.load_workbook(output)
        h_fill = PatternFill(start_color='10B981', end_color='10B981', fill_type='solid')
        h_font = Font(color='FFFFFF', bold=True, size=11)
        
        for name in wb.sheetnames:
            ws = wb[name]
            ws.views.sheetView[0].showGridLines = True
            for col_i, cell in enumerate(ws[1], 1):
                cell.fill = h_fill
                cell.font = h_font
                cell.alignment = Alignment(horizontal='center', vertical='center')
                ws.column_dimensions[get_column_letter(col_i)].width = 24
                
        out_f = io.BytesIO()
        wb.save(out_f)
        return out_f.getvalue()
        
    excel_file_bytes = generate_excel_suite(df_filtered)
    
    ec1, ec2, ec3 = st.columns([1.5, 1.5, 1])
    with ec1:
        st.download_button(
            label="📥 Exportar Relatório Executivo (.xlsx Multi-Abas)",
            data=excel_file_bytes,
            file_name=f"WTP_Comex_Executive_Report_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    with ec2:
        csv_bytes = df_filtered.to_csv(sep=';', decimal=',', index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Exportar Base Completa (.csv formatado)",
            data=csv_bytes,
            file_name=f"WTP_Comex_Data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # TABELA PRINCIPAL DE OPERAÇÕES COM TODAS AS CASAS DECIMAIS E SEPARADORES PERFEITOS
    # Pré-ordenação estrita por data decrescente (ano mais recente 2026 no topo)
    df_ledger_source = df_filtered.sort_values(by='DATA_REF', ascending=False).copy()

    df_ledger_display = pd.DataFrame()
    df_ledger_display['Data'] = df_ledger_source['DATA_REF'].dt.date
    df_ledger_display['Importador'] = df_ledger_source['PROVÁVEL IMPORTADOR']
    df_ledger_display['Exportador / Fabricante'] = df_ledger_source['PROVÁVEL EXPORTADOR']
    df_ledger_display['Origem'] = df_ledger_source['PAIS DE ORIGEM']
    df_ledger_display['NCM'] = df_ledger_source['NCM_FORMATADO']
    df_ledger_display['Modal'] = df_ledger_source['MODAL']
    df_ledger_display['Porto / URF'] = df_ledger_source['URF de Entrada']
    df_ledger_display[f'Valor FOB ({moeda_code})'] = (df_ledger_source['FOB_TOTAL'] * fx_rate).apply(lambda v: fmt_moeda_br(v, moeda=moeda_code))
    df_ledger_display['Peso Líquido'] = df_ledger_source['PESO_LIQUIDO'].apply(fmt_peso_br)
    df_ledger_display['Operações'] = df_ledger_source['QTD_OPS'].apply(fmt_int_br)

    ledger_cfg = {
        "Data": st.column_config.DateColumn(
            "Mês/Ano",
            format="MM/YYYY",
            help="Data de registro da operação. A ordenação é estritamente cronológica por ano e mês (2026 antes de 2024 na ordem decrescente)."
        ),
        "Importador": st.column_config.TextColumn("Importador", width="medium"),
        "Exportador / Fabricante": st.column_config.TextColumn("Exportador / Fabricante", width="medium"),
        "Origem": st.column_config.TextColumn("Origem", width="small"),
        "NCM": st.column_config.TextColumn("NCM", width="small"),
        "Modal": st.column_config.TextColumn("Modal", width="small"),
        "Porto / URF": st.column_config.TextColumn("Porto / URF", width="medium"),
        f"Valor FOB ({moeda_code})": st.column_config.TextColumn(f"Valor FOB ({moeda_code})", width="small"),
        "Peso Líquido": st.column_config.TextColumn("Peso Líquido", width="small"),
        "Operações": st.column_config.TextColumn("Operações", width="small"),
    }

    st.caption(f"Mostrando {len(df_ledger_display):,} operações ordenadas cronologicamente da mais recente (2026) para a mais antiga. Clique nos títulos das colunas para ordenar.")
    st.dataframe(
        df_ledger_display,
        column_config=ledger_cfg,
        use_container_width=True,
        height=540,
        hide_index=True
    )

# ----------------------------------------------------
# TAB 8: CATÁLOGO TÉCNICO WTP / LOGCOMEX
# ----------------------------------------------------
with tabs[7]:
    st.markdown(f"<div class='glass-card'><h4>📦 Catálogo Técnico Consolidado ({len(df_cat_raw):,} Itens Catalogados com Marcas e Modelos)</h4></div>", unsafe_allow_html=True)
    if df_cat_raw is not None and not df_cat_raw.empty:
        df_cat_disp = df_cat_raw.copy()
        
        # Formata colunas numéricas de valor se existirem
        for c in ['Estimativa de valor unitário', 'Provável quantidade estatística']:
            if c in df_cat_disp.columns:
                df_cat_disp[c] = df_cat_disp[c].apply(lambda v: f"{float(v):,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.') if pd.notna(v) else "")
                
        st.dataframe(df_cat_disp, use_container_width=True, height=550)
    else:
        st.warning("Catálogo técnico não carregado.")

# ==========================================
# 11. FOOTER
# ==========================================
st.markdown("<br><hr style='border-color: rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)
st.markdown(
    f"<div style='display: flex; justify-content: space-between; align-items: center; color: {THEME['text_muted']}; font-size: 0.82rem; font-family: \"Plus Jakarta Sans\";'>"
    f"<div>WTP ULTRASONICS • ENGENHARIA DE COMÉRCIO EXTERIOR & MAQUINÁRIO INDUSTRIAL</div>"
    f"<div>ENTERPRISE TERMINAL v85.0 • BY LUCIANO DINIZ • {datetime.now().year}</div>"
    f"</div>",
    unsafe_allow_html=True
)
