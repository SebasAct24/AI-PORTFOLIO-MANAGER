import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import requests
import random
import warnings
import sqlite3
import json
from datetime import datetime

# ==========================================
# 0. CONFIGURACIÓN DEL SISTEMA
# ==========================================
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Configuración visual ancha
st.set_page_config(page_title="Elite Quant Terminal", page_icon="🌍", layout="wide")

# Inicialización de la memoria de sesión
if 'ia_portafolio' not in st.session_state: 
    st.session_state['ia_portafolio'] = None

if 'datos_fundamentales' not in st.session_state: 
    st.session_state['datos_fundamentales'] = {}

if 'ia_resultados_completos' not in st.session_state: 
    st.session_state['ia_resultados_completos'] = None

# ==========================================
# 1. BASE DE DATOS LOCAL (SQLITE)
# ==========================================
def init_db():
    """Crea la tabla si no existe para guardar el historial."""
    conn = sqlite3.connect('portafolios_quant.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS historial_portafolios
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  fecha TEXT, 
                  perfil TEXT, 
                  horizonte TEXT, 
                  mercados TEXT, 
                  activos_json TEXT, 
                  retorno_esperado REAL, 
                  riesgo_esperado REAL)''')
    conn.commit()
    conn.close()

def guardar_en_db(perfil, horizonte, mercados, portafolio, retorno, riesgo):
    """Guarda el portafolio generado en la base de datos SQLite."""
    try:
        conn = sqlite3.connect('portafolios_quant.db')
        c = conn.cursor()
        fecha_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mercados_str = ", ".join(mercados)
        
        # Limpieza exhaustiva para que SQLite no explote con JSONs raros
        port_clean = {str(k): float(v) for k, v in portafolio.items()}
        activos_json = json.dumps(port_clean)
        
        # FIX: Evitar NaNs (Not a Number) que rompen la base de datos
        ret_val = 0.0 if pd.isna(retorno) else float(retorno)
        riesgo_val = 0.0 if pd.isna(riesgo) else float(riesgo)
        
        c.execute('''INSERT INTO historial_portafolios 
                     (fecha, perfil, horizonte, mercados, activos_json, retorno_esperado, riesgo_esperado) 
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (fecha_str, str(perfil), str(horizonte), mercados_str, activos_json, ret_val, riesgo_val))
        conn.commit()
        conn.close()
    except Exception as e:
        pass # Falla silenciosa para que no interrumpa el flujo del Dashboard

def cargar_historial_db():
    """Extrae el historial guardado ordenado del más reciente al más viejo."""
    conn = sqlite3.connect('portafolios_quant.db')
    df = pd.read_sql_query("SELECT * FROM historial_portafolios ORDER BY id DESC", conn)
    conn.close()
    return df

init_db()

# ==========================================
# 2. MATEMÁTICAS HIERARCHICAL RISK PARITY (HRP)
# ==========================================
def get_quasi_diag(link):
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        sort_ix.index = range(len(sort_ix))
    return sort_ix.tolist()

def get_cluster_var(cov, c_items):
    cov_ = cov.iloc[c_items, c_items]
    ivp = 1. / np.diag(cov_)
    ivp /= ivp.sum()
    return np.dot(np.dot(ivp.T, cov_), ivp)

def get_rec_bipart(cov, sort_ix):
    w = pd.Series(1.0, index=sort_ix)
    c_items = [sort_ix]
    while len(c_items) > 0:
        c_items = [i[j:k] for i in c_items for j, k in ((0, len(i)//2), (len(i)//2, len(i))) if len(i) > 1]
        for i in range(0, len(c_items), 2):
            c_1 = c_items[i]
            c_2 = c_items[i+1]
            c_1_v = get_cluster_var(cov, c_1)
            c_2_v = get_cluster_var(cov, c_2)
            alpha = 1 - c_1_v / (c_1_v + c_2_v)
            w[c_1] *= alpha
            w[c_2] *= 1 - alpha
    return w

def hierarchical_risk_parity(cov):
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1, 1)
    
    # 1. Distancia Euclidiana
    dist = np.sqrt(np.clip((1 - corr) / 2., 0, 1))
    dist_val = dist.values if hasattr(dist, 'values') else dist
    dist_euc = squareform(pdist(dist_val.T, metric='euclidean'))
    np.fill_diagonal(dist_euc, 0)
    
    # 2. Clustering Jerárquico
    link = sch.linkage(squareform(dist_euc), method='single')
    sort_ix = get_quasi_diag(link)
    sort_ix = [cov.index[i] for i in sort_ix]
    
    # 3. Bisección Recursiva
    cov_sorted = cov.loc[sort_ix, sort_ix]
    weights_num = get_rec_bipart(cov_sorted, list(range(len(sort_ix))))
    
    return pd.Series(weights_num.values, index=sort_ix)

# ==========================================
# 3. CONSTANTES Y LÍMITES POR PERFIL
# ==========================================
MERCADOS_POR_REGION = {
    "USA & LatAm": ["S&P 500 (USA)", "NASDAQ 100 (USA Tech)", "Bolsa Mexicana (IPC)"],
    "Europa": ["FTSE 100 (UK)", "DAX 40 (Alemania)"],
    "Asia": ["Nikkei 225 (Japón)", "Hang Seng (Hong Kong)"]
}

BENCHMARKS = {
    "S&P 500 (USA)": {"ticker": "SPY", "vol": "^VIX"},
    "NASDAQ 100 (USA Tech)": {"ticker": "QQQ", "vol": "^VXN"},
    "Bolsa Mexicana (IPC)": {"ticker": "^MXX", "vol": "^VIX"},
    "FTSE 100 (UK)": {"ticker": "^FTSE", "vol": "^VIX"},
    "DAX 40 (Alemania)": {"ticker": "^GDAXI", "vol": "VVDAX.DE"},
    "Nikkei 225 (Japón)": {"ticker": "^N225", "vol": "^NKVI.OS"},
    "Hang Seng (Hong Kong)": {"ticker": "^HSI", "vol": "^HSIL"}
}

LIMITES_PERFIL = {
    "Conservador": {"vol_max": 0.15, "beta_max": 0.8, "sharpe_min": 0.5, "p_core": 0.60},
    "Moderado":    {"vol_max": 0.30, "beta_max": 1.2, "sharpe_min": 0.3, "p_core": 0.35},
    "Agresivo":    {"vol_max": 0.60, "beta_max": 5.0,"sharpe_min": 0.0, "p_core": 0.15}
}

# ==========================================
# 4. BARRA LATERAL (CONTROLES UI)
# ==========================================
with st.sidebar:
    st.title("🌍 Elite Quant Terminal")
    st.caption("Quantamental + Beta/Sharpe + HRP 6D")
    
    perfil_usuario = st.selectbox("Perfil de Riesgo:", ["Conservador", "Moderado", "Agresivo"])
    horizonte_val = st.selectbox("Plazo de Inversión:", ["3 Meses", "6 Meses", "1 Año"])
    
    if horizonte_val == "3 Meses":
        H_DAYS = 63
    elif horizonte_val == "6 Meses":
        H_DAYS = 126
    else:
        H_DAYS = 252
    
    st.markdown("---")
    region_sel = st.radio("1. Región:", list(MERCADOS_POR_REGION.keys()))
    mercados_seleccionados = st.multiselect(
        "2. Mercados:", 
        MERCADOS_POR_REGION[region_sel], 
        default=[MERCADOS_POR_REGION[region_sel][0]]
    )
    
    st.markdown("---")
    st.info("💡 **Tip:** Apaga el Modo Prueba para un análisis Institucional Real.")
    modo_rapido = st.checkbox("Modo Prueba Rápida", value=True)
    ejecutar_btn = st.button("🚀 Ejecutar Análisis Quantamental", use_container_width=True)

# ==========================================
# 5. EXTRACCIÓN Y EVALUACIÓN FUNDAMENTAL
# ==========================================
@st.cache_data(show_spinner=False)
def obtener_datos_globales(mercados, modo_rapido):
    headers = {"User-Agent": "Mozilla/5.0"}
    tickers_totales = []
    sectores_totales = {}
    mercado_origen = {}

    def extraer_wikipedia(url, match_col, suffix, def_sec, nombre_mercado):
        try:
            html = requests.get(url, headers=headers, timeout=10).text
            tablas = pd.read_html(html)
            for df in tablas:
                posibles_cols = [c for c in df.columns if match_col.lower() in str(c).lower() or 'symbol' in str(c).lower()]
                if len(posibles_cols) > 0:
                    col_t = posibles_cols[0]
                    t_list = (df[col_t].astype(str).str.replace('.', '-') + suffix).tolist()
                    
                    c_sec = [c for c in df.columns if 'sector' in str(c).lower()]
                    if len(c_sec) > 0:
                        s_list = df[c_sec[0]].tolist()
                    else:
                        s_list = [def_sec] * len(t_list)
                        
                    tickers_totales.extend(t_list)
                    sectores_totales.update(dict(zip(t_list, s_list)))
                    mercado_origen.update(dict(zip(t_list, [nombre_mercado] * len(t_list))))
                    return True
        except Exception:
            pass
        return False

    # Función de emergencia si Wikipedia falla
    def agregar_emergencia(lista_tk, sector_name, nombre_mercado):
        tickers_totales.extend(lista_tk)
        sectores_totales.update({tk: sector_name for tk in lista_tk})
        mercado_origen.update({tk: nombre_mercado for tk in lista_tk})

    for m in mercados:
        if m == "S&P 500 (USA)": 
            if not extraer_wikipedia('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 'symbol', '', 'S&P 500', m):
                agregar_emergencia(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'], 'US Large Cap', m)
        elif m == "NASDAQ 100 (USA Tech)": 
            if not extraer_wikipedia('https://en.wikipedia.org/wiki/Nasdaq-100', 'ticker', '', 'Technology', m):
                agregar_emergencia(['META', 'TSLA', 'NFLX', 'PEP', 'COST'], 'Tech/Growth', m)
        elif m == "Bolsa Mexicana (IPC)": 
            if not extraer_wikipedia('https://en.wikipedia.org/wiki/Indice_de_Precios_y_Cotizaciones', 'ticker', '.MX', 'IPC', m):
                agregar_emergencia(['WALMEX.MX', 'AMXB.MX', 'GFNORTEO.MX', 'FEMSAUBD.MX', 'GMEXICOB.MX'], 'Mexico IPC', m)
        elif m == "FTSE 100 (UK)": 
            if not extraer_wikipedia('https://en.wikipedia.org/wiki/FTSE_100_Index', 'ticker', '.L', 'FTSE', m):
                agregar_emergencia(['SHEL.L', 'AZN.L', 'HSBA.L', 'ULVR.L', 'BP.L'], 'UK FTSE', m)
        elif m == "DAX 40 (Alemania)": 
            if not extraer_wikipedia('https://en.wikipedia.org/wiki/DAX', 'ticker', '.DE', 'DAX', m):
                agregar_emergencia(['SAP.DE', 'SIE.DE', 'ALV.DE', 'AIR.DE', 'MBG.DE'], 'Germany DAX', m)
        elif m == "Nikkei 225 (Japón)": 
            if not extraer_wikipedia('https://en.wikipedia.org/wiki/Nikkei_225', 'ticker', '.T', 'Nikkei', m):
                agregar_emergencia(['7203.T', '6758.T', '9984.T', '8306.T', '6861.T'], 'Japan Nikkei', m)
        elif m == "Hang Seng (Hong Kong)": 
            if not extraer_wikipedia('https://en.wikipedia.org/wiki/Hang_Seng_Index', 'ticker', '.HK', 'Hang Seng', m):
                agregar_emergencia(['0700.HK', '9988.HK', '0939.HK', '1299.HK', '0005.HK'], 'HK Hang Seng', m)
    tks = list(set(tickers_totales))
    if len(tks) == 0:
        st.error("Error de conexión con servidores externos de Wikipedia.")
        st.stop()

    if modo_rapido and len(tks) > 60: 
        tks = random.sample(tks, 60)
    
    benchmarks_list = list(set([v['ticker'] for v in BENCHMARKS.values()] + [v['vol'] for v in BENCHMARKS.values()]))
    activos_core = ['TLT', 'GLD', 'SHY', 'IEF', '^TNX']
    for c in activos_core: 
        mercado_origen[c] = "Macro/Refugio"
    
    todos_los_activos = list(set(tks + activos_core + benchmarks_list))
    data_raw = yf.download(todos_los_activos, start='2020-01-01', progress=False)
    data_close = data_raw['Close'].ffill()
    data_volume = data_raw['Volume'].fillna(0)
    
    return tks, sectores_totales, mercado_origen, data_close, data_volume

def evaluar_fundamentales_avanzado(ticker, perfil):
    try:
        info = yf.Ticker(ticker).info
        
        # FIX INTERNACIONAL: Si la API no tiene datos clave, la perdonamos para no borrar el país
        if info.get('trailingPE') is None and info.get('returnOnEquity') is None and info.get('revenueGrowth') is None:
            min_sc = 7.0 if perfil == 'Conservador' else 6.0 if perfil == 'Moderado' else 5.5
            return {"score": min_sc, "moat": "Regular", "pe": 15.0, "pasa": True}
        
        rev_g = info.get('revenueGrowth', 0)
        if rev_g is None: rev_g = 0
            
        earn_g = info.get('earningsGrowth', 0)
        if earn_g is None: earn_g = 0
            
        roe = info.get('returnOnEquity', 0)
        if roe is None: roe = 0
            
        de = info.get('debtToEquity', 0)
        if de is None: de = 0 
            
        fcf = info.get('freeCashflow', 0) or info.get('operatingCashflow', 0)
        if fcf is None: fcf = 0
            
        mcap = info.get('marketCap', 1)
        if mcap is None or mcap == 0: mcap = 1
            
        pe = info.get('trailingPE', 0) or info.get('forwardPE', 0)
        if pe is None: pe = 15
            
        margins = info.get('grossMargins', 0)
        if margins is None: margins = 0

        score_growth = np.clip(((rev_g + earn_g) / 2) * 200, 0, 100)
        score_roe = np.clip(roe * 300, 0, 100)
        score_fcf = np.clip((fcf / mcap) * 1000, 0, 100)
        score_de = np.clip(100 - (de / 2), 0, 100)
        
        if pe <= 0: score_val = 0
        else: score_val = np.clip(100 - ((pe - 10) * 2), 0, 100)
            
        score_moat = np.clip(margins * 200, 0, 100)

        total_score = ((score_growth * 0.25) + (score_roe * 0.20) + (score_fcf * 0.15) + 
                       (score_de * 0.10) + (score_val * 0.15) + (score_moat * 0.15)) / 10

        if margins >= 0.40: moat_cat = "Muy Bueno"
        elif margins >= 0.25: moat_cat = "Bueno"
        elif margins >= 0.10: moat_cat = "Regular"
        elif margins >= 0.05: moat_cat = "Malo"
        else: moat_cat = "Muy Malo"

        if perfil == 'Conservador': min_score = 7.0
        elif perfil == 'Moderado': min_score = 6.0
        else: min_score = 5.5
            
        pasa_filtro = (total_score >= min_score)
        return {"score": total_score, "moat": moat_cat, "pe": pe, "pasa": pasa_filtro}
        
    except Exception:
        min_sc = 7.0 if perfil == 'Conservador' else 6.0 if perfil == 'Moderado' else 5.5
        return {"score": min_sc, "moat": "Regular", "pe": 15.0, "pasa": True}
    
# ==========================================
# 6. CEREBRO: OPTIMIZADOR HRP + SHARPE (FIX PESOS Y CASH)
# ==========================================
def ensamblar_portafolio_ia(oraculo_df, data_close, sectores_dict, mercado_dict, mercados_sel, perfil, en_crisis):
    if perfil == 'Conservador': 
        metrica = 'peor_escenario'
    elif perfil == 'Agresivo': 
        metrica = 'mejor_escenario'
    else: 
        metrica = 'esperado'
    
    # 1. Filtro estricto: Solo lo que proyecte retornos positivos
    ret_proy = oraculo_df[metrica]
    positivos = ret_proy[ret_proy > 0.001].sort_values(ascending=False)

    # Si la IA dice que todo va a caer, nos vamos directo a Efectivo
    if positivos.empty: 
        return {'CASH (Efectivo USD)': 1.0}

    # 2. Macro Asignación
    limites_p = LIMITES_PERFIL[perfil]
    if en_crisis:
        p_core_lim = 0.8 
    else:
        p_core_lim = limites_p['p_core']
        
    p_sat_lim = 1.0 - p_core_lim

    # 3. Core Dinámico
    candidatos_core = ['QQQ', 'TLT', 'GLD', 'SHY', 'IEF']
    validos_core = [c for c in candidatos_core if c in positivos.index]
    
    port = {}
    if len(validos_core) == 0:
        p_core_real = 0.0
    else:
        p_core_real = p_core_lim
        for c in validos_core: 
            port[c] = p_core_real / len(validos_core)

    # 4. Satélites (Multi-mercado y Descorrelación)
    candidatos_sat = [t for t in positivos.index if t not in candidatos_core and t in data_close.columns]
    
    if len(candidatos_sat) == 0:
        port['CASH (Efectivo USD)'] = max(0, 1.0 - sum(port.values()))
        return port

    # Filtro Geográfico de Correlación
    df_ret_sat = data_close[candidatos_sat].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    mat_corr = df_ret_sat.corr()
    
    viables = []
    for t in candidatos_sat:
        if len(viables) == 0: 
            viables.append(t)
        else:
            mercado_t = mercado_dict.get(t, "Macro Global")
            rechazar = False
            for v in viables:
                corr_val = mat_corr.loc[t, v]
                mercado_v = mercado_dict.get(v, "Macro Global")
                
                if mercado_t == mercado_v:
                    limite_corr = 0.60
                else:
                    limite_corr = 0.85
                    
                if corr_val > limite_corr:
                    rechazar = True
                    break
            
            if not rechazar: 
                viables.append(t)
                
        if len(viables) == 20: 
            break

    # 5. HRP + Sharpe Adjustment
    v_ret = df_ret_sat[viables]
    try: 
        pesos_hrp = hierarchical_risk_parity(v_ret.cov() * 252)
    except Exception: 
        pesos_hrp = pd.Series(1.0 / len(viables), index=viables)

    volatilidades = np.clip(np.sqrt(np.diag(v_ret.cov() * 252)), 1e-8, 5)
    sharpe_norm = (ret_proy[viables] / volatilidades)
    
    if sharpe_norm.max() != sharpe_norm.min():
        sharpe_norm = (sharpe_norm - sharpe_norm.min()) / (sharpe_norm.max() - sharpe_norm.min() + 1e-8)
    else:
        sharpe_norm = pd.Series(0.0, index=viables)
    
    if perfil == 'Agresivo':
        factor = 1.5
    elif perfil == 'Conservador':
        factor = 0.5 
    else:
        factor = 1.0
        
    pesos_sat_raw = pesos_hrp * (1 + (sharpe_norm * factor))
    pesos_sat_raw = pesos_sat_raw / pesos_sat_raw.sum()

    # 6. Límites Finales
    p_sec = {"Refugio/Core": p_core_real}
    for t in viables:
        w = np.clip(pesos_sat_raw[t] * (1.0 - p_core_real), 0, 0.15)
        sec = sectores_dict.get(t, 'N/A')
        p_act_sec = p_sec.get(sec, 0)
        
        if p_act_sec + w <= 0.25:
            port[t] = w
            p_sec[sec] = p_act_sec + w
            
    # 7. Cierre al 100% con CASH
    total_asignado = sum(port.values())
    if total_asignado < 1.0:
        port['CASH (Efectivo USD)'] = 1.0 - total_asignado
    
    return port

# ==========================================
# 7. INTERFAZ Y FLUJO DE EJECUCIÓN (TABS)
# ==========================================
tab1, tab2, tab3 = st.tabs(["🚀 Estrategia Quantamental", "⚖️ Recalibración", "🗄️ Historial DB"])

with tab1:
    if ejecutar_btn:
        if len(mercados_seleccionados) == 0: 
            st.error("Selecciona al menos un mercado.")
            st.stop()
            
        tks, secs, m_origen, d_close, d_vol = obtener_datos_globales(mercados_seleccionados, modo_rapido)
        
        st.subheader("🌍 Monitores de Mercado")
        cols_mon = st.columns(len(mercados_seleccionados))
        crisis_global = False
        
        for i_m, m_nm in enumerate(mercados_seleccionados):
            m_cfg = BENCHMARKS[m_nm]
            if m_cfg['ticker'] in d_close.columns:
                p_val = d_close[m_cfg['ticker']].iloc[-1]
                s200_val = d_close[m_cfg['ticker']].rolling(200).mean().iloc[-1]
            else:
                p_val = 0
                s200_val = 0
                
            if m_cfg['vol'] in d_close.columns:
                v_val = d_close[m_cfg['vol']].iloc[-1]
            else:
                v_val = 0
                
            if v_val > 25 and p_val < s200_val: 
                crisis_global = True
                
            with cols_mon[i_m]:
                if s200_val > 0:
                    st.metric(m_nm, f"{p_val:,.2f}", f"{((p_val/s200_val)-1)*100:.2f}% vs SMA")
                else:
                    st.metric(m_nm, f"{p_val:,.2f}", "N/A")
                st.caption(f"VIX: {v_val:.2f}")

        if crisis_global: 
            st.error("🚨 ALERTA ROJA: Pánico detectado. Modo Refugio activado.")
        else: 
            st.success("✅ Régimen de mercado estable.")

        # --- FILTRO 1: TRIPLE GUILLOTINA (CON GARANTÍA GEOGRÁFICA) ---
        with st.spinner("🕵️‍♂️ Ejecutando Guillotina de Riesgo y Selección Geográfica..."):
            t_filtrados = [t for t in tks if t in d_close.columns]
            returns_all = d_close[t_filtrados].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
            vol_hist = returns_all.iloc[-252:].std() * np.sqrt(252)
            momentum = d_close[t_filtrados].pct_change(126).iloc[-1]
            lim_p = LIMITES_PERFIL[perfil_usuario]
            
            # Cálculo de Betas y Sharpe
            betas, sharpes = {}, {}
            for t_tk in t_filtrados:
                m_name = m_origen.get(t_tk, "S&P 500 (USA)")
                b_ticker = BENCHMARKS.get(m_name, BENCHMARKS["S&P 500 (USA)"])['ticker']
                if b_ticker in d_close.columns:
                    b_ret = d_close[b_ticker].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
                    a_ret = returns_all[t_tk]
                    idx_common = a_ret.index.intersection(b_ret.index)
                    if len(idx_common) > 30:
                        cv = np.cov(a_ret.loc[idx_common], b_ret.loc[idx_common])[0][1]
                        vr = np.var(b_ret.loc[idx_common])
                        betas[t_tk] = cv / vr if vr != 0 else 1.0
                    else: betas[t_tk] = 1.0
                else: betas[t_tk] = 1.0
                
                valid_data = d_close[t_tk].dropna()
                ret_ann = (valid_data.iloc[-1] / valid_data.iloc[-252]) - 1 if len(valid_data) >= 252 else returns_all[t_tk].mean() * 252
                sharpes[t_tk] = ret_ann / vol_hist[t_tk] if vol_hist[t_tk] > 0 else 0

            # SELECCIÓN GARANTIZADA POR MERCADO (La cura al monopolio de USA)
            candidatas_pre_fund = []
            for m in mercados_seleccionados:
                tks_mercado = [t for t in t_filtrados if m_origen.get(t) == m]
                
                # 1. Filtramos los mejores de ESTA bolsa específica
                pasan_m = [t for t in tks_mercado if vol_hist[t] <= lim_p['vol_max'] and betas[t] <= lim_p['beta_max'] and sharpes[t] >= lim_p['sharpe_min']]
                
                # 2. Salvavidas: Si la bolsa (ej. Japón) está en crisis y nadie pasa, agarramos a los 10 más seguros
                if not pasan_m:
                    pasan_m = vol_hist[tks_mercado].nsmallest(10).index.tolist()
                
                # 3. Agarramos a los de mejor Momentum de esta bolsa
                score_m = (momentum[pasan_m] / vol_hist[pasan_m]).dropna()
                top_m = score_m.nlargest(10 if modo_rapido else 20).index.tolist()
                candidatas_pre_fund.extend(top_m)

            # --- FILTRO FUNDAMENTAL ---
            final_candidatas, f_data = [], {}
            for t_c in set(candidatas_pre_fund):
                res_f = evaluar_fundamentales_avanzado(t_c, perfil_usuario)
                if res_f["pasa"]: 
                    final_candidatas.append(t_c)
                f_data[t_c] = res_f
            st.session_state['datos_fundamentales'] = f_data

        # --- MOTOR LSTM 6D ---
        with st.spinner("🧠 IA Analizando 6 Dimensiones (R, V, RSI, MACD, TNX, PE)..."):
            res_ia = {}
            prog_bar = st.progress(0)
            ai_list = list(set(final_candidatas + ['TLT', 'GLD', 'SHY', 'IEF', 'QQQ', 'SPY']))
            
            if modo_rapido:
                num_sims = 100
                epocas = 2
            else:
                num_sims = 1000
                epocas = 15
            
            for i_ai, t_ai in enumerate(ai_list):
                if t_ai not in d_close.columns: 
                    continue
                    
                df_f = pd.DataFrame()
                df_f['C'] = d_close[t_ai]
                df_f['V'] = d_vol.get(t_ai, 0)
                df_f['R'] = d_close[t_ai].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
                
                delta = df_f['C'].diff()
                gn = delta.clip(lower=0).ewm(14).mean()
                ls = (-delta.clip(upper=0)).ewm(14).mean()
                df_f['RSI'] = 100 - (100 / (1 + (gn / (ls + 1e-8))))
                df_f['MACD'] = df_f['C'].ewm(span=12).mean() - df_f['C'].ewm(span=26).mean()
                df_f['TNX'] = d_close.get('^TNX', 4.0)
                df_f['PE'] = f_data.get(t_ai, {}).get("pe", 15.0)
                df_f['T'] = df_f['C'].pct_change(H_DAYS).shift(-H_DAYS)
                df_tr = df_f.dropna()
                
                if len(df_tr) > 90:
                    scaler = MinMaxScaler()
                    X_vals = scaler.fit_transform(df_tr[['R', 'V', 'RSI', 'MACD', 'TNX', 'PE']].values)
                    
                    X_s = []
                    for j in range(len(X_vals)-90):
                        X_s.append(X_vals[j:j+90])
                    X_s = np.array(X_s)
                    
                    model = Sequential([
                        LSTM(32, return_sequences=True, input_shape=(90, 6)),
                        Dropout(0.2),
                        LSTM(16),
                        Dense(1)
                    ])
                    model.compile(optimizer='adam', loss='mse')
                    
                    if modo_rapido:
                        model.fit(X_s[-60:], df_tr['T'].values[90:][-60:], epochs=epocas, verbose=0)
                    else:
                        model.fit(X_s, df_tr['T'].values[90:], epochs=epocas, verbose=0)
                        
                    X_l = scaler.transform(df_f[['R', 'V', 'RSI', 'MACD', 'TNX', 'PE']].iloc[-90:].values)
                    sm = model(np.repeat(np.array([X_l]), num_sims, axis=0), training=True).numpy().flatten()
                    
                    res_ia[t_ai] = {
                        'peor_escenario': np.percentile(sm, 5), 
                        'esperado': np.mean(sm), 
                        'mejor_escenario': np.percentile(sm, 95)
                    }
                    
                prog_bar.progress((i_ai+1)/len(ai_list))
                tf.keras.backend.clear_session()
            
            oraculo = pd.DataFrame(res_ia).T
            portafolio = ensamblar_portafolio_ia(oraculo, d_close, secs, m_origen, mercados_seleccionados, perfil_usuario, crisis_global)
            st.session_state['ia_portafolio'] = portafolio
            st.session_state['ia_resultados_completos'] = oraculo

        # --- GRÁFICAS DE BACKTEST ---
        df_ret_1y = d_close.iloc[-252:].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
        p_ret = pd.Series(0.0, index=df_ret_1y.index)
        
        for tk_p, w_p in portafolio.items():
            if tk_p in df_ret_1y.columns: 
                p_ret += df_ret_1y[tk_p] * w_p
                
        st.subheader("📈 Backtest: Portafolio AI vs Mercado")
        c_data = pd.DataFrame({"Portafolio": (1+p_ret).cumprod()*100})
        
        for m_nm in mercados_seleccionados: 
            t_bch = BENCHMARKS[m_nm]['ticker']
            if t_bch in df_ret_1y: 
                c_data[m_nm] = (1+df_ret_1y[t_bch]).cumprod()*100
                
        st.line_chart(c_data)

        # --- TABLA Y RESULTADOS ---
        st.subheader("📊 Distribución Quantamental con Score, Beta y Sharpe")
        r_tab = []
        r_tot = 0.0
        
        for tk_f, w_f in portafolio.items():
            if tk_f == 'CASH (Efectivo USD)': 
                r_tab.append({
                    "Activo": tk_f, "Mercado": "Global", "Sector": "Liquidez", "Peso": f"{w_f*100:.2f}%", 
                    "Precio": "USD $ 1.00", "Ret. Proy": "0.00%", "Beta": "0.00", "Sharpe": "0.00", 
                    "Riesgo": "0.00%", "Score Fund": "N/A", "MOAT": "N/A", "P_Num": w_f, "S_Key": "Liquidez"
                })
            else:
                if tk_f in oraculo.index:
                    rv = oraculo.loc[tk_f, 'esperado']
                else:
                    rv = 0.0
                    
                r_tot += rv * w_f
                
                if tk_f in df_ret_1y.columns:
                    vi = df_ret_1y[tk_f].std() * np.sqrt(252) * 100 
                else:
                    vi = 0.0
                    
                bi = betas.get(tk_f, 1.0)
                sh = sharpes.get(tk_f, 0.0)
                
                if tk_f not in ['SHY', 'TLT', 'GLD', 'IEF', 'QQQ', 'SPY']:
                    sc = secs.get(tk_f, "Macro/Refugio")
                else:
                    sc = "Refugio/Core"
                    
                m_o = m_origen.get(tk_f, "USA")
                
                if str(tk_f).endswith('.MX'): m_str = "MXN $"
                elif str(tk_f).endswith('.L'): m_str = "GBP £"
                elif str(tk_f).endswith('.DE'): m_str = "EUR €"
                elif str(tk_f).endswith('.T'): m_str = "JPY ¥"
                elif str(tk_f).endswith('.HK'): m_str = "HKD $"
                elif str(tk_f).endswith('.SA'): m_str = "BRL R$"
                else: m_str = "USD $"
                
                if tk_f in d_close.columns:
                    pr = d_close[tk_f].iloc[-1]
                else:
                    pr = 0.0
                    
                f_d = f_data.get(tk_f, {"score": 0.0, "moat": "N/A"})
                if f_d['score'] > 0:
                    s_txt = f"{f_d['score']:.1f}/10"
                else:
                    s_txt = "N/A"
                
                r_tab.append({
                    "Activo": tk_f, "Mercado": m_o, "Sector": sc, "Peso": f"{w_f*100:.2f}%", 
                    "Precio": f"{m_str} {pr:,.2f}", "Ret. Proy": f"{rv*100:.2f}%", 
                    "Beta": f"{bi:.2f}", "Sharpe": f"{sh:.2f}", "Riesgo": f"{vi:.2f}%", 
                    "Score Fund": s_txt, "MOAT": f_d['moat'], "P_Num": w_f, "S_Key": sc
                })
        
        df_final = pd.DataFrame(r_tab)
        p_sec_tot = df_final.groupby("S_Key")["P_Num"].sum()
        df_final["Total Sector"] = df_final["S_Key"].map(p_sec_tot).apply(lambda x: f"{x*100:.2f}%")
        
        st.dataframe(df_final.drop(columns=["P_Num", "S_Key"]).set_index("Activo"), use_container_width=True)
        
        # --- CÁLCULO DE RIESGO DE PORTAFOLIO FINAL ---
        c_res1, c_res2 = st.columns(2)
        c_res1.success(f"### 🚀 Retorno Esperado: {r_tot*100:.2f}%")
        
        activos_calc = [k for k in portafolio.keys() if k != 'CASH (Efectivo USD)']
        if len(activos_calc) > 1:
            cov = df_ret_1y[activos_calc].cov() * 252
            pesos_calc = np.array([portafolio[a] for a in activos_calc])
            riesgo_hor = np.sqrt(np.dot(pesos_calc.T, np.dot(cov, pesos_calc))) * np.sqrt(H_DAYS/252)
        else: 
            riesgo_hor = p_ret.std() * np.sqrt(H_DAYS)
        
        c_res2.error(f"### ⚠️ Riesgo Portafolio: {riesgo_hor*100:.2f}%")
        guardar_en_db(perfil_usuario, horizonte_val, mercados_seleccionados, portafolio, r_tot, riesgo_hor)
        st.balloons()

with tab2:
    if st.session_state['ia_portafolio']:
        st.header("⚖️ Recalibración")
        n_rec = st.number_input("Activos Actuales", 1, 15, 3)
        u_p = {}
        for i_rec in range(int(n_rec)):
            c1_r, c2_r = st.columns(2)
            tk_in = c1_r.text_input(f"Ticker {i_rec}", key=f"rt{i_rec}").upper()
            ps_in = c2_r.number_input(f"W%", 0.0, 100.0, 10.0, key=f"rw{i_rec}") / 100
            if tk_in: 
                u_p[tk_in] = ps_in
                
        if st.button("Analizar Diferencias"):
            idl = st.session_state['ia_portafolio']
            todos_activos = set(list(u_p.keys()) + list(idl.keys()))
            comp = []
            for a in todos_activos:
                p_a = u_p.get(a, 0)
                p_i = idl.get(a, 0)
                if p_i == 0: 
                    act = "🔴 VENDER TOTAL"
                elif p_a == 0: 
                    act = "🟢 COMPRAR NUEVO"
                elif abs(p_i - p_a) < 0.03: 
                    act = "🟡 MANTENER"
                else: 
                    act = "🔵 AJUSTAR PESO"
                comp.append({"Activo": a, "Actual": f"{p_a*100:.1f}%", "IA": f"{p_i*100:.1f}%", "Acción": act})
            st.table(pd.DataFrame(comp).set_index("Activo"))

with tab3:
    st.header("🗄️ Historial y Análisis de Riesgo")
    try:
        df_h = cargar_historial_db()
        if not df_h.empty:
            df_h['retorno_esperado'] = pd.to_numeric(df_h['retorno_esperado'], errors='coerce').fillna(0)
            df_h['riesgo_esperado'] = pd.to_numeric(df_h['riesgo_esperado'], errors='coerce').fillna(0)
            df_h['retorno_esperado'] = (df_h['retorno_esperado'] * 100).apply(lambda x: f"{x:.2f}%")
            df_h['riesgo_esperado'] = (df_h['riesgo_esperado'] * 100).apply(lambda x: f"{x:.2f}%")
            
            id_insp = st.selectbox("Inspeccionar ID:", df_h['id'].tolist())
            if id_insp:
                dat_v = df_h[df_h['id'] == id_insp].iloc[0]
                dic_act = json.loads(dat_v['activos_json'])
                df_des = pd.DataFrame(list(dic_act.items()), columns=['Activo', 'Peso'])
                df_des['Peso'] = df_des['Peso'].apply(lambda x: f"{x*100:.2f}%")
                
                col_t, col_p = st.columns(2)
                col_t.table(df_des.set_index("Activo"))
                
                fig_p = px.pie(df_des, values=[float(x.strip('%')) for x in df_des['Peso']], names='Activo', title=f"Mix ID {id_insp}")
                col_p.plotly_chart(fig_p)
    except Exception as e: 
        st.info(f"Historial vacío o error al cargar: {e}")
    
    st.markdown("---")
    if st.session_state['ia_portafolio']:
        st.subheader("🔍 Mapa de Calor (Descorrelación Geográfica)")
        act_h = [k for k in st.session_state['ia_portafolio'].keys() if 'CASH' not in k]
        if len(act_h) > 1:
            try:
                cm_data = yf.download(act_h, start='2023-01-01', progress=False)['Close'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0).corr()
                st.plotly_chart(px.imshow(cm_data, text_auto=".2f", color_continuous_scale="RdBu_r"))
            except Exception as e: 
                st.warning(f"Error Heatmap: {e}")

# ==========================================
# 8. GLOSARIO INSTITUCIONAL
# ==========================================
st.markdown("---")
with st.expander("📚 Glosario de Métricas Institucionales (Documentación del Modelo)"):
    st.markdown("""
    ### 🛡️ Hierarchical Risk Parity (HRP)
    A diferencia de la Optimización de Markowitz (Media-Varianza), el **HRP** no requiere la inversión de una matriz de covarianza, lo que lo hace extremadamente estable ante el ruido del mercado. Agrupa los activos como un árbol jerárquico basándose en su correlación de distancia euclidiana y distribuye el presupuesto de riesgo de forma recursiva.

    ### 📈 Beta Multimercado
    La Beta mide la sensibilidad de un activo frente a su mercado de origen.
    * **Beta < 1.0:** Menos volátil que el mercado (Defensivo).
    * **Beta > 1.0:** Amplifica los movimientos del mercado (Agresivo).
    * *Nota:* En esta terminal, cada acción se calcula contra su índice regional oficial (SPY, QQQ, IPC, DAX, etc.).

    ### 🎯 Ratio de Sharpe
    Representa el retorno excedente por cada unidad de riesgo asumida. Utilizamos el Sharpe proyectado para "inclinar" los pesos matemáticos del HRP hacia las acciones con mayor probabilidad de ganancia ajustada al riesgo.

    ### 🧠 LSTM Quantamental de 6 Dimensiones
    Nuestra red neuronal Deep Learning procesa una matriz tridimensional en ventanas de 90 días, incluyendo:
    1. **Return:** Variación del precio.
    2. **Volume:** Flujo de liquidez.
    3. **RSI:** Indicador de momentum.
    4. **MACD:** Divergencia de tendencia.
    5. **TNX:** Tasa de los bonos a 10 años.
    6. **P/E:** Valoración fundamental de la empresa.
    """)