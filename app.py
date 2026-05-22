# ============================================================
# CALCULADORA DE VaR — GESTÃO DE MESAS DE TRADING
# Projeto Final — Gestão de Risco e Derivativos
# ============================================================
# Estrutura:
#   1. Imports e configuração
#   2. CSS customizado
#   3. Funções de cálculo (Black-Scholes, VaR Histórico, Paramétrico, Monte Carlo)
#   4. Geração de dados de exemplo
#   5. Cálculo de VaR por mesa
#   6. Páginas da aplicação (Home, Upload, Parâmetros, Cálculo, Limites, Dashboard)
#   7. Navegação principal (main)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from datetime import datetime
import warnings
import io

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="VaR Terminal | Mesas de Trading",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 2. CSS — TEMA BLOOMBERG / TERMINAL FINANCEIRO
# ============================================================

st.markdown(
    """
    <style>
        /* ── Fundo geral ── */
        .stApp { background-color: #0d0d0d; }

        section[data-testid="stSidebar"] {
            background-color: #111111;
            border-right: 1px solid #FF6600;
        }

        /* ── Texto geral ── */
        html, body, [class*="css"], p, span, label, div {
            color: #e0e0e0;
            font-family: 'Courier New', Courier, monospace;
        }
        h1, h2, h3, h4 { color: #FF6600 !important; font-family: 'Courier New', monospace; }

        /* ── Header principal ── */
        .main-header {
            font-size: 2rem;
            font-weight: 700;
            color: #FF6600 !important;
            text-align: center;
            padding: 0.8rem 0 0.1rem 0;
            letter-spacing: 2px;
            text-transform: uppercase;
            border-bottom: 2px solid #FF6600;
            margin-bottom: 0.3rem;
        }
        .sub-header {
            font-size: 0.9rem;
            color: #888 !important;
            text-align: center;
            letter-spacing: 3px;
            text-transform: uppercase;
            margin-bottom: 1.5rem;
        }

        /* ── Cards semáforo ── */
        .card-green {
            background: #0a1a0a;
            border: 1px solid #00FF41;
            border-left: 4px solid #00FF41;
            padding: 0.9rem 1.2rem;
            border-radius: 4px;
            margin: 0.4rem 0;
            box-shadow: 0 0 8px rgba(0,255,65,0.15);
        }
        .card-yellow {
            background: #1a1500;
            border: 1px solid #FFD700;
            border-left: 4px solid #FFD700;
            padding: 0.9rem 1.2rem;
            border-radius: 4px;
            margin: 0.4rem 0;
            box-shadow: 0 0 8px rgba(255,215,0,0.15);
        }
        .card-red {
            background: #1a0a0a;
            border: 1px solid #FF3333;
            border-left: 4px solid #FF3333;
            padding: 0.9rem 1.2rem;
            border-radius: 4px;
            margin: 0.4rem 0;
            box-shadow: 0 0 8px rgba(255,51,51,0.2);
        }
        .card-green h3, .card-green p { color: #00FF41 !important; }
        .card-yellow h3, .card-yellow p { color: #FFD700 !important; }
        .card-red h3, .card-red p { color: #FF3333 !important; }

        /* ── Metric box ── */
        .metric-box {
            background: #141414;
            border: 1px solid #FF6600;
            border-radius: 4px;
            padding: 1.2rem;
            text-align: center;
            box-shadow: 0 0 10px rgba(255,102,0,0.1);
        }
        .metric-box h2 { color: #FF6600 !important; font-size: 2rem; }
        .metric-box h4 { color: #FF6600 !important; }
        .metric-box p  { color: #aaa !important; font-size: 0.82rem; }

        /* ── Botões ── */
        .stButton > button {
            background: #FF6600 !important;
            color: #000 !important;
            font-weight: 700 !important;
            font-family: 'Courier New', monospace !important;
            border: none !important;
            border-radius: 3px !important;
            letter-spacing: 1px;
            text-transform: uppercase;
        }
        .stButton > button:hover {
            background: #FF8C00 !important;
            box-shadow: 0 0 12px rgba(255,102,0,0.5) !important;
        }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab"] {
            background: #1a1a1a;
            color: #FF6600;
            border-bottom: 2px solid #333;
            font-family: 'Courier New', monospace;
        }
        .stTabs [aria-selected="true"] {
            background: #FF6600 !important;
            color: #000 !important;
        }

        /* ── Expander ── */
        .streamlit-expanderHeader {
            background: #1a1a1a !important;
            color: #FF6600 !important;
            border: 1px solid #333 !important;
            font-family: 'Courier New', monospace;
        }

        /* ── Dataframe ── */
        .stDataFrame { border: 1px solid #333; }

        /* ── Métricas nativas ── */
        [data-testid="metric-container"] {
            background: #141414;
            border: 1px solid #FF6600;
            border-radius: 4px;
            padding: 0.8rem;
            box-shadow: 0 0 8px rgba(255,102,0,0.1);
        }
        [data-testid="metric-container"] label { color: #888 !important; font-size: 0.75rem; }
        [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #FF6600 !important; font-size: 1.4rem; }

        /* ── Divider ── */
        hr { border-color: #333 !important; }

        /* ── Alertas ── */
        .stAlert { border-radius: 3px; font-family: 'Courier New', monospace; }

        /* ── Progress bar ── */
        .stProgress > div > div { background-color: #FF6600 !important; }

        /* ── Ticker ── */
        .ticker {
            font-family: 'Courier New', monospace;
            font-size: 0.8rem;
            color: #00FF41;
            letter-spacing: 2px;
            text-align: center;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 3. FUNÇÕES DE CÁLCULO
# ============================================================

# -----------------------------------------------------------
# 3.1  BLACK-SCHOLES — precificação de opções europeias
# -----------------------------------------------------------

def black_scholes(S, K, T, r, sigma, tipo="call"):
    """
    Calcula preço e gregas de uma opção europeia pelo modelo Black-Scholes.

    Parâmetros
    ----------
    S     : preço atual do ativo subjacente
    K     : preço de exercício (strike)
    T     : tempo até o vencimento em anos
    r     : taxa livre de risco anual (decimal)
    sigma : volatilidade anualizada do ativo (decimal)
    tipo  : 'call' ou 'put'

    Retorna
    -------
    preco : preço justo da opção
    delta : sensibilidade do preço da opção ao preço do ativo
    gamma : taxa de variação do delta
    vega  : sensibilidade ao preço à volatilidade
    """
    # Vencimento expirado → valor intrínseco
    if T <= 0:
        if tipo == "call":
            return max(S - K, 0.0), (1.0 if S > K else 0.0), 0.0, 0.0
        else:
            return max(K - S, 0.0), (-1.0 if S < K else 0.0), 0.0, 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if tipo == "call":
        preco = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        preco = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # por 1% de vol

    return preco, delta, gamma, vega


# -----------------------------------------------------------
# 3.2  VaR HISTÓRICO
# -----------------------------------------------------------

def var_historico(retornos, confianca, valor_carteira):
    """
    VaR Histórico: usa a distribuição empírica dos retornos passados.

    Metodologia: ordena os retornos e seleciona o percentil (1 - confiança).
    Nenhuma distribuição é assumida — os dados falam por si.

    VaR = -Percentil_α(R_p) × Valor_Carteira
    """
    alpha = 1 - confianca
    percentil = np.percentile(retornos, alpha * 100)
    var = -percentil * valor_carteira

    # Expected Shortfall (CVaR): média das perdas além do VaR
    cauda = retornos[retornos <= percentil]
    es = -cauda.mean() * valor_carteira if len(cauda) > 0 else var

    return max(var, 0.0), max(es, 0.0)


# -----------------------------------------------------------
# 3.3  VaR PARAMÉTRICO (NORMAL)
# -----------------------------------------------------------

def var_parametrico(retornos, confianca, valor_carteira):
    """
    VaR Paramétrico: assume que os retornos seguem distribuição Normal.

    VaR = z_α × σ_p × V
    onde z_α é o quantil da normal padrão no nível de confiança.

    Vantagem : simples e rápido.
    Limitação: subestima risco quando retornos têm caudas pesadas (fat tails).
    """
    z_alpha = norm.ppf(confianca)
    sigma = retornos.std()
    var = z_alpha * sigma * valor_carteira

    # ES paramétrico analítico
    es = (norm.pdf(norm.ppf(confianca)) / (1 - confianca)) * sigma * valor_carteira

    return max(var, 0.0), max(es, 0.0)


# -----------------------------------------------------------
# 3.4  VaR MONTE CARLO
# -----------------------------------------------------------

def var_monte_carlo(retornos, confianca, valor_carteira, n_sim=10_000):
    """
    VaR por Simulação de Monte Carlo.

    Simula n_sim trajetórias do preço usando Movimento Browniano Geométrico (GBM):
        ST = S0 · exp[(μ - ½σ²)·T + σ·√T·Z],  Z ~ N(0,1)

    Os retornos simulados são usados para calcular o percentil de perda.
    """
    mu = retornos.mean()
    sigma = retornos.std()
    T = 1 / 252  # horizonte de 1 dia útil

    np.random.seed(42)
    Z = np.random.standard_normal(n_sim)
    retornos_sim = np.exp((mu - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z) - 1

    alpha = 1 - confianca
    percentil = np.percentile(retornos_sim, alpha * 100)
    var = -percentil * valor_carteira

    cauda = retornos_sim[retornos_sim <= percentil]
    es = -cauda.mean() * valor_carteira if len(cauda) > 0 else var

    return max(var, 0.0), max(es, 0.0), retornos_sim


# -----------------------------------------------------------
# 3.5  VaR DE OPÇÃO — Aproximação Delta
# -----------------------------------------------------------

def var_opcao_delta(S, K, T, r, sigma, tipo, quantidade, confianca, retornos_ativo):
    """
    VaR da opção pela Aproximação Delta (linearização).

    VaR_opção ≈ |Δ| × VaR_ativo_unitário × |quantidade|

    O Delta (Δ) mede quanto o preço da opção varia para cada R$ 1
    de variação no ativo subjacente. Multiplicando pelo VaR do ativo,
    obtemos a perda esperada na posição em opção.

    Limitação: ignora o efeito Gamma (curvatura), adequado para opções
    próximas do dinheiro (ATM) e horizontes curtos.
    """
    _, delta, _, _ = black_scholes(S, K, T, r, sigma, tipo)

    # VaR paramétrico unitário do ativo subjacente
    z_alpha = norm.ppf(confianca)
    sigma_diaria = retornos_ativo.std()
    var_ativo_unit = z_alpha * sigma_diaria * S  # VaR por 1 ação

    var_opcao = abs(delta) * var_ativo_unit * abs(quantidade)
    return max(var_opcao, 0.0), delta


# ============================================================
# 4. DADOS DE EXEMPLO
# ============================================================

@st.cache_data
def gerar_retornos_simulados(n_dias=252):
    """
    Gera série histórica simulada para 5 ações brasileiras típicas.
    Utiliza decomposição de Cholesky para introduzir correlação realista entre ativos.
    """
    np.random.seed(7)
    ativos = ["PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3"]
    vols = [0.028, 0.025, 0.020, 0.022, 0.018]
    mus = [0.0003, 0.0002, 0.0001, 0.00015, 0.0001]

    # Matriz de correlação realista (bancos correlacionados entre si, etc.)
    corr = np.array([
        [1.00, 0.45, 0.35, 0.38, 0.20],
        [0.45, 1.00, 0.30, 0.32, 0.18],
        [0.35, 0.30, 1.00, 0.75, 0.25],
        [0.38, 0.32, 0.75, 1.00, 0.22],
        [0.20, 0.18, 0.25, 0.22, 1.00],
    ])

    L = np.linalg.cholesky(corr)
    Z = np.random.standard_normal((n_dias, len(ativos)))
    Z_corr = Z @ L.T

    retornos = pd.DataFrame(
        Z_corr * vols + mus,
        columns=ativos,
        index=pd.date_range(end=datetime.today(), periods=n_dias, freq="B"),
    )

    # Série de preços acumulados
    precos_ini = {"PETR4": 38.50, "VALE3": 68.20, "ITUB4": 32.80, "BBDC4": 16.40, "ABEV3": 11.20}
    precos = pd.DataFrame(precos_ini, index=[retornos.index[0]])
    for dt in retornos.index[1:]:
        novo = precos.iloc[-1] * (1 + retornos.loc[dt])
        precos = pd.concat([precos, pd.DataFrame([novo], index=[dt])])

    return retornos, precos, precos_ini

@st.cache_data(ttl=3600)
def buscar_dados_yahoo(ativos, periodo="1y"):
    """
    Busca retornos históricos reais via Yahoo Finance.
    Fallback automático para dados simulados em caso de erro.
    """
    import yfinance as yf

    tickers = [f"{a}.SA" for a in ativos]
    try:
        raw = yf.download(tickers, period=periodo, auto_adjust=True, progress=False)
        precos = raw["Close"].copy()
        precos.columns = [c.replace(".SA", "") for c in precos.columns]
        precos = precos.dropna()
        retornos = precos.pct_change().dropna()
        return retornos, precos, True   # True = dados reais
    except Exception as e:
        st.warning(f"⚠️ Yahoo Finance indisponível ({e}). Usando dados simulados.")
        retornos, precos, precos_ini = gerar_retornos_simulados()
        return retornos, precos, False  # False = dados simulados

def criar_posicoes_exemplo():
    """
    Retorna DataFrame de posições de exemplo para as 5 mesas de trading.
    Inclui ações (long e short) e opções (call e put).
    """
    df = pd.DataFrame({
        "Mesa": [
            "Ações Brasil", "Ações Brasil", "Ações Brasil",
            "Opções",        "Opções",
            "Long & Short",  "Long & Short",
            "Volatilidade",  "Volatilidade",
            "Mesa Proprietária", "Mesa Proprietária",
        ],
        "Ativo": [
            "PETR4", "VALE3", "ITUB4",
            "PETR4", "VALE3",
            "ITUB4", "BBDC4",
            "ABEV3", "PETR4",
            "VALE3", "ITUB4",
        ],
        "Tipo": [
            "Ação",  "Ação",  "Ação",
            "Call",  "Put",
            "Ação",  "Ação",
            "Ação",  "Call",
            "Ação",  "Put",
        ],
        "Quantidade": [
            50_000,  30_000,  40_000,
             1_000,     800,
            25_000, -20_000,   # Long & Short: ITUB4 comprado, BBDC4 vendido
            35_000,     500,
            15_000,     600,
        ],
        "Preço": [
            38.50, 68.20, 32.80,
             3.20,  2.80,
            32.80, 16.40,
            11.20,  4.10,
            68.20,  1.50,
        ],
        "Strike": [
            None, None, None,
            40.00, 65.00,
            None, None,
            None, 42.00,
            None, 30.00,
        ],
        "Vencimento_dias": [
            None, None, None,
            30, 45,
            None, None,
            None, 60,
            None, 30,
        ],
        "Limite_VaR": [
            600_000, 600_000, 600_000,
            300_000, 300_000,
            400_000, 400_000,
            500_000, 500_000,
            350_000, 350_000,
        ],
    })

    df["Valor_Posicao"] = df["Quantidade"].abs() * df["Preço"]
    return df


# ============================================================
# 5. CÁLCULO DE VaR POR MESA
# ============================================================

def calcular_var_mesa(mesa_df, retornos_df, params):
    """
    Calcula o VaR de todos os ativos de uma mesa e consolida o total.

    Para ações  → metodologia selecionada pelo usuário (Histórico / Paramétrico / MC).
    Para opções → Aproximação Delta (delta × VaR do ativo subjacente).

    O VaR total da mesa é a soma dos VaRs individuais.
    (abordagem conservadora — assume correlação perfeita entre posições)
    """
    metodologia = params["metodologia"]
    confianca    = params["confianca"]
    r            = params["taxa_rf"]
    horizonte    = params["horizonte"]
    n_sim        = params.get("n_sim", 10_000)

    linhas = []
    var_total = 0.0

    for _, row in mesa_df.iterrows():
        ativo      = row["Ativo"]
        tipo       = row["Tipo"]
        qtd        = row["Quantidade"]
        preco      = row["Preço"]
        val_pos    = abs(qtd) * preco

        if ativo not in retornos_df.columns:
            continue

        ret = retornos_df[ativo].dropna()

        if tipo == "Ação":
            if metodologia == "Histórico":
                var, es = var_historico(ret, confianca, val_pos)
            elif metodologia == "Paramétrico":
                var, es = var_parametrico(ret, confianca, val_pos)
            else:  # Monte Carlo
                var, es, _ = var_monte_carlo(ret, confianca, val_pos, n_sim)

            # Ajuste de raiz do tempo: VaR_T = VaR_1d × √T
            var = var * np.sqrt(horizonte)
            delta_val = 1.0 if qtd > 0 else -1.0
            preco_opcao = None

        else:  # Opção Call ou Put
            strike   = row.get("Strike")   or preco * 1.05
            venc     = row.get("Vencimento_dias") or 30
            T        = max(venc, 1) / 252
            sigma_an = ret.std() * np.sqrt(252)   # vol anualizada
            opt_tipo = "call" if tipo == "Call" else "put"

            preco_opcao, delta_val, _, _ = black_scholes(preco, strike, T, r, sigma_an, opt_tipo)
            var, delta_val = var_opcao_delta(preco, strike, T, r, sigma_an, opt_tipo, qtd, confianca, ret)
            var = var * np.sqrt(horizonte)
            es  = var * 1.25  # ES aproximado para opções

        linhas.append({
            "Ativo":         ativo,
            "Tipo":          tipo,
            "Quantidade":    qtd,
            "Preço":         preco,
            "Valor Posição": val_pos,
            "VaR (R$)":      round(var, 2),
            "Delta":         round(delta_val, 4),
            "Preço Opção":   round(preco_opcao, 4) if preco_opcao else "—",
        })
        var_total += var

    return pd.DataFrame(linhas), var_total


# ============================================================
# 6. PÁGINAS
# ============================================================

# ----------------------------------------------------------
# 6.1  HOME
# ----------------------------------------------------------

def page_home():
    st.markdown('<div class="main-header">📡 VaR Terminal — Mesas de Trading</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">▸ sistema de monitoramento de risco de mercado ◂</div>', unsafe_allow_html=True)
    st.markdown('<div class="ticker">PETR4 &nbsp;|&nbsp; VALE3 &nbsp;|&nbsp; ITUB4 &nbsp;|&nbsp; BBDC4 &nbsp;|&nbsp; ABEV3 &nbsp;|&nbsp; VaR 95% &nbsp;|&nbsp; HORIZON: 1D</div>', unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        ### ▸ OBJETIVO
        Simular a área de **risco de mercado** de uma instituição financeira,
        monitorando o **Value at Risk (VaR)** de diferentes mesas de trading
        e gerando alertas quando os limites aprovados são ultrapassados.
        """)
    with c2:
        st.markdown("""
        ### ▸ METODOLOGIAS
        - `HIST` — VaR Histórico (empírico)
        - `PARAM` — VaR Paramétrico (Normal)
        - `MC` — VaR Monte Carlo (GBM)
        - `BS+Δ` — Black-Scholes + Delta (opções)
        """)
    with c3:
        st.markdown("""
        ### ▸ MESAS MONITORADAS
        - `[01]` Ações Brasil
        - `[02]` Opções
        - `[03]` Long & Short
        - `[04]` Volatilidade
        - `[05]` Mesa Proprietária
        """)

    st.markdown("---")
    st.markdown("### ▸ FLUXO DE OPERAÇÃO")

    passos = [
        ("01", "UPLOAD",      "Importe CSV/Excel ou carregue dados de exemplo"),
        ("02", "PARÂMETROS",  "Defina confiança, horizonte e metodologia"),
        ("03", "CÁLCULO",     "Execute o VaR para todas as mesas"),
        ("04", "DASHBOARD",   "Monitore gráficos, alertas e ranking"),
    ]

    cols = st.columns(4)
    for i, (num, titulo, desc) in enumerate(passos):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-box">
                <h2 style="color:#FF6600;font-size:2.5rem;margin:0">{num}</h2>
                <h4 style="color:#FF6600;letter-spacing:2px;margin:0.3rem 0">{titulo}</h4>
                <p style="color:#888;font-size:0.8rem;margin:0">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ▸ SEMÁFORO DE RISCO")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card-green"><h3>● VERDE</h3><p>Utilização ≤ 70% — Risco confortável</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card-yellow"><h3>● AMARELO</h3><p>Utilização 70%–100% — Atenção redobrada</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card-red"><h3>● VERMELHO</h3><p>Utilização > 100% — Excesso de limite</p></div>', unsafe_allow_html=True)


# ----------------------------------------------------------
# 6.2  UPLOAD DE POSIÇÕES
# ----------------------------------------------------------

def page_upload():
    st.title("📁 Upload de Posições")
    st.markdown("Importe o arquivo de posições ou use os dados de exemplo para testar o sistema.")

    tab_upload, tab_exemplo = st.tabs(["📂 Importar Arquivo", "🎲 Dados de Exemplo"])

    # --- Tab Upload ---
    with tab_upload:
        arq = st.file_uploader(
            "Selecione o arquivo CSV ou Excel",
            type=["csv", "xlsx"],
            help="Colunas obrigatórias: Mesa, Ativo, Tipo, Quantidade, Preço, Limite_VaR",
        )

        if arq:
            try:
                df_up = pd.read_csv(arq) if arq.name.endswith(".csv") else pd.read_excel(arq)
                st.session_state["posicoes"] = df_up
                st.success(f"✅ {len(df_up)} posições carregadas!")
                st.dataframe(df_up, use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao ler arquivo: {e}")

        # Template para download
        st.markdown("#### 📋 Template CSV")
        template = pd.DataFrame({
            "Mesa":          ["Ações Brasil", "Opções"],
            "Ativo":         ["PETR4",        "PETR4"],
            "Tipo":          ["Ação",          "Call"],
            "Quantidade":    [10_000,           500],
            "Preço":         [38.50,             3.20],
            "Strike":        ["",               40.00],
            "Vencimento_dias":["",              30],
            "Limite_VaR":    [600_000,         300_000],
        })
        csv_bytes = template.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Baixar Template CSV", csv_bytes, "template_posicoes.csv", "text/csv")

    # --- Tab Exemplo ---
        with tab_exemplo:
            st.markdown("### 📡 Fonte dos Dados Históricos")
    fonte = st.radio(
        "Escolha a fonte:",
        options=["🌐 Dados Reais (Yahoo Finance)", "🎲 Dados Simulados"],
        help="Dados reais puxam os últimos 12 meses da B3. Requer conexão com internet."
    )
    
    periodo = "1y"
    if fonte == "🌐 Dados Reais (Yahoo Finance)":
        periodo = st.selectbox(
            "Período histórico:",
            options=["6mo", "1y", "2y"],
            index=1,
            format_func=lambda x: {"6mo": "6 meses", "1y": "1 ano", "2y": "2 anos"}[x]
        )
    
    if st.button("🚀 Carregar Dados", type="primary"):
        posicoes = criar_posicoes_exemplo()
        ativos   = posicoes["Ativo"].unique().tolist()
    
        with st.spinner("Buscando dados..."):
            if fonte == "🌐 Dados Reais (Yahoo Finance)":
                retornos, precos, real = buscar_dados_yahoo(ativos, periodo)
            else:
                retornos, precos, _ = gerar_retornos_simulados()
                real = False
    
        st.session_state["posicoes"] = posicoes
        st.session_state["retornos"] = retornos
        st.session_state["precos"]   = precos
    
        if real:
            st.success("✅ Dados reais carregados via Yahoo Finance!")
        else:
            st.success("✅ Dados simulados carregados!")
    
            if "posicoes" in st.session_state:
                st.dataframe(st.session_state["posicoes"], use_container_width=True)


# ----------------------------------------------------------
# 6.3  PARÂMETROS DE RISCO
# ----------------------------------------------------------

def page_parametros():
    st.title("⚙️ Parâmetros de Risco")
    st.markdown("Configure os parâmetros que serão usados em todos os cálculos de VaR.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 📊 Configurações do VaR")

        confianca = st.select_slider(
            "Nível de Confiança",
            options=[0.90, 0.95, 0.99],
            value=0.95,
            format_func=lambda x: f"{x*100:.0f}%",
        )

        horizonte = st.selectbox(
            "Horizonte de Tempo",
            options=[1, 5, 10, 21],
            format_func=lambda x: f"{x} dia(s) útil(eis)",
        )

        metodologia = st.radio(
            "Metodologia de VaR",
            options=["Histórico", "Paramétrico", "Monte Carlo"],
            help=(
                "**Histórico**: distribuição empírica — sem hipótese distribucional.\n\n"
                "**Paramétrico**: assume normalidade — rápido, mas subestima caudas grossas.\n\n"
                "**Monte Carlo**: simula cenários pelo GBM — mais flexível, mais lento."
            ),
        )

        n_sim = 10_000
        if metodologia == "Monte Carlo":
            n_sim = st.number_input("Número de Simulações", 1_000, 100_000, 10_000, 1_000)

    with c2:
        st.markdown("### 📈 Parâmetros de Mercado")

        taxa_rf_pct = st.number_input(
            "Taxa Livre de Risco (% a.a.)",
            min_value=0.0, max_value=30.0,
            value=10.75, step=0.25,
            help="Selic atual (~10,75% a.a.)",
        )
        taxa_rf = taxa_rf_pct / 100

        janela = st.selectbox(
            "Janela Histórica",
            options=[60, 126, 252, 504],
            index=2,
            format_func=lambda x: f"{x} dias (~{x//21} meses)",
        )

        st.markdown("### ✅ Resumo")
        st.markdown(f"""
        | Parâmetro | Valor |
        |-----------|-------|
        | Nível de Confiança | **{confianca*100:.0f}%** |
        | Horizonte | **{horizonte} dia(s)** |
        | Metodologia | **{metodologia}** |
        | Taxa Livre de Risco | **{taxa_rf_pct:.2f}% a.a.** |
        | Janela Histórica | **{janela} dias** |
        | Simulações MC | **{n_sim:,}** |
        """)

    st.session_state["params"] = {
        "confianca":    confianca,
        "horizonte":    horizonte,
        "metodologia":  metodologia,
        "taxa_rf":      taxa_rf,
        "janela":       janela,
        "n_sim":        n_sim,
    }
    st.success("✅ Parâmetros salvos! Vá para a aba **Cálculo do VaR**.")


# ----------------------------------------------------------
# 6.4  CÁLCULO DO VaR
# ----------------------------------------------------------

def page_calculo():
    st.title("📐 Cálculo do VaR")

    if "posicoes" not in st.session_state:
        st.warning("⚠️ Carregue as posições primeiro (aba *Upload de Posições*).")
        return

    # Parâmetros default se o usuário não passou pela aba
    if "params" not in st.session_state:
        st.session_state["params"] = {
            "confianca": 0.95, "horizonte": 1, "metodologia": "Histórico",
            "taxa_rf": 0.1075, "janela": 252, "n_sim": 10_000,
        }

    posicoes = st.session_state["posicoes"]
    params   = st.session_state["params"]

    # Garantir que temos os retornos
    if "retornos" not in st.session_state:
        st.info("Gerando retornos históricos simulados...")
        retornos, precos, _ = gerar_retornos_simulados()
        st.session_state["retornos"] = retornos
        st.session_state["precos"]   = precos

    retornos = st.session_state["retornos"]

    st.markdown(f"""
    **Configuração atual:** Metodologia = `{params['metodologia']}` |
    Confiança = `{params['confianca']*100:.0f}%` |
    Horizonte = `{params['horizonte']} dia(s)`
    """)

    if st.button("🚀 Calcular VaR para Todas as Mesas", type="primary"):

        mesas = posicoes["Mesa"].unique()
        resultados = {}
        barra = st.progress(0)
        status = st.empty()

        for i, mesa in enumerate(mesas):
            status.text(f"Calculando: {mesa}...")
            barra.progress((i + 1) / len(mesas))

            mesa_df  = posicoes[posicoes["Mesa"] == mesa].copy()
            limite   = mesa_df["Limite_VaR"].iloc[0]
            val_cart = (mesa_df["Quantidade"].abs() * mesa_df["Preço"]).sum()

            detalhes, var_total = calcular_var_mesa(mesa_df, retornos, params)
            utilizacao = var_total / limite if limite > 0 else 0

            if utilizacao <= 0.70:
                status_cor, emoji = "Verde",    "🟢"
            elif utilizacao <= 1.00:
                status_cor, emoji = "Amarelo",  "🟡"
            else:
                status_cor, emoji = "Vermelho", "🔴"

            resultados[mesa] = {
                "VaR":          var_total,
                "Limite":       limite,
                "Utilizacao":   utilizacao,
                "Status":       status_cor,
                "Emoji":        emoji,
                "Valor_Cart":   val_cart,
                "Detalhes":     detalhes,
            }

        barra.progress(1.0)
        status.text("✅ Cálculo concluído!")
        st.session_state["resultados"] = resultados

        st.markdown("---")
        st.markdown("### 📋 Resultados por Mesa")

        for mesa, r in resultados.items():
            label_expander = f"{r['Emoji']} {mesa}  —  Utilização: {r['Utilizacao']*100:.1f}%"
            with st.expander(label_expander, expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("💼 Valor da Carteira", f"R$ {r['Valor_Cart']:,.0f}")
                c2.metric("⚠️ VaR Calculado",     f"R$ {r['VaR']:,.0f}")
                c3.metric("🎯 Limite Aprovado",    f"R$ {r['Limite']:,.0f}")
                c4.metric("📊 Utilização",          f"{r['Utilizacao']*100:.1f}%")

                pct = min(r["Utilizacao"], 1.0)
                st.progress(pct)

                if not r["Detalhes"].empty:
                    st.dataframe(
                        r["Detalhes"].style.format({
                            "Preço":        "R$ {:.2f}",
                            "Valor Posição":"R$ {:,.0f}",
                            "VaR (R$)":     "R$ {:,.0f}",
                        }),
                        use_container_width=True,
                    )


# ----------------------------------------------------------
# 6.5  MONITORAMENTO DE LIMITES
# ----------------------------------------------------------

def page_limites():
    st.title("🚦 Monitoramento de Limites")

    if "resultados" not in st.session_state:
        st.warning("⚠️ Execute o cálculo do VaR primeiro.")
        return

    res = st.session_state["resultados"]

    resumo = pd.DataFrame([{
        "Mesa":           mesa,
        "Valor Carteira": r["Valor_Cart"],
        "VaR Calculado":  r["VaR"],
        "Limite VaR":     r["Limite"],
        "Utilização (%)": r["Utilizacao"] * 100,
        "Status":         r["Emoji"] + " " + r["Status"],
    } for mesa, r in res.items()]).sort_values("Utilização (%)", ascending=False)

    # Contadores semáforo
    n_verde    = len(resumo[resumo["Utilização (%)"] <= 70])
    n_amarelo  = len(resumo[(resumo["Utilização (%)"] > 70) & (resumo["Utilização (%)"] <= 100)])
    n_vermelho = len(resumo[resumo["Utilização (%)"] > 100])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="card-green"><h3>🟢 {n_verde} Mesa(s)</h3><p>Risco Confortável (≤ 70%)</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="card-yellow"><h3>🟡 {n_amarelo} Mesa(s)</h3><p>Atenção (70% – 100%)</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="card-red"><h3>🔴 {n_vermelho} Mesa(s)</h3><p>Excesso de Limite (&gt; 100%)</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Alertas de excesso
    excessos = resumo[resumo["Utilização (%)"] > 100]
    if not excessos.empty:
        st.error("🚨 **ALERTAS DE EXCESSO DE LIMITE**")
        for _, row in excessos.iterrows():
            st.error(
                f"**{row['Mesa']}**  |  VaR R$ {row['VaR Calculado']:,.0f}  |  "
                f"Limite R$ {row['Limite VaR']:,.0f}  |  Utilização {row['Utilização (%)']:.1f}%"
            )

    # Tabela consolidada
    st.markdown("### 📋 Tabela Consolidada")
    st.dataframe(
        resumo.style.format({
            "Valor Carteira": "R$ {:,.0f}",
            "VaR Calculado":  "R$ {:,.0f}",
            "Limite VaR":     "R$ {:,.0f}",
            "Utilização (%)": "{:.1f}%",
        }),
        use_container_width=True,
    )

    # Ranking visual
    st.markdown("### 🏆 Ranking por Consumo de Risco")
    for rank, (_, row) in enumerate(resumo.iterrows(), 1):
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(f"**#{rank}  {row['Mesa']}** — {row['Utilização (%)']:.1f}%")
            st.progress(min(row["Utilização (%)"] / 100, 1.0))
        with c2:
            st.markdown(f"<br>{row['Status']}", unsafe_allow_html=True)

    # Export
    st.markdown("---")
    csv_out = resumo.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Exportar Resultados CSV", csv_out, "resultado_var.csv", "text/csv")


# ----------------------------------------------------------
# 6.6  DASHBOARD EXECUTIVO
# ----------------------------------------------------------

def page_dashboard():
    st.title("📊 Dashboard Executivo")

    if "resultados" not in st.session_state:
        st.warning("⚠️ Execute o cálculo do VaR primeiro.")
        return

    res    = st.session_state["resultados"]
    params = st.session_state.get("params", {"confianca": 0.95, "metodologia": "Histórico"})

    df = pd.DataFrame([{
        "Mesa":       mesa,
        "VaR":        r["VaR"],
        "Limite":     r["Limite"],
        "Utilizacao": r["Utilizacao"] * 100,
        "Disponivel": max(r["Limite"] - r["VaR"], 0),
        "Val_Cart":   r["Valor_Cart"],
        "Cor":        ("red" if r["Utilizacao"] > 1 else
                       "orange" if r["Utilizacao"] > 0.70 else "green"),
    } for mesa, r in res.items()]).sort_values("Utilizacao", ascending=False)

    DARK = "plotly_dark"
    ORANGE = "#FF6600"
    GREEN  = "#00FF41"
    RED    = "#FF3333"
    YELLOW = "#FFD700"

    def cor_plotly(c):
        return RED if c == "red" else (YELLOW if c == "orange" else GREEN)

    # ---- KPIs ----
    st.markdown("### ▸ INDICADORES PRINCIPAIS")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("TOTAL VaR",        f"R$ {df['VaR'].sum()/1e6:.2f} M")
    k2.metric("TOTAL LIMITES",    f"R$ {df['Limite'].sum()/1e6:.2f} M")
    k3.metric("UTILIZAÇÃO MÉDIA", f"{df['Utilizacao'].mean():.1f}%")
    k4.metric("MESAS EM EXCESSO", str(len(df[df["Utilizacao"] > 100])),
              delta="⚠️ ALERTA" if len(df[df["Utilizacao"] > 100]) > 0 else "✅ OK")

    st.markdown("---")

    # ---- Gráfico 1: VaR vs Limite ----
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="VaR Calculado",
            x=df["Mesa"], y=df["VaR"],
            marker_color=[cor_plotly(c) for c in df["Cor"]],
            text=[f"R$ {v/1e3:.0f}K" for v in df["VaR"]],
            textposition="outside",
            textfont=dict(color=ORANGE),
        ))
        fig.add_trace(go.Bar(
            name="Limite Aprovado",
            x=df["Mesa"], y=df["Limite"],
            marker_color="#333333",
            marker_line_color=ORANGE,
            marker_line_width=1,
            opacity=0.8,
        ))
        fig.update_layout(
            template=DARK,
            title=dict(text="VaR CALCULADO vs LIMITE", font=dict(color=ORANGE)),
            barmode="group",
            xaxis_tickangle=-25,
            yaxis_title="R$",
            height=420,
            paper_bgcolor="#0d0d0d",
            plot_bgcolor="#0d0d0d",
            font=dict(family="Courier New", color="#e0e0e0"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Gráfico 2: Utilização horizontal ----
    with c2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=df["Utilizacao"],
            y=df["Mesa"],
            orientation="h",
            marker_color=[cor_plotly(c) for c in df["Cor"]],
            text=[f"{u:.1f}%" for u in df["Utilizacao"]],
            textposition="outside",
            textfont=dict(color="#e0e0e0"),
        ))
        fig2.add_vline(x=100, line_dash="dash", line_color=RED,
                       annotation_text="LIMITE 100%",
                       annotation_font_color=RED)
        fig2.add_vline(x=70,  line_dash="dot",  line_color=YELLOW,
                       annotation_text="ATENÇÃO 70%",
                       annotation_font_color=YELLOW)
        fig2.update_layout(
            template=DARK,
            title=dict(text="UTILIZAÇÃO DO LIMITE POR MESA (%)", font=dict(color=ORANGE)),
            xaxis_title="Utilização (%)",
            height=420,
            paper_bgcolor="#0d0d0d",
            plot_bgcolor="#0d0d0d",
            font=dict(family="Courier New", color="#e0e0e0"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ---- Gráfico 3: Pizza + Evolução VaR ----
    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.pie(
            df, values="VaR", names="Mesa",
            title="DISTRIBUIÇÃO DO VaR POR MESA",
            color_discrete_sequence=[ORANGE, GREEN, RED, YELLOW, "#00BFFF", "#FF69B4"],
        )
        fig3.update_layout(
            template=DARK,
            height=400,
            paper_bgcolor="#0d0d0d",
            font=dict(family="Courier New", color="#e0e0e0"),
            title_font_color=ORANGE,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        if "retornos" in st.session_state:
            retornos = st.session_state["retornos"]
            ativo_p  = retornos.columns[0]
            ret_p    = retornos[ativo_p].dropna()

            var_roll, datas_roll = [], []
            for i in range(21, len(ret_p)):
                w = ret_p.iloc[i - 21: i]
                v, _ = var_historico(w, params["confianca"], 1_000_000)
                var_roll.append(v)
                datas_roll.append(ret_p.index[i])

            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=datas_roll[-60:], y=var_roll[-60:],
                mode="lines",
                line=dict(color=ORANGE, width=2),
                name="VaR Histórico (R$ 1M)",
                fill="tozeroy",
                fillcolor="rgba(255,102,0,0.08)",
            ))
            fig4.update_layout(
                template=DARK,
                title=dict(text="EVOLUÇÃO DO VaR — ÚLTIMOS 60 DIAS", font=dict(color=ORANGE)),
                xaxis_title="DATA", yaxis_title="R$",
                height=400,
                paper_bgcolor="#0d0d0d",
                plot_bgcolor="#0d0d0d",
                font=dict(family="Courier New", color="#e0e0e0"),
            )
            st.plotly_chart(fig4, use_container_width=True)

    # ---- Distribuição de retornos ----
    st.markdown("### ▸ ANÁLISE DE RETORNOS POR ATIVO")
    if "retornos" in st.session_state:
        retornos = st.session_state["retornos"]
        ativo_sel = st.selectbox("SELECIONE O ATIVO:", retornos.columns.tolist())
        ret_sel   = retornos[ativo_sel].dropna()

        c5, c6 = st.columns(2)
        with c5:
            pct_var = np.percentile(ret_sel, (1 - params["confianca"]) * 100)
            x_norm  = np.linspace(ret_sel.min(), ret_sel.max(), 300)
            y_norm  = norm.pdf(x_norm, ret_sel.mean(), ret_sel.std())
            y_norm  = y_norm / y_norm.max() * 40

            fig5 = go.Figure()
            fig5.add_trace(go.Histogram(
                x=ret_sel, nbinsx=50, name="RETORNOS",
                marker_color=ORANGE, opacity=0.6,
            ))
            fig5.add_trace(go.Scatter(
                x=x_norm, y=y_norm, mode="lines",
                line=dict(color=GREEN, width=2), name="NORMAL AJUSTADA",
            ))
            fig5.add_vline(
                x=pct_var, line_dash="dash", line_color=RED,
                annotation_text=f"VaR {params['confianca']*100:.0f}%: {pct_var*100:.2f}%",
                annotation_font_color=RED,
            )
            fig5.update_layout(
                template=DARK,
                title=dict(text=f"DISTRIBUIÇÃO DOS RETORNOS — {ativo_sel}", font=dict(color=ORANGE)),
                xaxis_title="RETORNO DIÁRIO", yaxis_title="FREQUÊNCIA",
                height=380,
                paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
                font=dict(family="Courier New", color="#e0e0e0"),
            )
            st.plotly_chart(fig5, use_container_width=True)

        with c6:
            if "precos" in st.session_state and ativo_sel in st.session_state["precos"].columns:
                precos = st.session_state["precos"]
                fig6 = go.Figure()
                fig6.add_trace(go.Scatter(
                    x=precos.index, y=precos[ativo_sel],
                    mode="lines", line=dict(color=ORANGE, width=1.5),
                    name=ativo_sel,
                    fill="tozeroy", fillcolor="rgba(255,102,0,0.06)",
                ))
                fig6.update_layout(
                    template=DARK,
                    title=dict(text=f"EVOLUÇÃO DO PREÇO — {ativo_sel}", font=dict(color=ORANGE)),
                    xaxis_title="DATA", yaxis_title="PREÇO (R$)",
                    height=380,
                    paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
                    font=dict(family="Courier New", color="#e0e0e0"),
                )
                st.plotly_chart(fig6, use_container_width=True)

    # ---- Comparação de metodologias ----
    st.markdown("### ▸ COMPARAÇÃO ENTRE METODOLOGIAS (R$ 1M EXPOSTO)")
    if "retornos" in st.session_state:
        retornos = st.session_state["retornos"]
        cnf      = params["confianca"]
        comp     = []
        for ativo in retornos.columns:
            rt = retornos[ativo].dropna()
            vh, _    = var_historico(rt, cnf, 1_000_000)
            vp, _    = var_parametrico(rt, cnf, 1_000_000)
            vm, _, _ = var_monte_carlo(rt, cnf, 1_000_000, 5_000)
            comp.append({"Ativo": ativo, "Histórico": vh, "Paramétrico": vp, "Monte Carlo": vm})

        df_comp = pd.DataFrame(comp)
        fig7 = go.Figure()
        cores_met = {"Histórico": ORANGE, "Paramétrico": GREEN, "Monte Carlo": "#00BFFF"}
        for met, cor in cores_met.items():
            fig7.add_trace(go.Bar(
                name=met, x=df_comp["Ativo"], y=df_comp[met],
                marker_color=cor,
                text=[f"R$ {v/1e3:.1f}K" for v in df_comp[met]],
                textposition="outside",
                textfont=dict(color="#e0e0e0"),
            ))
        fig7.update_layout(
            template=DARK,
            title=dict(text=f"COMPARAÇÃO VaR POR METODOLOGIA — IC {cnf*100:.0f}%", font=dict(color=ORANGE)),
            barmode="group", yaxis_title="R$", height=420,
            paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
            font=dict(family="Courier New", color="#e0e0e0"),
        )
        st.plotly_chart(fig7, use_container_width=True)

    # ---- Monte Carlo histograma ----
    st.markdown("### ▸ SIMULAÇÃO MONTE CARLO — HISTOGRAMA DE PERDAS")
    if "retornos" in st.session_state:
        retornos  = st.session_state["retornos"]
        ativo_mc  = st.selectbox("ATIVO PARA MONTE CARLO:", retornos.columns.tolist(), key="mc_ativo")
        val_mc    = st.number_input("VALOR DA POSIÇÃO (R$)", value=1_000_000, step=100_000, key="mc_val")

        if st.button("▶ RODAR SIMULAÇÃO MC", key="btn_mc"):
            ret_mc = retornos[ativo_mc].dropna()
            var_mc_val, es_mc, ret_sim = var_monte_carlo(ret_mc, params["confianca"], val_mc, 10_000)
            perdas_sim = -ret_sim * val_mc

            pct_line = np.percentile(perdas_sim, params["confianca"] * 100)

            fig_mc = go.Figure()
            fig_mc.add_trace(go.Histogram(
                x=perdas_sim, nbinsx=80,
                marker_color=ORANGE, opacity=0.7,
                name="PERDAS SIMULADAS",
            ))
            fig_mc.add_vline(
                x=pct_line, line_dash="dash", line_color=RED,
                annotation_text=f"VaR {params['confianca']*100:.0f}%: R$ {pct_line:,.0f}",
                annotation_font_color=RED,
            )
            fig_mc.update_layout(
                template=DARK,
                title=dict(text=f"DISTRIBUIÇÃO DAS PERDAS SIMULADAS — {ativo_mc}", font=dict(color=ORANGE)),
                xaxis_title="PERDA SIMULADA (R$)", yaxis_title="FREQUÊNCIA",
                height=400,
                paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
                font=dict(family="Courier New", color="#e0e0e0"),
            )
            st.plotly_chart(fig_mc, use_container_width=True)

            col_a, col_b = st.columns(2)
            col_a.metric("VaR MONTE CARLO", f"R$ {var_mc_val:,.0f}")
            col_b.metric("EXPECTED SHORTFALL (ES)", f"R$ {es_mc:,.0f}")


# ============================================================
# 6.7  BACKTESTING DO VaR
# ============================================================

def page_backtesting():
    st.title("🔬 Backtesting do VaR")
    st.markdown(
        "Verifica se o VaR calculado foi preciso historicamente — "
        "contando quantos dias a perda real **ultrapassou** o VaR previsto (**exceções**)."
    )

    if "retornos" not in st.session_state:
        st.warning("⚠️ Carregue os dados primeiro na aba *Upload de Posições*.")
        return

    retornos = st.session_state["retornos"]
    params   = st.session_state.get("params", {
        "confianca": 0.95, "metodologia": "Histórico", "horizonte": 1
    })

    # ── Configurações do backtesting ──
    st.markdown("### ⚙️ Configurações")
    c1, c2, c3 = st.columns(3)
    with c1:
        ativo_bt = st.selectbox("Ativo para backtesting:", retornos.columns.tolist())
    with c2:
        confianca_bt = st.select_slider(
            "Nível de Confiança",
            options=[0.90, 0.95, 0.99],
            value=params.get("confianca", 0.95),
            format_func=lambda x: f"{x*100:.0f}%",
        )
    with c3:
        janela_bt = st.selectbox(
            "Janela de estimação (dias)",
            options=[60, 126, 252],
            index=1,
            format_func=lambda x: f"{x} dias (~{x//21} meses)",
        )

    met_bt = st.radio(
        "Metodologia",
        options=["Histórico", "Paramétrico", "Monte Carlo"],
        horizontal=True,
    )

    if st.button("▶ Rodar Backtesting", type="primary"):

        ret_serie = retornos[ativo_bt].dropna().values
        n_total   = len(ret_serie)

        if n_total < janela_bt + 20:
            st.error("Série histórica muito curta para a janela escolhida.")
            return

        # ── Loop: calcula VaR em cada dia e verifica exceção ──
        datas, vars_bt, retornos_reais, excecoes = [], [], [], []

        for i in range(janela_bt, n_total):
            janela   = ret_serie[i - janela_bt: i]
            ret_real = ret_serie[i]          # retorno do dia seguinte (out-of-sample)

            if met_bt == "Histórico":
                v, _ = var_historico(pd.Series(janela), confianca_bt, 1.0)
            elif met_bt == "Paramétrico":
                v, _ = var_parametrico(pd.Series(janela), confianca_bt, 1.0)
            else:
                v, _, _ = var_monte_carlo(pd.Series(janela), confianca_bt, 1.0, 5_000)

            # VaR em termos de retorno (negativo = perda)
            var_retorno = -v   # ex: -0.032 significa perda de 3,2%
            excecao     = ret_real < var_retorno  # perda real > VaR

            datas.append(retornos.index[i])
            vars_bt.append(var_retorno)
            retornos_reais.append(ret_real)
            excecoes.append(excecao)

        df_bt = pd.DataFrame({
            "Data":         datas,
            "Retorno Real": retornos_reais,
            "VaR (limite)": vars_bt,
            "Exceção":      excecoes,
        })

        # ── Estatísticas ──
        T = len(df_bt)
        N = int(df_bt["Exceção"].sum())
        taxa_exc   = N / T
        taxa_esp   = 1 - confianca_bt

        # ── Teste de Kupiec (POF — Proportion of Failures) ──
        # H0: taxa de exceções = taxa esperada
        # LR ~ Chi-quadrado(1) sob H0
        from scipy.stats import chi2
        if N == 0:
            lr_stat = 0.0
        elif N == T:
            lr_stat = 9999.0
        else:
            lr_stat = -2 * (
                (T - N) * np.log(1 - taxa_esp) + N * np.log(taxa_esp)
            ) + 2 * (
                (T - N) * np.log(1 - taxa_exc) + N * np.log(taxa_exc)
            )

        p_valor   = 1 - chi2.cdf(lr_stat, df=1)
        aprovado  = p_valor > 0.05   # não rejeita H0 a 5%

        # ── KPIs ──
        st.markdown("---")
        st.markdown("### 📊 Resultados do Backtesting")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Dias Testados",         f"{T}")
        k2.metric("Exceções Observadas",   f"{N}")
        k3.metric("Taxa de Exceção",       f"{taxa_exc*100:.2f}%")
        k4.metric("Taxa Esperada",         f"{taxa_esp*100:.2f}%")
        k5.metric("p-valor Kupiec",        f"{p_valor:.4f}",
                  delta="✅ APROVADO" if aprovado else "❌ REPROVADO")

        # Interpretação do teste
        if aprovado:
            st.success(
                f"✅ **Teste de Kupiec APROVADO** (p-valor = {p_valor:.4f} > 0,05)  "
                f"— O modelo de VaR {met_bt} é estatisticamente adequado. "
                f"A taxa de exceções ({taxa_exc*100:.2f}%) é compatível com o nível de confiança de {confianca_bt*100:.0f}%."
            )
        else:
            st.error(
                f"❌ **Teste de Kupiec REPROVADO** (p-valor = {p_valor:.4f} < 0,05)  "
                f"— O modelo subestima ou superestima o risco. "
                f"Taxa observada: {taxa_exc*100:.2f}% vs esperada: {taxa_esp*100:.2f}%."
            )

        st.markdown("---")

        # ── Gráfico 1: Retornos reais vs VaR ──
        st.markdown("### 📈 Retornos Reais vs Limite do VaR")

        exc_datas = df_bt[df_bt["Exceção"]]["Data"]
        exc_rets  = df_bt[df_bt["Exceção"]]["Retorno Real"]

        fig1 = go.Figure()

        # Retornos reais
        fig1.add_trace(go.Scatter(
            x=df_bt["Data"], y=df_bt["Retorno Real"],
            mode="lines", name="Retorno Real",
            line=dict(color="#4A90D9", width=1),
        ))

        # Linha do VaR
        fig1.add_trace(go.Scatter(
            x=df_bt["Data"], y=df_bt["VaR (limite)"],
            mode="lines", name=f"VaR {confianca_bt*100:.0f}% (limite)",
            line=dict(color="#FF6600", width=1.5, dash="dash"),
        ))

        # Exceções em vermelho
        fig1.add_trace(go.Scatter(
            x=exc_datas, y=exc_rets,
            mode="markers", name="Exceções",
            marker=dict(color="#DC2626", size=7, symbol="x"),
        ))

        fig1.update_layout(
            template="plotly_dark" if "#0d0d0d" in st.session_state.get("tema","") else "plotly_white",
            height=420,
            xaxis_title="Data",
            yaxis_title="Retorno",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified",
        )
        st.plotly_chart(fig1, use_container_width=True)

        # ── Gráfico 2: Exceções ao longo do tempo (barras) ──
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### 📅 Exceções por Mês")
            df_bt["Mes"] = pd.to_datetime(df_bt["Data"]).dt.to_period("M").astype(str)
            exc_mes = df_bt.groupby("Mes")["Exceção"].sum().reset_index()
            exc_mes.columns = ["Mês", "Exceções"]

            fig2 = go.Figure(go.Bar(
                x=exc_mes["Mês"], y=exc_mes["Exceções"],
                marker_color="#DC2626",
                text=exc_mes["Exceções"],
                textposition="outside",
            ))
            fig2.update_layout(
                height=350,
                xaxis_title="Mês",
                yaxis_title="Nº de Exceções",
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig2, use_container_width=True)

        with c2:
            st.markdown("### 🥧 Proporção de Dias")
            fig3 = go.Figure(go.Pie(
                labels=["Dias sem exceção", "Exceções"],
                values=[T - N, N],
                marker_colors=["#16A34A", "#DC2626"],
                hole=0.4,
                textinfo="label+percent",
            ))
            fig3.update_layout(height=350)
            st.plotly_chart(fig3, use_container_width=True)

        # ── Tabela de exceções ──
        st.markdown("### 📋 Detalhes das Exceções")
        df_exc = df_bt[df_bt["Exceção"]].copy()
        df_exc["Retorno Real"] = df_exc["Retorno Real"].map("{:.4f}".format)
        df_exc["VaR (limite)"] = df_exc["VaR (limite)"].map("{:.4f}".format)
        df_exc = df_exc.drop(columns=["Exceção"])
        df_exc["Data"] = df_exc["Data"].astype(str)

        if df_exc.empty:
            st.success("Nenhuma exceção registrada no período!")
        else:
            st.dataframe(df_exc, use_container_width=True)

        # ── Interpretação do Teste de Kupiec ──
        st.markdown("---")
        st.markdown("### 📖 Sobre o Teste de Kupiec (POF)")
        st.markdown(f"""
        O **Teste de Kupiec** (1995) é o teste estatístico padrão para validar modelos de VaR.

        | Parâmetro | Valor |
        |-----------|-------|
        | Total de observações (T) | {T} |
        | Nº de exceções (N) | {N} |
        | Taxa observada (N/T) | {taxa_exc*100:.2f}% |
        | Taxa esperada (1 - IC) | {taxa_esp*100:.2f}% |
        | Estatística LR | {lr_stat:.4f} |
        | p-valor | {p_valor:.4f} |
        | Resultado | {"✅ Modelo adequado" if aprovado else "❌ Modelo inadequado"} |

        **Interpretação:** se o p-valor > 0,05, não rejeitamos H₀ — o modelo gera exceções
        na proporção esperada e é considerado **estatisticamente válido**.
        Um excesso de exceções indica que o VaR **subestima o risco** (modelo muito otimista).
        Poucas exceções indicam que o VaR **superestima o risco** (modelo muito conservador).
        """)


# ============================================================
# 7. NAVEGAÇÃO PRINCIPAL
# ============================================================

def main():
    st.sidebar.markdown("""
    <div style="text-align:center;padding:0.5rem 0 0.2rem 0;border-bottom:1px solid #FF6600;margin-bottom:0.8rem">
        <span style="color:#FF6600;font-family:'Courier New',monospace;font-size:1.1rem;font-weight:700;letter-spacing:3px">
        📡 VaR TERMINAL
        </span><br>
        <span style="color:#555;font-size:0.7rem;letter-spacing:2px">RISK MANAGEMENT SYSTEM</span>
    </div>
    """, unsafe_allow_html=True)

    paginas = {
        "▸ INÍCIO":                   page_home,
        "▸ UPLOAD DE POSIÇÕES":        page_upload,
        "▸ PARÂMETROS DE RISCO":       page_parametros,
        "▸ CÁLCULO DO VaR":            page_calculo,
        "▸ MONITORAMENTO DE LIMITES":  page_limites,
        "▸ DASHBOARD EXECUTIVO":       page_dashboard,
        "▸ BACKTESTING DO VaR":        page_backtesting,
    }

    pagina = st.sidebar.radio("", list(paginas.keys()))

    # Status das mesas na sidebar
    if "resultados" in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            "<span style='color:#FF6600;font-size:0.75rem;letter-spacing:2px'>■ STATUS DAS MESAS</span>",
            unsafe_allow_html=True
        )
        for mesa, r in st.session_state["resultados"].items():
            cor = "#00FF41" if r["Status"] == "Verde" else ("#FFD700" if r["Status"] == "Amarelo" else "#FF3333")
            util = r["Utilizacao"] * 100
            st.sidebar.markdown(
                f"<span style='color:{cor};font-size:0.8rem'>● {mesa[:14]:<14} {util:5.1f}%</span>",
                unsafe_allow_html=True
            )

    st.sidebar.markdown("---")
    params_sidebar = st.session_state.get("params", {})
    met = params_sidebar.get("metodologia", "—")
    cnf = params_sidebar.get("confianca", 0.95) * 100
    st.sidebar.markdown(
        f"<span style='color:#555;font-size:0.72rem;font-family:Courier New'>"
        f"MÉTODO : {met}<br>IC     : {cnf:.0f}%<br>"
        f"─────────────────────<br>"
        f"Gestão de Risco e Derivativos</span>",
        unsafe_allow_html=True
    )

    paginas[pagina]()


if __name__ == "__main__":
    main()
