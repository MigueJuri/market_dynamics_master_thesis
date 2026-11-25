# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: tesis
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Asignación long-only: 1/N, 1/σ e 1/σ² con EWMA y HAR-RV
#
# Este cuaderno compara tres esquemas de asignación long-only usando dos
# fuentes de volatilidad: EWMA histórica y predicción HAR-RV en log-vol.
# Todo el código de producción vive en ``thesis.portfolio``; aquí solo se
# orquesta la descarga, el backtest y las figuras.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis.config import TRADING_DAYS
from thesis.market_data import load_multiple_prices
from thesis.portfolio import (
    PORTFOLIO_NAMES,
    calculate_metrics,
    ewma_weights,
    plot_avg_weights,
    plot_equity_drawdown,
    plot_market_topology,
    plot_portfolio_composition,
    run_backtest,
)
from thesis.timeseries import log_returns

#: Raíz del repositorio (directorio que contiene ``styles/``).
_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "styles").is_dir())

#: Estilo de la tesis (155 mm, 12 pt, usetex); re-ejecutar tras editarlo.
STYLE = _ROOT / "styles" / "thesis.mplstyle"
plt.style.use(STYLE)
W = plt.rcParams["figure.figsize"][0]  # \textwidth en pulgadas


# %% [markdown]
# ## Configuración

# %%
CONFIG = {
    "tickers": [
        "SPY",
        "QQQ",
        "DIA",
        "IWM",
        "EEM",
        "FXI",
        "EWZ",
        "IEUR",
        "EFA",
        "ACWI",
        "ARKK",
        "XLE",
        "XLV",
        "XLF",
        "GLD",
        "URA",
        "SLV",
        "XLRE",
        "TLT",
        "DBC",
        "EDV",
        "AIA",
    ],
    "start": "2010-01-01",
    "end": "2026-01-01",
    "lookback": 252,
    "rebalance_freq": 21,
    "tx_cost": 0.005,
    "lambda_ewma": 0.94,
    "har_min_obs": 60,
    "har_lookback": 252 * 5,
}


# %% [markdown]
# ## Descarga de datos

# %%
prices = load_multiple_prices(
    CONFIG["tickers"], start=CONFIG["start"], end=CONFIG["end"]
)
prices = prices.dropna()
log_rets = log_returns(prices)

print(f"Muestra: {log_rets.index[0].date()} -> {log_rets.index[-1].date()}")
print(f"Días: {len(log_rets)}, activos: {log_rets.shape[1]}")

# Memoria efectiva del EWMA.
_w = ewma_weights(CONFIG["lookback"], CONFIG["lambda_ewma"])
print(
    f"EWMA λ={CONFIG['lambda_ewma']}: "
    f"vida media = {np.log(0.5) / np.log(CONFIG['lambda_ewma']):.1f} días, "
    f"N efectivo = {1 / np.sum(_w**2):.0f} de {CONFIG['lookback']}"
)


# %% [markdown]
# ## Backtest EWMA

# %%
port_rets_ewma, weights_ewma, turnover_ewma = run_backtest(
    log_rets,
    lookback=CONFIG["lookback"],
    rebalance_freq=CONFIG["rebalance_freq"],
    tx_cost=CONFIG["tx_cost"],
    volatility_model="ewma",
    use_ewma=True,
    lambda_=CONFIG["lambda_ewma"],
)


# %% [markdown]
# ## Backtest HAR-RV

# %%
port_rets_har, weights_har, turnover_har = run_backtest(
    log_rets,
    lookback=CONFIG["lookback"],
    rebalance_freq=CONFIG["rebalance_freq"],
    tx_cost=CONFIG["tx_cost"],
    volatility_model="har_rv",
    har_lookback=CONFIG["har_lookback"],
    har_min_obs=CONFIG["har_min_obs"],
)


# %% [markdown]
# ## Métricas

# %%
spy_rets = (np.exp(log_rets["SPY"]) - 1.0).loc[port_rets_ewma.index]

metrics_rows = []
for label, rets, to in [
    ("EWMA", port_rets_ewma, turnover_ewma),
    ("HAR-RV", port_rets_har, turnover_har),
]:
    for k in PORTFOLIO_NAMES:
        metrics_rows.append(
            calculate_metrics(rets[k], f"{label} {k}", to[k], TRADING_DAYS)
        )
metrics_rows.append(calculate_metrics(spy_rets, "SPY (Market)"))

df_metrics = pd.DataFrame(metrics_rows).set_index("Strategy")
print(df_metrics.round(4).to_string())


# %% [markdown]
# ## Figuras

# %%
# Curva de capital y drawdown (EWMA).
fig = plot_equity_drawdown(port_rets_ewma, benchmark=spy_rets)
fig.savefig("outputs/portfolio_ewma_equity_drawdown.pdf")
plt.show()


# %%
# Curva de capital y drawdown (HAR-RV).
fig = plot_equity_drawdown(port_rets_har, benchmark=spy_rets)
fig.savefig("outputs/portfolio_har_rv_equity_drawdown.pdf")
plt.show()


# %%
# Evolución de pesos 1/σ² (EWMA).
fig = plot_portfolio_composition(
    weights_ewma["1/σ²"], title=r"Evolución de pesos (1/$\sigma^2$, EWMA)"
)
fig.savefig("outputs/portfolio_ewma_weights_1var.pdf")
plt.show()


# %%
# Composición promedio (EWMA).
fig = plot_avg_weights(weights_ewma)
fig.savefig("outputs/portfolio_ewma_avg_weights.pdf")
plt.show()


# %%
# Composición promedio (HAR-RV).
fig = plot_avg_weights(weights_har)
fig.savefig("outputs/portfolio_har_rv_avg_weights.pdf")
plt.show()


# %%
# Topología de mercado por régimen.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, 0.45 * W), sharey=True)
plot_market_topology(
    log_rets,
    start_date="2019-01-01",
    end_date="2019-12-31",
    ax=ax1,
    title="Régimen Normal (2019)",
)
plot_market_topology(
    log_rets,
    start_date="2020-01-01",
    end_date="2020-04-30",
    ax=ax2,
    title="Régimen de Crisis (Q1 2020)",
)
fig.savefig("outputs/portfolio_market_topology.pdf")
plt.show()


# %%
# Pesos del último rebalanceo (cartera actual recomendada).
last_w = pd.DataFrame(
    {k: weights_ewma[k].iloc[-1] for k in PORTFOLIO_NAMES}
).sort_values("1/σ²", ascending=False)
print(f"\nFecha del último rebalanceo: {weights_ewma['1/N'].index[-1].date()}")
print(last_w.round(4).to_string())


# %%
if __name__ == "__main__":
    print("Ejecutado como script.")

# %%
