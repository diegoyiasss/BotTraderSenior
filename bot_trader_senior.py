import argparse
import html
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mplfinance as mpf
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

# ==============================================================================
# 1. CONFIGURACIÓN GENERAL Y MULTI-MERCADO
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "bot_trader.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("TraderSeniorBot")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

DATA_SOURCE = os.getenv("DATA_SOURCE", "yfinance").strip().lower()
STRICT_KILLZONE = os.getenv("STRICT_KILLZONE", "true").strip().lower() == "true"
SEND_YELLOW_ALERTS = os.getenv("SEND_YELLOW_ALERTS", "false").strip().lower() == "true"
ACTIVE_SYMBOLS = [
    s.strip().upper()
    for s in os.getenv("SYMBOLS", "EURUSD,GBPUSD,USDJPY,XAUUSD,NAS100").split(",")
    if s.strip()
]

# Parámetros técnicos por activo (escala de pip/punto, decimales y buffer de SL)
ASSETS_CATALOG = {
    "EURUSD": {"yf": "EURUSD=X", "mt5": "EURUSD", "pip": 0.0001, "decimals": 5, "buffer_pips": 2.0},
    "GBPUSD": {"yf": "GBPUSD=X", "mt5": "GBPUSD", "pip": 0.0001, "decimals": 5, "buffer_pips": 2.5},
    "USDJPY": {"yf": "USDJPY=X", "mt5": "USDJPY", "pip": 0.01,   "decimals": 3, "buffer_pips": 2.5},
    "XAUUSD": {"yf": "GC=F",     "mt5": "XAUUSD", "pip": 0.1,    "decimals": 2, "buffer_pips": 15.0},
    "NAS100": {"yf": "NQ=F",     "mt5": "NAS100", "pip": 1.0,    "decimals": 2, "buffer_pips": 10.0},
}

NY_TZ = ZoneInfo("America/New_York")
STATE_FILE = BASE_DIR / "last_signals.json"
STATE_LOCK = threading.Lock()
GEMINI_MODEL = "gemini-3.8-flash"

# ==============================================================================
# 2. SYSTEM PROMPT: PROTOCOLO TRADER-SENIOR + MANUAL ARJO / DODGYSDD / TTRADES
# ==============================================================================
TRADER_SENIOR_PROMPT = """
Eres un Trader Institucional Senior y Analista Técnico Jefe (>15 años de experiencia).
Aplicas estrictamente el Manual Operativo Institucional (Arjo + DodgysDD + TTrades) en 1m-5m:

CHECKLIST SECUENCIAL DE 7 FILTROS OBLIGATORIOS:
1. Draw on Liquidity (DOL): Objetivo claro (Tier S: EQH/EQL >3 velas o Data Wicks 8:30 AM NY; Tier A: Trendline >=3 toques o EQH/EQL cortos; Tier B: FVG viejos o NWOG/NDOG).
2. Filtro de Rango (Prima vs. Descuento): Compras SOLO en Descuento (<50% rango); Ventas SOLO en Prima (>50% rango).
3. Respeto de Zona HTF (15m/1h): Interacción con PD Array HTF sin cierre de cuerpo por fuera.
4. Validación del Sweep (Prueba de Arjo): Perforación mayoritaria con mecha y cierre dentro del rango, SIN ningún FVG solapado sobre el nivel tomado. Si existe FVG solapado es un LIQUIDITY RUN -> semáforo ROJO (prohibido revertir).
5. Derrota de la Matriz Impulsora: El OB/FVG previo que impulsó el barrido debe fallar y ser quebrado (salir de Zona Roja hacia Zona Verde).
6. Confirmación del IFVG (TTrades/DodgysDD): Vela de desplazamiento con cuerpo sólido cerrando al otro lado de un FVG previo singular y evidente (SIBI->IFVG Alcista / BISI->IFVG Bajista).
7. Parámetros de Riesgo: SL detrás de la mecha del barrido; R:R objetivo 1:1.5 a 1:2; Break Even mecánico al superar con aceleración el swing interno previo; salida temprana inmediata a mercado si una vela cierra con cuerpo al otro lado del IFVG.

REGLAS DE RESPUESTA:
- Cero emociones. Sé estrictamente frío y matemático.
- Si NO se cumplen los 7 filtros, NO inventes señales: responde AMARILLO (si falta cierre/IFVG pero hay barrido/zona) o ROJO (si es rango muerto, tierra de nadie o Liquidity Run).
- Devuelve EXCLUSIVAMENTE un JSON válido con esta estructura exacta:
{
  "semaforo": "VERDE" | "AMARILLO" | "ROJO",
  "direccion": "LONG" | "SHORT" | "NEUTRO",
  "checklist_cumplidos": 0,
  "resumen_ejecutivo": "Máximo 2 líneas claras con el estado actual y acción inmediata.",
  "contexto_estructura": "Sesión, Prima/Descuento, HTF 15m/1h y lectura de las últimas velas.",
  "mapa_liquidez": "DOL (Tier), Sweep vs Run (Prueba de Arjo), Zona Roja/Verde e IFVG/CE 50%.",
  "plan_trading": "Escenario principal paso a paso y contingencia.",
  "modalidad_entrada": "Modalidad A (Al cierre)" | "Modalidad B (Retesteo CE 50%)" | "NINGUNA",
  "precio_entrada": 0.0,
  "ifvg_alto": 0.0,
  "ifvg_bajo": 0.0,
  "ce_50": 0.0,
  "stop_loss": 0.0,
  "be_trigger": 0.0,
  "tp1": 0.0,
  "tp2": 0.0,
  "rr_estimado": "1:1.8",
  "nivel_barrido": 0.0,
  "mecha_barrido": 0.0,
  "tipo_barrido": "SSL_SWEEP" | "BSL_SWEEP" | "NINGUNO",
  "dol_objetivo_precio": 0.0,
  "dol_etiqueta": "Ej. EQH Tier S / Data Wick 8:30",
  "invalidacion_temprana": "Nivel exacto de cierre de vela que obliga a cerrar a mercado."
}
"""

# ==============================================================================
# 3. DESCARGA DE DATOS (MT5 PRIMARIO / YFINANCE RESPALDO)
# ==============================================================================
def fetch_ohlcv_yfinance(yf_symbol: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(
        tickers=yf_symbol,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        raise RuntimeError(f"yfinance sin datos para {yf_symbol} ({interval}).")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close"]].dropna().copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(NY_TZ)
    else:
        df.index = df.index.tz_convert(NY_TZ)
    return df


def fetch_ohlcv_mt5(mt5_symbol: str, timeframe_str: str, bars: int = 180) -> pd.DataFrame:
    import MetaTrader5 as mt5

    tf_map = {
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "1h": mt5.TIMEFRAME_H1,
    }
    if not mt5.initialize():
        raise RuntimeError(f"Error MT5: {mt5.last_error()}")

    rates = mt5.copy_rates_from_pos(mt5_symbol, tf_map[timeframe_str], 0, bars)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"MT5 sin velas para {mt5_symbol} ({timeframe_str}).")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(NY_TZ)
    df.set_index("time", inplace=True)
    df.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"},
        inplace=True,
    )
    return df[["Open", "High", "Low", "Close"]].dropna().copy()


def fetch_all_mt5(symbols_list: list[str]) -> dict:
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError(f"No se pudo conectar a MT5: {mt5.last_error()}")

    all_info = mt5.symbols_get() or []
    available_names = [s.name for s in all_info]
    results = {}

    try:
        for sym in symbols_list:
            cfg = ASSETS_CATALOG.get(sym)
            if not cfg:
                continue
            base_name = cfg["mt5"]
            # Autodetectar nombre exacto en el broker (ej. EURUSD, EURUSDm, EURUSD#, GOLD, USTEC)
            aliases = [base_name]
            if sym == "XAUUSD":
                aliases += ["GOLD", "XAUUSDm", "XAUUSD#"]
            elif sym == "NAS100":
                aliases += ["USTEC", "USTECm", "US100", "NAS100m", "NDX"]
            else:
                aliases += [f"{base_name}m", f"{base_name}#", f"{base_name}."]

            matched = next((a for a in aliases if a in available_names), None)
            if not matched:
                matched = next((n for n in available_names if n.startswith(base_name)), None)
            if not matched:
                log.warning("[%s] Simbolo no encontrado en Observacion del Mercado de MT5.", sym)
                continue

            mt5.symbol_select(matched, True)
            data_tfs = {}
            for tf_str, tf_const, bars in [("5m", mt5.TIMEFRAME_M5, 180), ("15m", mt5.TIMEFRAME_M15, 120), ("1h", mt5.TIMEFRAME_H1, 72)]:
                rates = mt5.copy_rates_from_pos(matched, tf_const, 0, bars)
                if rates is None or len(rates) == 0:
                    raise RuntimeError(f"Sin velas MT5 para {matched} ({tf_str})")
                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(NY_TZ)
                df.set_index("time", inplace=True)
                df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
                data_tfs[tf_str] = df[["Open", "High", "Low", "Close"]].dropna().iloc[:-1]

            results[sym] = (data_tfs["5m"], data_tfs["15m"], data_tfs["1h"])
            log.info("[%s] Velas cargadas desde MT5 (%s).", sym, matched)
    finally:
        mt5.shutdown()

    return results

def load_market_data(asset_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if DATA_SOURCE == "mt5":
        try:
            df_5m = fetch_ohlcv_mt5(asset_cfg["mt5"], "5m", 180)
            df_15m = fetch_ohlcv_mt5(asset_cfg["mt5"], "15m", 120)
            df_1h = fetch_ohlcv_mt5(asset_cfg["mt5"], "1h", 72)
            return df_5m.iloc[:-1], df_15m.iloc[:-1], df_1h.iloc[:-1]
        except Exception as exc:
            log.warning("[%s] Fallo MT5 (%s), usando yfinance.", asset_cfg["mt5"], exc)

    df_5m = fetch_ohlcv_yfinance(asset_cfg["yf"], "5m", "2d")
    df_15m = fetch_ohlcv_yfinance(asset_cfg["yf"], "15m", "5d")
    df_1h = fetch_ohlcv_yfinance(asset_cfg["yf"], "60m", "10d")
    return df_5m.iloc[:-1], df_15m.iloc[:-1], df_1h.iloc[:-1]


# ==============================================================================
# 4. MOTOR DETERMINÍSTICO INSTITUCIONAL (ARJO + DODGYSDD + TTRADES)
# ==============================================================================
def get_session_status(ts_ny: datetime) -> dict:
    hour = ts_ny.hour + ts_ny.minute / 60.0
    if 2.0 <= hour < 5.0:
        return {"session": "London Killzone", "active_killzone": True}
    if 7.0 <= hour < 11.5:
        return {"session": "New York Killzone (incl. 8:30 AM News)", "active_killzone": True}
    if 13.0 <= hour < 15.5:
        return {"session": "NY PM Session", "active_killzone": True}
    if 19.0 <= hour or hour < 2.0:
        return {"session": "Asian Range / Baja Liquidez", "active_killzone": False}
    return {"session": "Transición entre sesiones", "active_killzone": False}


def detect_fvgs(df: pd.DataFrame, pip_size: float, decimals: int) -> list[dict]:
    raw_fvgs = []
    highs = df["High"].values
    lows = df["Low"].values
    times = df.index

    for i in range(2, len(df)):
        if lows[i] > highs[i - 2]:
            top, bottom = float(lows[i]), float(highs[i - 2])
            if (top - bottom) >= 0.5 * pip_size:
                raw_fvgs.append({
                    "type": "BISI",
                    "index": i,
                    "time": str(times[i]),
                    "top": round(top, decimals),
                    "bottom": round(bottom, decimals),
                    "ce": round((top + bottom) / 2, decimals),
                })
        elif highs[i] < lows[i - 2]:
            top, bottom = float(lows[i - 2]), float(highs[i])
            if (top - bottom) >= 0.5 * pip_size:
                raw_fvgs.append({
                    "type": "SIBI",
                    "index": i,
                    "time": str(times[i]),
                    "top": round(top, decimals),
                    "bottom": round(bottom, decimals),
                    "ce": round((top + bottom) / 2, decimals),
                })

    merged = []
    for fvg in raw_fvgs:
        if (
            merged
            and merged[-1]["type"] == fvg["type"]
            and (fvg["index"] - merged[-1]["index"]) <= 2
        ):
            merged[-1]["top"] = round(max(merged[-1]["top"], fvg["top"]), decimals)
            merged[-1]["bottom"] = round(min(merged[-1]["bottom"], fvg["bottom"]), decimals)
            merged[-1]["ce"] = round((merged[-1]["top"] + merged[-1]["bottom"]) / 2, decimals)
            merged[-1]["index"] = fvg["index"]
            merged[-1]["time"] = fvg["time"]
        else:
            merged.append(fvg)
    return merged


def find_swings(df: pd.DataFrame, decimals: int, left: int = 2, right: int = 2) -> tuple[list[dict], list[dict]]:
    swing_highs, swing_lows = [], []
    highs = df["High"].values
    lows = df["Low"].values
    times = df.index

    for i in range(left, len(df) - right):
        if highs[i] == max(highs[i - left : i + right + 1]):
            swing_highs.append({"index": i, "time": str(times[i]), "price": round(float(highs[i]), decimals)})
        if lows[i] == min(lows[i - left : i + right + 1]):
            swing_lows.append({"index": i, "time": str(times[i]), "price": round(float(lows[i]), decimals)})
    return swing_highs, swing_lows


def find_dol_targets(df_15m: pd.DataFrame, df_5m: pd.DataFrame, current_price: float, pip_size: float, decimals: int) -> dict:
    sh, sl = find_swings(df_15m, decimals, left=2, right=2)
    tolerance = 2.0 * pip_size
    eqh, eql = [], []

    for i in range(len(sh)):
        for j in range(i + 1, len(sh)):
            if abs(sh[i]["price"] - sh[j]["price"]) <= tolerance:
                sep = sh[j]["index"] - sh[i]["index"]
                tier = "Tier S" if sep > 3 else "Tier A"
                level = round(max(sh[i]["price"], sh[j]["price"]), decimals)
                if level > current_price:
                    eqh.append({"level": level, "tier": tier, "candles_apart": sep})

    for i in range(len(sl)):
        for j in range(i + 1, len(sl)):
            if abs(sl[i]["price"] - sl[j]["price"]) <= tolerance:
                sep = sl[j]["index"] - sl[i]["index"]
                tier = "Tier S" if sep > 3 else "Tier A"
                level = round(min(sl[i]["price"], sl[j]["price"]), decimals)
                if level < current_price:
                    eql.append({"level": level, "tier": tier, "candles_apart": sep})

    data_wicks = []
    news_candles = df_5m[(df_5m.index.hour == 8) & (df_5m.index.minute == 30)]
    for ts, row in news_candles.tail(2).iterrows():
        data_wicks.append({
            "time": str(ts),
            "high": round(float(row["High"]), decimals),
            "low": round(float(row["Low"]), decimals),
            "tier": "Tier S (Data Wick 8:30 NY)",
        })

    return {
        "bsl_eqh": sorted(eqh, key=lambda x: x["level"])[:3],
        "ssl_eql": sorted(eql, key=lambda x: x["level"], reverse=True)[:3],
        "data_wicks_830_ny": data_wicks,
    }


def analyze_institutional_structure(
    symbol_name: str,
    asset_cfg: dict,
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
) -> dict:
    pip_size = asset_cfg["pip"]
    decimals = asset_cfg["decimals"]
    buffer_pips = asset_cfg["buffer_pips"]

    last_candle = df_5m.iloc[-1]
    last_time = df_5m.index[-1]
    current_price = round(float(last_candle["Close"]), decimals)

    session_info = get_session_status(last_time)

    recent_1h = df_1h.tail(24)
    range_high = round(float(recent_1h["High"].max()), decimals)
    range_low = round(float(recent_1h["Low"].min()), decimals)
    equilibrium = round((range_high + range_low) / 2, decimals)
    zone = "DISCOUNT" if current_price < equilibrium else "PREMIUM"

    dol = find_dol_targets(df_15m.tail(80), df_5m.tail(160), current_price, pip_size, decimals)

    recent_5m = df_5m.tail(40).copy()
    fvgs_5m = detect_fvgs(recent_5m, pip_size, decimals)
    sh_5m, sl_5m = find_swings(recent_5m.iloc[:-6], decimals, left=2, right=2)

    sweep_detected = None
    last_window = recent_5m.iloc[-8:]

    if sl_5m:
        key_low = min(s["price"] for s in sl_5m[-4:])
        wick_below = last_window["Low"].min() < key_low
        closed_inside = last_window.iloc[-1]["Close"] > key_low
        overlapping_sibi = any(
            f["type"] == "SIBI" and f["bottom"] <= key_low <= f["top"]
            for f in fvgs_5m
            if f["index"] >= len(recent_5m) - 10
        )
        if wick_below:
            sweep_detected = {
                "side": "SSL_SWEEP" if (closed_inside and not overlapping_sibi) else "SSL_LIQUIDITY_RUN",
                "swept_level": key_low,
                "extreme_wick": round(float(last_window["Low"].min()), decimals),
                "overlapping_fvg_arjo": overlapping_sibi,
                "closed_back_inside": bool(closed_inside),
            }

    if sh_5m:
        key_high = max(s["price"] for s in sh_5m[-4:])
        wick_above = last_window["High"].max() > key_high
        closed_inside = last_window.iloc[-1]["Close"] < key_high
        overlapping_bisi = any(
            f["type"] == "BISI" and f["bottom"] <= key_high <= f["top"]
            for f in fvgs_5m
            if f["index"] >= len(recent_5m) - 10
        )
        if wick_above:
            sweep_detected = {
                "side": "BSL_SWEEP" if (closed_inside and not overlapping_bisi) else "BSL_LIQUIDITY_RUN",
                "swept_level": key_high,
                "extreme_wick": round(float(last_window["High"].max()), decimals),
                "overlapping_fvg_arjo": overlapping_bisi,
                "closed_back_inside": bool(closed_inside),
            }

    ifvg_confirmed = None
    recent_close = float(recent_5m.iloc[-1]["Close"])
    prev_close = float(recent_5m.iloc[-2]["Close"])

    for fvg in reversed(fvgs_5m):
        if fvg["index"] < len(recent_5m) - 15 or fvg["index"] >= len(recent_5m) - 1:
            continue
        if fvg["type"] == "SIBI" and recent_close > fvg["top"] and prev_close <= fvg["top"]:
            internal_swing_high = round(float(recent_5m.iloc[fvg["index"] : -1]["High"].max()), decimals)
            ifvg_confirmed = {
                "type": "IFVG_ALCISTA",
                "top": fvg["top"],
                "bottom": fvg["bottom"],
                "ce_50": fvg["ce"],
                "internal_be_swing": internal_swing_high,
                "fvg_origin_time": fvg["time"],
            }
            break
        if fvg["type"] == "BISI" and recent_close < fvg["bottom"] and prev_close >= fvg["bottom"]:
            internal_swing_low = round(float(recent_5m.iloc[fvg["index"] : -1]["Low"].min()), decimals)
            ifvg_confirmed = {
                "type": "IFVG_BAJISTA",
                "top": fvg["top"],
                "bottom": fvg["bottom"],
                "ce_50": fvg["ce"],
                "internal_be_swing": internal_swing_low,
                "fvg_origin_time": fvg["time"],
            }
            break

    candidate_setup = None
    if ifvg_confirmed and sweep_detected:
        buf = buffer_pips * pip_size
        if ifvg_confirmed["type"] == "IFVG_ALCISTA" and sweep_detected["side"] == "SSL_SWEEP":
            sl = round(sweep_detected["extreme_wick"] - buf, decimals)
            entry_a = current_price
            entry_b = ifvg_confirmed["ce_50"]
            risk_b = max(entry_b - sl, pip_size)
            candidate_setup = {
                "direction": "LONG",
                "entry_modalidad_A": entry_a,
                "entry_modalidad_B_CE50": entry_b,
                "stop_loss": sl,
                "be_trigger": ifvg_confirmed["internal_be_swing"],
                "tp1_1_5R": round(entry_b + 1.5 * risk_b, decimals),
                "tp2_2_0R": round(entry_b + 2.0 * risk_b, decimals),
                "invalidation_close_below": ifvg_confirmed["bottom"],
            }
        elif ifvg_confirmed["type"] == "IFVG_BAJISTA" and sweep_detected["side"] == "BSL_SWEEP":
            sl = round(sweep_detected["extreme_wick"] + buf, decimals)
            entry_a = current_price
            entry_b = ifvg_confirmed["ce_50"]
            risk_b = max(sl - entry_b, pip_size)
            candidate_setup = {
                "direction": "SHORT",
                "entry_modalidad_A": entry_a,
                "entry_modalidad_B_CE50": entry_b,
                "stop_loss": sl,
                "be_trigger": ifvg_confirmed["internal_be_swing"],
                "tp1_1_5R": round(entry_b - 1.5 * risk_b, decimals),
                "tp2_2_0R": round(entry_b - 2.0 * risk_b, decimals),
                "invalidation_close_above": ifvg_confirmed["top"],
            }

    return {
        "symbol": symbol_name,
        "decimals": decimals,
        "timestamp_ny": last_time.strftime("%Y-%m-%d %H:%M %Z"),
        "current_price": current_price,
        "session": session_info,
        "htf_dealing_range_24h": {
            "high": range_high,
            "low": range_low,
            "equilibrium_50": equilibrium,
            "current_zone": zone,
        },
        "dol_targets": dol,
        "sweep_analysis_arjo": sweep_detected,
        "ifvg_confirmed": ifvg_confirmed,
        "candidate_setup": candidate_setup,
        "recent_fvgs_5m": fvgs_5m[-5:],
    }


# ==============================================================================
# 5. VALIDACIÓN ULTRARRÁPIDA CON GEMINI (HTTP DIRECTO SIN BLOQUEOS)
# ==============================================================================
def validate_with_gemini(df_5m: pd.DataFrame, df_15m: pd.DataFrame, analysis: dict) -> dict:
    import time
    dec = analysis["decimals"]
    candles_5m_str = df_5m.tail(25).round(dec).to_string()
    candles_15m_str = df_15m.tail(12).round(dec).to_string()

    user_prompt = (
        "Analiza los siguientes datos exactos de mercado y aplica el checklist secuencial de 7 filtros:\n\n"
        "=== RESUMEN DETERMINISTICO PRE-CALCULADO ===\n"
        + json.dumps(analysis, indent=2, ensure_ascii=False)
        + "\n\n=== ULTIMAS 12 VELAS HTF (15m - Hora NY) ===\n"
        + candles_15m_str
        + "\n\n=== ULTIMAS 25 VELAS OPERATIVAS (5m - Hora NY) ===\n"
        + candles_5m_str
        + "\n\nEmite tu veredicto institucional estrictamente en el formato JSON solicitado."
    )

    models_to_try = [GEMINI_MODEL, "gemini-3.8-flash-lite", "gemini-3.8-pro"]
    last_error = None

    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "systemInstruction": {"parts": [{"text": TRADER_SENIOR_PROMPT}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.1,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }

        for attempt in range(2):
            try:
                resp = requests.post(url, json=payload, timeout=20)
                if resp.status_code == 400 and "thinkingConfig" in resp.text:
                    payload["generationConfig"].pop("thinkingConfig", None)
                    resp = requests.post(url, json=payload, timeout=20)

                if resp.status_code in (429, 503):
                    log.warning("[%s] %s saturado (%s), reintentando...", analysis["symbol"], model_name, resp.status_code)
                    time.sleep(2 * (attempt + 1))
                    continue

                if resp.status_code == 404:
                    break

                if not resp.ok:
                    raise RuntimeError(f"Error Gemini {resp.status_code}: {resp.text}")

                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text)
            except Exception as exc:
                last_error = exc
                time.sleep(1)

    log.warning("[%s] API Gemini saturada (%s). Usando respaldo deterministico local.", analysis["symbol"], last_error)
    cand = analysis.get("candidate_setup") or {}
    ifvg = analysis.get("ifvg_confirmed") or {}
    sweep = analysis.get("sweep_analysis_arjo") or {}
    is_green = bool(cand) and analysis["session"]["active_killzone"]
    semaforo = "VERDE" if is_green else ("AMARILLO" if (sweep or ifvg) else "ROJO")
    return {
        "semaforo": semaforo,
        "direccion": cand.get("direction", "NEUTRO"),
        "checklist_cumplidos": 7 if is_green else (4 if (sweep or ifvg) else 1),
        "resumen_ejecutivo": f"Analisis deterministico (respaldo por alta demanda API): Precio en {analysis['htf_dealing_range_24h']['current_zone']} ({analysis['session']['session']}).",
        "contexto_estructura": f"Rango HTF 24h: {analysis['htf_dealing_range_24h']['low']} - {analysis['htf_dealing_range_24h']['high']} (EQ: {analysis['htf_dealing_range_24h']['equilibrium_50']}).",
        "mapa_liquidez": f"Sweep: {sweep.get('side', 'Ninguno')} | IFVG: {ifvg.get('type', 'Ninguno')}.",
        "plan_trading": "Esperar confirmacion completa de los 7 filtros del manual en Killzone antes de ejecutar.",
        "modalidad_entrada": "Modalidad B (Retesteo CE 50%)" if cand else "NINGUNA",
        "precio_entrada": cand.get("entry_modalidad_B_CE50", 0.0),
        "ifvg_alto": ifvg.get("top", 0.0),
        "ifvg_bajo": ifvg.get("bottom", 0.0),
        "ce_50": ifvg.get("ce_50", 0.0),
        "stop_loss": cand.get("stop_loss", 0.0),
        "be_trigger": cand.get("be_trigger", 0.0),
        "tp1": cand.get("tp1_1_5R", 0.0),
        "tp2": cand.get("tp2_2_0R", 0.0),
        "rr_estimado": "1:1.5 - 1:2.0" if cand else "N/A",
        "nivel_barrido": sweep.get("swept_level", 0.0),
        "mecha_barrido": sweep.get("extreme_wick", 0.0),
        "tipo_barrido": sweep.get("side", "NINGUNO"),
        "dol_objetivo_precio": (analysis["dol_targets"]["bsl_eqh"][0]["level"] if analysis["dol_targets"]["bsl_eqh"] else 0.0),
        "dol_etiqueta": "DOL 15m",
        "invalidacion_temprana": f"Cierre con cuerpo mas alla de {cand.get('invalidation_close_below') or cand.get('invalidation_close_above') or 'N/A'}",
    }
# ==============================================================================
# 6. GENERADOR DE GRÁFICO INSTITUCIONAL (ESTILO TRADINGVIEW)
# ==============================================================================
def generate_setup_chart(df_5m: pd.DataFrame, analysis: dict, verdict: dict) -> Path:
    symbol = analysis["symbol"]
    dec = analysis["decimals"]
    fmt = f".{dec}f"
    output_path = BASE_DIR / f"setup_chart_{symbol}.png"
    plot_df = df_5m.tail(55).copy()

    mc = mpf.make_marketcolors(
        up="#26a69a",
        down="#ef5350",
        edge="inherit",
        wick="inherit",
        volume="in",
    )
    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=mc,
        facecolor="#131722",
        figcolor="#131722",
        gridcolor="#1e222d",
        gridstyle="--",
        rc={"font.size": 8, "axes.edgecolor": "#2a2e39"},
    )

    semaforo = verdict.get("semaforo", "ROJO")
    direccion = verdict.get("direccion", "NEUTRO")
    rr = verdict.get("rr_estimado", "N/A")
    filtros = verdict.get("checklist_cumplidos", 0)
    zona_es = "DESCUENTO" if analysis["htf_dealing_range_24h"]["current_zone"] == "DISCOUNT" else "PRIMA"

    fig, axes = mpf.plot(
        plot_df,
        type="candle",
        style=style,
        ylabel="Precio",
        figsize=(13.5, 7.5),
        returnfig=True,
        tight_layout=False,
    )
    ax = axes[0]
    n_bars = len(plot_df)
    ymin, ymax = ax.get_ylim()
    y_span = ymax - ymin

    # Titulo limpio fuera del area de velas para evitar solapamientos
    fig.suptitle(
        f"{symbol} (5m) | {semaforo} ({filtros}/7) - {direccion} | Zona: {zona_es} | {analysis['timestamp_ny']}",
        color="#eceff1",
        fontsize=11,
        fontweight="bold",
        y=0.96,
    )

    # 1. Linea de Equilibrio 50% HTF (solo si entra en la escala visible)
    eq = analysis["htf_dealing_range_24h"]["equilibrium_50"]
    if ymin <= eq <= ymax:
        ax.axhline(eq, color="#90a4ae", linestyle="-.", linewidth=0.9, alpha=0.65)
        ax.text(
            0.5,
            eq + y_span * 0.008,
            f"Equilibrio 50% HTF ({zona_es}): {format(eq, fmt)}",
            color="#b0bec5",
            fontsize=7.5,
            va="bottom",
        )

    # 2. Dibujar FVG recientes desde la VELA EXACTA donde se formaron hasta el borde derecho
    for fvg in analysis.get("recent_fvgs_5m", [])[-3:]:
        f_top, f_bot = fvg["top"], fvg["bottom"]
        if f_top <= f_bot:
            continue
        try:
            fvg_ts = pd.Timestamp(fvg["time"])
            if fvg_ts < plot_df.index[0]:
                start_x = 0
            else:
                start_x = int(plot_df.index.get_indexer([fvg_ts], method="nearest")[0])
        except Exception:
            start_x = max(0, n_bars - 15)

        width = max(2, (n_bars - 1) - start_x)
        is_bull = fvg["type"] == "BISI"
        color_fvg = "#26a69a" if is_bull else "#ef5350"
        label_fvg = "FVG Alcista" if is_bull else "FVG Bajista"

        ax.add_patch(
            Rectangle(
                (start_x, f_bot),
                width,
                f_top - f_bot,
                facecolor=color_fvg,
                alpha=0.14,
                edgecolor=color_fvg,
                linestyle=":",
                linewidth=0.9,
            )
        )
        ax.text(
            start_x + 0.3,
            f_top,
            label_fvg,
            color=color_fvg,
            fontsize=7.5,
            alpha=0.9,
            va="bottom",
        )

    # 3. Anotaciones institucionales para AMARILLO y VERDE
    dol_note = ""
    if semaforo in ("VERDE", "AMARILLO"):
        sweep_info = analysis.get("sweep_analysis_arjo") or {}
        swept_lvl = float(verdict.get("nivel_barrido") or sweep_info.get("swept_level") or 0)
        wick_lvl = float(verdict.get("mecha_barrido") or sweep_info.get("extreme_wick") or 0)
        sweep_type = str(verdict.get("tipo_barrido") or sweep_info.get("side") or "")

        sweep_labels = {
            "SSL_SWEEP": "Barrido de Minimos (Sweep)",
            "BSL_SWEEP": "Barrido de Maximos (Sweep)",
            "SSL_LIQUIDITY_RUN": "Ruptura Bajista (Liquidity Run - No Revertir)",
            "BSL_LIQUIDITY_RUN": "Ruptura Alcista (Liquidity Run - No Revertir)",
        }
        sweep_text = sweep_labels.get(sweep_type, sweep_type)
        is_ssl = "SSL" in sweep_type

        if swept_lvl > 0 and ymin <= swept_lvl <= ymax:
            line_col = "#ff9800" if "SWEEP" in sweep_type else "#ff5252"
            ax.axhline(swept_lvl, color=line_col, linestyle="--", linewidth=1.1, alpha=0.9)
            ax.text(
                0.5,
                swept_lvl + y_span * 0.008,
                f"{sweep_text}: {format(swept_lvl, fmt)}",
                color=line_col,
                fontsize=8,
                fontweight="bold",
                va="bottom",
            )

        # Ubicar la flecha EXACTAMENTE en la vela que hizo la mecha del barrido
        if wick_lvl > 0 and ymin <= wick_lvl <= ymax:
            tail_12 = plot_df.iloc[-12:]
            if is_ssl:
                wick_ts = tail_12["Low"].idxmin()
                wick_y = float(tail_12.loc[wick_ts, "Low"])
                marker_style, va_dir, y_off = "^", "top", -y_span * 0.02
            else:
                wick_ts = tail_12["High"].idxmax()
                wick_y = float(tail_12.loc[wick_ts, "High"])
                marker_style, va_dir, y_off = "v", "bottom", y_span * 0.02

            wick_x = int(plot_df.index.get_loc(wick_ts))
            ax.plot([wick_x], [wick_y + y_off * 0.4], marker=marker_style, color="#ff9800", markersize=8)
            ax.text(
                max(0, wick_x - 6),
                wick_y + y_off,
                f"Mecha: {format(wick_y, fmt)}",
                color="#ffcc80",
                fontsize=7.5,
                fontweight="bold",
                va=va_dir,
            )

        # Objetivo de Liquidez (DOL): solo dibujar linea si cae dentro del grafico visible
        dol_lvl = float(verdict.get("dol_objetivo_precio") or 0)
        if dol_lvl > 0:
            if ymin <= dol_lvl <= ymax:
                ax.axhline(dol_lvl, color="#00e5ff", linestyle="-.", linewidth=1.1, alpha=0.85)
                ax.text(
                    0.5,
                    dol_lvl - y_span * 0.01,
                    f"Objetivo Liquidez (DOL): {format(dol_lvl, fmt)}",
                    color="#4dd0e1",
                    fontsize=8,
                    fontweight="bold",
                    va="top",
                )
            else:
                dol_note = f"Objetivo DOL (HTF): {format(dol_lvl, fmt)}"

        # Zona IFVG y Consequent Encroachment (CE 50%)
        ifvg_top = float(verdict.get("ifvg_alto") or 0)
        ifvg_bot = float(verdict.get("ifvg_bajo") or 0)
        ce_50 = float(verdict.get("ce_50") or 0)
        if ifvg_top > 0 and ifvg_bot > 0 and ifvg_top > ifvg_bot:
            ax.add_patch(
                Rectangle(
                    (n_bars - 14, ifvg_bot),
                    14,
                    ifvg_top - ifvg_bot,
                    facecolor="#2962ff",
                    alpha=0.28,
                    edgecolor="#448aff",
                    linewidth=1.2,
                )
            )
            ax.text(n_bars - 13.8, ifvg_top, "IFVG Confirmado", color="#82b1ff", fontsize=8, fontweight="bold", va="bottom")
            if ce_50 > 0:
                ax.axhline(ce_50, color="#ffd54f", linestyle="--", linewidth=1.0, alpha=0.9)
                ax.text(0.5, ce_50, f"Nivel 50% IFVG (Entrada B): {format(ce_50, fmt)}", color="#ffd54f", fontsize=7.5, va="bottom")

        # Cajas de Posicion (Entry -> TP en verde, Entry -> SL en rojo) y Break Even
        entry = float(verdict.get("precio_entrada") or 0)
        sl = float(verdict.get("stop_loss") or 0)
        tp1 = float(verdict.get("tp1") or 0)
        tp2 = float(verdict.get("tp2") or 0)
        be_trigger = float(verdict.get("be_trigger") or 0)

        if entry > 0 and sl > 0 and tp1 > 0 and direccion in ("LONG", "SHORT"):
            box_x = n_bars - 7
            box_w = 6.5
            tp_target = tp2 if tp2 > 0 else tp1

            ax.add_patch(Rectangle((box_x, min(entry, tp_target)), box_w, abs(tp_target - entry), facecolor="#26a69a", alpha=0.25))
            ax.add_patch(Rectangle((box_x, min(entry, sl)), box_w, abs(sl - entry), facecolor="#ef5350", alpha=0.28))

            ax.axhline(entry, color="#ffffff", linestyle="-", linewidth=1.0, alpha=0.85)
            ax.axhline(sl, color="#ef5350", linestyle="--", linewidth=1.1)
            ax.axhline(tp1, color="#26a69a", linestyle="--", linewidth=1.1)
            ax.text(box_x, entry, f" ENTRADA {format(entry, fmt)}", color="#ffffff", fontsize=8, fontweight="bold", va="bottom")
            ax.text(box_x, sl, f" STOP LOSS {format(sl, fmt)}", color="#ff8a80", fontsize=8, fontweight="bold", va="top")
            ax.text(box_x, tp1, f" TP1 {format(tp1, fmt)}", color="#69f0ae", fontsize=8, fontweight="bold", va="bottom")

            if tp2 > 0 and tp2 != tp1:
                ax.axhline(tp2, color="#00e676", linestyle=":", linewidth=1.0)
                ax.text(box_x, tp2, f" TP2 {format(tp2, fmt)}", color="#00e676", fontsize=7.5, va="bottom")

            if be_trigger > 0:
                ax.axhline(be_trigger, color="#ce93d8", linestyle=":", linewidth=1.1)
                ax.text(0.5, be_trigger, f"Activar Break Even: {format(be_trigger, fmt)}", color="#e1bee7", fontsize=7.5, va="bottom")

        # Tarjeta resumen limpia en la esquina inferior izquierda (nunca tapa el titulo)
        hud_lines = [f"ESTADO: {semaforo} ({filtros}/7) | {direccion}"]
        if entry > 0:
            hud_lines.append(f"Entrada: {format(entry, fmt)} | SL: {format(sl, fmt)}")
            hud_lines.append(f"TP1: {format(tp1, fmt)} | BE: {format(be_trigger, fmt)} | R:R {rr}")
        else:
            hud_lines.append("Esperando confirmacion de IFVG (sin entrada aun)")
        if dol_note:
            hud_lines.append(dol_note)

        ax.text(
            0.015,
            0.03,
            "\n".join(hud_lines),
            transform=ax.transAxes,
            fontsize=8,
            color="#eceff1",
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#1e222d", edgecolor="#448aff", alpha=0.92),
        )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#131722")
    plt.close(fig)
    return output_path
# ==============================================================================
# 7. ENVÍO INMEDIATO A TELEGRAM (TEXTO PRIMERO -> GRÁFICO DESPUÉS)
# ==============================================================================
def already_notified(symbol: str, signal_key: str) -> bool:
    with STATE_LOCK:
        if not STATE_FILE.exists():
            return False
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            return state.get(symbol, {}).get("last_signal_key") == signal_key
        except Exception:
            return False


def save_notified_state(symbol: str, signal_key: str) -> None:
    with STATE_LOCK:
        state = {}
        if STATE_FILE.exists():
            try:
                state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            except Exception:
                state = {}
        state[symbol] = {"last_signal_key": signal_key, "sent_at": datetime.now(NY_TZ).isoformat()}
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def send_telegram_alert(df_5m: pd.DataFrame, analysis: dict, verdict: dict) -> None:
    sem = verdict.get("semaforo", "ROJO")
    emoji = {"VERDE": "🟢", "AMARILLO": "🟡", "ROJO": "🔴"}.get(sem, "⚪")
    e = html.escape
    dec = analysis["decimals"]
    fmt = f".{dec}f"

    # 1) Enviar reporte textual INMEDIATAMENTE (0.3s) para cero retraso en decisión
    report = (
        f"{emoji} <b>{e(sem)} | {e(analysis['symbol'])} (5m) — {e(verdict.get('direccion', 'NEUTRO'))}</b>\n"
        f"🕒 {e(analysis['timestamp_ny'])} | {e(analysis['session']['session'])}\n"
        f"✅ Filtros checklist: <b>{verdict.get('checklist_cumplidos', 0)}/7</b> | R:R: <b>{e(str(verdict.get('rr_estimado', 'N/A')))}</b>\n\n"
        f"<b>1. 🚦 Veredicto Rápido</b>\n"
        f"{e(verdict.get('resumen_ejecutivo', ''))}\n\n"
        f"<b>2. 🌍 Contexto y Estructura</b>\n"
        f"{e(verdict.get('contexto_estructura', ''))}\n\n"
        f"<b>3. 🔍 Mapa de Liquidez y POI</b>\n"
        f"{e(verdict.get('mapa_liquidez', ''))}\n\n"
        f"<b>4. 🎯 Plan de Trading</b>\n"
        f"{e(verdict.get('plan_trading', ''))}\n\n"
        f"<b>5. ⚖️ Parámetros de Ejecución</b>\n"
        f"• <b>Modalidad:</b> {e(str(verdict.get('modalidad_entrada', 'N/A')))}\n"
        f"• <b>Entrada:</b> <code>{format(float(verdict.get('precio_entrada') or 0), fmt)}</code> (CE 50%: <code>{format(float(verdict.get('ce_50') or 0), fmt)}</code>)\n"
        f"• <b>Stop Loss:</b> <code>{format(float(verdict.get('stop_loss') or 0), fmt)}</code>\n"
        f"• <b>Break Even Trigger:</b> <code>{format(float(verdict.get('be_trigger') or 0), fmt)}</code>\n"
        f"• <b>TP1 / TP2:</b> <code>{format(float(verdict.get('tp1') or 0), fmt)}</code> / <code>{format(float(verdict.get('tp2') or 0), fmt)}</code>\n"
        f"• <b>Invalidación temprana:</b> {e(str(verdict.get('invalidacion_temprana', 'N/A')))}\n\n"
        f"<b>6. 🛡️ Gestión de Riesgo</b>\n"
        f"Arriesgar máx. 1%–2%. Cerrar a mercado si una vela cierra con cuerpo al otro lado del IFVG."
    )

    msg_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(
        msg_url,
        data={"chat_id": TELEGRAM_CHAT_ID, "text": report, "parse_mode": "HTML"},
        timeout=20,
    )
    resp.raise_for_status()
    log.info("[%s] Reporte enviado a Telegram (%s).", analysis["symbol"], sem)

    # 2) Generar y enviar gráfico anotado justo después
    chart_path = generate_setup_chart(df_5m, analysis, verdict)
    photo_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(chart_path, "rb") as img:
        resp2 = requests.post(
            photo_url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": f"📊 Gráfico {analysis['symbol']} (5m) — {sem}",
            },
            files={"photo": img},
            timeout=25,
        )
        resp2.raise_for_status()
    log.info("[%s] Gráfico enviado a Telegram.", analysis["symbol"])


# ==============================================================================
# 8. ORQUESTADOR PARALELO MULTI-ACTIVO
# ==============================================================================
def scan_single_asset(symbol_name: str, force_test: bool, mt5_cache: dict) -> dict | None:
    asset_cfg = ASSETS_CATALOG.get(symbol_name)
    if not asset_cfg:
        log.warning("Activo %s no definido en ASSETS_CATALOG.", symbol_name)
        return None

    df_5m, df_15m, df_1h = mt5_cache[symbol_name] if symbol_name in mt5_cache else load_market_data(asset_cfg)
    analysis = analyze_institutional_structure(symbol_name, asset_cfg, df_5m, df_15m, df_1h)
    dec = analysis["decimals"]

    log.info(
        "[%s] @ %s | Sesión: %s | Zona: %s | Sweep: %s | IFVG: %s",
        symbol_name,
        format(analysis["current_price"], f".{dec}f"),
        analysis["session"]["session"],
        analysis["htf_dealing_range_24h"]["current_zone"],
        analysis["sweep_analysis_arjo"]["side"] if analysis["sweep_analysis_arjo"] else "None",
        analysis["ifvg_confirmed"]["type"] if analysis["ifvg_confirmed"] else "None",
    )

    if not force_test:
        if STRICT_KILLZONE and not analysis["session"]["active_killzone"]:
            return None
        if not analysis["sweep_analysis_arjo"] and not analysis["ifvg_confirmed"]:
            return None

    return {"df_5m": df_5m, "df_15m": df_15m, "analysis": analysis}


def run(force_test: bool = False, test_all: bool = False) -> None:
    if not GEMINI_API_KEY or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("Faltan credenciales en el archivo .env (GEMINI_API_KEY / TELEGRAM_*).")

    symbols_to_scan = ACTIVE_SYMBOLS if (not force_test or test_all) else [ACTIVE_SYMBOLS[0]]
    log.info("Iniciando escaneo paralelo para: %s", ", ".join(symbols_to_scan))

    mt5_cache = {}
    if DATA_SOURCE == "mt5":
        try:
            mt5_cache = fetch_all_mt5(symbols_to_scan)
        except Exception as exc:
            log.warning("Fallo conexion MT5 (%s). Usando respaldo yfinance.", exc)

    candidates = []
    with ThreadPoolExecutor(max_workers=min(5, len(symbols_to_scan))) as executor:
        futures = {executor.submit(scan_single_asset, sym, force_test, mt5_cache): sym for sym in symbols_to_scan}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                result = future.result()
                if result:
                    candidates.append(result)
            except Exception as exc:
                log.error("[%s] Error durante escaneo: %s", sym, exc)

    if not candidates:
        log.info("Ciclo completado: ningún activo presentó Sweep/IFVG activo en Killzone.")
        return

    for item in candidates:
        analysis = item["analysis"]
        sym = analysis["symbol"]
        try:
            log.info("[%s] Consultando Gemini (%s)...", sym, GEMINI_MODEL)
            verdict = validate_with_gemini(item["df_5m"], item["df_15m"], analysis)
            semaforo = verdict.get("semaforo", "ROJO")
            log.info(
                "[%s] Veredicto: %s (%s) | Filtros: %s/7",
                sym,
                semaforo,
                verdict.get("direccion"),
                verdict.get("checklist_cumplidos"),
            )

            should_notify = (
                force_test
                or semaforo == "VERDE"
                or (SEND_YELLOW_ALERTS and semaforo == "AMARILLO")
            )
            if not should_notify:
                continue

            signal_key = f"{sym}_{semaforo}_{verdict.get('direccion')}_{verdict.get('ce_50')}"
            if not force_test and already_notified(sym, signal_key):
                log.info("[%s] Señal duplicada omitida.", sym)
                continue

            send_telegram_alert(item["df_5m"], analysis, verdict)

            if not force_test:
                save_notified_state(sym, signal_key)
        except Exception as exc:
            log.error("[%s] Error validando/enviando señal: %s", sym, exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bot Institucional Multi-Activo Trader-Senior")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Fuerza análisis y envío a Telegram del primer activo configurado.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Combinado con --test, fuerza el envío de los 5 mercados a la vez.",
    )
    args = parser.parse_args()
    run(force_test=args.test, test_all=args.all)



