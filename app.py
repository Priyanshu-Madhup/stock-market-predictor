"""
Stock Market Predictor & Suggestor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
XGBoost + Groq AI  ·  pywebview Edition
• Natural-language ticker resolution
• Feature-rich XGBoost forecasting (30-day iterative)
• Modern web UI with Plotly charts & marked.js AI rendering
"""

import webview
import os, csv, json, re, threading
from openai import OpenAI
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import yfinance as yf

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Config ─────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "")
AI_CACHE_FILE      = "ai_cache.json"

# Chart colour constants (mirrored in CSS / Plotly)
ACCENT     = "#58a6ff"
ACCENT_GRN = "#3fb950"
ACCENT_RED = "#f85149"
BG_ROOT    = "#0d1117"
BG_CARD    = "#161b22"
BG_INPUT   = "#21262d"
BORDER     = "#30363d"
TEXT_SEC   = "#8b949e"

# ── AI cache ───────────────────────────────────────────────────────────────
def _load_ai_cache() -> dict:
    try:
        with open(AI_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _save_ai_cache(cache: dict) -> None:
    with open(AI_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

# ── Groq (OpenAI-compatible) helpers ─────────────────────────────────────────
def _get_client() -> OpenAI:
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        raise ValueError("GROQ_API_KEY is not set.\nAdd it to the .env file.")
    if not GROQ_MODEL or GROQ_MODEL == "your_model_name_here":
        raise ValueError("GROQ_MODEL is not set.\nAdd it to the .env file.")
    return OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )

def _post(messages: list, max_tokens: int = 2048) -> str:
    client = _get_client()
    # Extract the user message content — both callers pass a single user message
    user_input = messages[0]["content"]
    response = client.responses.create(
        model=GROQ_MODEL,
        input=user_input,
    )
    content = response.output_text
    if not content:
        raise ValueError(
            f"Model returned empty response ({GROQ_MODEL}).\n"
            "Possible rate limit — wait a moment and retry."
        )
    return content.strip()


def resolve_ticker(name: str) -> str:
    raw = _post(
        [{"role": "user", "content":
          f"What is the exact Yahoo Finance ticker symbol for '{name}'? "
          "Reply ONLY the ticker symbol, nothing else."}],
        max_tokens=16,
    )
    m = re.search(r'\b[A-Z][A-Z0-9]*(?:\.[A-Z]{1,3})?\b', raw)
    return m.group(0) if m else raw.strip().upper()

def get_ai_analysis(ticker: str, csv_content: str) -> str:
    return _post(
        [{"role": "user", "content": (
            f"Stock ticker: {ticker}\n"
            f"XGBoost-predicted closing prices for the next ~20 trading days:\n"
            f"{csv_content}\n\n"
            "Provide a structured financial analysis with EXACTLY these markdown sections:\n"
            "## 1. Company Overview\n"
            "## 2. Predicted Trend Analysis\n"
            "## 3. Investment Recommendation (Buy / Hold / Sell)\n"
            "## 4. Key Risks\n"
            "## 5. Recent Developments\n\n"
            "Rules: Start IMMEDIATELY with '## 1. Company Overview'. "
            "No intro sentence. Be concise, professional, data-driven. "
            "Use markdown: ## headings, **bold**, bullet lists, tables."
        )}],
        max_tokens=2048,
    )

# ── Feature engineering ────────────────────────────────────────────────────
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def build_features(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df["DayOfWeek"]  = df.index.dayofweek
    df["Month"]      = df.index.month
    df["DayOfYear"]  = df.index.dayofyear
    df["WeekOfYear"] = df.index.isocalendar().week.astype(int)
    for lag in [1, 2, 3, 5, 7, 10, 20]:
        df[f"Close_lag_{lag}"] = df["Close"].shift(lag)
    for w in [5, 10, 20]:
        df[f"SMA_{w}"] = df["Close"].rolling(w).mean()
        df[f"STD_{w}"] = df["Close"].rolling(w).std()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]   = df["EMA_12"] - df["EMA_26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["RSI_14"] = _rsi(df["Close"], 14)
    df["Return_1d"]  = df["Close"].pct_change(1)
    df["Return_5d"]  = df["Close"].pct_change(5)
    df["HL_spread"]  = df["High"] - df["Low"]
    df["OC_spread"]  = df["Close"] - df["Open"]
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        df["Volume_log"] = np.log1p(df["Volume"])
    df.dropna(inplace=True)
    return df

BASE_FEATURES = [
    "DayOfWeek", "Month", "DayOfYear", "WeekOfYear",
    "Close_lag_1", "Close_lag_2", "Close_lag_3",
    "Close_lag_5", "Close_lag_7", "Close_lag_10", "Close_lag_20",
    "SMA_5", "SMA_10", "SMA_20",
    "STD_5", "STD_10", "STD_20",
    "EMA_12", "EMA_26", "MACD", "Signal",
    "RSI_14", "Return_1d", "Return_5d",
    "HL_spread", "OC_spread",
]

def train_and_predict(df: pd.DataFrame):
    feat_cols = BASE_FEATURES.copy()
    if "Volume_log" in df.columns:
        feat_cols.append("Volume_log")

    model = xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.04, max_depth=6,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0, tree_method="hist",
    )
    model.fit(df[feat_cols], df["Close"], verbose=False)

    rolling    = df.copy()
    last_date  = rolling.index[-1]
    future_dates, predictions = [], []
    cal_day    = 1

    while len(future_dates) < 22:
        nd = last_date + timedelta(days=cal_day); cal_day += 1
        if nd.weekday() >= 5:
            continue
        future_dates.append(nd)
        cs = rolling["Close"]

        def _lag(n): return cs.iloc[-n] if len(cs) >= n else cs.iloc[0]
        def _rm(n):  return cs.tail(n).mean()
        def _rs(n):  return cs.tail(n).std()

        ema12 = rolling["EMA_12"].iloc[-1] * (1 - 2/13) + cs.iloc[-1] * (2/13)
        ema26 = rolling["EMA_26"].iloc[-1] * (1 - 2/27) + cs.iloc[-1] * (2/27)
        macd  = ema12 - ema26
        sig   = rolling["Signal"].iloc[-1] * (1 - 2/10) + macd * (2/10)

        row = dict(
            DayOfWeek=nd.weekday(), Month=nd.month,
            DayOfYear=nd.timetuple().tm_yday, WeekOfYear=int(nd.isocalendar()[1]),
            Close_lag_1=_lag(1), Close_lag_2=_lag(2), Close_lag_3=_lag(3),
            Close_lag_5=_lag(5), Close_lag_7=_lag(7), Close_lag_10=_lag(10),
            Close_lag_20=_lag(20),
            SMA_5=_rm(5), SMA_10=_rm(10), SMA_20=_rm(20),
            STD_5=_rs(5), STD_10=_rs(10), STD_20=_rs(20),
            EMA_12=ema12, EMA_26=ema26, MACD=macd, Signal=sig,
            RSI_14=rolling["RSI_14"].iloc[-1],
            Return_1d=(cs.iloc[-1]-cs.iloc[-2])/cs.iloc[-2] if len(cs) >= 2 else 0,
            Return_5d=(cs.iloc[-1]-cs.iloc[-5])/cs.iloc[-5] if len(cs) >= 5 else 0,
            HL_spread=rolling["HL_spread"].iloc[-1],
            OC_spread=rolling["OC_spread"].iloc[-1],
        )
        if "Volume_log" in feat_cols:
            row["Volume_log"] = rolling["Volume_log"].iloc[-1]

        pred = float(model.predict(pd.DataFrame([{c: row[c] for c in feat_cols}]))[0])
        predictions.append(pred)

        nr = rolling.iloc[[-1]].copy(); nr.index = [nd]
        nr["Close"] = pred; nr["EMA_12"] = ema12; nr["EMA_26"] = ema26
        nr["MACD"] = macd; nr["Signal"] = sig
        nr["HL_spread"] = rolling["HL_spread"].iloc[-1]; nr["OC_spread"] = 0.0
        rolling = pd.concat([rolling, nr])

    return model, future_dates, predictions


# ── JSON encoder ───────────────────────────────────────────────────────────
class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        if isinstance(o, datetime):    return o.strftime('%Y-%m-%d')
        return super().default(o)

def _jdump(obj) -> str:
    return json.dumps(obj, cls=_Enc)


# ── pywebview API ──────────────────────────────────────────────────────────
# Keep the window reference at module level so pywebview's API introspector
# never tries to scan a Window object inside the Api class (causes crash).
_window_ref = None

class Api:
    def __init__(self):
        self._hist         = None
        self._future_dates = None
        self._predictions  = None
        self._ticker       = None
        self._ai_cache     = _load_ai_cache()

    def _js(self, code: str):
        global _window_ref
        if _window_ref:
            try:
                _window_ref.evaluate_js(code)
            except Exception:
                pass

    # ── Called from JS ───────────────────────────────────────────────────
    def analyze(self, query: str):
        """Full pipeline — runs directly on pywebview's API thread.
        evaluate_js() is reliable from pywebview's own threads (COM-affinity safe).
        Do NOT wrap in threading.Thread — that breaks evaluate_js on EdgeChromium."""
        self._pipeline(query)

    def get_chart_data(self, chart_type: str):
        """Return serialisable chart data dict — pywebview resolves as Promise."""
        if self._hist is None:
            return None
        h = self._hist
        t = self._ticker

        if chart_type == "closing":
            return {
                "type": "single",
                "title": f"{t} — 1-Year Closing Price",
                "dates":  [str(d.date()) for d in h.index],
                "values": h["Close"].tolist(),
                "name":   "Close Price",
                "color":  ACCENT,
                "fill":   "rgba(88,166,255,0.08)",
            }
        if chart_type == "opening":
            return {
                "type": "single",
                "title": f"{t} — 1-Year Opening Price",
                "dates":  [str(d.date()) for d in h.index],
                "values": h["Open"].tolist(),
                "name":   "Open Price",
                "color":  ACCENT_GRN,
                "fill":   "rgba(63,185,80,0.08)",
            }
        if chart_type == "predicted":
            return {
                "type":       "predicted",
                "title":      f"{t} — Historical + XGBoost 30-Day Forecast",
                "hist_dates": [str(d.date()) for d in h.index],
                "hist_vals":  h["Close"].tolist(),
                "pred_dates": [str(d.date()) for d in self._future_dates],
                "pred_vals":  [float(p) for p in self._predictions],
            }
        return None

    def exit_app(self):
        global _window_ref
        if _window_ref:
            _window_ref.destroy()

    # ── Pipeline ─────────────────────────────────────────────────────────
    def _pipeline(self, query: str):
        def prog(msg: str):
            self._js(f"onProgress({_jdump(msg)})")

        try:
            prog(f'Resolving ticker for "{query}"…')
            ticker = resolve_ticker(query)
            self._js(f"onTickerResolved({_jdump(ticker)})")

            prog(f"Fetching 1-year history for {ticker}…")
            hist = yf.Ticker(ticker).history(period="1y")
            if hist.empty:
                raise ValueError(
                    f"No data found for '{ticker}'.\n"
                    "Check the symbol or try the full company name."
                )
            if hasattr(hist.index, "tz") and hist.index.tz:
                hist.index = hist.index.tz_localize(None)

            prog("Engineering 26 technical features…")
            df = build_features(hist)

            prog("Training XGBoost model & forecasting 30 days…")
            model, future_dates, predictions = train_and_predict(df)

            feat_cols = BASE_FEATURES.copy()
            if "Volume_log" in df.columns:
                feat_cols.append("Volume_log")
            y_hat = model.predict(df[feat_cols])
            mae   = float(mean_absolute_error(df["Close"], y_hat))
            r2    = float(r2_score(df["Close"], y_hat))

            csv_path = "stock_pred.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["DATE", "PREDICTED CLOSING PRICE"])
                for d, p in zip(future_dates, predictions):
                    w.writerow([d.strftime("%Y-%m-%d"), f"{p:.2f}"])

            from_cache = ticker in self._ai_cache
            if from_cache:
                prog("Loading cached AI analysis…")
                ai_text = self._ai_cache[ticker]
            else:
                prog("Fetching AI analysis from Groq…")
                csv_content = pd.read_csv(csv_path).to_string(index=False)
                ai_text = get_ai_analysis(ticker, csv_content)
                self._ai_cache[ticker] = ai_text
                _save_ai_cache(self._ai_cache)

            # Store for chart requests
            self._hist         = hist
            self._future_dates = future_dates
            self._predictions  = predictions
            self._ticker       = ticker

            chg     = float(hist["Close"].iloc[-1] - hist["Close"].iloc[-2])
            chg_pct = float(chg / hist["Close"].iloc[-2] * 100)

            prev  = float(hist["Close"].iloc[-1])
            preds = []
            for i, (d, p) in enumerate(zip(future_dates, predictions)):
                pf = float(p)
                preds.append({"day": i + 1, "date": d.strftime("%Y-%m-%d"),
                              "price": pf, "delta": pf - prev})
                prev = pf

            result = {
                "ticker":      ticker,
                "lastClose":   float(hist["Close"].iloc[-1]),
                "high52w":     float(hist["High"].max()),
                "low52w":      float(hist["Low"].min()),
                "change1d":    chg,
                "changePct":   chg_pct,
                "mae":         mae,
                "r2":          r2,
                "aiText":      ai_text,
                "fromCache":   from_cache,
                "predictions": preds,
            }
            self._js(f"onAnalysisComplete({_jdump(result)})")

        except Exception as exc:
            self._js(f"onError({_jdump(str(exc))})")


# ── HTML ───────────────────────────────────────────────────────────────────
def get_html() -> str:
    model_short = GROQ_MODEL.split("/")[-1] if GROQ_MODEL else "—"

    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Stock Market Predictor &amp; Suggestor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
<script src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js"></script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
:root {
  --bg:    #0d1117; --card:  #161b22; --input: #21262d;
  --border:#30363d; --accent:#58a6ff; --green: #3fb950;
  --red:   #f85149; --yellow:#e3b341; --purple:#d2a8ff;
  --text:  #e6edf3; --sec:   #8b949e; --mut:   #484f58;
  --r: 6px; --tr: 0.16s ease;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Inter','Segoe UI',system-ui,sans-serif;
  font-size:13px;line-height:1.6;height:100vh;display:flex;flex-direction:column;
  overflow:hidden;-webkit-font-smoothing:antialiased;user-select:none}

/* ── Topbar ── */
.topbar{height:56px;background:var(--card);border-bottom:1px solid var(--border);
  display:flex;align-items:center;padding:0 18px;gap:10px;flex-shrink:0;position:relative}
.topbar::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;
  background:var(--accent);border-radius:0 2px 2px 0}
.tb-logo{display:flex;align-items:center;justify-content:center;
  width:32px;height:32px;background:rgba(88,166,255,0.12);
  border:1px solid rgba(88,166,255,0.2);border-radius:6px;margin-left:6px;flex-shrink:0}
.tb-logo i{color:var(--accent);font-size:16px}
.tb-title{font-size:15px;font-weight:700;letter-spacing:-0.2px}
.tb-sep{color:var(--mut);font-size:15px;margin:0 2px}
.tb-sub{font-size:15px;color:var(--sec);font-weight:400}
.tb-model{margin-left:auto;font-size:11px;color:var(--mut);background:var(--input);
  padding:3px 10px;border-radius:4px;border:1px solid var(--border);
  font-family:'Consolas',monospace;letter-spacing:0.2px}

/* ── Layout ── */
.body{display:flex;flex:1;overflow:hidden}

/* ── Sidebar ── */
.sidebar{width:288px;background:var(--card);border-right:1px solid var(--border);
  display:flex;flex-direction:column;flex-shrink:0;overflow-y:auto}
.si{padding:16px 14px;display:flex;flex-direction:column;flex:1;gap:0}
.slabel{font-size:10px;font-weight:700;letter-spacing:1.4px;text-transform:uppercase;
  color:var(--mut);margin-bottom:8px}

/* Search */
.sw{position:relative;margin-bottom:6px}
.si-icon{position:absolute;left:9px;top:50%;transform:translateY(-50%);
  color:var(--mut);pointer-events:none;font-size:13px;line-height:1}
#search{width:100%;background:var(--input);border:1px solid var(--border);
  border-radius:var(--r);color:var(--text);font-family:inherit;font-size:13px;
  padding:8px 10px 8px 30px;outline:none;transition:border-color var(--tr);user-select:text}
#search:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(88,166,255,0.1)}
#search::placeholder{color:var(--mut);font-size:12px}
.hint{font-size:11px;color:var(--mut);margin-bottom:10px}

/* Buttons */
.btn{display:flex;align-items:center;justify-content:center;gap:6px;width:100%;
  padding:8px 12px;border:1px solid transparent;border-radius:var(--r);
  font-family:inherit;font-size:12px;font-weight:600;cursor:pointer;
  transition:all var(--tr);outline:none;letter-spacing:0.1px}
.btn i{font-size:13px;line-height:1}
.btn:disabled{opacity:0.4;cursor:not-allowed}
.btn-p{background:var(--accent);color:#0d1117;border-color:var(--accent)}
.btn-p:hover:not(:disabled){background:#79b8ff;border-color:#79b8ff;
  box-shadow:0 2px 10px rgba(88,166,255,0.25)}
.btn-p:active:not(:disabled){transform:translateY(1px)}
.btn-s{background:transparent;color:var(--sec);border-color:var(--border)}
.btn-s:hover:not(:disabled){background:var(--input);color:var(--text);border-color:var(--mut)}
.btn-g{background:rgba(63,185,80,0.08);color:var(--green);border-color:rgba(63,185,80,0.3)}
.btn-g:hover:not(:disabled){background:rgba(63,185,80,0.15)}
.btn-d{background:transparent;color:var(--mut);border-color:var(--border)}
.btn-d:hover:not(:disabled){background:rgba(248,81,73,0.08);color:var(--red);border-color:rgba(248,81,73,0.3)}

.div{height:1px;background:var(--border);margin:14px 0}

/* Info grid */
.igrid{display:flex;flex-direction:column;gap:5px;margin-bottom:2px}
.irow{display:flex;justify-content:space-between;align-items:center;
  padding:5px 8px;border-radius:4px;background:var(--bg);
  border:1px solid transparent}
.ik{color:var(--sec);font-size:11px}
.iv{font-weight:600;font-size:12px;color:var(--text);text-align:right;
  font-variant-numeric:tabular-nums}
.iv.acc{color:var(--accent)}.iv.pos{color:var(--green)}.iv.neg{color:var(--red)}
@keyframes flash{0%,100%{background:var(--bg)}50%{background:rgba(88,166,255,0.08)}}
.flash{animation:flash 0.5s ease}

/* Sidebar loading */
.sload{display:none;flex-direction:column;align-items:center;gap:8px;padding:14px 0}
.sload.on{display:flex}
@keyframes spin{to{transform:rotate(360deg)}}
.sp{width:22px;height:22px;border:2px solid var(--border);border-top-color:var(--accent);
  border-radius:50%;animation:spin 0.7s linear infinite}
.sload-txt{font-size:11px;color:var(--sec);text-align:center;line-height:1.4}

.spacer{flex:1;min-height:8px}
.cbns{display:flex;flex-direction:column;gap:6px}

/* ── Main panel ── */
.main{flex:1;display:flex;flex-direction:column;overflow:hidden;padding:14px 16px}

/* Tabs */
.tabbar{display:flex;align-items:flex-end;border-bottom:1px solid var(--border);flex-shrink:0}
.tbtn{padding:8px 16px 9px;background:none;border:none;border-bottom:2px solid transparent;
  margin-bottom:-1px;color:var(--sec);font-family:inherit;font-size:12px;font-weight:600;
  cursor:pointer;transition:all var(--tr);display:flex;align-items:center;gap:6px;
  letter-spacing:0.2px;text-transform:uppercase}
.tbtn i{font-size:13px}
.tbtn:hover{color:var(--text)}
.tbtn.on{color:var(--accent);border-bottom-color:var(--accent)}

.tpanel{display:none;flex:1;flex-direction:column;overflow:hidden;padding-top:14px}
.tpanel.on{display:flex}

/* AI panel */
.ai-hdr{display:flex;align-items:baseline;gap:8px;margin-bottom:2px;flex-shrink:0}
.ai-title{font-size:15px;font-weight:700}
.ai-sub{font-size:11px;color:var(--sec);margin-bottom:10px;flex-shrink:0}

/* AI loading */
.aiload{display:none;flex-direction:column;align-items:center;justify-content:center;
  gap:16px;padding:60px 20px;flex:1}
.aiload.on{display:flex}
.aisp{width:36px;height:36px;border:3px solid var(--border);border-top-color:var(--accent);
  border-radius:50%;animation:spin 0.8s linear infinite}
.aistep{font-size:12px;color:var(--sec);text-align:center;max-width:300px;line-height:1.5}
.pb{height:2px;background:var(--border);border-radius:2px;width:180px;overflow:hidden}
@keyframes pbanim{0%{transform:translateX(-100%)}100%{transform:translateX(280%)}}
.pbf{height:100%;width:35%;background:var(--accent);border-radius:2px;
  animation:pbanim 1.5s ease-in-out infinite}

/* AI content area */
.aiwrap{flex:1;overflow-y:auto;display:none;border-radius:var(--r);
  border:1px solid var(--border);background:var(--card)}
.aiwrap.on{display:block}
#aitxt{padding:20px 24px;line-height:1.8;user-select:text}
#aitxt h1,#aitxt h2{color:var(--accent);border-bottom:1px solid var(--border);
  padding-bottom:6px;margin:20px 0 10px;font-size:14px;font-weight:700;letter-spacing:-0.1px}
#aitxt h2:first-child,#aitxt h1:first-child{margin-top:0}
#aitxt h3{color:var(--purple);margin:14px 0 6px;font-size:13px;font-weight:600}
#aitxt p{margin:6px 0;color:var(--text)}
#aitxt strong{color:#fff;font-weight:600}
#aitxt em{color:var(--yellow)}
#aitxt ul,#aitxt ol{padding-left:20px;margin:6px 0}
#aitxt li{margin:3px 0;color:var(--text)}
#aitxt table{border-collapse:collapse;width:100%;margin:14px 0;font-size:12px}
#aitxt th{background:var(--input);color:var(--sec);padding:7px 12px;
  border:1px solid var(--border);text-align:left;font-size:10px;font-weight:700;
  text-transform:uppercase;letter-spacing:0.6px}
#aitxt td{padding:7px 12px;border:1px solid var(--border);color:var(--text)}
#aitxt tr:nth-child(even) td{background:rgba(33,38,45,0.5)}
#aitxt code{background:var(--input);color:#79c0ff;padding:1px 5px;border-radius:3px;
  font-family:'Consolas',monospace;font-size:11px}
#aitxt pre{background:var(--input);border:1px solid var(--border);border-radius:var(--r);
  padding:12px;overflow-x:auto;margin:8px 0}
#aitxt pre code{background:none;padding:0}
#aitxt blockquote{border-left:3px solid var(--accent);padding:6px 12px;
  margin:10px 0;color:var(--sec);background:var(--input);border-radius:0 4px 4px 0}
#aitxt hr{border:none;border-top:1px solid var(--border);margin:16px 0}
#aitxt a{color:var(--accent);text-decoration:none}

/* Placeholder */
.ph{display:flex;flex-direction:column;align-items:center;justify-content:center;
  height:100%;gap:12px;padding:60px}
.ph-ico{font-size:40px;color:var(--mut);opacity:0.6;line-height:1}
.ph-t{font-size:14px;font-weight:600;color:var(--sec)}
.ph-s{font-size:11px;color:var(--mut);text-align:center;line-height:1.7}

/* Forecast table */
.fc-hdr{flex-shrink:0;margin-bottom:4px}
.fc-title{font-size:15px;font-weight:700;margin-bottom:2px}
.fc-sub{font-size:11px;color:var(--sec);margin-bottom:8px}
.twrap{flex:1;overflow-y:auto;border-radius:var(--r);border:1px solid var(--border);background:var(--card)}
table.pt{width:100%;border-collapse:collapse}
table.pt thead th{position:sticky;top:0;z-index:1;background:var(--input);
  color:var(--sec);font-size:10px;font-weight:700;letter-spacing:0.8px;
  text-transform:uppercase;padding:10px 14px;text-align:center;
  border-bottom:1px solid var(--border)}
table.pt tbody tr{border-bottom:1px solid var(--border);transition:background var(--tr)}
table.pt tbody tr:last-child{border-bottom:none}
table.pt tbody tr:hover{background:rgba(88,166,255,0.04)}
table.pt tbody tr:nth-child(even){background:rgba(33,38,45,0.35)}
table.pt td{padding:9px 14px;text-align:center;font-size:12px;color:var(--text)}
table.pt td:first-child{color:var(--mut);font-size:11px;font-weight:500}
table.pt .price{font-weight:700;font-size:13px;font-variant-numeric:tabular-nums}
table.pt .up{color:var(--green);font-weight:600}
table.pt .dn{color:var(--red);font-weight:600}
.t-empty{text-align:center;padding:50px;color:var(--mut);font-size:12px}

/* Status bar */
.sbar{height:28px;background:var(--card);border-top:1px solid var(--border);
  display:flex;align-items:center;padding:0 14px;gap:8px;flex-shrink:0}
.sdot{width:6px;height:6px;border-radius:50%;background:var(--mut);
  flex-shrink:0;transition:background var(--tr)}
.sdot.ready{background:var(--green)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
.sdot.busy{background:var(--yellow);animation:pulse 1.2s ease-in-out infinite}
.sdot.err{background:var(--red)}
.stxt{font-size:11px;color:var(--sec);flex:1}
.sbrand{font-size:10px;color:var(--mut);letter-spacing:0.3px}

/* Chart modal */
.moverlay{display:none;position:fixed;inset:0;background:rgba(1,4,9,0.9);
  z-index:100;align-items:center;justify-content:center;backdrop-filter:blur(6px)}
.moverlay.on{display:flex}
@keyframes fis{from{opacity:0;transform:scale(0.95)}to{opacity:1;transform:scale(1)}}
.mcard{background:var(--card);border:1px solid var(--border);border-radius:10px;
  width:90vw;max-width:1080px;max-height:88vh;display:flex;flex-direction:column;
  animation:fis 0.18s ease;overflow:hidden}
.mhdr{display:flex;align-items:center;justify-content:space-between;
  padding:12px 16px;border-bottom:1px solid var(--border)}
.mttl{font-weight:700;font-size:13px;color:var(--text)}
.mclose{background:transparent;border:1px solid var(--border);border-radius:4px;
  color:var(--sec);width:26px;height:26px;display:flex;align-items:center;
  justify-content:center;cursor:pointer;font-size:14px;transition:all var(--tr)}
.mclose:hover{background:rgba(248,81,73,0.1);border-color:rgba(248,81,73,0.3);color:var(--red)}
.mbody{flex:1;overflow:hidden;min-height:480px}
#chartdiv{width:100%;height:100%;min-height:480px}

/* Toast */
@keyframes sd{from{transform:translate(-50%,-110%);opacity:0}to{transform:translate(-50%,0);opacity:1}}
.toast{display:none;position:fixed;top:65px;left:50%;transform:translateX(-50%);
  background:var(--card);border:1px solid var(--border);border-left:3px solid var(--red);
  border-radius:var(--r);padding:10px 16px;color:var(--text);font-size:12px;z-index:200;
  max-width:500px;text-align:left;box-shadow:0 8px 32px rgba(0,0,0,0.5);
  line-height:1.5;display:flex;align-items:flex-start;gap:8px}
.toast.on{display:flex;animation:sd 0.2s ease}
.toast i{color:var(--red);font-size:14px;flex-shrink:0;margin-top:1px}

/* Scrollbar */
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
::-webkit-scrollbar-thumb:hover{background:var(--mut)}
</style>
</head>
<body>

<!-- Top bar -->
<div class="topbar">
  <div class="tb-logo"><i class="bi bi-graph-up-arrow"></i></div>
  <span class="tb-title">Stock Market Predictor</span>
  <span class="tb-sep">/</span>
  <span class="tb-sub">Suggestor</span>
  <span class="tb-model">__MODEL_SHORT__</span>
</div>

<div class="body">
  <!-- Sidebar -->
  <aside class="sidebar">
    <div class="si">
      <div class="slabel">Search Stock</div>
      <div class="sw">
        <i class="bi bi-search si-icon"></i>
        <input id="search" type="text" placeholder='Company name or ticker symbol'
               onkeydown="if(event.key==='Enter')analyze()">
      </div>
      <div class="hint">e.g. Apple, Reliance Industries, TSLA</div>
      <button id="bta" class="btn btn-p" onclick="analyze()">
        <i class="bi bi-lightning-charge-fill"></i> Run Analysis
      </button>

      <div class="sload" id="sload">
        <div class="sp"></div>
        <div class="sload-txt" id="ptxt">Starting…</div>
      </div>

      <div class="div"></div>
      <div class="slabel">Stock Info</div>
      <div class="igrid">
        <div class="irow"><span class="ik">Ticker</span>    <span class="iv acc" id="i-t">—</span></div>
        <div class="irow"><span class="ik">Last Close</span><span class="iv"      id="i-c">—</span></div>
        <div class="irow"><span class="ik">52W High</span>  <span class="iv pos"  id="i-h">—</span></div>
        <div class="irow"><span class="ik">52W Low</span>   <span class="iv neg"  id="i-l">—</span></div>
        <div class="irow"><span class="ik">1D Change</span> <span class="iv"      id="i-d">—</span></div>
        <div class="irow"><span class="ik">Model MAE</span> <span class="iv"      id="i-m">—</span></div>
        <div class="irow"><span class="ik">Model R²</span>  <span class="iv"      id="i-r">—</span></div>
      </div>

      <div class="div"></div>
      <div class="slabel">Charts</div>
      <div class="cbns">
        <button id="bc1" class="btn btn-s" onclick="openChart('closing')"  disabled>
          <i class="bi bi-bar-chart-line"></i> Closing Price (1Y)
        </button>
        <button id="bc2" class="btn btn-s" onclick="openChart('opening')"  disabled>
          <i class="bi bi-bar-chart"></i> Opening Price (1Y)
        </button>
        <button id="bc3" class="btn btn-g" onclick="openChart('predicted')" disabled>
          <i class="bi bi-cpu"></i> XGBoost Forecast (30d)
        </button>
      </div>

      <div class="spacer"></div>
      <div class="div"></div>
      <button class="btn btn-d" onclick="window.pywebview.api.exit_app()">
        <i class="bi bi-box-arrow-right"></i> Exit Application
      </button>
    </div>
  </aside>

  <!-- Main -->
  <main class="main">
    <div class="tabbar">
      <button class="tbtn on" id="tab-ai-btn" onclick="switchTab('ai')">
        <i class="bi bi-stars"></i> AI Analysis
      </button>
      <button class="tbtn" id="tab-fc-btn" onclick="switchTab('fc')">
        <i class="bi bi-table"></i> 30-Day Forecast
      </button>
    </div>

    <!-- AI tab -->
    <div class="tpanel on" id="tab-ai">
      <div class="ai-hdr">
        <span class="ai-title">AI Analysis &amp; Recommendation</span>
      </div>
      <div class="ai-sub">Powered by Groq — company overview, trend analysis, and investment recommendation</div>

      <div class="aiload" id="aiload">
        <div class="aisp"></div>
        <div class="aistep" id="aistep">Initializing…</div>
        <div class="pb"><div class="pbf"></div></div>
      </div>

      <div class="aiwrap" id="aiwrap">
        <div id="aitxt">
          <div class="ph">
            <i class="bi bi-robot ph-ico"></i>
            <div class="ph-t">No analysis loaded</div>
            <div class="ph-s">Enter a company name or ticker symbol<br>and click Run Analysis to get started</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Forecast tab -->
    <div class="tpanel" id="tab-fc">
      <div class="fc-hdr">
        <div class="fc-title">30-Day Price Forecast</div>
        <div class="fc-sub">XGBoost model · 26 engineered features · iterative day-by-day prediction (trading days only)</div>
      </div>
      <div class="twrap">
        <table class="pt">
          <thead><tr><th>#</th><th>Date</th><th>Predicted Close</th><th>&Delta; vs Previous</th></tr></thead>
          <tbody id="ptbody">
            <tr><td colspan="4" class="t-empty">Run an analysis to populate the forecast table</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </main>
</div>

<!-- Status bar -->
<div class="sbar">
  <div class="sdot" id="sdot"></div>
  <div class="stxt" id="stxt">Ready — enter a company name or ticker symbol to begin</div>
  <div class="sbrand">XGBoost · Groq AI</div>
</div>

<!-- Chart modal -->
<div class="moverlay" id="moverlay" onclick="if(event.target===this)closeChart()">
  <div class="mcard">
    <div class="mhdr">
      <span class="mttl" id="mttl">Chart</span>
      <button class="mclose" onclick="closeChart()"><i class="bi bi-x-lg"></i></button>
    </div>
    <div class="mbody"><div id="chartdiv"></div></div>
  </div>
</div>

<!-- Toast -->
<div class="toast" id="toast">
  <i class="bi bi-exclamation-circle"></i>
  <span id="toast-msg"></span>
</div>

<script>
let done = false;

// ── Tabs ──────────────────────────────────────────────────────────────────────
function switchTab(n) {
  document.querySelectorAll('.tpanel').forEach(e => e.classList.remove('on'));
  document.querySelectorAll('.tbtn').forEach(e => e.classList.remove('on'));
  document.getElementById('tab-' + n).classList.add('on');
  document.getElementById('tab-' + n + '-btn').classList.add('on');
}

// ── Analyze ───────────────────────────────────────────────────────────────────
function analyze() {
  const q = document.getElementById('search').value.trim();
  if (!q) { showToast('Please enter a company name or ticker symbol.'); return; }
  if (!window.pywebview?.api) { showToast('App not ready yet — please wait a moment.'); return; }
  setLoading(true);
  window.pywebview.api.analyze(q);
}

function setLoading(on) {
  document.getElementById('bta').disabled = on;
  document.getElementById('sload').classList.toggle('on', on);
  if (on) {
    document.getElementById('aiload').classList.add('on');
    document.getElementById('aiwrap').classList.remove('on');
    setStatus('Running analysis…', 'busy');
    ['bc1','bc2','bc3'].forEach(id => document.getElementById(id).disabled = true);
    switchTab('ai');
  }
}

// ── Python → JS callbacks ──────────────────────────────────────────────────────
function onProgress(msg) {
  document.getElementById('ptxt').textContent   = msg;
  document.getElementById('aistep').textContent = msg;
  setStatus(msg, 'busy');
}

function onTickerResolved(t) {
  document.getElementById('search').value = t;
}

function onAnalysisComplete(d) {
  done = true;
  av('i-t', d.ticker, 'acc');
  av('i-c', '$' + d.lastClose.toFixed(2));
  av('i-h', '$' + d.high52w.toFixed(2), 'pos');
  av('i-l', '$' + d.low52w.toFixed(2),  'neg');
  const s = d.change1d >= 0 ? '+' : '';
  const ce = document.getElementById('i-d');
  ce.textContent = s + d.change1d.toFixed(2) + ' (' + s + d.changePct.toFixed(2) + '%)';
  ce.className = 'iv ' + (d.change1d >= 0 ? 'pos' : 'neg');
  av('i-m', '$' + d.mae.toFixed(2));
  av('i-r', d.r2.toFixed(4));

  document.getElementById('aiload').classList.remove('on');
  document.getElementById('aiwrap').classList.add('on');

  const box = document.getElementById('aitxt');
  try {
    box.innerHTML = marked.parse(d.aiText);
  } catch(e) {
    box.innerHTML = '<pre style="white-space:pre-wrap;color:#e6edf3;user-select:text">'
      + d.aiText.replace(/</g,'&lt;') + '</pre>';
  }
  box.parentElement.scrollTop = 0;

  // Forecast table
  const tb = document.getElementById('ptbody');
  tb.innerHTML = '';
  d.predictions.forEach(r => {
    const s = r.delta >= 0 ? '+' : '';
    const tr = document.createElement('tr');
    tr.innerHTML = '<td>' + r.day + '</td><td>' + r.date + '</td>'
      + '<td class="price">$' + r.price.toFixed(2) + '</td>'
      + '<td class="' + (r.delta >= 0 ? 'up' : 'dn') + '">' + s + r.delta.toFixed(2) + '</td>';
    tb.appendChild(tr);
  });

  ['bc1','bc2','bc3'].forEach(id => document.getElementById(id).disabled = false);
  setLoading(false);
  setStatus('Analysis complete for ' + d.ticker, 'ready');
}

function onError(msg) {
  setLoading(false);
  document.getElementById('aiload').classList.remove('on');
  if (done) document.getElementById('aiwrap').classList.add('on');
  showToast(msg);
  setStatus('Error: ' + msg.substring(0, 90), 'err');
}

// ── Charts ─────────────────────────────────────────────────────────────────────
async function openChart(type) {
  if (!window.pywebview?.api) return;
  const d = await window.pywebview.api.get_chart_data(type);
  if (!d) return;

  const base = {
    paper_bgcolor: '#0d1117', plot_bgcolor: '#161b22',
    font: { color: '#8b949e', family: 'Inter,Segoe UI,system-ui', size: 12 },
    xaxis: { gridcolor: '#21262d', linecolor: '#30363d', tickcolor: '#8b949e', zeroline: false },
    yaxis: { gridcolor: '#21262d', linecolor: '#30363d', tickcolor: '#8b949e',
             tickprefix: '$', zeroline: false },
    legend: { bgcolor: '#161b22', bordercolor: '#30363d', font: { color: '#e6edf3' } },
    margin: { t: 20, r: 20, b: 50, l: 75 },
    hovermode: 'x unified',
    hoverlabel: { bgcolor: '#21262d', bordercolor: '#30363d',
                  font: { color: '#e6edf3', size: 12 } },
  };

  let traces;
  if (d.type === 'predicted') {
    traces = [
      { x: d.hist_dates, y: d.hist_vals, name: 'Historical Close',
        type: 'scatter', mode: 'lines',
        line: { color: '#58a6ff', width: 2 },
        fill: 'tozeroy', fillcolor: 'rgba(88,166,255,0.07)',
        hovertemplate: '%{x}<br><b>$%{y:.2f}</b><extra>Historical</extra>' },
      { x: d.pred_dates, y: d.pred_vals, name: 'XGBoost Forecast',
        type: 'scatter', mode: 'lines',
        line: { color: '#f85149', width: 2.5, dash: 'dot' },
        fill: 'tozeroy', fillcolor: 'rgba(248,81,73,0.07)',
        hovertemplate: '%{x}<br><b>$%{y:.2f}</b><extra>Forecast</extra>' },
    ];
  } else {
    traces = [{
      x: d.dates, y: d.values, name: d.name,
      type: 'scatter', mode: 'lines',
      line: { color: d.color, width: 2 },
      fill: 'tozeroy', fillcolor: d.fill,
      hovertemplate: '%{x}<br><b>$%{y:.2f}</b><extra>' + d.name + '</extra>'
    }];
  }

  document.getElementById('mttl').textContent = d.title;
  document.getElementById('moverlay').classList.add('on');
  Plotly.newPlot('chartdiv', traces, base,
    { responsive: true, displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d','select2d','autoScale2d'] });
}

function closeChart() {
  document.getElementById('moverlay').classList.remove('on');
  Plotly.purge('chartdiv');
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function setStatus(msg, type) {
  document.getElementById('stxt').textContent = msg;
  document.getElementById('sdot').className = 'sdot ' + (type || '');
}

function av(id, val, cls) {
  const el = document.getElementById(id);
  el.textContent = val;
  if (cls) el.className = 'iv ' + cls;
  el.classList.remove('flash');
  void el.offsetWidth;
  el.classList.add('flash');
}

let _tt;
function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('on');
  clearTimeout(_tt);
  _tt = setTimeout(() => t.classList.remove('on'), 6500);
}

document.addEventListener('keydown', e => { if (e.key === 'Escape') closeChart(); });
setTimeout(() => document.getElementById('search').focus(), 600);
</script>
</body>
</html>""".replace("__MODEL_SHORT__", model_short)


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    api    = Api()
    window = webview.create_window(
        title    = "Stock Market Predictor & Suggestor",
        html     = get_html(),
        js_api   = api,
        width    = 1440,
        height   = 860,
        min_size = (1100, 700),
    )
    _window_ref = window          # store globally so Api._js() can reach it
    webview.start(debug=False)
