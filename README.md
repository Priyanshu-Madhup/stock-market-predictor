# Stock Market Predictor & Suggestor

A desktop application that combines **XGBoost machine learning** with **OpenRouter AI** to forecast stock prices and generate investment analysis — all inside a modern, web-based UI powered by `pywebview`.

---

## Features

- **Natural-language ticker resolution** — type a company name (e.g. "Apple", "Reliance Industries") and the AI resolves the correct Yahoo Finance ticker
- **XGBoost forecasting** — 26 engineered technical features, trained on 1 year of daily OHLCV data, iteratively predicts the next 30 trading days
- **AI analysis** — structured markdown report via OpenRouter covering company overview, trend analysis, investment recommendation (Buy / Hold / Sell), key risks, and recent developments
- **AI response caching** — results are stored in `ai_cache.json` so repeated lookups are instant and free
- **Interactive Plotly charts** — zoomable, hoverable candlestick-style line charts for closing price, opening price, and the 30-day forecast
- **Crash-free tab switching** — fully web-based UI; no Tkinter, no tkinterweb instability

---

## Tech Stack

| Layer | Technology |
|---|---|
| Desktop window | `pywebview` (EdgeChromium / WebView2) |
| Charts | Plotly.js (CDN) |
| Markdown rendering | marked.js (CDN) |
| Icons | Bootstrap Icons (CDN) |
| ML model | XGBoost + scikit-learn |
| Market data | yfinance |
| AI API | OpenRouter (OpenAI-compatible SDK) |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Priyanshu-Madhup/stock-market-predictor.git
cd stock-market-predictor

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Configure environment variables (see below)
cp .env.example .env
```

---

## Configuration

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=your_model_name_here
```

- Get a free API key at [openrouter.ai](https://openrouter.ai)
- Recommended model: `google/gemini-flash-1.5` (fast, cheap) or any model listed on OpenRouter

---

## Usage

```bash
python stock_predictor_01.py
```

1. Type a **company name or ticker symbol** in the search box
2. Click **Run Analysis** (or press Enter)
3. The app will:
   - Resolve the ticker via AI
   - Fetch 1 year of price history from Yahoo Finance
   - Engineer 26 technical features and train the XGBoost model
   - Generate a 30-day iterative forecast
   - Fetch (or load from cache) an AI analysis report
4. Switch between **AI Analysis** and **30-Day Forecast** tabs — no reloading, instant

---

## Project Structure

```
stock-market-predictor/
├── stock_predictor_01.py   # Main application (single-file)
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed)
├── ai_cache.json           # Cached AI responses (auto-generated)
└── README.md
```

---

## Requirements

- Python 3.10+
- Windows (WebView2 / EdgeChromium is required by pywebview on Windows)
- Internet connection (for yfinance data and OpenRouter API)

---

## License

[MIT](LICENSE)
