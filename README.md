# Binance Futures Validation Chain Engine — Mobile Website

Responsive dashboard for Android Chrome. BTCUSDT USDⓈ-M perpetual, 5-minute candles, public Binance data, signal-only. No API key and no order execution.

## Deploy online (phone-friendly workflow)
1. Download and extract this ZIP.
2. Create a GitHub repository and upload the project files.
3. In Render, create a new **Web Service** and connect that repository.
4. Render can use `render.yaml`, or set:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Wait for deployment, open the generated HTTPS URL in Android Chrome.

You must perform the account/repository/deployment steps yourself; this package does not create or publish a public URL automatically. Free hosting may sleep, have cold starts, or impose limits.

## Local run
Python 3.11+:
```bash
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```
Open `http://127.0.0.1:8000`.

## Data source
Official USDⓈ-M Futures public REST API:
- Base: `https://fapi.binance.com`
- Klines: `/fapi/v1/klines`
- Server time: `/fapi/v1/time`
Only fully closed candles are retained. If Binance data is unavailable, the UI shows an error and does not substitute demo/fabricated values.

## Dashboard
- `/` responsive dashboard
- `/docs` API documentation
- `/health` service status
- `/api/market` latest closed candles and WAIT-first signal status
- `/api/backtest` POST research backtest
- `/api/validate` POST walk-forward/OOS/Monte Carlo diagnostics

## Safety and research limitations
- No trading API key, private account endpoints, order endpoint, or automatic order execution.
- Directional signals remain locked; current signal intentionally stays WAIT until strategy has been independently reviewed.
- Fee and slippage are configurable assumptions. Funding history is NOT included in baseline backtest. This must be added before using results to judge profitability.
- The simple simulator approximates execution and does not fully model intrabar stop/target ordering, spread, market impact, liquidation, or serial dependence.
- Monte Carlo resamples individual trade returns independently, so it is only a rough diagnostic.
- A result marked as research gates passed is not proof of profitability.
- Do not enter secret keys in the dashboard or repository.
