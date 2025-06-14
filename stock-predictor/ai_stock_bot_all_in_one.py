# ai_stock_bot_all_in_one.py

import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime, timedelta, date, time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon", quiet=True)

# === Configuration ===
NEWS_API_KEY          = os.getenv("NEWS_API_KEY", "357a8a8f71d84227ad83268caa623330")
RAW_NEWS_DIR          = "data/raw_news"
FEATURES_CSV          = "data/features.csv"
MODEL_CLASS_PATH      = "models/stock_classifier.pkl"
MODEL_REG_PATH        = "models/stock_regressor.pkl"
HIST_PRICE_START      = "2022-01-01"
HIST_PRICE_END        = "2025-03-29"
DEFAULT_TICKER        = "NVDA"

# === 1) Helper: Fetch a 30-day “window” from NewsAPI (cached) ===

def fetch_news_window(ticker: str, start_dt: datetime, end_dt: datetime, page_size: int = 100) -> pd.DataFrame:
    """
    Fetch all NewsAPI articles for `ticker` between start_dt (inclusive) and end_dt (exclusive).
    Returns a DataFrame of columns ["ticker","publishedAt","title","description","url"], 
    or empty DataFrame on any HTTP error (including 429).
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q":        ticker,
        "from":     start_dt.isoformat(),
        "to":       end_dt.isoformat(),
        "sortBy":   "publishedAt",
        "language": "en",
        "pageSize": page_size,
        "apiKey":   NEWS_API_KEY,
    }

    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", [])
    except requests.exceptions.HTTPError as e:
        print(f"[fetch_news_window]: HTTP error for {ticker} from {start_dt.date()} to {end_dt.date()}: {e}")
        return pd.DataFrame(columns=["ticker","publishedAt","title","description","url"])

    records = []
    for art in articles:
        records.append({
            "ticker":      ticker,
            "publishedAt": art.get("publishedAt"),
            "title":       art.get("title"),
            "description": art.get("description"),
            "url":         art.get("url"),
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    return df


def compute_vader(df: pd.DataFrame, text_col: str = "title") -> pd.DataFrame:
    """
    Add a 'sentiment_score' column via NLTK VADER to the DataFrame `df` of headlines.
    """
    sid = SentimentIntensityAnalyzer() # W variable name :)
    df = df.copy()
    df["sentiment_score"] = df[text_col].fillna("").apply(
        lambda t: sid.polarity_scores(t)["compound"]
    )
    return df



def download_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily stock data for `ticker` between `start` and `end` (YYYY-MM-DD),
    compute technical features, and return DataFrame with:
      date, Open, High, Low, Close, Volume,
      daily_return, rolling_5d_return, rolling_10d_vol, rolling_20d_vol,
      ma_diff_10, ma_diff_20, volume_change,
      label_up, future_return
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        progress=False,
        auto_adjust=True
    ).reset_index()

    # Flatten multi-level columns if any
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.columns = [str(col).strip() for col in df.columns]

    # Required columns: Date, Open, High, Low, Close, Volume
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    df["date"] = pd.to_datetime(df["Date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)

    # Previous day
    df["prev_close"]  = df["Close"].shift(1)
    df["prev_volume"] = df["Volume"].shift(1)

    # Daily return
    df["daily_return"] = (df["Close"] - df["prev_close"]) / df["prev_close"]
    df["daily_return"] = df["daily_return"].fillna(0.0)

    # Rolling volatilities
    df["rolling_5d_return"] = df["daily_return"].rolling(window=5).mean().fillna(0.0)
    df["rolling_10d_vol"]   = df["daily_return"].rolling(window=10).std().fillna(0.0)
    df["rolling_20d_vol"]   = df["daily_return"].rolling(window=20).std().fillna(0.0)

    # Moving averages (10d/20d) with backfill for first few rows
    df["ma_10"] = df["Close"].rolling(window=10).mean().bfill()
    df["ma_20"] = df["Close"].rolling(window=20).mean().bfill()

    df["ma_diff_10"] = df["Close"] - df["ma_10"]
    df["ma_diff_20"] = df["Close"] - df["ma_20"]

    # Volume change ratio
    df["volume_change"] = (df["Volume"] - df["prev_volume"]) / df["prev_volume"]
    df["volume_change"] = df["volume_change"].fillna(0.0)

    # Classification label: next-day up or down
    df["future_close"] = df["Close"].shift(-1)
    df["label_up"]     = np.where(df["future_close"] > df["Close"], 1, 0)

    # Regression target: next-day return
    df["future_return"] = (df["future_close"] - df["Close"]) / df["Close"]

    df = df.dropna(subset=["future_close", "prev_close"]).reset_index(drop=True)

    return df[[
        "date", "Open", "High", "Low", "Close", "Volume",
        "daily_return", "rolling_5d_return", "rolling_10d_vol", "rolling_20d_vol",
        "ma_diff_10", "ma_diff_20", "volume_change",
        "label_up", "future_return"
    ]]



def build_features(ticker: str) -> pd.DataFrame:
    """
    1) Download price history with technical features.
    2) Break the date range into 30-day windows, fetch once per window, cache under data/raw_news/.
    3) Concatenate all window CSVs, compute VADER, group by date → (avg_sentiment, std_sentiment, count_articles).
    4) Merge those daily sentiment aggregates into the price DataFrame.
    5) Compute sentiment_x_return and save to data/features.csv.
    """

    # ── a) Download and preprocess price data ──
    df_price = download_prices(ticker, HIST_PRICE_START, HIST_PRICE_END)
    all_dates = sorted(df_price["date"].unique())

    # ── b) Break the full range into 30-day windows ──
    window_size = 30
    windows = []
    start_date = all_dates[0]
    while start_date <= all_dates[-1]:
        end_date = min(start_date + timedelta(days=window_size - 1), all_dates[-1])
        windows.append((start_date, end_date))
        start_date = end_date + timedelta(days=1)

    # Ensure the raw_news directory exists
    if not os.path.isdir(RAW_NEWS_DIR):
        os.makedirs(RAW_NEWS_DIR, exist_ok=True)

    # ── c) For each window, fetch & cache if not already present ──
    for (win_start, win_end) in windows:
        start_dt = datetime.combine(win_start, time.min)
        end_dt   = datetime.combine(win_end + timedelta(days=1), time.min)
        fname    = f"{RAW_NEWS_DIR}/{ticker}_{win_start.isoformat()}_{win_end.isoformat()}.csv"

        if not os.path.isfile(fname):
            df_window = fetch_news_window(ticker, start_dt, end_dt, page_size=100)
            df_window.to_csv(fname, index=False)
            print(f"[build_features]: Fetched {len(df_window)} articles for {win_start} → {win_end} → {fname}")
        else:
            print(f"[build_features]: Window file exists, skipping fetch: {fname}")

    # ── d) Concatenate all raw-news CSVs and compute daily aggregates ──
    all_raw = []
    for fname in sorted(os.listdir(RAW_NEWS_DIR)):
        if fname.startswith(f"{ticker}_") and fname.endswith(".csv"):
            path = os.path.join(RAW_NEWS_DIR, fname)
            dfw = pd.read_csv(path, parse_dates=["publishedAt"])
            all_raw.append(dfw)

    if all_raw:
        df_all_news = pd.concat(all_raw, ignore_index=True)
    else:
        df_all_news = pd.DataFrame(columns=["ticker","publishedAt","title","description","url"])

    if not df_all_news.empty:
        df_all_news = compute_vader(df_all_news, text_col="title")
        df_all_news["published_date"] = df_all_news["publishedAt"].dt.date

        daily_agg = df_all_news.groupby("published_date").agg(
            avg_sentiment  = ("sentiment_score", "mean"),
            std_sentiment  = ("sentiment_score", "std"),
            count_articles = ("sentiment_score", "count")
        ).reset_index()
        daily_agg["std_sentiment"] = daily_agg["std_sentiment"].fillna(0.0)
    else:
        daily_agg = pd.DataFrame(columns=["published_date","avg_sentiment","std_sentiment","count_articles"])

    daily_agg.rename(columns={"published_date": "date"}, inplace=True)

    # ── e) Merge price DataFrame with daily_agg ──
    df_merge = pd.merge(df_price, daily_agg, on="date", how="left")
    df_merge["avg_sentiment"]   = df_merge["avg_sentiment"].fillna(0.0)
    df_merge["std_sentiment"]   = df_merge["std_sentiment"].fillna(0.0)
    df_merge["count_articles"]  = df_merge["count_articles"].fillna(0)

    # ── f) Compute interaction term ──
    df_merge["sentiment_x_return"] = df_merge["avg_sentiment"] * df_merge["daily_return"]

    # ── g) Ensure output directory, then save to CSV ──
    parent = os.path.dirname(FEATURES_CSV)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)

    df_merge.to_csv(FEATURES_CSV, index=False)
    print(f"[build_features]: Saved merged {len(df_merge)} rows to {FEATURES_CSV}")

    return df_merge



def train_models():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "daily_return",
        "rolling_5d_return",
        "rolling_10d_vol",
        "rolling_20d_vol",
        "ma_diff_10",
        "ma_diff_20",
        "volume_change",
        "avg_sentiment",
        "std_sentiment",
        "count_articles",
        "sentiment_x_return"
    ]

    X = df[feature_cols]
    y_class = df["label_up"]
    y_reg   = df["future_return"]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, shuffle=False)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, shuffle=False)

    # ── md 1
    rf_clf = RandomForestClassifier(random_state=42)
    param_grid = {"n_estimators": [100, 200], "max_depth": [None, 5, 10]}
    grid = GridSearchCV(rf_clf, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=0)
    grid.fit(X_train_c, y_train_c)
    best_clf = grid.best_estimator_
    pred_c = best_clf.predict(X_test_c)
    acc   = accuracy_score(y_test_c, pred_c)
    print(f"[train_models]: Classifier best params: {grid.best_params_}")
    print(f"[train_models]: Classification test accuracy = {acc:.4f}")

    # ── model 2 ;) ──
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_reg.fit(X_train_r, y_train_r)

    parent = os.path.dirname(MODEL_CLASS_PATH)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)

    joblib.dump(best_clf, MODEL_CLASS_PATH)
    print(f"[train_models]: Saved classifier to {MODEL_CLASS_PATH}")

    joblib.dump(rf_reg, MODEL_REG_PATH)
    print(f"[train_models]: Saved regressor to {MODEL_REG_PATH}")



def predict_today(ticker: str = DEFAULT_TICKER):
    """
    1) Define “yesterday” as date.today() - 2 (so our price features come from the last trading day).
    2) Fetch exactly that day’s headlines (24h UTC window), compute VADER, save raw/filtered CSVs, 
       compute per-article impact = sentiment_score * yesterday’s daily_return.
    3) Download last ~90 days of price data up to “yesterday”, compute yesterday’s feature row.
    4) Predict direction (classifier) + next-day return (regressor), print out recommendation + point change.
    """

    yesterday_date = date.today() - timedelta(days=2)

    start_dt = datetime.combine(yesterday_date, time.min)
    end_dt   = start_dt + timedelta(days=1)
    df_news_all = fetch_news_window(ticker, start_dt, end_dt, page_size=200)

    if not os.path.isdir("data"):
        os.makedirs("data", exist_ok=True)

    raw_file = f"data/raw_articles_{ticker}_{yesterday_date.isoformat()}.csv"
    df_news_all.to_csv(raw_file, index=False)
    print(f"[predict]: Saved raw fetch ({len(df_news_all)} rows) to {raw_file}")

    if not df_news_all.empty:
        df_news_all["date"] = df_news_all["publishedAt"].dt.date
        df_news_yesterday = df_news_all[df_news_all["date"] == yesterday_date].reset_index(drop=True)
    else:
        df_news_yesterday = df_news_all.copy()

    if df_news_yesterday.empty and not df_news_all.empty:
        df_news_yesterday = df_news_all.copy()
        print(f"[predict]: No articles stamped exactly {yesterday_date}, using all {len(df_news_all)} rows.")

    if not df_news_yesterday.empty:
        df_news_yesterday = compute_vader(df_news_yesterday, text_col="title")
        filtered_file = f"data/filtered_articles_{ticker}_{yesterday_date.isoformat()}.csv"
        df_news_yesterday.to_csv(filtered_file, index=False)
        print(f"[predict]: Saved filtered ({len(df_news_yesterday)} rows) to {filtered_file}")

        avg_s = df_news_yesterday["sentiment_score"].mean()
        std_s = df_news_yesterday["sentiment_score"].std() if len(df_news_yesterday) > 1 else 0.0
        c     = len(df_news_yesterday)
    else:
        avg_s, std_s, c = 0.0, 0.0, 0

    end_date   = yesterday_date
    start_date = end_date - timedelta(days=90)
    df_hist = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        interval="1d",
        progress=False,
        auto_adjust=True
    ).reset_index()

    df_hist.columns = [col[0] if isinstance(col, tuple) else col for col in df_hist.columns]
    df_hist.columns = [str(col).strip() for col in df_hist.columns]
    df_hist["date"] = pd.to_datetime(df_hist["Date"]).dt.date
    df_hist = df_hist.sort_values("date").reset_index(drop=True)

    df_hist["prev_close"]  = df_hist["Close"].shift(1)
    df_hist["daily_return"] = (df_hist["Close"] - df_hist["prev_close"]) / df_hist["prev_close"]
    df_hist["daily_return"] = df_hist["daily_return"].fillna(0.0)

    df_hist["rolling_5d_return"] = df_hist["daily_return"].rolling(window=5).mean().fillna(0.0)
    df_hist["rolling_10d_vol"]   = df_hist["daily_return"].rolling(window=10).std().fillna(0.0)
    df_hist["rolling_20d_vol"]   = df_hist["daily_return"].rolling(window=20).std().fillna(0.0)

    df_hist["ma_10"] = df_hist["Close"].rolling(window=10).mean().bfill()
    df_hist["ma_20"] = df_hist["Close"].rolling(window=20).mean().bfill()
    df_hist["ma_diff_10"] = df_hist["Close"] - df_hist["ma_10"]
    df_hist["ma_diff_20"] = df_hist["Close"] - df_hist["ma_20"]

    df_hist["volume_change"] = (df_hist["Volume"] - df_hist["Volume"].shift(1)) / df_hist["Volume"].shift(1)
    df_hist["volume_change"] = df_hist["volume_change"].fillna(0.0)

    if yesterday_date not in set(df_hist["date"]):
        raise ValueError(f"No price data for {ticker} on {yesterday_date} (must be a trading day).")

    row = df_hist.loc[df_hist["date"] == yesterday_date].iloc[0]
    daily_return_yesterday = float(row["daily_return"])
    five_day_return        = float(row["rolling_5d_return"])
    ten_day_volatility     = float(row["rolling_10d_vol"])
    twenty_day_volatility  = float(row["rolling_20d_vol"])
    ma_diff_10             = float(row["ma_diff_10"])
    ma_diff_20             = float(row["ma_diff_20"])
    volume_change          = float(row["volume_change"])
    close_price_yesterday  = float(row["Close"])

    # impact calculation **** needs refinement ****
    if not df_news_yesterday.empty:
        df_imp = df_news_yesterday.copy()
        df_imp["impact"] = df_imp["sentiment_score"] * daily_return_yesterday
        imp_file = f"data/articles_with_impact_{ticker}_{yesterday_date.isoformat()}.csv"
        df_imp[[
            "ticker", "publishedAt", "title", "description", "url",
            "sentiment_score", "impact"
        ]].to_csv(imp_file, index=False)
        print(f"[predict]: Saved articles+impact ({len(df_imp)}) to {imp_file}")

    feat_dict = {
        "daily_return":       daily_return_yesterday,
        "rolling_5d_return":  five_day_return,
        "rolling_10d_vol":    ten_day_volatility,
        "rolling_20d_vol":    twenty_day_volatility,
        "ma_diff_10":         ma_diff_10,
        "ma_diff_20":         ma_diff_20,
        "volume_change":      volume_change,
        "avg_sentiment":      avg_s,
        "std_sentiment":      std_s,
        "count_articles":     c,
        "sentiment_x_return": avg_s * daily_return_yesterday
    }
    df_feat = pd.DataFrame([feat_dict])

    clf = joblib.load(MODEL_CLASS_PATH)
    reg = joblib.load(MODEL_REG_PATH)

    p_up       = clf.predict_proba(df_feat)[0][1]
    pred_class = int(clf.predict(df_feat)[0])

    pred_ret = float(reg.predict(df_feat)[0])
    pred_point_change = pred_ret * close_price_yesterday

    rec = "HOLD"
    if pred_class == 1 and p_up > 0.6:
        rec = "BUY"
    elif pred_class == 0 and p_up > 0.6:
        rec = "SELL"

    print(f"[predict]: Ticker: {ticker}")
    print(f"  → P(up) = {p_up:.3f}  (direction = {'UP' if pred_class==1 else 'DOWN'})")
    print(f"  Rec: {rec}")
    print(f"  → Yesterday’s close ({yesterday_date}): ${close_price_yesterday:,.2f}")
    print(f"  → Predicted next-day return = {pred_ret*100:.2f}%")
    print(f"  → Predicted next-day point change = ${pred_point_change:,.2f}")



if __name__ == "__main__":
    ticker = "NVDA"

    print("=== Building features ===")
    build_features(ticker)

    print("=== Training models ===")
    train_models()

    print("=== Predicting today ===")
    predict_today(ticker)
