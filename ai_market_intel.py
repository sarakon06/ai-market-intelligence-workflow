from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    target_ticker: str = "^GSPC"
    peer_tickers: Tuple[str, ...] = ("MSFT", "GOOGL", "AMZN")
    start_date: str = "2015-01-01"
    end_date: str = "2025-01-01"
    frequency: str = "ME"  # MonthEnd alias compatible with modern pandas
    test_horizon_months: int = 24
    min_train_months: int = 48
    n_estimators: int = 300
    random_state: int = 42
    positive_threshold: float = 0.55
    output_dir: str = "yellowwood_outputs"
    fred_indicators: Dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.fred_indicators is None:
            self.fred_indicators = {
                "DFF": "FedFundsRate",
                "CPIAUCSL": "CPI",
                "UNRATE": "UnemploymentRate",
                "UMCSENT": "ConsumerSentiment",
                "RSAFS": "RetailSales",
            }


class MarketIntelligencePipeline:
    """
    Decision-support oriented ML pipeline for market / macro signal screening.

    Updated to:
    - avoid pandas_datareader entirely
    - download FRED data directly from FRED CSV endpoints
    - use pandas-compatible month-end frequency alias (ME)
    - make yfinance behavior explicit with auto_adjust=False
    - produce recruiter-friendly structured outputs
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _clean_ticker(ticker: str) -> str:
        return ticker.replace("^", "")

    @staticmethod
    def _pct_change(series: pd.Series, periods: int) -> pd.Series:
        return series.pct_change(periods=periods, fill_method=None)

    def download_price_data(self, tickers: List[str]) -> pd.DataFrame:
        logger.info("Downloading price data for %s", ", ".join(tickers))
        raw = yf.download(
            tickers,
            start=self.cfg.start_date,
            end=self.cfg.end_date,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if raw.empty:
            raise ValueError("No market data returned from Yahoo Finance.")

        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" not in raw.columns.get_level_values(0):
                raise KeyError("Downloaded data did not include a Close field.")
            close = raw["Close"].copy()
        else:
            if "Close" not in raw.columns:
                raise KeyError("Downloaded data did not include a Close field.")
            close = raw[["Close"]].copy()
            close.columns = [tickers[0]]

        close = close.dropna(axis=1, how="all")
        close.columns = [f"{self._clean_ticker(str(c))}_Close" for c in close.columns]
        close.index = pd.to_datetime(close.index)
        close = close.sort_index()
        return close

    def _download_single_fred_series(self, code: str, friendly_name: str) -> pd.DataFrame:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
        df = pd.read_csv(url)
        if "DATE" not in df.columns or code not in df.columns:
            raise ValueError(f"Unexpected FRED response format for {code}.")

        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.rename(columns={"DATE": "Date", code: friendly_name}).dropna(subset=["Date"])
        df[friendly_name] = pd.to_numeric(df[friendly_name], errors="coerce")
        df = df.set_index("Date").sort_index()
        return df.loc[self.cfg.start_date:self.cfg.end_date]

    def download_fred_data(self) -> pd.DataFrame:
        logger.info("Downloading FRED indicators")
        frames: List[pd.DataFrame] = []

        for code, friendly_name in self.cfg.fred_indicators.items():
            try:
                frames.append(self._download_single_fred_series(code, friendly_name))
            except Exception as exc:
                logger.warning("Skipping FRED series %s: %s", code, exc)

        if not frames:
            logger.warning("No FRED data available; pipeline will continue with price-only features.")
            return pd.DataFrame()

        fred = pd.concat(frames, axis=1).sort_index()
        return fred.ffill().bfill()

    def merge_data(self, prices: pd.DataFrame, fred: pd.DataFrame) -> pd.DataFrame:
        monthly_prices = prices.resample(self.cfg.frequency).last().sort_index()
        if fred.empty:
            return monthly_prices

        merged = pd.merge_asof(
            monthly_prices,
            fred.sort_index(),
            left_index=True,
            right_index=True,
            direction="backward",
        )
        return merged.ffill().bfill()

    def engineer_features(self, df: pd.DataFrame):
        target = self._clean_ticker(self.cfg.target_ticker)
        peers = [self._clean_ticker(t) for t in self.cfg.peer_tickers]
        target_col = f"{target}_Close"

        if target_col not in df.columns:
            raise KeyError(f"Target column {target_col} not found in dataset.")

        work = df.copy()

        # Target series features
        work[f"{target}_ret_1m"] = self._pct_change(work[target_col], 1)
        work[f"{target}_ret_3m"] = self._pct_change(work[target_col], 3)
        work[f"{target}_ret_6m"] = self._pct_change(work[target_col], 6)
        work[f"{target}_vol_3m"] = work[f"{target}_ret_1m"].rolling(3).std()
        work[f"{target}_vol_6m"] = work[f"{target}_ret_1m"].rolling(6).std()
        work[f"{target}_ma_gap_3m"] = work[target_col] / work[target_col].rolling(3).mean() - 1
        work[f"{target}_ma_gap_6m"] = work[target_col] / work[target_col].rolling(6).mean() - 1
        work[f"{target}_drawdown_12m"] = work[target_col] / work[target_col].rolling(12).max() - 1

        # Peer-relative performance
        for peer in peers:
            peer_col = f"{peer}_Close"
            if peer_col in work.columns:
                work[f"{peer}_ret_1m"] = self._pct_change(work[peer_col], 1)
                work[f"rel_perf_{target}_vs_{peer}_3m"] = self._pct_change(work[target_col], 3) - self._pct_change(work[peer_col], 3)
                work[f"rel_perf_{target}_vs_{peer}_6m"] = self._pct_change(work[target_col], 6) - self._pct_change(work[peer_col], 6)

        # Macro level + changes
        for macro_col in self.cfg.fred_indicators.values():
            if macro_col in work.columns:
                work[f"{macro_col}_chg_1m"] = self._pct_change(work[macro_col], 1)
                work[f"{macro_col}_chg_3m"] = self._pct_change(work[macro_col], 3)
                work[f"{macro_col}_lag_1"] = work[macro_col].shift(1)
                work[f"{macro_col}_lag_3"] = work[macro_col].shift(3)

        # Next-period binary target
        work["target_up_next_period"] = (work[f"{target}_ret_1m"].shift(-1) > 0).astype(int)

        metadata_cols = [c for c in work.columns if c.endswith("_Close")]
        non_feature_cols = metadata_cols + ["target_up_next_period"]

        X = work.drop(columns=non_feature_cols, errors="ignore")
        y = work["target_up_next_period"]

        valid = X.notna().all(axis=1) & y.notna()
        X = X.loc[valid].copy()
        y = y.loc[valid].copy()
        aligned = work.loc[valid].copy()
        return X, y, aligned

    def walk_forward_backtest(self, X: pd.DataFrame, y: pd.Series):
        min_train = self.cfg.min_train_months
        horizon = self.cfg.test_horizon_months

        if len(X) < min_train + 6:
            raise ValueError("Not enough observations for walk-forward validation.")

        start_idx = max(min_train, len(X) - horizon)
        predictions = []
        feature_buckets = []

        for i in range(start_idx, len(X)):
            X_train, y_train = X.iloc[:i], y.iloc[:i]
            X_test, y_test = X.iloc[[i]], y.iloc[[i]]

            model = RandomForestClassifier(
                n_estimators=self.cfg.n_estimators,
                random_state=self.cfg.random_state,
                class_weight="balanced",
                min_samples_leaf=2,
                max_depth=6,
            )
            model.fit(X_train, y_train)

            prob_up = float(model.predict_proba(X_test)[0, 1])
            pred = int(prob_up >= self.cfg.positive_threshold)
            actual = int(y_test.iloc[0])

            predictions.append(
                {
                    "date": X_test.index[0],
                    "actual": actual,
                    "predicted": pred,
                    "prob_up": prob_up,
                    "confidence": abs(prob_up - 0.5) * 2,
                }
            )
            feature_buckets.append(pd.Series(model.feature_importances_, index=X.columns))

        pred_df = pd.DataFrame(predictions).set_index("date")
        metrics = {
            "accuracy": float(accuracy_score(pred_df["actual"], pred_df["predicted"])),
            "precision": float(precision_score(pred_df["actual"], pred_df["predicted"], zero_division=0)),
            "recall": float(recall_score(pred_df["actual"], pred_df["predicted"], zero_division=0)),
            "f1": float(f1_score(pred_df["actual"], pred_df["predicted"], zero_division=0)),
            "mean_prob_up": float(pred_df["prob_up"].mean()),
            "coverage_months": int(len(pred_df)),
        }
        avg_importance = pd.concat(feature_buckets, axis=1).mean(axis=1).sort_values(ascending=False)
        return pred_df, metrics, avg_importance

    def train_final_model(self, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
        model = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            random_state=self.cfg.random_state,
            class_weight="balanced",
            min_samples_leaf=2,
            max_depth=6,
        )
        model.fit(X, y)
        return model

    def latest_signal(self, model: RandomForestClassifier, X: pd.DataFrame) -> Dict[str, object]:
        latest_idx = X.index[-1]
        latest_row = X.iloc[[-1]]
        prob_up = float(model.predict_proba(latest_row)[0, 1])
        signal = "Up" if prob_up >= self.cfg.positive_threshold else "Down"
        return {
            "as_of": latest_idx.strftime("%Y-%m-%d"),
            "signal": signal,
            "prob_up": round(prob_up, 4),
            "confidence": round(abs(prob_up - 0.5) * 2, 4),
        }

    def create_research_brief(self, metrics: Dict[str, float], latest_signal: Dict[str, object], feature_importance: pd.Series) -> str:
        top_features = feature_importance.head(5)
        lines = [
            "# Market Intelligence Brief",
            "",
            "## Executive Summary",
            f"- Latest model signal: **{latest_signal['signal']}** as of {latest_signal['as_of']}",
            f"- Probability of positive next-period move: **{latest_signal['prob_up']:.1%}**",
            f"- Signal confidence score: **{latest_signal['confidence']:.1%}**",
            f"- Walk-forward accuracy over the most recent {metrics['coverage_months']} months: **{metrics['accuracy']:.1%}**",
            f"- Precision / Recall / F1: **{metrics['precision']:.1%} / {metrics['recall']:.1%} / {metrics['f1']:.1%}**",
            "",
            "## Most Influential Drivers",
        ]
        for feature, value in top_features.items():
            lines.append(f"- {feature}: {value:.4f}")
        lines += [
            "",
            "## Interpretation",
            "- This workflow is intended as a research prioritization and signal-screening tool, not a stand-alone investment engine.",
            "- The strongest value is in automating data ingestion, standardizing feature generation, and surfacing interpretable drivers for analyst review.",
            "- A natural next step for private equity use would be pairing this with LLM-based summarization of earnings calls, consumer reviews, retailer pages, and competitor updates.",
        ]
        return "\n".join(lines)

    def save_outputs(self, predictions: pd.DataFrame, metrics: Dict[str, float], latest_signal: Dict[str, object], feature_importance: pd.Series, brief: str) -> None:
        predictions.to_csv(self.output_dir / "predictions.csv")
        feature_importance.rename("importance").to_csv(self.output_dir / "feature_importance.csv")
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": asdict(self.cfg),
                    "metrics": metrics,
                    "latest_signal": latest_signal,
                    "top_features": feature_importance.head(10).round(6).to_dict(),
                },
                f,
                indent=2,
            )
        (self.output_dir / "research_brief.md").write_text(brief, encoding="utf-8")

    def run(self) -> None:
        all_tickers = [self.cfg.target_ticker, *self.cfg.peer_tickers]
        prices = self.download_price_data(all_tickers)
        fred = self.download_fred_data()
        merged = self.merge_data(prices, fred)
        X, y, _aligned = self.engineer_features(merged)

        logger.info("Feature matrix shape: %s", X.shape)
        logger.info("Date range after feature engineering: %s to %s", X.index.min().date(), X.index.max().date())

        predictions, metrics, avg_importance = self.walk_forward_backtest(X, y)
        final_model = self.train_final_model(X, y)
        latest = self.latest_signal(final_model, X)
        brief = self.create_research_brief(metrics, latest, avg_importance)
        self.save_outputs(predictions, metrics, latest, avg_importance, brief)

        logger.info("Run complete. Outputs saved to %s", self.output_dir.resolve())
        logger.info(
            "Latest signal: %s | P(up)=%.2f | Accuracy=%.2f",
            latest["signal"],
            latest["prob_up"],
            metrics["accuracy"],
        )


if __name__ == "__main__":
    pipeline = MarketIntelligencePipeline(Config())
    pipeline.run()
