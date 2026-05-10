from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from app.core.observability import (
    mark_ml_prediction_finished,
    mark_ml_training_finished,
    mark_ml_training_started,
)
from app.core.settings import settings
from app.dtos.ml import (
    ModelMetricsDto,
    PredictionInputDto,
    PredictionOutputDto,
    TrainingRequestDto,
    TrainingResultDto,
)


class MachineLearningService:
    def __init__(self):
        self.data_dir = Path(settings.MODEL_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.data_dir / "lstm_stock_model.keras"
        self.scaler_path = self.data_dir / "close_scaler.pkl"
        self.metadata_path = self.data_dir / "training_metadata.pkl"

    def _prepare_sequences(
        self, scaled_prices: np.ndarray, sequence_length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(scaled_prices) <= sequence_length:
            raise ValueError(
                "Histórico insuficiente para a janela informada. "
                "Aumente o período de coleta ou reduza sequence_length."
            )

        features = []
        labels = []
        for index in range(sequence_length, len(scaled_prices)):
            features.append(scaled_prices[index - sequence_length : index, 0])
            labels.append(scaled_prices[index, 0])

        return np.array(features), np.array(labels)

    def _download_close_series(self, ticker: str, start_date: str, end_date: str | None) -> np.ndarray:
        dataset = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
        )
        if dataset.empty:
            raise ValueError("Não foi possível coletar dados para o ticker e período informados.")

        close_series = dataset["Close"].dropna().to_numpy(dtype=np.float32)
        if close_series.size == 0:
            raise ValueError("A série de fechamento retornou vazia para os parâmetros informados.")
        return close_series.reshape(-1, 1)

    def _build_model(self, sequence_length: int, lstm_units: int, learning_rate: float):
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Input
        from tensorflow.keras.optimizers import Adam

        model = Sequential(
            [
                Input(shape=(sequence_length, 1)),
                LSTM(lstm_units),
                Dense(1),
            ]
        )
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
        return model

    def train_model(self, payload: TrainingRequestDto) -> TrainingResultDto:
        mark_ml_training_started()
        training_started_at = perf_counter()
        training_succeeded = False
        try:
            close_prices = self._download_close_series(
                ticker=payload.ticker,
                start_date=payload.start_date,
                end_date=payload.end_date,
            )

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_prices = scaler.fit_transform(close_prices).astype(np.float32)

            x_data, y_data = self._prepare_sequences(scaled_prices, payload.lookback_window_size)

            split_index = int(len(x_data) * payload.training_data_ratio)
            if split_index <= 0 or split_index >= len(x_data):
                raise ValueError("training_data_ratio gerou divisão inválida entre treino e validação.")

            x_train, x_validation = x_data[:split_index], x_data[split_index:]
            y_train, y_validation = y_data[:split_index], y_data[split_index:]

            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
            x_validation = x_validation.reshape((x_validation.shape[0], x_validation.shape[1], 1))

            model = self._build_model(
                sequence_length=payload.lookback_window_size,
                lstm_units=payload.lstm_hidden_units,
                learning_rate=payload.learning_rate,
            )

            model.fit(
                x_train,
                y_train,
                validation_data=(x_validation, y_validation),
                epochs=payload.training_epochs,
                batch_size=payload.samples_per_batch,
                verbose=0,
            )

            predicted_scaled = model.predict(x_validation, verbose=0)
            predicted_values = scaler.inverse_transform(predicted_scaled)
            expected_values = scaler.inverse_transform(y_validation.reshape(-1, 1))

            mae = float(mean_absolute_error(expected_values, predicted_values))
            rmse = float(np.sqrt(mean_squared_error(expected_values, predicted_values)))
            mape = float(mean_absolute_percentage_error(expected_values, predicted_values) * 100)

            model.save(self.model_path)
            joblib.dump(scaler, self.scaler_path)

            metadata = {
                "ticker": payload.ticker,
                "start_date": payload.start_date,
                "end_date": payload.end_date,
                "lookback_window_size": payload.lookback_window_size,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "validation_samples": int(len(expected_values)),
            }
            joblib.dump(metadata, self.metadata_path)

            training_succeeded = True
            return TrainingResultDto(**metadata)
        finally:
            training_duration = perf_counter() - training_started_at
            mark_ml_training_finished(duration_seconds=training_duration, success=training_succeeded)

    def _load_artifacts(self):
        if not self.model_path.exists() or not self.scaler_path.exists() or not self.metadata_path.exists():
            raise ValueError("Modelo não treinado. Execute /api/v1/ml/train antes da inferência.")

        from tensorflow.keras.models import load_model

        model = load_model(self.model_path)
        scaler: MinMaxScaler = joblib.load(self.scaler_path)
        metadata: dict = joblib.load(self.metadata_path)
        return model, scaler, metadata

    def predict(self, payload: PredictionInputDto) -> PredictionOutputDto:
        prediction_succeeded = False
        try:
            model, scaler, metadata = self._load_artifacts()
            lookback_window_size = metadata.get("lookback_window_size", metadata.get("sequence_length"))
            if lookback_window_size is None:
                raise ValueError("Metadados inválidos: janela de entrada não encontrada.")

            if len(payload.recent_closing_prices) != lookback_window_size:
                raise ValueError(
                    f"O histórico deve conter exatamente {lookback_window_size} valores de fechamento."
                )

            historical = np.array(payload.recent_closing_prices, dtype=np.float32).reshape(-1, 1)
            scaled_historical = scaler.transform(historical)
            x_input = scaled_historical.reshape((1, lookback_window_size, 1))

            prediction_scaled = model.predict(x_input, verbose=0)
            predicted_close = float(scaler.inverse_transform(prediction_scaled)[0][0])

            prediction_succeeded = True
            return PredictionOutputDto(
                predicted_close=predicted_close,
                ticker=metadata["ticker"],
                lookback_window_size=lookback_window_size,
            )
        finally:
            mark_ml_prediction_finished(success=prediction_succeeded)

    def get_latest_metrics(self) -> ModelMetricsDto:
        if not self.metadata_path.exists():
            raise ValueError("Nenhum treinamento encontrado.")
        metadata: dict = joblib.load(self.metadata_path)
        if "lookback_window_size" not in metadata and "sequence_length" in metadata:
            metadata["lookback_window_size"] = metadata["sequence_length"]
        return ModelMetricsDto(**metadata)
