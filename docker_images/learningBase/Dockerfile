FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir pandas numpy scikit-learn statsmodels tensorflow joblib

COPY data /app/data
COPY scripts /app/scripts

RUN mkdir /app/models

CMD ["python", "/app/scripts/train_models.py"]
