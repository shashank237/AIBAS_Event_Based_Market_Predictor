FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    tensorflow \
    statsmodels \
    scikit-learn \
    joblib

COPY run_prediction.py /app/run_prediction.py

CMD ["python", "run_prediction.py"]
