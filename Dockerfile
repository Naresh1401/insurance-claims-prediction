FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/raw data/processed data/models plots

EXPOSE 8000 8501

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
