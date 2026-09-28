FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY overspend_api.py overspend_api.pkl ./
EXPOSE 8000
CMD ["uvicorn", "overspend_api:app", "--host", "0.0.0.0", "--port", "8000"]
