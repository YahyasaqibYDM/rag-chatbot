FROM python:3.11-slim

WORKDIR /app

# FFmpeg install karo (video processing ke liye)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Requirements install karo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Baaki code copy karo
COPY . .

# Port (Render $PORT deta hai)
ENV PORT=8000
EXPOSE 8000

# Server chalao
CMD uvicorn app:app --host 0.0.0.0 --port $PORT