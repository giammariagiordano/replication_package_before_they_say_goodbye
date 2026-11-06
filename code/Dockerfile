# Frontend build
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Backend (serves the built UI too)
FROM python:3.12-slim
# Java runs Senti4SD's feature extractor; the text-classification tools live in the mounted external/ (README)
RUN apt-get update && apt-get install -y --no-install-recommends git default-jre-headless && rm -rf /var/lib/apt/lists/*
WORKDIR /app/backend
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/app ./app
COPY --from=web /web/dist /app/frontend/dist
ENV DATA_DIR=/data EXTERNAL_DIR=/external
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
