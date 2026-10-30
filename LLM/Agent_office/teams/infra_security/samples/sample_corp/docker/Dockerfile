# 샘플상사(가상 회사) 웹 이미지 — 시험용.
FROM python:3.12-slim

ARG APP_VERSION=1.0.0
ENV APP_ENV=production
ENV API_SECRET_KEY=PLACEHOLDER

ADD https://downloads.example.com/agent/agent.tar.gz /opt/agent/

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

USER root
CMD ["python", "-m", "app"]
