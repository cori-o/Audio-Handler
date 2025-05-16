# Dockerfile for Audio Handler  
# write by Jaedong, Oh (2025.05.16)
# --- Builder stage ---
FROM python:3.12-slim-bullseye as builder
WORKDIR /app
ARG GITHUB_TOKEN

# 필수 도구만 설치
RUN apt-get update && apt-get install -y --no-install-recommends python3-venv build-essential git locales ffmpeg && \
    python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .

# 필요 패키지 설치
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install git+https://github.com/wenet-e2e/wespeaker.git && \
    pip install git+https://${GITHUB_TOKEN}@github.com/Jaedong95/nsnet2-denoiser.git

# --- Inference image ---
FROM python:3.12-slim-bullseye
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /audio-handler

COPY --from=builder /opt/venv /opt/venv
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg
# CMD ["python", "main.py"]