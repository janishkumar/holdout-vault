FROM python:3.13-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml /app/
COPY src /app/src
RUN pip install /app \
 && useradd --uid 10001 --create-home vault \
 && mkdir -p /vault /scratch \
 && chown vault:vault /vault /scratch
USER 10001
ENV HOLDOUT_VAULT_HOME=/vault HOLDOUT_VAULT_MODE=sealed
EXPOSE 7341
CMD ["holdout-vault", "vault-api", "--host", "0.0.0.0", "--port", "7341"]
