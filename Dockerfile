FROM ghcr.io/astral-sh/uv:0.12.23 AS uv
FROM python:3.12-slim AS base
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /home/app/mlflow \
    && chown -R app:app /app /home/app
FROM base AS tracking
RUN uv sync --frozen --no-dev --extra tracking
USER app
EXPOSE 5000
CMD ["python", "-m", "mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
FROM base AS api
RUN uv sync --frozen --no-dev --extra ml
USER app
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "trajectory_mlops.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
FROM base AS ui
RUN uv sync --frozen --no-dev --extra ui
COPY app ./app
USER app
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
