FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
RUN pip install --no-cache-dir uv==0.11.14
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY raya_maas ./raya_maas
RUN uv sync --frozen --no-dev && useradd --create-home --uid 10001 raya
ENV PATH="/opt/venv/bin:$PATH" RAYA_HOST=0.0.0.0
USER raya
EXPOSE 8000
HEALTHCHECK --interval=30s --start-period=180s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz',timeout=2)"
CMD ["raya-serve"]
