FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONTINUUM_LIBRARY_DIR=/var/lib/continuum/books

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY skills ./skills
COPY .codex-plugin ./.codex-plugin
COPY .mcp.json ./

RUN python -m pip install --no-cache-dir '.[plugin]' \
    && addgroup --system continuum \
    && adduser --system --ingroup continuum --home /var/lib/continuum continuum \
    && mkdir -p /var/lib/continuum/books \
    && chown -R continuum:continuum /var/lib/continuum

USER continuum
VOLUME ["/var/lib/continuum"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health', timeout=3)"

CMD ["sh", "-c", "exec continuum-memory-mcp --transport streamable-http --host 0.0.0.0 --port ${PORT:-8000} --path /mcp"]
