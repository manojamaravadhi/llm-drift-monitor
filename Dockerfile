FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEEPEVAL_TELEMETRY_OPT_OUT=1 \
    LLMDRIFT_DB=/app/data/results.db

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY config ./config
COPY datasets ./datasets
COPY dashboard ./dashboard
COPY data ./data
COPY .streamlit ./.streamlit

RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8501 7860
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 0
CMD ["sh", "-c", "streamlit run dashboard/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
