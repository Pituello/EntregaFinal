# Una sola imagen para la API y los workers (cambia solo el comando, ver docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencias primero: si solo cambia el código, Docker reutiliza esta capa.
# requirements.lock = las versiones exactas con las que se probó la entrega.
COPY requirements.txt requirements.lock ./
RUN pip install -r requirements.lock

COPY orquestador ./orquestador
COPY app ./app
COPY data ./data
COPY scripts ./scripts
COPY tests ./tests
COPY pytest.ini ./

# Sin privilegios de root dentro del contenedor.
RUN useradd --create-home --uid 1000 vw && mkdir -p evidencia && chown -R vw /app
USER vw

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
