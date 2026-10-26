#!/usr/bin/env bash
# Levanta el sistema completo con un solo comando (Linux, macOS o Git Bash en Windows):
#
#   ./run.sh              -> Redis + Phoenix + API + 1 worker
#   ./run.sh 3            -> lo mismo con 3 workers independientes
#
# En una máquina limpia solo hace falta Docker. Si no existe el .env, lo crea desde
# .env.example y pide la OPENAI_API_KEY (no se muestra al escribirla y nunca se sube al repo).
set -euo pipefail
cd "$(dirname "$0")"

WORKERS="${1:-1}"

if ! docker info >/dev/null 2>&1; then
  echo "❌ Docker no está corriendo. Abrí Docker Desktop y volvé a ejecutar ./run.sh"
  exit 1
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "📝 Se creó .env desde .env.example"
fi
if ! grep -qE '^OPENAI_API_KEY=.+' .env; then
  read -rsp "🔑 Pegá tu OPENAI_API_KEY (no se muestra): " clave; echo
  [ -z "$clave" ] && { echo "❌ Sin API key no se puede usar el LLM."; exit 1; }
  # Reemplaza la línea vacía sin depender de "sed -i" (distinto en macOS y en Git Bash).
  grep -vE '^OPENAI_API_KEY=' .env > .env.tmp && { echo "OPENAI_API_KEY=$clave"; cat .env.tmp; } > .env && rm .env.tmp
fi

echo "🐳 Construyendo y levantando (workers: $WORKERS)..."
docker compose up -d --build --scale worker="$WORKERS"

printf "⏳ Esperando a la API"
for _ in $(seq 1 60); do
  if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
    echo " ✅"
    curl -s http://localhost:8000/health; echo
    echo
    echo "API (Swagger):  http://localhost:8000/docs"
    echo "Phoenix:        http://localhost:6006"
    echo "Logs:           docker compose logs -f worker"
    echo "Tests:          docker compose run --rm -e REDIS_URL_TESTS=redis://redis:6379/0 api python -m pytest -q"
    echo "Apagar:         docker compose down      (los datos quedan en los volúmenes)"
    exit 0
  fi
  printf "."; sleep 2
done
echo
echo "❌ La API no respondió en 2 minutos. Revisá: docker compose logs api worker"
exit 1
