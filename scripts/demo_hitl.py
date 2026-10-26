"""
Demostración del Human-in-the-loop desde la línea de comandos.

    python scripts/demo_hitl.py                 # pregunta si aprobás
    python scripts/demo_hitl.py --aprobar       # aprueba sin preguntar
    python scripts/demo_hitl.py --rechazar --comentario "el cliente no dejó seña"

1. Pide una reserva (acción crítica) a la API.
2. Espera a que el job quede en WAITING_APPROVAL y muestra el pedido.
3. Envía la decisión a POST /tasks/{id}/approve.
4. Espera el resultado final: la reserva confirmada, o el rechazo.

Para probar que la pausa sobrevive a un reinicio: cortá la API (Ctrl+C)
mientras el script espera tu respuesta, volvé a levantarla y recién ahí
respondé. El estado del grafo está en Redis, no en la memoria del proceso.
"""
import argparse
import json
import sys
import time

import httpx

sys.stdout.reconfigure(encoding="utf-8")

TAREA = "Reservá una Saveiro Extreme cabina doble en la sucursal Córdoba para el cliente Juan Pérez."


def esperar(cliente: httpx.Client, job_id: str, estados: set[str], limite_s: float = 300) -> dict:
    ultimo, inicio = None, time.time()
    while time.time() - inicio < limite_s:
        try:
            job = cliente.get(f"/tasks/{job_id}").json()
        except httpx.ConnectError:
            print("   (la API no responde: ¿la estás reiniciando? sigo esperando...)")
            time.sleep(2)
            continue
        if job["estado"] != ultimo:
            ultimo = job["estado"]
            print(f"   estado: {ultimo}")
        if ultimo in estados:
            return job
        time.sleep(1)
    raise TimeoutError(f"el job no llegó a {estados} en {limite_s} s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo de aprobación humana")
    decision = parser.add_mutually_exclusive_group()
    decision.add_argument("--aprobar", action="store_true")
    decision.add_argument("--rechazar", action="store_true")
    parser.add_argument("--revisor", default="Damián")
    parser.add_argument("--comentario", default="")
    parser.add_argument("--tarea", default=TAREA)
    parser.add_argument("--api", default="http://localhost:8000")
    args = parser.parse_args()

    with httpx.Client(base_url=args.api, timeout=30) as cliente:
        job_id = cliente.post("/tasks", json={"tarea": args.tarea}).json()["job_id"]
        print(f"📝 Tarea encolada: job {job_id}\n   {args.tarea}")
        job = esperar(cliente, job_id, {"WAITING_APPROVAL", "DONE", "FAILED"})
        if job["estado"] != "WAITING_APPROVAL":
            print(json.dumps(job.get("resultado") or job.get("error"), ensure_ascii=False, indent=2))
            return

        pedido = job["aprobacion"]
        print(f"\n⏸️  El grafo está en pausa (estado guardado en Redis). Pedido de aprobación:")
        print(f"   {pedido['pregunta']}")
        print(f"   Por qué es crítica: {pedido['motivo']}")
        print(f"   Stock en el catálogo: {pedido['stock_en_catalogo']}")

        if args.aprobar or args.rechazar:
            aprobado = args.aprobar
        else:
            aprobado = input("\n¿Aprobás la reserva? [s/n]: ").strip().lower() in ("s", "si", "sí", "y")
        r = cliente.post(f"/tasks/{job_id}/approve",
                         json={"aprobado": aprobado, "revisor": args.revisor, "comentario": args.comentario})
        print(f"\n{'✅ Aprobada' if aprobado else '❌ Rechazada'} -> HTTP {r.status_code}: {r.json().get('detalle', r.json())}")

        final = esperar(cliente, job_id, {"DONE", "FAILED"})
        print(f"\n📨 Respuesta final:\n{(final.get('resultado') or {}).get('respuesta') or final.get('error')}")
        if final.get("resultado", {}).get("reserva"):
            print(f"\nReserva: {json.dumps(final['resultado']['reserva'], ensure_ascii=False)}")
        print("\nHistorial del job:")
        for evento in final["historial"]:
            print(f"   {evento['en']}  {evento['estado']:<17} {evento['detalle']}")


if __name__ == "__main__":
    main()
