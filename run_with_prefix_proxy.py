\
import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, StreamingResponse

INTERNAL_HOST = "127.0.0.1"
INTERNAL_PORT = int(os.environ.get("INTERNAL_VLLM_PORT", "8001"))
PUBLIC_HOST = "0.0.0.0"
PUBLIC_PORT = int(os.environ.get("PUBLIC_PORT", "8000"))
MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "Qwen3.5-2B")
PREFIX_PATH = Path(os.environ.get("PREFIX_PATH", "/app/prefix_system.txt"))

VLLM_CMD = [
    "python3", "-m", "vllm.entrypoints.openai.api_server",
    "--model=/model",
    f"--served-model-name={MODEL_NAME}",
    f"--host={INTERNAL_HOST}",
    f"--port={INTERNAL_PORT}",
    "--dtype=bfloat16",
    "--max-model-len=32768",
    "--gpu-memory-utilization=0.98",
    "--tensor-parallel-size=1",
    "--enable-prefix-caching",
    "--kv-cache-dtype=fp8",
    "--enable-chunked-prefill",
    "--max-num-batched-tokens=65536",
    "--max-num-seqs=8",
    "--performance-mode=interactivity",
    "--scheduling-policy=fcfs",
    "--no-enable-log-requests",
    "--trust-remote-code",
]


def start_vllm() -> subprocess.Popen:
    print("[runner] starting internal vLLM on port", INTERNAL_PORT, flush=True)
    print("[runner] cmd:", " ".join(VLLM_CMD), flush=True)
    return subprocess.Popen(VLLM_CMD)


async def wait_for_vllm(proc: subprocess.Popen, timeout_s: int = 900) -> None:
    deadline = time.time() + timeout_s
    url = f"http://{INTERNAL_HOST}:{INTERNAL_PORT}/v1/models"
    async with httpx.AsyncClient(timeout=10.0) as client:
        last_err = None
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"vLLM exited early with code {proc.returncode}")
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    print("[runner] internal vLLM is ready", flush=True)
                    return
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception as e:
                last_err = repr(e)
            await asyncio.sleep(2)
    raise TimeoutError(f"Timed out waiting for vLLM. Last error: {last_err}")


async def warm_prefix_cache() -> None:
    if not PREFIX_PATH.exists():
        print("[runner] no prefix file found; skip warmup", flush=True)
        return

    prefix = PREFIX_PATH.read_text(encoding="utf-8")
    print(f"[runner] warming common system prefix cache: {len(prefix)} chars", flush=True)
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": prefix},
            {"role": "user", "content": "Reply with one short word."},
        ],
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
    }
    url = f"http://{INTERNAL_HOST}:{INTERNAL_PORT}/v1/chat/completions"
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(url, json=payload)
        print("[runner] warmup status:", r.status_code, flush=True)
        if r.status_code >= 400:
            print("[runner] warmup body:", r.text[:1000], flush=True)
            r.raise_for_status()
    print("[runner] prefix warmup done; starting public proxy", flush=True)


def make_proxy_app(proc: subprocess.Popen) -> FastAPI:
    app = FastAPI()

    @app.middleware("http")
    async def check_backend_alive(request: Request, call_next):
        if proc.poll() is not None:
            return JSONResponse({"error": "internal vLLM process exited", "code": proc.returncode}, status_code=503)
        return await call_next(request)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def proxy(path: str, request: Request):
        target = f"http://{INTERNAL_HOST}:{INTERNAL_PORT}/{path}"
        if request.url.query:
            target += f"?{request.url.query}"

        # Preserve most headers but let httpx set hop-by-hop/content length.
        headers = dict(request.headers)
        for h in ["host", "content-length", "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"]:
            headers.pop(h, None)

        body = await request.body()
        client = httpx.AsyncClient(timeout=None)
        req = client.build_request(request.method, target, headers=headers, content=body)
        resp = await client.send(req, stream=True)

        excluded = {"content-encoding", "content-length", "transfer-encoding", "connection"}
        response_headers = {k: v for k, v in resp.headers.items() if k.lower() not in excluded}

        async def close_response():
            await resp.aclose()
            await client.aclose()

        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers=response_headers,
            media_type=resp.headers.get("content-type"),
            background=BackgroundTask(close_response),
        )

    return app


async def main_async() -> None:
    proc = start_vllm()

    def _shutdown(*_):
        print("[runner] shutdown signal received", flush=True)
        if proc.poll() is None:
            proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    await wait_for_vllm(proc)
    await warm_prefix_cache()

    app = make_proxy_app(proc)
    config = uvicorn.Config(app, host=PUBLIC_HOST, port=PUBLIC_PORT, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main_async())
