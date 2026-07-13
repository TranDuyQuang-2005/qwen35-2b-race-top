# Qwen3.5-2B Race v4 - Prefix Warmup Proxy

This image is based on `vllm/vllm-openai:v0.22.1` and keeps the real vLLM OpenAI-compatible server.

Main changes versus v3:

- Run vLLM internally on port 8001.
- Warm the public trace common system prefix before exposing port 8000.
- Expose a lightweight proxy on port 8000 only after vLLM is ready and prefix warmup is finished.
- Keep BF16 model weights for accuracy.
- Keep FP8 KV cache, prefix caching, chunked prefill, and low-latency scheduling.

Docker Hub target image:

```text
duquang/qwen35-2b-race-top:v4
```

Submit `docker-compose.yml` after GitHub Actions pushes the image.
