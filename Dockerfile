FROM vllm/vllm-openai:v0.22.1

WORKDIR /app

# httpx is used by the lightweight local proxy and warmup client.
RUN python3 -m pip install --no-cache-dir httpx

COPY run_with_prefix_proxy.py /app/run_with_prefix_proxy.py
COPY prefix_system.txt /app/prefix_system.txt

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

ENTRYPOINT ["python3", "/app/run_with_prefix_proxy.py"]
