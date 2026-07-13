# Qwen3.5-2B Race Top Serving Stack
Mục tiêu cấu hình:

- Serve trực tiếp bằng vLLM OpenAI-compatible API.
- Giữ BF16 model weights để bảo toàn accuracy.
- Dùng FP8 KV cache để tăng khả năng xử lý đồng thời.
- Bật prefix caching để giảm prefill khi prompt có phần lặp.
- Bật chunked prefill để cân bằng TTFT và TPOT.
- Giới hạn `max_model_len=32768` để tránh lãng phí KV cache trên 18GB VRAM.
- Giảm overhead log khi benchmark.


