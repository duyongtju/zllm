
curl -v -XPOST http://127.0.0.1:8000/generate \
  -d '{"temperature":0.95, "top_p":0.8, max_tokens=50, "prompt":"上海有什么好玩的地方？","stream":true}'