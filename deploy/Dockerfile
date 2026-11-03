FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY zarrbridge ./zarrbridge
RUN pip install --no-cache-dir . && useradd -m zb
USER zb
EXPOSE 8800
VOLUME ["/data"]
# Non-loopback bind: put a reverse proxy or --token in front of this in production.
CMD ["zarrbridge", "serve", "/data", "--host", "0.0.0.0", "--port", "8800", "--insecure", "--cors-origin", "*"]
