
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

COPY app/requirements.txt ./requirements.txt
RUN python3 -m pip install --no-cache-dir -r ./requirements.txt

COPY app/ ./
COPY models/qwen2.5-vl-3b/ /models/qwen2.5-vl-3b/

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

CMD ["python3", "/app/worker.py"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=3m --retries=6 CMD python3 -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1',45831)); s.close()" || exit 1