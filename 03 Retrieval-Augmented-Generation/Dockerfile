FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

COPY app/requirements.txt ./requirements.txt
RUN python3 -m pip install --no-cache-dir -r ./requirements.txt

COPY app/app.py ./
COPY app/worker.py ./
COPY app/config.py ./

COPY models/qwen2.5-vl-3b/ /models/qwen2.5-vl-3b/
COPY models/qwen3-embedding-0.6b/ /models/qwen3-embedding-0.6b/

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

RUN mkdir -p ./corpus ./output

CMD ["python3", "worker.py"]