# Build stage: install all Python deps then strip inference-irrelevant packages.
# Using a venv so the runtime stage snapshots only the final cleaned-up filesystem.
FROM python:3.12-slim AS builder

# GPU support:
#   NVIDIA (default, Turing+ incl. Blackwell):       set CUDA=true
#   NVIDIA legacy (Maxwell/Pascal/Volta, no Blackwell): set CUDA=true and CUDA_LEGACY=true
#   AMD:    set ROCM=true  (requires ROCm drivers on the host)
#   Rockchip NPU: set RKNN=true (arm64 only, requires the rknpu driver on the host)
#   None:   leave all false (CPU-only, slow but works)
ARG CUDA=false
ARG CUDA_LEGACY=false
ARG ROCM=false
ARG RKNN=false

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install torch first so it gets its own cached layer.
# cu128 wheels (default) drop sm_50/60/70 to fit PyPI size limits; cu126 wheels keep
# Maxwell through Hopper but lack Blackwell (sm_100/120). See pytorch/pytorch#145544.
RUN if [ "$CUDA" = "true" ] && [ "$CUDA_LEGACY" = "true" ]; then \
      pip install --no-cache-dir \
        torch==2.7.0+cu126 \
        torchvision==0.22.0+cu126 \
        --extra-index-url https://download.pytorch.org/whl/cu126; \
    elif [ "$CUDA" = "true" ]; then \
      pip install --no-cache-dir \
        torch==2.7.0+cu128 \
        torchvision==0.22.0+cu128 \
        --extra-index-url https://download.pytorch.org/whl/cu128; \
    elif [ "$ROCM" = "true" ]; then \
      pip install --no-cache-dir \
        torch==2.7.0 \
        torchvision==0.22.0 \
        --index-url https://download.pytorch.org/whl/rocm6.3; \
    else \
      pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 \
        --index-url https://download.pytorch.org/whl/cpu; \
    fi

COPY requirements.txt .
# All nvidia-*-cu12 packages except triton are hard-required by torch at import time:
# torch.__init__.py preloads them via ctypes before loading torch._C, and libtorch_cuda.so
# has them in its NEEDED list. Triton is only used by torch.compile(), not inference.
# opencv-python (GUI variant, installed by ultralytics) is replaced by headless;
# explicit uninstall removes the orphaned opencv_python.libs directory.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir opencv-python-headless \
    && pip uninstall -y triton 2>/dev/null || true

# rknn-toolkit-lite2 is only the RKNPU runtime: it loads and runs already-converted
# .rknn models. Rockchip's converter package (rknn-toolkit2) is deliberately not
# installed alongside it, because it pins numpy<=1.26.4 and torch<=2.4.0, which
# this project's numpy 2 / torch 2.7 cannot satisfy. Conversion therefore happens
# off-device on x86 and the container only ever loads finished models.
# onnx is not used at runtime; torch.onnx.export needs it to write the file that
# the converter then builds from, and that export has to happen here so the graph
# comes from the same torch and open_clip the app itself runs.
RUN if [ "$RKNN" = "true" ]; then pip install --no-cache-dir rknn-toolkit-lite2==2.3.2 onnx==1.22.0; fi

# Runtime stage: clean base + only the final venv state (no ghost install layers).
FROM python:3.12-slim

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# rknn-toolkit-lite2 dlopens librknnrt.so at runtime init and refuses to run
# against a mismatched version, so it is fetched from the same v2.3.2 tag as the
# wheel instead of relying on whatever the host or base image happens to ship.
# Downloaded with python rather than curl/wget: neither is in the slim base image,
# and ADD cannot be made conditional on the build arg.
ARG RKNN=false
RUN if [ "$RKNN" = "true" ]; then \
      python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so', '/usr/lib/librknnrt.so')"; \
    fi

# Set cache directories to /data to support read-only root FS.
# HOME is required: open_clip downloads the openai CLIP weights to a hardcoded
# ~/.cache/clip path that ignores XDG_CACHE_HOME, TORCH_HOME and HF_HOME, so
# without HOME they fall back to /root and crash a read-only root filesystem.
ENV HOME=/data \
    TORCH_HOME=/data/.cache/torch \
    HF_HOME=/data/.cache/huggingface \
    XDG_CACHE_HOME=/data/.cache \
    YOLO_CONFIG_DIR=/data/.ultralytics

# glibc's malloc gives each thread its own arena and malloc_trim() only trims the
# main arena, so freed memory from YOLO/CLIP worker threads stays mapped even after
# gc.collect() + malloc_trim(). Forcing a single arena makes freed heap reliably
# return to the OS when inference_session() unloads models.
ENV MALLOC_ARENA_MAX=1

# Copy code to /app
WORKDIR /app
COPY VERSION .
COPY app/ .

# Use /data as the working directory so downloads (like YOLO models) go there
WORKDIR /data
VOLUME ["/data"]

EXPOSE 8000

CMD ["python", "/app/main.py"]
