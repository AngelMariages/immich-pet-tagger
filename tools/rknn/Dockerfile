# ONNX -> RKNN converter.
#
# Deliberately its own image: rknn-toolkit2 pins numpy<=1.26.4 and torch<=2.4.0,
# which the app's numpy 2 / torch 2.7 cannot satisfy, so the two cannot share a
# venv. Wheels exist for both x86_64 and aarch64, so this runs either on a PC or
# on the board itself.
#
# Build and run from the repo root:
#   docker build -t rknn-convert tools/rknn
#   docker run --rm -v $PWD/data:/data rknn-convert /data/rknn/clip_ViT_B_16_openai.onnx rk3588
FROM python:3.12-slim

# opencv-python arrives as a dependency and imports libGL, which the slim base
# does not carry; the headless build is the same library without that link.
RUN pip install --no-cache-dir rknn-toolkit2==2.3.2 \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir opencv-python-headless

COPY convert.py /convert.py
WORKDIR /data
ENTRYPOINT ["python", "/convert.py"]
