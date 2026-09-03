# ONNX -> RKNN converter.
#
# Deliberately its own image: rknn-toolkit2 pins numpy<=1.26.4 and torch<=2.4.0,
# which the app's numpy 2 / torch 2.7 cannot satisfy, so the two cannot share a
# venv. Wheels exist for both x86_64 and aarch64, and building a model does not
# need NPU hardware, so this runs either on a PC or on the board itself.
#
# Build and run from the repo root:
#   docker build -t rknn-convert tools/rknn
#   docker run --rm -v $PWD/data:/data rknn-convert /data/rknn/clip_ViT_B_16_openai.onnx rk3588
FROM python:3.12-slim

# setuptools is held below 81 because rknn-toolkit2 still imports pkg_resources,
# which 81 removed. opencv-python arrives as one of its dependencies and imports
# libGL, which the slim base does not carry, so it is swapped for the headless
# build of the same library: uninstall first (both ship the same cv2 package, so
# installing over it and then uninstalling would take the replacement with it),
# and --no-deps so that install cannot drag numpy past rknn-toolkit2's pin and
# break its compiled extensions.
RUN pip install --no-cache-dir rknn-toolkit2==2.3.2 "setuptools<81" \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --no-deps opencv-python-headless

COPY convert.py /convert.py
WORKDIR /data
ENTRYPOINT ["python", "/convert.py"]
