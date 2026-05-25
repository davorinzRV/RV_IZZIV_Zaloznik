FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    nano \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy \
    opencv-python-headless \
    matplotlib \
    pandas \
    scipy \
    tabulate \
    mediapipe==0.10.14 \
    "protobuf<5"

COPY . .

CMD ["bash"]
