# ==================================================
# OTW Simulation Environment (lightweight + stable)
# ==================================================
FROM python:3.10-slim

# 1. Install system dependencies (pygame, matplotlib)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    # python-dev libsdl-image1.2-dev libsdl-mixer1.2-dev \
    # libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev python-numpy subversion libportmidi-dev \
    # ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev gcc \
    x11-apps x11-utils x11-xserver-utils \
    && rm -rf /var/lib/apt/lists/*

# 2. Install python packages
RUN pip install --no-cache-dir pygame numpy matplotlib pyyaml

# 3. GUI
ENV DISPLAY=:0
ENV SDL_VIDEODRIVER=x11

# 3. Set working directory inside container
WORKDIR /workspace

# 4. Default command
CMD ["/bin/bash"]

# match default user UID/GID (1000:1000) với phat
RUN useradd -u 1000 -m phat
USER phat
