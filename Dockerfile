# ==================================================
# OTW Simulation Environment (lightweight + stable)
# ==================================================
FROM python:3.10-slim

# 1. Install system dependencies (pygame, matplotlib)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
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
