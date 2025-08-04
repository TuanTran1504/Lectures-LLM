FROM continuumio/miniconda3:latest

# Set environment for Miniconda
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH
SHELL ["/bin/bash", "-c"]

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget git curl build-essential software-properties-common libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y postgresql-client
# Set working directory
WORKDIR /project

# Copy environment definition and install dependencies
COPY environment.yml .

# Accept TOS and create environment
RUN conda tos accept --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --channel https://repo.anaconda.com/pkgs/r && \
    conda update -n base -c defaults conda && \
    conda env create -f environment.yml

# Activate environment and install PyTorch with CUDA + psycopg2
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate rag-env && \
    pip install psycopg[binary] && \
    pip install torch torchvision torchaudio &&\
    python -m nltk.downloader punkt

# Copy application code
COPY . .

# Make startup script executable
RUN chmod +x start.sh

# Run Flask app
CMD ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate rag-env && ./start.sh"]
