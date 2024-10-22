# Use the micromamba image as base
FROM docker.io/mambaorg/micromamba:bookworm-slim

# Switch to root to install system packages
USER root

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip \
    curl \
    vim
    
# Switch back to the mambauser
USER $MAMBA_USER

# Install Python packages using micromamba
RUN micromamba install -y -n base -c conda-forge \
    python \
    pip \
    pip-tools \
    pandas \
    numpy \
    fsspec \
    gcsfs \
    ujson \
    xarray \
    kerchunk \
    climpred \
    dask \
    xarray \
    xesmf \
    regionmask \
    xhistogram \
    scikit-learn \
    xskillscore \
    cartopy \
    flask \
    cfgrib \
    requests \
    s3fs \
    boto3 \
    pyarrow \
    google-cloud-storage \
    pytest \
    gunicorn \
    altair \
    matplotlib \
    && micromamba clean --all --yes

# Set the entrypoint to use micromamba activate
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]

# Install additional Python packages
# Install additional Python packages
RUN /opt/conda/envs/base/bin/python -m pip install --upgrade pip && \
    /opt/conda/envs/base/bin/pip install xbootstrap xhistogram calendar \
    python-dotenv vl-convert-python coiled# Set the working directory

WORKDIR /app

# Create directory for AWS credentials

