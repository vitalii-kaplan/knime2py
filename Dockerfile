# ---------- Builder ----------
    FROM python:3.11-slim AS builder
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libxml2-dev libxslt1-dev zlib1g-dev \
     && rm -rf /var/lib/apt/lists/*
    WORKDIR /app
    COPY pyproject.toml README.md /app/
    COPY src/ /app/src/
    RUN python -m pip install --upgrade pip wheel build \
     && python -m build --wheel --outdir /app/dist \
     && pip wheel --no-cache-dir --wheel-dir /app/wheels /app/dist/*.whl
    
    # ---------- Runtime ----------
    FROM python:3.11-slim
    # If you ever render via the `dot` binary, uncomment the next two lines:
    # RUN apt-get update && apt-get install -y --no-install-recommends graphviz \
    #  && rm -rf /var/lib/apt/lists/*
    RUN apt-get update && apt-get install -y --no-install-recommends \
        libxml2 libxslt1.1 zlib1g \
     && rm -rf /var/lib/apt/lists/*
    WORKDIR /app
    COPY --from=builder /app/dist /wheels_app
    COPY --from=builder /app/wheels /wheels
    RUN python -m pip install --no-cache-dir /wheels_app/*.whl && \
        python -m pip install --no-cache-dir /wheels/*
    # Default entrypoint = your CLI
    ENTRYPOINT ["k2p"]
    CMD ["--help"]
    