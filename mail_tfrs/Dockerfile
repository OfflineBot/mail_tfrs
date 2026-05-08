# Multi-stage build for the `mail_tfrs_server` HTTP service.
#
# Build context must be the workspace root (containing `Cargo.toml`,
# `tfrs/` and `mail_tfrs/`):
#
#   cd <repo-root>
#   docker build -f mail_tfrs/Dockerfile -t mail_tfrs:latest .
#
# The GitHub Actions workflow does this automatically on every push to main.

# ---------- builder ----------
FROM rust:1.85-slim AS build
WORKDIR /src

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev cmake clang ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Bring in the workspace manifest and both member crates.
COPY Cargo.toml  Cargo.toml
COPY tfrs/       tfrs/
COPY mail_tfrs/  mail_tfrs/

RUN cargo build --release --bin mail_tfrs_server

# ---------- runtime ----------
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -r -u 1000 -m app

COPY --from=build /src/target/release/mail_tfrs_server /usr/local/bin/mail_tfrs_server

USER app
WORKDIR /home/app
RUN mkdir -p /home/app/data

EXPOSE 8080
ENV MAIL_TFRS_BIND=0.0.0.0:8080 \
    MAIL_TFRS_MODEL=/home/app/data/mail_model.bin \
    MAIL_TFRS_DATASETS=/home/app/data \
    MAIL_TFRS_D_MODEL=64 \
    MAIL_TFRS_N_HEADS=4 \
    MAIL_TFRS_D_FF=128 \
    MAIL_TFRS_N_ENC_LAYERS=2 \
    MAIL_TFRS_SEQ_LEN=512

VOLUME ["/home/app/data"]

CMD ["mail_tfrs_server"]
