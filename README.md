# mail_tfrs

Multi-label email classifier built on top of a small encoder-only transformer.

This repo contains two crates:

- **`tfrs/`** — the generic transformer library + classifier trainer. Domain-agnostic.
- **`mail_tfrs/`** — email-specific glue (RON dataset loader, mail-to-text, HTTP server) that depends on `tfrs`.

## Quick start (local)

```sh
cargo build --release            # builds both crates
./target/release/mail_tfrs_server   # starts on 127.0.0.1:8080
```

## Run as a Docker service

After pushing this repo to GitHub the included GitHub Action builds and pushes a
container image to GHCR (`ghcr.io/<owner>/<repo>:latest`).

Drop this into your `docker-compose.yml`:

```yaml
services:
  mail_tfrs:
    image: ghcr.io/<your-gh-user>/<this-repo-name>:latest
    ports:
      - "127.0.0.1:8090:8080"
    volumes:
      - mail_tfrs_data:/home/app/data
    restart: unless-stopped

volumes:
  mail_tfrs_data:
```

Steps:
1. Push this repo to GitHub.
2. Wait for the **build & push container** workflow to finish.
3. Open the resulting package in GitHub → Packages → Package settings → set visibility to public (or set up GHCR auth on your host).
4. Drop the snippet above into your existing compose file, replace `<your-gh-user>` and `<this-repo-name>`, then `docker compose pull && docker compose up -d`.

## HTTP API

| Method | Path       | Body                                       | Returns                          |
|--------|------------|--------------------------------------------|----------------------------------|
| GET    | `/health`  | —                                          | `ok`                             |
| GET    | `/info`    | —                                          | model arch + categories          |
| POST   | `/predict` | `{subject, sender_name, sender_email, body, threshold?}` | `[{name, probability, predicted}]` |
| POST   | `/train`   | `{files?, fresh?, max_steps?, max_epochs?, target_label_acc?, eval_every_steps?, lr?, seed?, log_every?}` | training summary |

Training is synchronous and serialised: a second `/train` while one is running
returns `409 Conflict`.

Place `dataset_*.ron` files into the `mail_tfrs_data` volume (mount path:
`/home/app/data`) before calling `/train` with `{}`, or pass absolute paths
inside the volume via `files`.

### Environment variables

| Var                       | Default                              |
|---------------------------|--------------------------------------|
| `MAIL_TFRS_BIND`          | `0.0.0.0:8080`                       |
| `MAIL_TFRS_MODEL`         | `/home/app/data/mail_model.bin`      |
| `MAIL_TFRS_DATASETS`      | `/home/app/data`                     |
| `MAIL_TFRS_THRESHOLD`     | `0.5`                                |
| `MAIL_TFRS_D_MODEL`       | `64`                                 |
| `MAIL_TFRS_N_HEADS`       | `4`                                  |
| `MAIL_TFRS_D_FF`          | `128`                                |
| `MAIL_TFRS_N_ENC_LAYERS`  | `2`                                  |
| `MAIL_TFRS_SEQ_LEN`       | `512`                                |

The architecture vars must stay constant across train/load — the loader
asserts they match the on-disk header.
