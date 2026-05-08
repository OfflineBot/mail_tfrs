# `make`                                  — build release binaries (cli + server)
# `make train [FILES="path/*.ron"]`       — fresh training run
# `make resume [FILES="..."] [STEPS=N]`   — continue training the existing checkpoint
# `make eval`                             — evaluate the checkpoint
# `make serve [PORT=8080]`                — run the HTTP server
# `make lines`                            — line counts
#
# If FILES is unset, the binary scans ~/Downloads for `dataset*.ron`.

CLI    := bin/mail_tfrs
SERVER := bin/mail_tfrs_server
MODEL  := bin/mail_model.bin

FILES ?=
STEPS ?= 20000
PORT  ?= 8080

FILES_ARG := $(if $(FILES),--files $(FILES),)

.PHONY: make train resume eval serve lines

make: $(CLI) $(SERVER)

$(CLI) $(SERVER): $(shell find src -name '*.rs') Cargo.toml
	@cargo build --release
	@mkdir -p bin
	@cp -f target/release/mail_tfrs        $(CLI)
	@cp -f target/release/mail_tfrs_server $(SERVER)
	@echo "built $(CLI) and $(SERVER)"

train: $(CLI)
	@./$(CLI) train --max-steps $(STEPS) --target-label-acc 0.95 --eval-every 500 \
		--model $(MODEL) $(FILES_ARG)

resume: $(CLI)
	@if [ ! -f "$(MODEL)" ]; then echo "error: no checkpoint at $(MODEL) — run 'make train' first"; exit 2; fi
	@./$(CLI) continue --max-steps $(STEPS) --eval-every 500 --model $(MODEL) $(FILES_ARG)

eval: $(CLI)
	@./$(CLI) eval --model $(MODEL) $(FILES_ARG)

serve: $(SERVER)
	@MAIL_TFRS_BIND=0.0.0.0:$(PORT) MAIL_TFRS_MODEL=$(MODEL) ./$(SERVER)

lines:
	@cloc . --exclude-dir=target,bin,dist
