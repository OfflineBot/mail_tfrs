# `make`         — build release lib + smoke-test bin
# `make check`   — cargo check
# `make test`    — cargo test
# `make smoke`   — run the copy-task smoke test
# `make lines`   — line counts

BIN := bin/tfrs

.PHONY: make check test smoke lines

make: $(BIN)

$(BIN): $(shell find src -name '*.rs') Cargo.toml
	@cargo build --release
	@mkdir -p bin
	@cp -f target/release/tfrs $(BIN)
	@echo "built $(BIN)"

check:
	@cargo check

test:
	@cargo test

smoke: $(BIN)
	@./$(BIN) copy

lines:
	@cloc . --exclude-dir=target,bin,dist
