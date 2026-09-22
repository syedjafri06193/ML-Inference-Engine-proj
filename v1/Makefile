PY ?= python3
NPM ?= npm

.PHONY: all
all: build fixtures test

# --- build ------------------------------------------------------------------

.PHONY: build
build: wasm js

.PHONY: wasm
wasm:
	./scripts/build.sh

.PHONY: wasm-bench
wasm-bench:
	WITH_BENCH=1 ./scripts/build.sh

.PHONY: js
js:
	cd js && $(NPM) run build

.PHONY: fixtures
fixtures:
	$(PY) tests/make_fixtures.py

# --- test -------------------------------------------------------------------

# Three suites, and all three matter:
#   rust     the kernels' own unit tests, natively
#   python   wasm vs an independent NumPy implementation, bit for bit
#   js       the public API, and a browser actually classifying a digit
.PHONY: test
test: test-rust test-python test-js

.PHONY: test-rust
test-rust:
	cargo test --manifest-path crates/lnn/Cargo.toml --lib

.PHONY: test-python
test-python:
	$(PY) -m unittest discover -s tests -p 'test_*.py' -v

.PHONY: test-js
test-js:
	cd js && $(NPM) test

.PHONY: test-browser
test-browser:
	node tests/browser_test.mjs

# --- measure ----------------------------------------------------------------

.PHONY: bench
bench: wasm-bench fixtures
	node bench/bench.mjs

.PHONY: size
size:
	./scripts/size-report.sh --fail-over scalar=25000 simd128=48000 relaxed=56000

.PHONY: probes
probes:
	./scripts/gen-probes.sh

# --- housekeeping -----------------------------------------------------------

.PHONY: fmt
fmt:
	cargo fmt --manifest-path crates/lnn/Cargo.toml

.PHONY: clean
clean:
	rm -rf dist js/dist crates/lnn/target bench/results
	rm -f examples/digits/*.wasm examples/digits/*.lnn examples/digits/index.js \
	      examples/digits/detect.js examples/digits/format.js js/test/fixtures.json

.PHONY: serve
serve:
	@echo "http://127.0.0.1:8000/  (Ctrl-C to stop)"
	cd examples/digits && $(PY) -m http.server 8000
