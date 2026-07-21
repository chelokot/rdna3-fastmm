# Benchmark artifacts

Reports in this directory are produced from clean commits by the benchmark
harnesses one level above. Each JSON file contains raw timings, execution order,
runtime and device provenance, memory accounting, certificate and generated-code
hashes, and FP32 correctness samples.

- [`rx7900xtx-rank343-prepacked-16384-f5f3e88.json`](rx7900xtx-rank343-prepacked-16384-f5f3e88.json):
  clean rank-343 MFMA prepacked result at `16384³`.
- [`rx7900xtx-rank49-both-16384-83dbaf7.json`](rx7900xtx-rank49-both-16384-83dbaf7.json):
  clean rank-49 dynamic and prepacked result at `16384³`.
- [`rx7900xtx-rank49-external-16384-afdacb4.json`](rx7900xtx-rank49-external-16384-afdacb4.json):
  clean rank-49 Inductor external-callable result at `16384³`.
- [`rx7900xtx-linear-ideogram4-8214-up-3993605.json`](rx7900xtx-linear-ideogram4-8214-up-3993605.json)
  and [`rx7900xtx-linear-ideogram4-8214-down-3993605.json`](rx7900xtx-linear-ideogram4-8214-down-3993605.json):
  clean BF16 Triton-op results for the real high-resolution Ideogram shape.
- [`rx7900xtx-linear-ideogram4-5120-up-3993605.json`](rx7900xtx-linear-ideogram4-5120-up-3993605.json),
  [`rx7900xtx-linear-ideogram4-9216-up-3993605.json`](rx7900xtx-linear-ideogram4-9216-up-3993605.json),
  [`rx7900xtx-linear-ideogram4-5632-down-3993605.json`](rx7900xtx-linear-ideogram4-5632-down-3993605.json),
  and [`rx7900xtx-linear-ideogram4-9216-down-3993605.json`](rx7900xtx-linear-ideogram4-9216-down-3993605.json):
  clean lower and upper boundary evidence for the dispatched Ideogram families.
- [`rx7900xtx-linear-ltx23-19968-up-3993605.json`](rx7900xtx-linear-ltx23-19968-up-3993605.json)
  and [`rx7900xtx-linear-ltx23-19968-down-3993605.json`](rx7900xtx-linear-ltx23-19968-down-3993605.json):
  clean BF16 Triton-op results for the LTX-2.3 upscaled second stage.
