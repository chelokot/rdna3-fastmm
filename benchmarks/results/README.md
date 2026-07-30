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
- [`rx7900xtx-rank7-linear-hidream3600-up-45ad03b.json`](rx7900xtx-rank7-linear-hidream3600-up-45ad03b.json),
  [`rx7900xtx-rank7-linear-hidream4096-up-45ad03b.json`](rx7900xtx-rank7-linear-hidream4096-up-45ad03b.json),
  and [`rx7900xtx-rank7-linear-hidream4096-down-45ad03b.json`](rx7900xtx-rank7-linear-hidream4096-down-45ad03b.json):
  clean rank-7 HiDream operator wins.
- [`rx7900xtx-rank7-linear-ideogram4704-up-45ad03b.json`](rx7900xtx-rank7-linear-ideogram4704-up-45ad03b.json),
  [`rx7900xtx-rank7-linear-ideogram4704-down-45ad03b.json`](rx7900xtx-rank7-linear-ideogram4704-down-45ad03b.json),
  [`rx7900xtx-rank7-linear-ideogram8214-up-45ad03b.json`](rx7900xtx-rank7-linear-ideogram8214-up-45ad03b.json),
  and [`rx7900xtx-rank7-linear-ideogram8214-down-45ad03b.json`](rx7900xtx-rank7-linear-ideogram8214-down-45ad03b.json):
  clean rank-7 Ideogram operator wins at local and high-resolution row counts.
- [`rx7900xtx-rank7-linear-ltx4992-up-45ad03b.json`](rx7900xtx-rank7-linear-ltx4992-up-45ad03b.json):
  clean rank-7 biased LTX first-stage up-projection result.
- [`rx7900xtx-rank7-linear-qwen8192-up-45ad03b.json`](rx7900xtx-rank7-linear-qwen8192-up-45ad03b.json),
  [`rx7900xtx-rank7-linear-qwen16384-up-45ad03b.json`](rx7900xtx-rank7-linear-qwen16384-up-45ad03b.json),
  and [`rx7900xtx-rank7-linear-qwen16384-down-45ad03b.json`](rx7900xtx-rank7-linear-qwen16384-down-45ad03b.json):
  clean rank-7 biased Qwen operator wins retained by the exact-shape gate.
- [`rx7900xtx-rank7-linear-ideogram-up-4704-both-da03834.json`](rx7900xtx-rank7-linear-ideogram-up-4704-both-da03834.json),
  [`rx7900xtx-rank7-linear-ideogram-down-4704-both-da03834.json`](rx7900xtx-rank7-linear-ideogram-down-4704-both-da03834.json),
  and [`rx7900xtx-rank7-linear-hidream-up-4096-both-da03834.json`](rx7900xtx-rank7-linear-hidream-up-4096-both-da03834.json):
  clean native-weight dynamic and prepacked plan results.
- [`rx7900xtx-rank7-linear-ltx-down-4992-triton-da03834.json`](rx7900xtx-rank7-linear-ltx-down-4992-triton-da03834.json):
  clean public-operator evidence for the newly admitted first-stage LTX down
  shape.
- [`rx7900xtx-rank7-linear-ltx-down-4992-both-da03834.json`](rx7900xtx-rank7-linear-ltx-down-4992-both-da03834.json)
  and [`rx7900xtx-rank7-linear-ltx-up-720-prepacked-da03834.json`](rx7900xtx-rank7-linear-ltx-up-720-prepacked-da03834.json):
  clean explicit prepacking results for the first-stage down and low-envelope
  up LTX shapes.
