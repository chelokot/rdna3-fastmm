# 2x2x2 rank-7, 15-addition circuit

This certificate computes a `2x2` matrix times a `2x2` matrix using seven
scalar multiplications and 15 additions or subtractions. It backs the
mixed-precision rank-7 Linear operator; runtime dispatch remains restricted to
separately measured shape families.

The reduced circuit is copied without modification from
[FastMatrixMultiplication](https://github.com/dronperminov/FastMatrixMultiplication)
commit [`98ba522`](https://github.com/dronperminov/FastMatrixMultiplication/tree/98ba522db92b74f1f8c561a78038ff3091356d73), file
[`schemes/results/addition_reduced_ZT/2x2x2_m7_cr15_cn24_ZT_reduced.json`](https://github.com/dronperminov/FastMatrixMultiplication/blob/98ba522db92b74f1f8c561a78038ff3091356d73/schemes/results/addition_reduced_ZT/2x2x2_m7_cr15_cn24_ZT_reduced.json).
Both files have SHA-256
`b374fcc797ea014994453b7502625e24f70ca2f1dcf1bdbe74a716342bfefb1e`.

Verify it from the project root:

```bash
python -m tools.verify_reduced_scheme \
  certificates/2x2x2_rank7_15add/certificate.json
```

Regenerate the research kernel deterministically with:

```bash
python -m tools.generate_triton_scheme \
  certificates/2x2x2_rank7_15add/certificate.json \
  src/rdna3_fastmm/generated/rank7_2x2x2.py
```

Run one bounded real-shape comparison with:

```bash
python -m research.prototypes.benchmark_rank7_linear \
  --shape 4097,4096,12288 \
  --mode both \
  --output /tmp/rank7-hidream.json
```
