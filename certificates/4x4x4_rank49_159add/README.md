# Exact 4x4x4 rank-49, 159-addition circuit

This certificate computes a `4x4` matrix times a `4x4` matrix using 49 scalar multiplications and 159 additions or subtractions. Its three linear-map costs are 42, 42, and 75. Every circuit edge has coefficient `-1` or `1`, and the Brent tensor identity holds over the integers.

The reduced circuit is copied without modification from FastMatrixMultiplication commit [`98ba522`](https://github.com/dronperminov/FastMatrixMultiplication/tree/98ba522db92b74f1f8c561a78038ff3091356d73), file [`schemes/results/addition_reduced_ZT/4x4x4_m49_cr159_fv100_cn474_ZT_reduced.json`](https://github.com/dronperminov/FastMatrixMultiplication/blob/98ba522db92b74f1f8c561a78038ff3091356d73/schemes/results/addition_reduced_ZT/4x4x4_m49_cr159_fv100_cn474_ZT_reduced.json). Both the upstream file and this certificate have SHA-256 `a3c4121dfd09607045255628dd94b60133c24c46b65f1089e89c6564ba522561`.

Verify from the project root:

```bash
python -m tools.verify_reduced_scheme certificates/4x4x4_rank49_159add/certificate.json
```

The verifier independently expands the schedule, recounts its operations, and checks the exact 64-entry Brent tensor. This is a public factorization and circuit, not a new rank or addition-count result. The new result in this repository is its generated RDNA3 implementation and measured wall-clock behavior.
