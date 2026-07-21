# Exact 8x8x8 rank-343, 1661-addition circuit

This certificate computes an `8x8` matrix times an `8x8` matrix using 343
scalar multiplications and 1661 additions or subtractions. Its directional
linear-map costs are 462, 456, and 743. Every edge has coefficient `-1` or `1`,
and the Brent tensor identity holds over the integers.

The reduced circuit is copied without modification from FastMatrixMultiplication
commit [`98ba522`](https://github.com/dronperminov/FastMatrixMultiplication/tree/98ba522db92b74f1f8c561a78038ff3091356d73), file
[`8x8x8_m343_cr1661_fv1015_cn4434_ZT_reduced.json`](https://github.com/dronperminov/FastMatrixMultiplication/blob/98ba522db92b74f1f8c561a78038ff3091356d73/schemes/results/addition_reduced_ZT/8x8x8_m343_cr1661_fv1015_cn4434_ZT_reduced.json).
The upstream file and this certificate both have SHA-256
`c7afccb09f8491e484af1696c25c809757c4af649ee7220e79f25d1c7afe3955`.

```bash
python -m tools.verify_reduced_scheme \
  certificates/8x8x8_rank343_1661add/certificate.json
```

The public circuit is not a new rank record. Its MFMA reconstruction and RDNA3
runtime implementation are the new experimental result in this repository.
