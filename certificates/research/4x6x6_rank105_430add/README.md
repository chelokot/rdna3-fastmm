# 4x6x6 rank-105, 430-addition research circuit

This exact ternary circuit multiplies a `4x6` block matrix by a `6x6` block
matrix with 105 leaf products and 430 additions or subtractions. Its
directional costs are `(112, 131, 187)`. It is retained only for bounded RDNA3
screening and is not part of runtime dispatch.

The circuit executes `105/144` of classical multiplication. On the Ideogram
`4704x4608 @ 4608x12288` up projection, its 105 leaf products have shape
`1176x768 @ 768x2048`.

The certificate is copied without modification from Andrew Perminov's
[`FastMatrixMultiplication`](https://github.com/dronperminov/FastMatrixMultiplication)
repository at commit
[`98ba522`](https://github.com/dronperminov/FastMatrixMultiplication/tree/98ba522db92b74f1f8c561a78038ff3091356d73),
file
`schemes/results/addition_reduced_ZT/4x6x6_m105_cr430_fv243_cn894_ZT_reduced.json`.
The construction belongs to the exact ternary scheme family described in
[*Fast Matrix Multiplication: from GPUs to Tensor Decompositions*](https://arxiv.org/abs/2606.02480).
The upstream repository and certificate are MIT-licensed; its license is
preserved in [`SOURCE_LICENSE`](SOURCE_LICENSE).

Verify and regenerate the research kernel from the project root with:

```bash
python -m tools.verify_reduced_scheme \
  certificates/research/4x6x6_rank105_430add/certificate.json
python -m tools.generate_triton_scheme \
  certificates/research/4x6x6_rank105_430add/certificate.json \
  research/prototypes/generated_rank105_4x6x6.py
```
