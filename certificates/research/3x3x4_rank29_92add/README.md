# 3x3x4 rank-29, 92-addition research circuit

This exact ternary circuit multiplies a `3x3` block matrix by a `3x4` block
matrix with 29 leaf products and 92 additions or subtractions. Its directional
costs are `(21, 26, 45)`. It is retained only for bounded RDNA3 screening and
is not part of runtime dispatch.

The `3x3x4` aspect ratio is a close match for the Ideogram up projection:
`4704x4608 @ 4608x12288` becomes 29 leaf products of
`1568x1536 @ 1536x3072`. The circuit executes `29/36` of classical
multiplication, compared with `7/8` for the current rank-7 operator.

The certificate is copied without modification from Andrew Perminov's
[`FastMatrixMultiplication`](https://github.com/dronperminov/FastMatrixMultiplication)
repository at commit
[`98ba522`](https://github.com/dronperminov/FastMatrixMultiplication/tree/98ba522db92b74f1f8c561a78038ff3091356d73),
file
`schemes/results/addition_reduced_ZT/3x3x4_m29_cr92_cn134_ZT_reduced.json`.
The upstream repository and certificate are MIT-licensed; its license is
preserved in [`SOURCE_LICENSE`](SOURCE_LICENSE).

Verify and regenerate the research kernel from the project root with:

```bash
python -m tools.verify_reduced_scheme \
  certificates/research/3x3x4_rank29_92add/certificate.json
python -m tools.generate_triton_scheme \
  certificates/research/3x3x4_rank29_92add/certificate.json \
  research/prototypes/generated_rank29_3x3x4.py
```
