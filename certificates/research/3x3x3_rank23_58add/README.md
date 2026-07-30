# 3x3x3 rank-23 research circuit

This exact ternary circuit multiplies `3x3` block matrices with 23 leaf
products instead of 27. It is retained only for bounded RDNA3 screening and is
not part of runtime dispatch.

The certificate is copied without semantic modification from
[FastMatrixMultiplication](https://github.com/dronperminov/FastMatrixMultiplication)
commit [`98ba522`](https://github.com/dronperminov/FastMatrixMultiplication/tree/98ba522db92b74f1f8c561a78038ff3091356d73), file
[`schemes/results/addition_reduced_ZT/3x3x3_m23_cr58_cn122_ZT_reduced.json`](https://github.com/dronperminov/FastMatrixMultiplication/blob/98ba522db92b74f1f8c561a78038ff3091356d73/schemes/results/addition_reduced_ZT/3x3x3_m23_cr58_cn122_ZT_reduced.json).
The upstream CRLF file has SHA-256
`80bb4187ddf0cfc04654ca3e524f4d311c5a62245ee231a1101280b43fd2ae3a`;
this LF-normalized copy has
`a3b18e1e7bb6406ec32225c5c9e84b9cc0fb05b2637490e5dcba16e73aa9ab33`.
The upstream MIT license is preserved in [`SOURCE_LICENSE`](SOURCE_LICENSE).

Verify and regenerate the research kernel from the project root with:

```bash
python -m tools.verify_reduced_scheme \
  certificates/research/3x3x3_rank23_58add/certificate.json
python -m tools.generate_triton_scheme \
  certificates/research/3x3x3_rank23_58add/certificate.json \
  research/prototypes/generated_rank23_3x3x3.py
```
