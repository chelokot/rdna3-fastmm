# 3x3x3 rank-23, 55-addition research circuit

This exact ternary circuit multiplies `3x3` block matrices with 23 leaf
products and 55 additions or subtractions. Its directional costs are
`(13, 14, 28)`. It is retained only for bounded RDNA3 screening and is not
part of runtime dispatch.

The circuit was independently reconstructed on July 20, 2026. A provenance
audit found the same `13 + 14 + 28` bound in Greg Sidebottom and Claude Fable
5's earlier [July 6 commit](https://github.com/gsidebottom/logic/commit/1ca82c93c55545f7060dc636d6b2ac8adea58e9f),
so this artifact makes no priority claim. It realizes the same expanded factor
maps as the existing 58-addition research circuit with one fewer operation in
each stage.

[`source_certificate.json`](source_certificate.json) contains the independently
verified signed-gate representation. It uses the rank-23 factorization from
Andrew Perminov's `FastMatrixMultiplication` repository at commit
[`98ba522`](https://github.com/dronperminov/FastMatrixMultiplication/tree/98ba522db92b74f1f8c561a78038ff3091356d73).
[`certificate.json`](certificate.json) is its deterministic conversion to this
project's reduced-circuit schema. The upstream factorization is MIT-licensed;
its license is preserved in [`SOURCE_LICENSE`](SOURCE_LICENSE).

Convert, verify, and regenerate the research kernel from the project root with:

```bash
python -m tools.convert_linear_circuit \
  certificates/research/3x3x3_rank23_55add/source_certificate.json \
  certificates/research/3x3x3_rank23_55add/certificate.json
python -m tools.verify_reduced_scheme \
  certificates/research/3x3x3_rank23_55add/certificate.json
python -m tools.generate_triton_scheme \
  certificates/research/3x3x3_rank23_55add/certificate.json \
  research/prototypes/generated_rank23_55add_3x3x3.py
```
