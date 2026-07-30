# 3x3x3 rank-23, 56-addition research circuit

This exact ternary circuit multiplies `3x3` block matrices with 23 leaf
products and 56 additions or subtractions. Its directional costs are
`(13, 13, 30)`. It is retained only for bounded RDNA3 screening and is not
part of runtime dispatch.

The schedule is mechanically transcribed from the `SIDES` certificate in
Yinqi Sun's [`3by3r23-56a`](https://github.com/sunyinqi0508/3by3r23-56a)
repository at commit
[`2917e6d`](https://github.com/sunyinqi0508/3by3r23-56a/tree/2917e6dedb624340a7a75fbb0214627ed545ea84).
The nine output expressions are reordered from the source's row-major
presentation to this project's column-major certificate convention; the
linear forms and arithmetic are otherwise unchanged.

The accompanying paper, [*An Exact 56-Addition, Rank-23 Scheme for General
3x3 Matrix Multiplication*](https://arxiv.org/abs/2604.27645), establishes the
`13 + 13 + 30` schedule and integer Brent certificate. The upstream reference
implementation is MIT-licensed; its license is preserved in
[`SOURCE_LICENSE`](SOURCE_LICENSE).

Verify and regenerate the research kernel from the project root with:

```bash
python -m tools.verify_reduced_scheme \
  certificates/research/3x3x3_rank23_56add/certificate.json
python -m tools.generate_triton_scheme \
  certificates/research/3x3x3_rank23_56add/certificate.json \
  research/prototypes/generated_rank23_56add_3x3x3.py
```
