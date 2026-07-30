# Third-party notices

The exact rank-7, rank-49, and rank-343 certificates are copied from
[FastMatrixMultiplication](https://github.com/dronperminov/FastMatrixMultiplication)
at commit `98ba522db92b74f1f8c561a78038ff3091356d73`. The upstream project and the
copied certificate are distributed under the MIT License; the original license
text is included beside the certificate.

The corresponding generated Triton code is derived mechanically from those
certificates. The RDNA3 runtime, benchmark harness, generator, and integration
code in this repository are licensed under this repository's MIT License.

The rank-48 straight-line programs are copied from
[Fast-Matrix-Multiplication](https://github.com/jgdumas/Fast-Matrix-Multiplication)
at commit `8bb354d63061504d1a712efafc8d06a0e8fa3f07`. The upstream programs and
the generated rank-48 Triton code are distributed under the CeCILL-B License;
the original license text is included beside the certificate. The algorithm is
described by Jean-Guillaume Dumas, Clément Pernet, and Alexandre Sedoglavic in
[A more accurate rational non-commutative algorithm for multiplying 4x4
matrices using 48 multiplications](https://arxiv.org/abs/2603.18699v3).
