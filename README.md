# rblas

A pure-Rust, competitive BLAS (Basic Linear Algebra Subprograms). No C, no
Fortran, no `unsafe` outside the arch-gated SIMD kernels.

## Status

Early but real. The GEMM core is cache-blocked + packed (GotoBLAS/BLIS
structure) with a hand-written NEON microkernel, and single-threaded throughput
is in the same ballpark as tuned native BLAS on Apple Silicon.

Measured on an Apple M2 Max (single thread), vs. `matrixmultiply` 0.3 — the
de-facto pure-Rust GEMM baseline (the engine behind `ndarray`). Run it yourself
with `cargo bench --bench compare`.

| Routine     | Size  | rblas       | matrixmultiply | rblas / mm |
|-------------|-------|-------------|----------------|------------|
| SGEMM (f32) | 256³  | 78 GFLOP/s  | 90 GFLOP/s     | 86%        |
| SGEMM (f32) | 512³  | 83 GFLOP/s  | 93 GFLOP/s     | 89%        |
| SGEMM (f32) | 1024³ | 85 GFLOP/s  | ~90 GFLOP/s    | **~95%**   |
| DGEMM (f64) | 256³  | 42 GFLOP/s  | 45 GFLOP/s     | 95%        |
| DGEMM (f64) | 1024³ | 43 GFLOP/s  | 43 GFLOP/s     | **~100%**  |

DGEMM is on par with matrixmultiply at large sizes; SGEMM is within ~5–10%.
Two optimizations got us here: thread-local reuse + right-sizing of the packing
buffers (killed the small-matrix penalty — 256³ SGEMM went 53% → 86%), and an
f64 microkernel that processes 8 columns in two 4-wide groups (16 live
accumulators instead of 32, no register spill — 1024³ DGEMM went 68% → ~100%).
Core peak ≈ 100 GFLOP/s f32 / ~50 GFLOP/s f64 per P-core.

### Multi-threading (`--features threads`)

The macrokernel partitions C into column chunks (≈4 per worker for load balance)
and runs them on a rayon pool — each chunk writes disjoint C columns and gets its
own packing scratch, so there's no synchronization on the hot path.

On the M2 Max (8 P + 4 E cores), a 4608³ SGEMM:

| Mode      | Throughput   | Speedup |
|-----------|--------------|---------|
| serial    | 87 GFLOP/s   | 1.0×    |
| threaded  | 606 GFLOP/s  | **7.0×**|

```
cargo run --release --features threads --example threads_speedup
```

## Implemented

### Real (`f32`/`f64`)

- **Level 1**: `axpy`, `dot`, `scal`, `nrm2` (robust scaled), `asum`, `copy`,
  `swap` — contiguous fast paths dispatch to NEON / AVX2, with strided fallbacks.
- **Level 2**: `gemv` (notrans/trans, row- & col-major), `ger`.
- **Level 3**: `gemm` (all transpose combos, row- & col-major, `alpha`/`beta`,
  packed + SIMD microkernel for f32/f64), `syrk`, `symm`, `trsm`.
  Checked entry point `try_gemm` returns `Result` instead of panicking.

### Complex (`C32`/`C64`)

- **Level 1**: `axpy`, `dotu`, `dotc`, `scal`, `nrm2`.
- **Level 2**: `gemv` (incl. conjugate-transpose), `geru`, `gerc`.
- **Level 3**: `gemm` (4M decomposition — reuses the fast real kernel via
  real/imag planes), `symm`, `hemm`, `trsm`.

Generic over `f32`/`f64` via the `Float` trait, with a `Complex<T>` element type
for the `c`/`z` routines. Runtime CPU dispatch: AVX2+FMA on x86-64, NEON on
aarch64, portable scalar everywhere else. `#![no_std]`-compatible (needs
`alloc`). All `unsafe` is confined to `src/kernel/{avx2,neon,gemm_neon,gemm_avx2}.rs`
behind feature gates.

## Architecture

```
src/
  lib.rs        Float trait, arch-kernel supertrait plumbing
  types.rs      Layout (row/col major), Transpose
  level1.rs     vector–vector ops
  level2.rs     matrix–vector ops
  level3.rs     gemm: normalize layout/transpose -> packed col-major kernel
  kernel/
    mod.rs      runtime dispatch (Element trait), Level-1 entry points
    scalar.rs   portable fallback kernels
    neon.rs     AArch64 NEON Level-1 kernels
    avx2.rs     x86-64 AVX2+FMA Level-1 kernels
    gemm.rs     cache-blocking driver + packing + scalar microkernel
    gemm_neon.rs hand-written NEON 8×8 GEMM microkernels
```

## Roadmap

- Benchmark/tune the AVX2 GEMM microkernel on real x86-64 hardware (see note).
- Close the remaining SGEMM gap (packing prefetch; the f64 tile already matches).
- Auto-tuned cache-block params (`MC`/`KC`/`NC`) per detected cache sizes.
- Remaining Level-3: `trsm`, `syrk`, `symm`, `trmm`.
- AVX-512 path; runtime tile selection.
- Robust `nrm2` (scaled sum of squares), complex types.

### Done

- ✅ Packed + cache-blocked GEMM with hand-written NEON microkernels (f32/f64).
- ✅ Thread-local, right-sized packing scratch (no per-call alloc).
- ✅ f64 microkernel retuned to 4-wide column groups — on par with matrixmultiply.
- ✅ Multi-threaded GEMM (`threads` feature) — 7× on the M2 Max, no hot-path sync.
- ✅ AVX2+FMA GEMM microkernel for x86-64 (8×8 f32/f64), runtime-dispatched with
  scalar fallback. Compiles and the dispatch/fallback paths pass on x86-64;
  the AVX2 intrinsics themselves are exercised by a self-skipping unit test
  (`avx2_microkernel_matches_scalar`) that needs AVX2-capable hardware — not yet
  run on a real AVX2 host (this dev machine is Apple Silicon; Rosetta is
  SSE-only), so AVX2 *performance* is still unbenchmarked.

## Testing

```
cargo test                          # unit + edge-case correctness
cargo test --release --test correctness          # large random GEMM vs naive
cargo test --release --features threads --test parallel   # threaded vs serial
cargo bench                         # criterion GFLOP/s reports
cargo bench --bench compare         # vs matrixmultiply
```

## License

MIT OR Apache-2.0.
