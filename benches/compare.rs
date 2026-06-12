//! Head-to-head GEMM: rblas vs. `matrixmultiply` (the pure-Rust baseline behind
//! ndarray). Both run single-threaded on identical column-major buffers, same
//! sizes, with throughput reported as FLOP/s (2·m·n·k) so the bars compare
//! directly.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rblas::{level3, Layout, Transpose};

const SIZES: &[usize] = &[64, 128, 256, 512, 1024];

fn flops(n: usize) -> u64 {
    2 * (n as u64).pow(3)
}

fn bench_sgemm(c: &mut Criterion) {
    let mut g = c.benchmark_group("sgemm_vs_matrixmultiply");
    for &n in SIZES {
        let a = vec![1.0f32; n * n];
        let b = vec![2.0f32; n * n];
        let mut cm = vec![0.0f32; n * n];
        g.throughput(Throughput::Elements(flops(n)));

        g.bench_with_input(BenchmarkId::new("rblas", n), &n, |bn, &n| {
            bn.iter(|| {
                level3::gemm(
                    Layout::ColMajor,
                    Transpose::None,
                    Transpose::None,
                    n,
                    n,
                    n,
                    black_box(1.0f32),
                    black_box(&a),
                    n,
                    black_box(&b),
                    n,
                    0.0,
                    black_box(&mut cm),
                    n,
                );
            });
        });

        g.bench_with_input(BenchmarkId::new("matrixmultiply", n), &n, |bn, &n| {
            bn.iter(|| unsafe {
                // Column-major: row stride 1, col stride n (lda).
                matrixmultiply::sgemm(
                    n,
                    n,
                    n,
                    1.0,
                    a.as_ptr(),
                    1,
                    n as isize,
                    b.as_ptr(),
                    1,
                    n as isize,
                    0.0,
                    cm.as_mut_ptr(),
                    1,
                    n as isize,
                );
            });
        });
    }
    g.finish();
}

fn bench_dgemm(c: &mut Criterion) {
    let mut g = c.benchmark_group("dgemm_vs_matrixmultiply");
    for &n in SIZES {
        let a = vec![1.0f64; n * n];
        let b = vec![2.0f64; n * n];
        let mut cm = vec![0.0f64; n * n];
        g.throughput(Throughput::Elements(flops(n)));

        g.bench_with_input(BenchmarkId::new("rblas", n), &n, |bn, &n| {
            bn.iter(|| {
                level3::gemm(
                    Layout::ColMajor,
                    Transpose::None,
                    Transpose::None,
                    n,
                    n,
                    n,
                    black_box(1.0f64),
                    black_box(&a),
                    n,
                    black_box(&b),
                    n,
                    0.0,
                    black_box(&mut cm),
                    n,
                );
            });
        });

        g.bench_with_input(BenchmarkId::new("matrixmultiply", n), &n, |bn, &n| {
            bn.iter(|| unsafe {
                matrixmultiply::dgemm(
                    n,
                    n,
                    n,
                    1.0,
                    a.as_ptr(),
                    1,
                    n as isize,
                    b.as_ptr(),
                    1,
                    n as isize,
                    0.0,
                    cm.as_mut_ptr(),
                    1,
                    n as isize,
                );
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench_sgemm, bench_dgemm);
criterion_main!(benches);
