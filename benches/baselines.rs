//! rblas vs. the real competitors: **faer** (pure Rust) and **OpenBLAS**
//! (C/Fortran). All single-threaded, identical column-major buffers, same
//! sizes, throughput as FLOP/s (2·m·n·k). Enable with `--features bench-baselines`.
//!
//! OpenBLAS is the honest bar — a mature, assembly-tuned BLAS. faer is the
//! state-of-the-art pure-Rust crate. matrixmultiply (in `compare.rs`) is only a
//! pure-Rust *peer*; these two are what "competitive" should really mean.

// Linking `openblas-src` brings the OpenBLAS symbols in for `cblas`.
extern crate openblas_src;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rblas::{level3, Layout, Transpose};

const SIZES: &[usize] = &[256, 512, 1024];

fn flops(n: usize) -> u64 {
    2 * (n as u64).pow(3)
}

fn bench_dgemm(c: &mut Criterion) {
    // Pin OpenBLAS to a single thread for a fair single-core comparison.
    unsafe {
        openblas_set_num_threads(1);
    }
    let mut g = c.benchmark_group("dgemm");
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
                    1.0,
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

        g.bench_with_input(BenchmarkId::new("faer", n), &n, |bn, &n| {
            let fa = faer::Mat::from_fn(n, n, |i, j| a[i + j * n]);
            let fb = faer::Mat::from_fn(n, n, |i, j| b[i + j * n]);
            bn.iter(|| {
                let prod = black_box(&fa) * black_box(&fb);
                black_box(prod);
            });
        });

        g.bench_with_input(BenchmarkId::new("openblas", n), &n, |bn, &n| {
            bn.iter(|| unsafe {
                cblas::dgemm(
                    cblas::Layout::ColumnMajor,
                    cblas::Transpose::None,
                    cblas::Transpose::None,
                    n as i32,
                    n as i32,
                    n as i32,
                    1.0,
                    black_box(&a),
                    n as i32,
                    black_box(&b),
                    n as i32,
                    0.0,
                    black_box(&mut cm),
                    n as i32,
                );
            });
        });
    }
    g.finish();
}

fn bench_sgemm(c: &mut Criterion) {
    unsafe {
        openblas_set_num_threads(1);
    }
    let mut g = c.benchmark_group("sgemm");
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
                    1.0,
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

        g.bench_with_input(BenchmarkId::new("faer", n), &n, |bn, &n| {
            let fa = faer::Mat::from_fn(n, n, |i, j| a[i + j * n]);
            let fb = faer::Mat::from_fn(n, n, |i, j| b[i + j * n]);
            bn.iter(|| {
                let prod = black_box(&fa) * black_box(&fb);
                black_box(prod);
            });
        });

        g.bench_with_input(BenchmarkId::new("openblas", n), &n, |bn, &n| {
            bn.iter(|| unsafe {
                cblas::sgemm(
                    cblas::Layout::ColumnMajor,
                    cblas::Transpose::None,
                    cblas::Transpose::None,
                    n as i32,
                    n as i32,
                    n as i32,
                    1.0,
                    black_box(&a),
                    n as i32,
                    black_box(&b),
                    n as i32,
                    0.0,
                    black_box(&mut cm),
                    n as i32,
                );
            });
        });
    }
    g.finish();
}

extern "C" {
    // Provided by OpenBLAS; pin threads so the comparison is single-core.
    fn openblas_set_num_threads(n: i32);
}

criterion_group!(benches, bench_dgemm, bench_sgemm);
criterion_main!(benches);
