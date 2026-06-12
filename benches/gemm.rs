//! GEMM benchmark reporting GFLOP/s for square sizes.
//!
//! FLOPs for an m×n×k multiply-add GEMM = 2*m*n*k. Criterion's element
//! throughput is set to that so the report reads directly in FLOP/s.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rblas::{level3, Layout, Transpose};

fn bench_sgemm(c: &mut Criterion) {
    let mut g = c.benchmark_group("sgemm_square");
    for &n in &[64usize, 128, 256, 512, 1024] {
        let a = vec![1.0f32; n * n];
        let b = vec![2.0f32; n * n];
        let mut cm = vec![0.0f32; n * n];
        g.throughput(Throughput::Elements(2 * (n as u64).pow(3)));
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |bn, &n| {
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
    }
    g.finish();
}

fn bench_dgemm(c: &mut Criterion) {
    let mut g = c.benchmark_group("dgemm_square");
    for &n in &[64usize, 128, 256, 512, 1024] {
        let a = vec![1.0f64; n * n];
        let b = vec![2.0f64; n * n];
        let mut cm = vec![0.0f64; n * n];
        g.throughput(Throughput::Elements(2 * (n as u64).pow(3)));
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |bn, &n| {
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
    }
    g.finish();
}

criterion_group!(benches, bench_sgemm, bench_dgemm);
criterion_main!(benches);
