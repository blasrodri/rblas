//! Level-1 throughput benchmarks (axpy / dot / scal) across sizes.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rblas::level1;

fn bench_axpy(c: &mut Criterion) {
    let mut g = c.benchmark_group("axpy_f32");
    for &n in &[1_024usize, 16_384, 262_144, 1_048_576] {
        let x = vec![1.0f32; n];
        let mut y = vec![2.0f32; n];
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| level1::axpy(n, black_box(3.0f32), black_box(&x), 1, black_box(&mut y), 1));
        });
    }
    g.finish();
}

fn bench_dot(c: &mut Criterion) {
    let mut g = c.benchmark_group("dot_f32");
    for &n in &[1_024usize, 16_384, 262_144, 1_048_576] {
        let x = vec![1.0f32; n];
        let y = vec![2.0f32; n];
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| black_box(level1::dot(n, black_box(&x), 1, black_box(&y), 1)));
        });
    }
    g.finish();
}

fn bench_dot_f64(c: &mut Criterion) {
    let mut g = c.benchmark_group("dot_f64");
    for &n in &[16_384usize, 262_144, 1_048_576] {
        let x = vec![1.0f64; n];
        let y = vec![2.0f64; n];
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| black_box(level1::dot(n, black_box(&x), 1, black_box(&y), 1)));
        });
    }
    g.finish();
}

criterion_group!(benches, bench_axpy, bench_dot, bench_dot_f64);
criterion_main!(benches);
