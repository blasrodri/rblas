//! Quick wall-clock speedup check for the threaded GEMM path.
//!
//! Run serial:    `cargo run --release --example threads_speedup`
//! Run threaded:  `cargo run --release --features threads --example threads_speedup`
//!
//! Uses a large `n` (> NC = 4096) so the column-band partition actually splits
//! across the rayon pool. Reports GFLOP/s for an `n³` GEMM.

use std::time::Instant;

use rblas::{level3, Layout, Transpose};

fn main() {
    let n = 4608; // multiple of 256 and > NC so several bands are produced
    let (m, k) = (n, n);
    println!("GEMM {m}×{n}×{k} (f32)");

    let a = vec![1.0f32; m * k];
    let b = vec![2.0f32; k * n];
    let mut c = vec![0.0f32; m * n];

    // Warm up (and let rayon spin up its pool).
    run(m, n, k, &a, &b, &mut c);

    let iters = 3;
    let t0 = Instant::now();
    for _ in 0..iters {
        run(m, n, k, &a, &b, &mut c);
    }
    let secs = t0.elapsed().as_secs_f64() / iters as f64;
    let gflops = 2.0 * (n as f64).powi(3) / secs / 1e9;

    #[cfg(feature = "threads")]
    let mode = format!("threaded ({} threads)", rblas::level3::num_threads());
    #[cfg(not(feature = "threads"))]
    let mode = "serial".to_string();

    println!("{mode}: {:.2} ms/iter, {gflops:.1} GFLOP/s", secs * 1e3);
}

fn run(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    level3::gemm(
        Layout::ColMajor,
        Transpose::None,
        Transpose::None,
        m,
        n,
        k,
        1.0,
        a,
        m,
        b,
        k,
        0.0,
        c,
        m,
    );
}
