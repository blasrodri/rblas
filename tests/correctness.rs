//! Integration-level correctness: large random GEMM against a naive reference,
//! exercising the full packed NEON path under `--release`.

use rblas::{level3, Layout, Transpose};

fn naive(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for j in 0..n {
        for i in 0..m {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i + p * m] * b[p + j * k];
            }
            c[i + j * m] = acc;
        }
    }
    c
}

// Cheap deterministic PRNG so the test needs no dev-dependency.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

#[test]
fn large_random_sgemm_matches_naive() {
    let (m, n, k) = (200, 180, 150);
    let mut s = 0x1234_5678u64;
    let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut s)).collect();
    let b: Vec<f32> = (0..k * n).map(|_| lcg(&mut s)).collect();
    let mut c = vec![0.0f32; m * n];
    level3::gemm(
        Layout::ColMajor,
        Transpose::None,
        Transpose::None,
        m,
        n,
        k,
        1.0,
        &a,
        m,
        &b,
        k,
        0.0,
        &mut c,
        m,
    );
    let want = naive(m, n, k, &a, &b);
    let mut max_rel = 0.0f32;
    for (g, w) in c.iter().zip(&want) {
        let denom = w.abs().max(1e-4);
        max_rel = max_rel.max((g - w).abs() / denom);
    }
    assert!(max_rel < 1e-3, "max relative error {max_rel} too large");
}
