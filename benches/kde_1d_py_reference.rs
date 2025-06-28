use std::fs::File;

use criterion::{criterion_group, criterion_main, Criterion};

use pprof::criterion::{Output, PProfProfiler};

use ndarray::Array1;
use ndarray_npy::NpzReader;

use kde_diffusion::kde_1d;

fn data_from_npz(path: &str) -> Vec<f64> {
    let file = File::open(path).unwrap();
    let mut npz = NpzReader::new(file).unwrap();
    let x: Array1<f64> = npz.by_name("x").unwrap();
    x.to_vec()
}

pub fn criterion_benchmark_vs_py(c: &mut Criterion) {
    // This will be constant for all benchmarks
    let grid_size = 1024;
    let limits = (None, None);

    // Get this specific distribution because it's the one
    // with the most datapoints (total of 10^6 datapoints)
    let reference1d: Vec<f64> = data_from_npz(
        "test/reference_densities/reference1d.npz"
    );

    c.bench_function("1D vs Python", |b| b.iter(||
        kde_1d(&reference1d, grid_size, limits)
    ));
} 

criterion_group!(
    name = benches;
    config = Criterion::default()
        .measurement_time(std::time::Duration::new(10, 0))
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = criterion_benchmark_vs_py
);
criterion_main!(benches);