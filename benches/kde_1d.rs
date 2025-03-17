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

pub fn criterion_benchmark(c: &mut Criterion) {
    let grid_size = 1024;
    let limits = (None, None);

    let x_botev_01_claw: Vec<f64> = data_from_npz("test/reference_densities/botev_01_claw.npz");
    let x_botev_02_strongly_skewed: Vec<f64> = data_from_npz("test/reference_densities/botev_01_claw.npz");
    let x_botev_03_kurtotic_unimodal: Vec<f64> = data_from_npz("test/reference_densities/botev_03_kurtotic_unimodal.npz");
    let x_botev_04_double_claw: Vec<f64> = data_from_npz("test/reference_densities/botev_04_double_claw.npz");

    // Try to get rid of the clones inside the closure

    c.bench_function("01. Claw", |b| b.iter(||
        kde_1d(&x_botev_01_claw, grid_size, limits)
    ));

    c.bench_function("02. Strongly skewed", |b| b.iter(||
        kde_1d(&x_botev_02_strongly_skewed, grid_size, limits)
    ));

    c.bench_function("03. Strongly skewed", |b| b.iter(||
        kde_1d(&x_botev_03_kurtotic_unimodal, grid_size, limits)
    ));

    c.bench_function("04. Double claw", |b| b.iter(||
        kde_1d(&x_botev_04_double_claw, grid_size, limits)
    ));
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .with_profiler(PProfProfiler::new(1000, Output::Flamegraph(None)));
    targets = criterion_benchmark
);
criterion_main!(benches);