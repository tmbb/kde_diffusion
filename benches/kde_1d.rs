use std::fs::File;

use criterion::{
    criterion_group, criterion_main,
    Criterion, Throughput, BenchmarkId
};

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
    // This will be constant for all benchmarks
    let grid_size = 1024;
    let limits = (None, None);

    // Get this specific distribution because it's the one
    // with the most datapoints (total of 10^6 datapoints)
    let x_botev_04_double_claw: Vec<f64> = data_from_npz(
        "test/reference_densities/botev_04_double_claw.npz"
    );

    let mut group = c.benchmark_group("Double claw");
    for size in (1..=10).map(|i| i * 20_000 as usize) {
        let size_in_k = size / 1000;
        let parameter_string = format!("{:04}k", size_in_k);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("Throughput", parameter_string), &size, |b, &size| {
            // Take the first `size` values from the distribution
            let x: Vec<f64> = x_botev_04_double_claw.iter().map(|x_i| *x_i).take(size).collect();
            b.iter(|| kde_1d(&x, grid_size, limits));
        });
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .measurement_time(std::time::Duration::new(10, 0))
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = criterion_benchmark
);
criterion_main!(benches);