use std::f64::consts::PI;
use std::option::Option;

use argmin::{
    core::{OptimizationResult, CostFunction, Executor, State, Error as ArgminError},
    solver::brent::BrentRoot,
};

use rustdct::DctPlanner;

use ndarray::Array1;

// Struct to handle evaluating the root of t - ξγ(t)
struct ZetaGammaLMinusT {
    // Cached square indices
    k2: Array1<f64>,
    // Cached transformed components of the type II DCT
    a2: Array1<f64>,
    // Nr of datapoints (not nr of grid points)
    n_datapoints: f64,
    // The Nr of times the ξ function is applied
    l: i32,
    // Cached constants that depend only on the iteration index
    // and do neither depend on the data nor on values from
    // previous iterations.
    // The values are cached not only because of performance
    // (the performance impact would probably be minimal)
    // but mainly to make it clear which parts of the calculation
    // depend on the data and which parts don't.
    cached_c: Array1<f64>
}

struct Histogram {
    counts: Vec<u32>,
    bins: Vec<f64>,
    data_range: f64
}

#[derive(Debug)]
pub struct Kde1DResult {
    pub density: Vec<f64>,
    pub grid: Vec<f64>,
    pub bandwidth: f64
}

// The product of the first `n` odd numbers.
// This is closely related to the 'alternate factorial' or 'odd factorial'.
fn product_of_first_n_odd_numbers(n: i32) -> f64 {
    // Convert integers into floats to avoid integer overflow errors.
    (1..=n)
    .map(|k| (2*k - 1) as f64)
    .product()
}

impl ZetaGammaLMinusT {
    pub fn new(k2: &Array1<f64>, transformed: &Array1<f64>, n_datapoints: usize, l: i32) -> Self {
        // Unlike the Python implementation, we don't divide by 2.0 here
        // because we are not applying the same kind of post-processing.
        // This is due to different normalization operations in the Python
        // and Rust imiplementations of the discrete cosinde transform.
        //
        // NOTE: Thanks to Frank Steffahn (@steffahn) who found out that
        // a different kind of post-processing was needed through old-fashioned
        // experimentation (although his suggestion was a bit different).
        let a2: Array1<f64> = transformed.powi(2);

        // Slowly build up the constants we'll need.
        // To make it easier to compare our implementation to the "reference"
        // python implementation, we will compute these constants the same way,
        // in two steps instead of all at once.
        //
        // Because these computations are run only once and don't enter the
        // solution loop, we can be as inefficient as we want (within reason).
        //
        // The whole point of this is to cache constants that will be reused
        // in all loops instead of computing them anew in each loop iteration.
        // It also helps simplify the expressions inside the loop iteration.

        // This constant is called `K` in the Python code.
        let cached_c1: Array1<f64> =
            (2..=(l - 1))
            .rev()
            .map(|s| product_of_first_n_odd_numbers(s) / (2.0 * PI).sqrt())
            .collect::<Vec<_>>()
            .into();

        // This constant is called `C` in the Python code.
        let cached_c2: Array1<f64> =
            (2..=(l - 1))
            .rev()
            .map(|s| (1.0 + 0.5_f64.powf((s as f64) + 0.5)) / 3.0)
            .collect::<Vec<_>>()
            .into();

        // The cached product, which is a constant for each iteration
        let cached_c: Array1<f64> = 2.0 * cached_c1 * cached_c2 / (n_datapoints as f64);

        Self {
            k2: k2.clone(),
            a2: a2,
            n_datapoints: (n_datapoints as f64),
            l: l,
            // Cached here for efficiency and to increase code clarity
            cached_c: cached_c
        }
    }
}

impl CostFunction for ZetaGammaLMinusT {
    type Param = f64;
    type Output = f64;

    fn cost(&self, t: &Self::Param) -> Result<Self::Output, ArgminError> {
        // While this function cold probably be implemented as a `.fold()`
        // operating on the index of iteration, we have chosen to implement it
        // as a mutable variable `f` that is continuously updated in order
        // to be closer to the original Python implementation.
        let mut f: f64 = 2.0 * PI.powi(2 * self.l) * 
            (&self.k2.powi(self.l) *
             &self.a2 *
             (- PI.powi(2) * &self.k2 * *t).exp())
            .sum();
        
        // Let's ignore the cached values and calculate them anew in each iteration,
        // so that the expression is more similar than the python reference.
        // Once we understand where the results difference comes from, we'll get back
        // to caching the results
        for (s, c_s) in (2..=(self.l - 1)).rev().zip(&self.cached_c) {
            // The original Python implementation re-evaluates `k` and `c`
            // on each iteration. While this is not super inefficient, it
            // hides the fact that these are just constants that only depend
            // on the index of iteration `s` and not on any of the previous
            // iterations of the loop (or even on the transformed values)
            //
            // The original Python code is equivalent to the following lines:
            // 
            //     let k = product_of_first_n_odd_numbers(s) / (2.0 * PI).sqrt();
            //     let c = (1.0 + 0.5_f64.powf((s as f64) + 0.5)) / 3.0;
            //     let t_s: f64 = (2.0 * k * c / (self.n_datapoints * f)).powf(2.0 / (3.0 + 2.0 * (s as f64)))
            //
            // All the constants in the product are encapsulated in the `c_s` value,
            // which we get from the cached vector, resulting in the following expression:
            let t_s: f64 = (c_s / f).powf(2.0 / (3.0 + 2.0 * (s as f64)));
            
            f = 2.0 * PI.powi(2 * s) *
               (&self.k2.powi(s) * 
                &self.a2 *
                (- PI.powi(2) * &self.k2 * t_s).exp())
                .sum()
        }

        // Return the final difference (the "fixed point")
        Ok(t - (2.0 * self.n_datapoints * PI.sqrt() * f).powf(-0.4))
    }
}

fn histogram(mut x: Vec<f64>, grid_size: usize, lim_low: Option<f64>, lim_high: Option<f64>) -> Histogram {
    // Sort the array so that placing the values in bins becomes easier
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    // This can only fail if x is empty or the result contains NaNs.
    // We should deal with that case before these lines.
    let x_min0 = x.first().unwrap();
    let x_max0 = x.last().unwrap();
    let delta0 = x_max0 - x_min0;

    // TODO: remove when we publish the crate
    for x_i in x.iter() {
        assert!(x_i >= x_min0);
        assert!(x_i <= x_max0);
    }

    let x_min = match lim_low {
        Some(value) => value,
        None => x_min0 - delta0 * 0.1
    };

    let x_max = match lim_high {
        Some(value) => value,
        None => x_max0 + delta0 * 0.1
    };

    // Due to the way we build the edges, it's true that `edges.len() == grid_size + 1`
    let edges: Vec<f64> = Array1::linspace(x_min, x_max, grid_size + 1).to_vec();
    // TODO: try to convert this into a slice
    let bins: Vec<f64> = (0..grid_size).map(|i| edges[i]).collect::<Vec<f64>>();
    
    let mut counts: Vec<u32> = vec![0; grid_size];

    // j is the index for the bin and for the edges
    let mut j = 0;
    for x_i in x.iter() {
        // Increase the bin index until we are in the right bin.
        // We know we are in the right bin when we are past the relevant left edge. 
        while *x_i > edges[j + 1] {
            j += 1
        }

        // Now that we are in the right bin, increase the count for the bin
        counts[j] += 1
    }

    let result = Histogram{
        counts: counts,
        bins: bins,
        data_range: x_max - x_min
    };

    result
}

/// Evaluated the KDE for a 1D distribution from a vector of observations.
/// Requires a suggested grid size and optional upper or lower limits.
///
/// If the limits are not given, they will be evaluated from the data.
pub fn kde_1d(x: Vec<f64>, suggested_grid_size: usize, limits: (Option<f64>, Option<f64>)) ->
          Result<Kde1DResult, ArgminError> {
    // Round up the `grid_size` to the next power of two, masking the old value.
    let grid_size = (2_i32.pow((suggested_grid_size as f64).log2().ceil() as u32)) as usize;
    // This is the same as variable `N` from the python implementation
    let n_datapoints = x.len(); 
    
    let (lim_low, lim_high) = limits;

    // Squared indices
    let k2: Array1<f64> =
        (0..grid_size)
        .map(|k| (k as f64).powi(2))
        .collect::<Vec<f64>>()
        .into();

    // Bin the data points into equally spaced bins
    let histogram_result: Histogram = histogram(x, grid_size, lim_low, lim_high);

    let counts = histogram_result.counts;
    let bins = histogram_result.bins;
    let delta_x = histogram_result.data_range;

    // Get the raw density applied at the grid points.
    let mut transformed: Vec<f64> =
        counts
        .iter()
        .map(|count| (*count as f64) / (n_datapoints as f64))
        .collect();

    // Compute type II discrete cosine transform (DCT), then adjust first component.
    let mut planner: DctPlanner<f64> = DctPlanner::new();
    let dct = planner.plan_dct2(grid_size);
    dct.process_dct2(&mut transformed);

    // Adjust the first component of the DCT.
    // I'm not sure what role this plays here, so I'm just copying
    // the Python implementation.
    transformed[0] /= 2.0;
    // Convert the transformed vector into an array to facilitate numerical operations
    let transformed: Array1<f64> = transformed.into();

    // Create the function such that the root is the optimal t_star which we will use
    // to evaluated the kernel bandwidth.
    let fixed_point = ZetaGammaLMinusT::new(&k2, &transformed, n_datapoints, 7_i32);

    // Initial parameter for the root-finding algorithm (this is almost arbitrary)
    let init_param: f64 = 0.05;
    // Find the root with the Brent algorithm (which doesn't require derivative information)
    let solver = BrentRoot::new(0.0, 0.1, 2e-12);
    let result: OptimizationResult<_, _, _> =
        Executor::new(fixed_point, solver)
        .configure(|state| state.param(init_param).max_iters(100))
        // For now, we just bubble up the ArgminError.
        // TODO: find a better error model for this.
        .run()?;

    let t_star: f64 = result.state.get_best_param().copied().unwrap_or(init_param);

    // Smooth the transformation using the `t_star` as a smoothing parameter.
    let mut smoothed: Array1<f64> = transformed * (- 0.5 * PI.powi(2) * t_star * &k2).exp();
    
    // Reverse transformation after adjusting first component
    smoothed[0] *= 2.0;
    // Invert the smoothed transformation to get the smoothed grid values.
    let mut inverse = smoothed.to_vec();
    // The DCT type III is the inverse of the DCT type II.
    // The Python implementation calls this function the IDCT.
    let idct = planner.plan_dct3(grid_size);
    idct.process_dct3(&mut inverse);
    
    // Normalize the smoothed density.
    // Compare with the python implementation, which uses a different
    // normalization constant. This is due to different implementations
    // of the DCT in the libraries in both languages.
    let density: Array1<f64> = 2.0 * Array1::from(inverse) / delta_x;

    // Translate the smoothing parameter into a bandwidth.
    let bandwidth = t_star.sqrt() * delta_x;

    let kde_1d_result = Kde1DResult{
        density: density.to_vec(),
        grid: bins,
        bandwidth: bandwidth
    };

    Ok(kde_1d_result)
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::f64::consts::PI;

    // Enable reading reference data from the python package
    // (numbers must be read as an `Array0`)
    use ndarray::{Array1, Array0};
    use ndarray_npy::NpzReader;
    // Approximate comparisons between floating point
    use approx::assert_relative_eq;
    // Enable plots for visual comparison of curves
    use plotly::{Plot, Scatter, Layout};
    use plotly::common::{Anchor, Mode, Font};
    use plotly::layout::Annotation;
    // Random distributions to reproduce the tests from Botev et al.
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Normal, Distribution};

    use super::*;

    #[test]
    fn test_against_reference() {
        // The (long!) first part of this function varuiables saved
        // into the `.npz` file format by numpy.
        // Because Python's type system is much more lax than
        // rust's own type system, we will need to performe some manual
        // conversions instead of just using the values as they are.
        // While we could convert this into a more friendly format,
        // there is some value in just reusing the reference files
        // in a reference implementation.

        let file = File::open("test/reference1d.npz").unwrap();
        let mut npz = NpzReader::new(file).unwrap();

        // Extract the (sometimes weirdly typed) variables from the npz file.
        // Everything is an array (although maybe a zero-dimensional one)

        // First, we extract the actual arrays
        let x: Array1<f64> = npz.by_name("x").unwrap();
        let reference_density: Array1<f64> = npz.by_name("density").unwrap();
        let reference_grid: Array1<f64> = npz.by_name("grid").unwrap();
        
        // Variables ending in `_array` still need to be converted
        // into their actual type (integer or float)

        // Here, `n` will be the variable referred to as `N` in the original
        // python implementation that serves as reference as well as the
        // respective tests.
        let n_array: Array0<u16> = npz.by_name("N").unwrap();
        // Here, `grid_size` is the variable `n` in the original python tests
        let grid_size_array: Array0<u16> = npz.by_name("n").unwrap();
        let x_min_array: Array0<i16> = npz.by_name("xmin").unwrap();
        let x_max_array: Array0<u8> = npz.by_name("xmax").unwrap();
        let reference_bandwidth_array: Array0<f64> = npz.by_name("bandwidth").unwrap();

        // Convert everything into the right types
        let n: usize = *n_array.get(()).unwrap() as usize;
        let grid_size: usize = *grid_size_array.get(()).unwrap() as usize;
        let x_min: f64 = *x_min_array.get(()).unwrap() as f64;
        let x_max: f64 = *x_max_array.get(()).unwrap() as f64;
        let reference_bandwidth: f64 = *reference_bandwidth_array.get(()).unwrap();

        let limits = (Some(x_min), Some(x_max));
        let kde_result: Kde1DResult = kde_1d(x.to_vec(), grid_size, limits).unwrap();

        assert_eq!(n, x.len());
        assert_relative_eq!(reference_bandwidth, kde_result.bandwidth);

        for i in 0..grid_size {
            assert_relative_eq!(reference_density[i], kde_result.density[i]);
            assert_relative_eq!(reference_grid[i], kde_result.grid[i])
        }
      
        // Invert the formula for the bandwidth to get the t_star from the root
        // finding algorithm, to see how different it is from the reference implementation
        let calculated_t_star = (kde_result.bandwidth / (x_max - x_min)).powi(2);
        let reference_t_star = (reference_bandwidth / (x_max - x_min)).powi(2);
        
        // Plot the data so that it can be visually inspected.
        // Note that all automated tests that comapre numbers have been done before.
        // We only add this to help debug things if anything goes wrong.
        let mut plot = Plot::new();

        let calculated_density_trace =
            Scatter::new(kde_result.grid, kde_result.density)
            .mode(Mode::LinesMarkers)
            .name("Calculated density (.rs)");

        let reference_density_trace =
            Scatter::new(reference_grid.to_vec(), reference_density.to_vec())
            .mode(Mode::LinesMarkers)
            .name("Reference density (.py)");

        plot.add_trace(calculated_density_trace);
        plot.add_trace(reference_density_trace);

        let layout = Layout::new().annotations(vec![
            Annotation::new()
                .text(format!("
                    Reference BW = {:.6}<br>
                    Calculated BW = {:.6}<br><br>
                    Reference t⋆ = {:.6}<br>
                    Calculated t⋆ = {:.6}",
                    reference_t_star,
                    calculated_t_star,
                    &reference_bandwidth,
                    &kde_result.bandwidth))
                .x(0.95)
                .y(0.95)
                .x_ref("paper")
                .y_ref("paper")
                .x_anchor(Anchor::Right)
                .y_anchor(Anchor::Top)
                .show_arrow(false)
                .font(Font::new().size(8))
        ]);

        plot.set_layout(layout);

        plot.write_html("test/plots/kde1py_reference.html");
    }

    const BOTEV_SEED: u64 = 31415;
    const BOTEV_GRID_SIZE: usize = 2048;

    #[derive(Clone)]
    struct MixtureOfNormals {
        parameters: Vec<(f64, (f64, f64))>,
        weights: Vec<f64>,
        distributions: Vec<Normal<f64>>
    }

    impl MixtureOfNormals {
        fn new(parameters: Vec<(f64, (f64, f64))>) -> Self {
            let mut distributions = Vec::with_capacity(parameters.len());
            let mut weights = Vec::with_capacity(parameters.len());
            
            for (weight, (mean, std_dev)) in parameters.iter() {
                weights.push(*weight);
                distributions.push(Normal::new(*mean, *std_dev).unwrap());
            }

            Self {
                parameters: parameters,
                weights: weights,
                distributions: distributions
            }
        }
        
        fn pdf(&self, x: f64) -> f64 {
            // Sum this expression over the weights and distributions:
            //
            //      w * (1 / sqrt(2π σ²)) * exp(-((x - μ)² / (2σ²)))
            
            self.parameters
                .iter()
                .map(|(w, (mean, std_dev))|
                    w * (- (x - mean).powi(2) / (2.0 * std_dev.powi(2))).exp() /
                    (std_dev * (2.0 * PI).sqrt())
                )
                .sum()
        }
    }

    impl Distribution<f64> for MixtureOfNormals {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
            let r: f64 = rng.random();
            
            let mut weight: f64 = self.weights[0];
            let mut i: usize = 0;
            while r > weight {
                weight += self.weights[i + 1];
                i += 1;
            }

            let normal = self.distributions[i];
            normal.sample(rng)
        }
    }

    fn plot_density(path: &str, title: &str, dist: &MixtureOfNormals, kde_result: Kde1DResult) {
        let mut plot = Plot::new();

        let actual_density: Vec<f64> = kde_result.grid
            .clone()
            .iter()
            .map(|x| dist.pdf(*x))
            .collect();
        
        let calculated_density_trace =
            Scatter::new(kde_result.grid.clone(), kde_result.density)
            .mode(Mode::Lines)
            .name(format!("{} (calculated density)", title));

        let actual_density_trace =
            Scatter::new(kde_result.grid, actual_density)
            .mode(Mode::Lines)
            .name(format!("{} (actual density)", title));


        let layout = Layout::new().title(title);

        plot.add_trace(actual_density_trace);
        plot.add_trace(calculated_density_trace);
        plot.set_layout(layout);

        plot.write_html(path);
    }

    #[test]
    fn test_botev_01_claw() {
        let claw = MixtureOfNormals::new(vec![
            (0.5, (0.0, 1.0)),
            (0.1, (-1.0, 0.1)),
            (0.1, (-0.5, 0.1)),
            (0.1, (0.0, 0.1)),
            (0.1, (0.5, 0.1)),
            (0.1, (1.0, 0.1))
        ]);

        let mut rng = ChaCha8Rng::seed_from_u64(BOTEV_SEED);
        let x: Vec<f64> = claw.clone().sample_iter(&mut rng).take(100_000).collect();

        let kde_result: Kde1DResult = kde_1d(x, BOTEV_GRID_SIZE, (None, None)).unwrap();

        plot_density(
            "test/plots/botev_01_claw.html",
            "Claw",
            &claw,
            kde_result
        );
    }

    #[test]
    fn test_botev_02_strongly_skewed() {
        let strongly_skewed = MixtureOfNormals::new(
            (0..=7)
            .map(|k| (1./8., (3. * (2./3. as f64).powi(k) - 1., (2./3. as f64).powi(2*k))))
            .collect()
        );

        let mut rng = ChaCha8Rng::seed_from_u64(BOTEV_SEED);
        let x: Vec<f64> = strongly_skewed.clone().sample_iter(&mut rng).take(10_000).collect();

        let kde_result: Kde1DResult = kde_1d(x, BOTEV_GRID_SIZE, (None, None)).unwrap();

        plot_density(
            "test/plots/botev_02_strongly_skewed.html",
            "Strongly skewed",
            &strongly_skewed,
            kde_result
        );
    }

    #[test]
    fn test_botev_03_kurtotic_unimodal() {
        let kurtotic_unimodal = MixtureOfNormals::new(vec![
            (2./3., (0.0, 1.0)),
            (1./3., (0.0, 0.1))
        ]);

        let mut rng = ChaCha8Rng::seed_from_u64(BOTEV_SEED);
        let x: Vec<f64> = kurtotic_unimodal.clone().sample_iter(&mut rng).take(10_000).collect();

        let kde_result: Kde1DResult = kde_1d(x, BOTEV_GRID_SIZE, (None, None)).unwrap();

        plot_density(
            "test/plots/botev_03_kurtotic_unimodal.html",
            "Kurtotic unimodal",
            &kurtotic_unimodal,
            kde_result
        );
    }

    #[test]
    fn test_botev_04_double_claw() {
        let mut parameters = vec![
            (49./100., (-1.0, 2./3.)),
            (49./100., (1.0, 2./3.))
        ];

        for k in 0..=6 {
            parameters.push((1./350., ((k as f64 - 3.0) / 2.0, 1./100.)))
        }

        let double_claw = MixtureOfNormals::new(parameters);

        let mut rng = ChaCha8Rng::seed_from_u64(BOTEV_SEED);
        let x: Vec<f64> = double_claw.clone().sample_iter(&mut rng).take(1_000_000).collect();

        let kde_result: Kde1DResult = kde_1d(x, BOTEV_GRID_SIZE, (None, None)).unwrap();

        plot_density(
            "test/plots/botev_04_double_claw.html",
            "Double claw",
            &double_claw,
            kde_result
        );
    }

    #[test]
    fn test_botev_05_discrete_comb() {
        let mut parameters = vec![];

        for k in 0..=2 {
            parameters.push((2./7., ((12.*(k as f64) - 15.)/7., 2./7.)))
        }

        for k in 8..=10 {
            parameters.push((1./21., (2.*(k as f64)/7., 1./21.)))
        }

        let discrete_comb = MixtureOfNormals::new(parameters);

        let mut rng = ChaCha8Rng::seed_from_u64(BOTEV_SEED);
        let x: Vec<f64> = discrete_comb.clone().sample_iter(&mut rng).take(100_000).collect();

        let kde_result: Kde1DResult = kde_1d(x, BOTEV_GRID_SIZE, (None, None)).unwrap();

        plot_density(
            "test/plots/botev_05_discrete_comb.html",
            "Discrete comb",
            &discrete_comb,
            kde_result
        );
    }
    
    #[test]
    fn test_botev_06_asymmetric_double_claw() {
        let mut parameters = vec![];

        for k in 0..=1 {
            parameters.push((46./100., (2.*(k as f64) - 1., 2./3.)))
        }

        for k in 1..=3 {
            parameters.push((1./300., (-(k as f64)/2., 1./100.)))
        }

        for k in 1..=3 {
            parameters.push((7./300., ((k as f64)/2., 7./100.)))
        }

        let asymmetric_double_claw = MixtureOfNormals::new(parameters);

        let mut rng = ChaCha8Rng::seed_from_u64(BOTEV_SEED);
        let x: Vec<f64> = asymmetric_double_claw.clone().sample_iter(&mut rng).take(1_000_000).collect();

        let kde_result: Kde1DResult = kde_1d(x, BOTEV_GRID_SIZE, (None, None)).unwrap();

        plot_density(
            "test/plots/botev_06_asymmetric_double_claw.html",
            "Discrete comb",
            &asymmetric_double_claw,
            kde_result
        );
    }
    
    #[test]
    fn test_botev_07_outlier() {
        let parameters = vec![
            (1./10., (0., 1.)),
            (9./10., (0., 0.1))
        ];

        let outlier = MixtureOfNormals::new(parameters);

        let mut rng = ChaCha8Rng::seed_from_u64(BOTEV_SEED);
        let x: Vec<f64> = outlier.clone().sample_iter(&mut rng).take(100_000).collect();

        let kde_result: Kde1DResult = kde_1d(x, BOTEV_GRID_SIZE, (None, None)).unwrap();

        plot_density(
            "test/plots/botev_07_outlier.html",
            "Outlier",
            &outlier,
            kde_result
        );
    }

    #[test]
    fn test_botev_08_separated_bimodal() {
        let parameters = vec![
            (1./2., (-12., 1./2.)),
            (1./2., (12., 1./2.))
        ];

        let separated_bimodal = MixtureOfNormals::new(parameters);

        let mut rng = ChaCha8Rng::seed_from_u64(BOTEV_SEED);
        let x: Vec<f64> = separated_bimodal.clone().sample_iter(&mut rng).take(1_000).collect();

        let kde_result: Kde1DResult = kde_1d(x, BOTEV_GRID_SIZE, (None, None)).unwrap();

        plot_density(
            "test/plots/botev_08_separated_bimodal.html",
            "Separated bimodal",
            &separated_bimodal,
            kde_result
        );
    }

    #[test]
    fn test_botev_09_skewed_bimodal() {
        let parameters = vec![
            (3./4., (0., 1.)),
            (1./4., (3./2., 1./3.))
        ];

        let skewed_bimodal = MixtureOfNormals::new(parameters);

        let mut rng = ChaCha8Rng::seed_from_u64(BOTEV_SEED);
        let x: Vec<f64> = skewed_bimodal.clone().sample_iter(&mut rng).take(10_000).collect();

        let kde_result: Kde1DResult = kde_1d(x, BOTEV_GRID_SIZE, (None, None)).unwrap();

        plot_density(
            "test/plots/botev_09_skewed_bimodal.html",
            "Skewed bimodal",
            &skewed_bimodal,
            kde_result
        );
    }

    #[test]
    fn test_botev_10_bimodal() {
        let parameters = vec![
            (1./2., (0., 0.1)),
            (1./2., (5., 1.))
        ];

        let bimodal = MixtureOfNormals::new(parameters);

        let mut rng = ChaCha8Rng::seed_from_u64(BOTEV_SEED);
        let x: Vec<f64> = bimodal.clone().sample_iter(&mut rng).take(10_000).collect();

        let kde_result: Kde1DResult = kde_1d(x, BOTEV_GRID_SIZE, (None, None)).unwrap();

        plot_density(
            "test/plots/botev_10_bimodal.html",
            "Bimodal",
            &bimodal,
            kde_result
        );
    }
}
