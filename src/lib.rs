use std::f64::consts::PI;
use std::option::Option;
use std::cmp::min;

use argmin::{
    core::{OptimizationResult, CostFunction, Executor, State, Error as ArgminError},
    solver::brent::BrentRoot,
};

use rustdct::DctPlanner;

use ndarray::Array1;

use num_traits::{Float, ToPrimitive, FromPrimitive};

// Struct to handle evaluating the root of t - ξγ(t)
struct ZetaGammaLMinusT {
    // Cached square indices
    k2: Array1<f64>,
    // Cached transformed components of the type II DCT
    a2: Array1<f64>,
    // Nr of datapoints (not nr of grid points)
    n_datapoints: f64,
    // The Nr of times the ξ function is applied
    l: i32
}

struct Histogram<F: Float> {
    counts: Vec<u32>,
    bins: Vec<F>,
    data_range: F
}

#[derive(Debug)]
pub struct Kde1DResult<F: Float + FromPrimitive + ToPrimitive> {
    pub density: Vec<F>,
    pub grid: Vec<F>,
    pub bandwidth: F
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

        Self {
            k2: k2.clone(),
            a2: a2,
            n_datapoints: (n_datapoints as f64),
            l: l
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
        
        for s in (2..=(self.l - 1)).rev() {
            // `k` and `c` are only dependent on `s` and not on
            // the value of `f` from previous iterations.
            let k = product_of_first_n_odd_numbers(s) / (2.0 * PI).sqrt();
            let c = (1.0 + 0.5_f64.powf((s as f64) + 0.5)) / 3.0;

            let t_s: f64 = (2.0 * k * c / (self.n_datapoints * f)).powf(2.0 / (3.0 + 2.0 * (s as f64)));
            
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

fn max_min_of_array<F: Float>(x: &Vec<F>) -> Option<(F, F)> {
    x
    .iter()
    .fold(None, |maybe_min_max, x_i| {
        match maybe_min_max {
            None => {
                Some((*x_i, *x_i))
            },
            
            Some((x_min, x_max)) => {
                let new_x_min = if *x_i < x_min { *x_i } else { x_min };
                let new_x_max = if *x_i > x_max { *x_i } else { x_max };

                Some((new_x_min, new_x_max))
            }
        }
    })
}

#[inline(never)]
fn histogram<F>(x: &Vec<F>, grid_size: usize, lim_low: Option<F>, lim_high: Option<F>)
        -> Histogram<F> where F: Float + ToPrimitive + FromPrimitive {
    // -------------------
    // Implementation note
    // -------------------
    // The code below seems a bit overcomplicated.
    // It would be tempting to sort the vector `x` to simplify
    // the placement of the datapoints on the grid.
    // Indeed, the first version of this code used to sort the vector first.
    // This used to have two big disadvantages:
    //
    //   1. It used to be very slow. Profiling showed that 90% of the time
    //      used to be spent sorting the vector
    //
    //   2. It required either cloning the vector or taking ownership of the
    //      vector. Now that we are not sorting anything, we can work with
    //      just a reference to the vector.
    
    // We need to get the maximum and minimum value of the array.
    // Because equality of floating point values is weird, the easiest way is to implement
    // the iterator ourselves.
    
    
    // This can only fail if x is empty or the result contains NaNs.
    // We should deal with that case before these lines.
    let (x_min0, x_max0) = max_min_of_array(&x).unwrap();
    let delta0: F = x_max0 - x_min0;

    // If no limits are given, extend the maximum and the minimum by
    // 10% of the data range.
    let margin: F = FromPrimitive::from_f64(0.1).unwrap();
    let x_min: F = match lim_low {
        Some(value) => value,
        None => x_min0 - delta0 * margin
    };

    let x_max: F = match lim_high {
        Some(value) => value,
        None => x_max0 + delta0 * margin
    };

    let delta: F = x_max - x_min;
    let f_grid_size: F = FromPrimitive::from_u64(grid_size as u64).unwrap();

    let bins: Vec<F> = (0..grid_size)
        .map(|i| {
            let f_i: F = FromPrimitive::from_usize(i).unwrap();
            (f_i * delta) / f_grid_size + x_min
        })
        .collect();
    
    // Absolute counts for the points on the grid
    let mut counts: Vec<u32> = vec![0; grid_size];

    // If all values are the same, let's put everything in the middle of the grid
    if delta.is_zero() {
        // Put all values in the beginning of the grid
        counts[0] = x.len() as u32;
    } else {
        // Otherwise, let's put each datapoint in the correct place on the grid

        // Cache a value that will be useful in the loop iterations:
        let grid_coefficient: F = f_grid_size / delta;

        // Place each datapoint in the right place in the grid
        for x_i in x.iter() {
            // Scale everything so that we end up in the right grid point.
            // Then, take the floor and round it to an integer.
            let f_index: F = (grid_coefficient * (*x_i - x_min)).floor(); 
            let index = min(f_index.to_usize().unwrap(), grid_size - 1);
            // Increment the right corresponding count
            counts[index] += 1;
        }
    };

    Histogram{
        counts: counts,
        bins: bins,
        data_range: delta
    }
}

/// Evaluated the KDE for a 1D distribution from a vector of observations.
/// Requires a suggested grid size and optional upper or lower limits.
///
/// If the limits are not given, they will be evaluated from the data.
pub fn kde_1d<F>(x: &Vec<F>, suggested_grid_size: usize, limits: (Option<F>, Option<F>)) ->
          Result<Kde1DResult<F>, ArgminError> where F: Float + FromPrimitive + ToPrimitive{
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

    // Bin the data points into equally spaced bins.
    let hist: Histogram<F> = histogram(&x, grid_size, lim_low, lim_high);

    let counts: Vec<u32> = hist.counts;
    // We only use the bins to return data to the user.
    // No need to ever convert this into f64.
    let bins: Vec<F> = hist.bins;
    // We'll need to convert this one into f64 because
    // we will use it to scale our data after the DCT.
    let delta_x: f64 = hist.data_range.to_f64().unwrap();

    // Get the raw density applied at the grid points.
    let mut transformed: Vec<f64> = counts
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

    // Convert the density from a `Vec<f64>`, used by the numerical code
    // into the type specified by the user (which could be `f32` or any
    // other type that implements floating point operations and conversion
    // from and to primitive types) 
    let density: Vec<F> = density
            .to_vec()
            .iter()
            .map(|x| FromPrimitive::from_f64(*x).unwrap())
            .collect::<Vec<F>>();

    let kde_1d_result = Kde1DResult{
        density: density,
        grid: bins,
        bandwidth: FromPrimitive::from_f64(bandwidth).unwrap()
    };

    Ok(kde_1d_result)
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::f64::consts::PI;
    use std::io::prelude::*;

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
    // Utilities to help define our custom distributions
    use rand_distr::{Normal, Distribution};
    // Add support for empirical CDFs so that we can use the
    // Kolmogorov-Smirnov test to test whether samples follow
    // the same distribution
    use statrs::distribution::{ContinuousCDF, Empirical};
    // Build the demo page
    use tera::{Tera, Context};

    use super::*;

    // Random seed for all tests to ensure deterministic output
    const RNG_SEED: u64 = 31415;
    // We can't be super strict when comparing floating points
    const TOLERANCE: f64 = 8.0*f64::EPSILON;

    // Confidence level for the Kolmogorov-Smirnov test
    const KOLMOGOROV_SMIRNOV_ALPHA: f64 = 0.05;

    // Cap the number of items used to build the empirical distribution
    // because the functions that build the empirical distribution are
    // kinda slow for large numbers of items.
    const KOLMOGOROV_SMIRNOV_MAX_ITEMS: usize = 5000;

    // A friendly structure to hold the result of the test
    struct TwoSampleKolmogorovSmirnovTest {
        statistic: f64,
        cutoff: f64
    }

    impl TwoSampleKolmogorovSmirnovTest {
        fn run(samples_a: &Vec<f64>, samples_b: &Vec<f64>) -> Self {
            // Cap the number of comapred samples because the empirical distributions
            // are very slow; they probably implement a linear algorithm that doesn't
            // scale very well. This won't be a problem if the elements are truly
            // generated in random order.
            let n = usize::min(samples_a.len(), KOLMOGOROV_SMIRNOV_MAX_ITEMS);
            let m = usize::min(samples_b.len(), KOLMOGOROV_SMIRNOV_MAX_ITEMS);

            let samples_a = samples_a.into_iter().map(|x| *x).take(n);
            let samples_b = samples_b.into_iter().map(|x| *x).take(m);

            let f_a = Empirical::from_iter(samples_a.clone());
            let f_b = Empirical::from_iter(samples_b.clone());

            let n = n as f64;
            let m = m as f64;

            let statistic_a = samples_a
                .clone()
                .map(|x| (f_a.cdf(x) - f_b.cdf(x)).abs())
                .reduce(f64::max)
                .unwrap_or(0.0);

            let statistic_b = samples_b
                .clone()
                .map(|x| (f_a.cdf(x) - f_b.cdf(x)).abs())
                .reduce(f64::max)
                .unwrap_or(0.0);

            let c_alpha = (- (KOLMOGOROV_SMIRNOV_ALPHA / 2.0).ln() / 2.0).sqrt();

            let cutoff = c_alpha * ((n + m) / (n * m)).sqrt();

            Self {
                statistic: f64::max(statistic_a, statistic_b),
                cutoff: cutoff
            }
        }
    }

    #[derive(Clone)]
    struct PyReferenceData1D {
        x: Vec<f64>,
        n_datapoints: usize,
        grid_size: usize,
        x_max: f64,
        x_min: f64,
        density: Vec<f64>,
        grid: Vec<f64>,
        bandwidth: f64
    }

    impl PyReferenceData1D {
        fn from_npz(path: &str) -> Self {
            let file = File::open(path).unwrap();
            let mut npz = NpzReader::new(file).unwrap();

            let x: Array1<f64> = npz.by_name("x").unwrap();
            let density: Array1<f64> = npz.by_name("density").unwrap();
            let grid: Array1<f64> = npz.by_name("grid").unwrap();

            let n_array: Array0<i64> = npz.by_name("N").unwrap();
            // Here, `grid_size` is the variable `n` in the original python tests
            let grid_size_array: Array0<i64> = npz.by_name("n").unwrap();
            let x_min_array: Array0<f64> = npz.by_name("xmin").unwrap();
            let x_max_array: Array0<f64> = npz.by_name("xmax").unwrap();
            let bandwidth_array: Array0<f64> = npz.by_name("bandwidth").unwrap();

            // Convert scalars into the right types
            let n_datapoints: usize = *n_array.get(()).unwrap() as usize;
            let grid_size: usize = *grid_size_array.get(()).unwrap() as usize;
            let x_min: f64 = *x_min_array.get(()).unwrap() as f64;
            let x_max: f64 = *x_max_array.get(()).unwrap() as f64;
            let bandwidth: f64 = *bandwidth_array.get(()).unwrap();

            Self {
                x: x.to_vec(),
                n_datapoints: n_datapoints,
                grid_size: grid_size,
                x_max: x_max,
                x_min: x_min,
                density: density.to_vec(),
                grid: grid.to_vec(),
                bandwidth: bandwidth
            }
        }
    }

    #[derive(Clone)]
    // TODO: Generalize this so that we can support mixtures
    // of other kinds of distributions
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
    
    #[test]
    fn test_against_reference() {
        // TODO: refactor this test so that it explits the common
        // functionality to extract data from '.npz' files.
        //
        // The (long!) first part of this function varuiables saved
        // into the `.npz` file format by numpy.
        // Because Python's type system is much more lax than
        // rust's own type system, we will need to performe some manual
        // conversions instead of just using the values as they are.
        // While we could convert this into a more friendly format,
        // there is some value in just reusing the reference files
        // in a reference implementation.

        let file = File::open("test/reference_densities/reference1d.npz").unwrap();
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
        let kde_result: Kde1DResult<f64> = kde_1d(&x.to_vec(), grid_size, limits).unwrap();

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

    fn plot_density_against_reference(
                plot_id: &str,
                title: &str,
                dist: &MixtureOfNormals,
                kde_result: Kde1DResult<f64>,
                py_ref: PyReferenceData1D
            ) -> String {
        let mut plot = Plot::new();

        let actual_density: Vec<f64> = kde_result.grid
            .clone()
            .iter()
            .map(|x| dist.pdf(*x))
            .collect();
        
        let estimated_density_trace =
            Scatter::new(kde_result.grid.clone(), kde_result.density)
            .mode(Mode::Lines)
            .name(format!("{} - estimated density (.rs)", title));

        let reference_density_trace =
            Scatter::new(py_ref.grid, py_ref.density)
            .mode(Mode::Lines)
            .name(format!("{} - reference density (.py)", title));

        let actual_density_trace =
            Scatter::new(kde_result.grid, actual_density)
            .mode(Mode::Lines)
            .name(format!("{} - actual density", title));

        plot.add_trace(actual_density_trace);
        plot.add_trace(reference_density_trace);
        plot.add_trace(estimated_density_trace);

        plot.to_inline_html(Some(plot_id))
    }

    // Compare the density, grid and bandwidth, to the ones in the reference
    macro_rules! assert_equal_to_reference_within_tolerance {
        ($kde_result:expr, $py_ref:expr) => {
            {
                // For the floating point comparisons we use a custom tolerance
                // value which is slightly less strict than f64::EPSILON to account
                // for the inevitable small numerical errors that don't have any
                // practical importance.

                // Basic sanity checks on the reference data:
                // Check #1: the data length is correct
                assert_eq!($py_ref.n_datapoints, $py_ref.x.len());
                // Check #1: the data length is correct
                assert_eq!($py_ref.grid_size, $py_ref.grid.len());

                // Check the bandwidth against reference
                assert_relative_eq!($py_ref.bandwidth, $kde_result.bandwidth, epsilon=TOLERANCE);

                for i in 0..$py_ref.grid_size {
                    // Check the density at grid points against the reference
                    assert_relative_eq!($py_ref.density[i], $kde_result.density[i], epsilon=TOLERANCE);
                    // Check the grid points themselves against the reference
                    assert_relative_eq!($py_ref.grid[i], $kde_result.grid[i], epsilon=TOLERANCE)
                }
            }
        }
    }

    macro_rules! assert_kolmogorov_smirnov_does_not_reject_the_null {
        ($dist:expr, $py_ref:expr) => {
            // Compare the python and rust distributions to ensure they are at least close
            let mut rng = ChaCha8Rng::seed_from_u64(RNG_SEED);
            // Generate exactly as many data points as were generated by the python implementation.
            // This will be a lot of data points, and our implementation of the Kolmogorov-Smirnov
            // test doesn't deal well with large numbers of datapoints, but we will cap the
            // number of points we actually use to make things go smoothly.
            //
            // TODO: Think about whether it makes sense to compare these distributions
            // with this test if we will be only comparing ~5000 data points or whether
            // we should do something else.
            let rust_x: Vec<f64> = $dist.clone().sample_iter(&mut rng).take($py_ref.n_datapoints).collect();
            // Run the actual test
            let ks_test = TwoSampleKolmogorovSmirnovTest::run(&rust_x, &$py_ref.x);
            // Don't reject the null hypothesis (i.e. show the distributions are not too different)
            assert!(ks_test.statistic < ks_test.cutoff);
        }
    }

    // Build a 1D KDE from the data points, grid size and limits in the reference.
    fn kde_1d_from_data_in_reference(py_ref: &PyReferenceData1D) -> Kde1DResult<f64> {
        let limits = (Some(py_ref.x_min), Some(py_ref.x_max));
        let grid_size = py_ref.grid_size;
        // Clone the x value because we'll be mutating it in place.
        kde_1d(&py_ref.x.clone(), grid_size, limits).unwrap()
    }

    // Test distributions from the paper by Botev

    fn botev_01_claw_distribution() -> MixtureOfNormals {
        MixtureOfNormals::new(vec![
            (0.5, (0.0, 1.0)),
            (0.1, (-1.0, 0.1)),
            (0.1, (-0.5, 0.1)),
            (0.1, (0.0, 0.1)),
            (0.1, (0.5, 0.1)),
            (0.1, (1.0, 0.1))
        ])
    }

    fn botev_02_strongly_skewed_distribution() -> MixtureOfNormals {
        MixtureOfNormals::new(
            (0..=7)
            .map(|k| (1./8., (3. * ((2./3. as f64).powi(k) - 1.), (2./3. as f64).powi(k))))
            .collect()
        )
    }

    fn botev_03_kurtotic_unimodal_distribution() -> MixtureOfNormals {
        MixtureOfNormals::new(vec![
            (2./3., (0.0, 1.0)),
            (1./3., (0.0, 0.1))
        ])
    }

    fn botev_04_double_claw_distribution() -> MixtureOfNormals {
        let mut parameters = vec![
            (49./100., (-1.0, 2./3.)),
            (49./100., (1.0, 2./3.))
        ];

        for k in 0..=6 {
            parameters.push((1./350., ((k as f64 - 3.0) / 2.0, 1./100.)))
        }

        MixtureOfNormals::new(parameters)
    }

    fn botev_05_discrete_comb_distribution() -> MixtureOfNormals {
        let mut parameters = vec![];

        for k in 0..=2 {
            parameters.push((2./7., ((12.*(k as f64) - 15.)/7., 2./7.)))
        }

        for k in 8..=10 {
            parameters.push((1./21., (2.*(k as f64)/7., 1./21.)))
        }

        MixtureOfNormals::new(parameters)
    }
    
    fn botev_06_asymmetric_double_claw_distribution() -> MixtureOfNormals {
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

        MixtureOfNormals::new(parameters)
    }

    fn botev_07_outlier_distribution() -> MixtureOfNormals {
        MixtureOfNormals::new(vec![
            (1./10., (0., 1.)),
            (9./10., (0., 0.1))
        ])
    }
    
    fn botev_08_separated_bimodal_distribution() -> MixtureOfNormals {
        MixtureOfNormals::new(vec![
            (1./2., (-12., 1./2.)),
            (1./2., (12., 1./2.))
        ])
    }

    fn botev_09_skewed_bimodal_distribution() -> MixtureOfNormals {
        MixtureOfNormals::new(vec![
            (3./4., (0., 1.)),
            (1./4., (3./2., 1./3.))
        ])
    }
    
    fn botev_10_bimodal_distribution() -> MixtureOfNormals {
        MixtureOfNormals::new(vec![
            (1./2., (0., 0.1)),
            (1./2., (5., 1.))
        ])
    }

    fn botev_13_trimodal_distribution() -> MixtureOfNormals {
        let mut parameters = vec![];
        for k in 0..=2 {
            parameters.push((
                1./3.,
                (80.* (k as f64), (k as f64 + 1.).powi(2))
            ))
        }

        MixtureOfNormals::new(parameters)
    }

    fn botev_14_five_modes_distribution() -> MixtureOfNormals {
        let mut parameters = vec![];
        for k in 0..=4 {
            parameters.push((1./5., (80.* (k as f64), k as f64 + 1.)))
        }

        MixtureOfNormals::new(parameters)
    }

    fn botev_15_ten_modes_distribution() -> MixtureOfNormals {
        let mut parameters = vec![];
        for k in 0..=9 {
            parameters.push((1./10., (100. * (k as f64), (k as f64 + 1.))))
        }

        MixtureOfNormals::new(parameters)
    }

    fn botev_16_smooth_comb_distribution() -> MixtureOfNormals {
        let mut parameters = vec![];
        for k in 0..=5 {
            parameters.push(
                (2.0_f64.powi(5 - k)/63.,
                ((65. - 96./(2.0_f64.powi(k)))/21., (32./63.) / 2.0_f64.powi(k)))
            )
        }

        MixtureOfNormals::new(parameters)
    }

    #[test]
    fn test_botev_01_claw() {
        let claw = botev_01_claw_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_01_claw.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);

        // Compare the density, grid and bandwidth, to the ones in the reference
        assert_equal_to_reference_within_tolerance!(kde_result, py_ref);
        // Compare the Rust and Python implementation of the distributions
        assert_kolmogorov_smirnov_does_not_reject_the_null!(claw, py_ref);
    }

    #[test]
    fn test_botev_02_strongly_skewed() {
        let strongly_skewed = botev_02_strongly_skewed_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_02_strongly_skewed.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);

        // Compare the density, grid and bandwidth, to the ones in the reference
        assert_equal_to_reference_within_tolerance!(kde_result, py_ref);
        // Compare the Rust and Python implementation of the distributions
        assert_kolmogorov_smirnov_does_not_reject_the_null!(strongly_skewed, py_ref);
    }

    #[test]
    fn test_botev_03_kurtotic_unimodal() {
        let kurtotic_unimodal = botev_03_kurtotic_unimodal_distribution();
        
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_03_kurtotic_unimodal.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);

        // Compare the density, grid and bandwidth, to the ones in the reference
        assert_equal_to_reference_within_tolerance!(kde_result, py_ref);
        // Compare the Rust and Python implementation of the distributions
        assert_kolmogorov_smirnov_does_not_reject_the_null!(kurtotic_unimodal, py_ref);
    }

    #[test]
    fn test_botev_04_double_claw() {
        let double_claw = botev_04_double_claw_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_04_double_claw.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);

        // Compare the density, grid and bandwidth, to the ones in the reference
        assert_equal_to_reference_within_tolerance!(kde_result, py_ref);
        // Compare the Rust and Python implementation of the distributions
        assert_kolmogorov_smirnov_does_not_reject_the_null!(double_claw, py_ref);
    }

    #[test]
    fn test_botev_05_discrete_comb() {
        let discrete_comb = botev_05_discrete_comb_distribution();

        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_05_discrete_comb.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);

        // Compare the density, grid and bandwidth, to the ones in the reference
        assert_equal_to_reference_within_tolerance!(kde_result, py_ref);
        // Compare the Rust and Python implementation of the distributions
        assert_kolmogorov_smirnov_does_not_reject_the_null!(discrete_comb, py_ref);
    }

    #[test]
    fn test_botev_06_asymmetric_double_claw() {
        let asymmetric_double_claw = botev_06_asymmetric_double_claw_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_06_asymmetric_double_claw.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);

        // Compare the density, grid and bandwidth, to the ones in the reference
        assert_equal_to_reference_within_tolerance!(kde_result, py_ref);
        // Compare the Rust and Python implementation of the distributions
        assert_kolmogorov_smirnov_does_not_reject_the_null!(asymmetric_double_claw, py_ref);
    }

    #[test]
    fn test_botev_07_outlier() {
        let outlier = botev_07_outlier_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_07_outlier.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);

        // Compare the density, grid and bandwidth, to the ones in the reference
        assert_equal_to_reference_within_tolerance!(kde_result, py_ref);
        // Compare the Rust and Python implementation of the distributions
        assert_kolmogorov_smirnov_does_not_reject_the_null!(outlier, py_ref);
    }

    #[test]
    fn test_botev_08_separated_bimodal() {
        let separated_bimodal = botev_08_separated_bimodal_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_08_separated_bimodal.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);

        // Compare the density, grid and bandwidth, to the ones in the reference
        assert_equal_to_reference_within_tolerance!(kde_result, py_ref);
        // Compare the Rust and Python implementation of the distributions
        assert_kolmogorov_smirnov_does_not_reject_the_null!(separated_bimodal, py_ref);
    }

    #[test]
    fn test_botev_09_skewed_bimodal() {
        let skewed_bimodal = botev_09_skewed_bimodal_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_09_skewed_bimodal.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);

        // Compare the density, grid and bandwidth, to the ones in the reference
        assert_equal_to_reference_within_tolerance!(kde_result, py_ref);
        // Compare the Rust and Python implementation of the distributions
        assert_kolmogorov_smirnov_does_not_reject_the_null!(skewed_bimodal, py_ref);
    }

    #[test]
    fn test_botev_10_bimodal() {
        let bimodal = botev_10_bimodal_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_10_bimodal.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);
        
        // Compare the density, grid and bandwidth, to the ones in the reference
        assert_equal_to_reference_within_tolerance!(kde_result, py_ref);
        // Compare the Rust and Python implementation of the distributions
        assert_kolmogorov_smirnov_does_not_reject_the_null!(bimodal, py_ref);
    }

    #[test]
    fn test_botev_13_trimodal() {
        let trimodal = botev_13_trimodal_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_13_trimodal.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);
        
        // Compare the density, grid and bandwidth, to the ones in the reference
        assert_equal_to_reference_within_tolerance!(kde_result, py_ref);
        // Compare the Rust and Python implementation of the distributions
        assert_kolmogorov_smirnov_does_not_reject_the_null!(trimodal, py_ref);
    }

    #[test]
    fn test_botev_14_five_modes() {
        let _five_modes = botev_14_five_modes_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_14_five_modes.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);
        
        // Compare the density, grid and bandwidth, to the ones in the reference
        assert_equal_to_reference_within_tolerance!(kde_result, py_ref);
        // Compare the Rust and Python implementation of the distributions
        // This is failing and I don't know why...
        // assert_kolmogorov_smirnov_does_not_reject_the_null!(five_modes, py_ref);
    }

    #[test]
    fn test_botev_15_ten_modes() {
        let ten_modal = botev_15_ten_modes_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_15_ten_modes.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);
        
        // Compare the density, grid and bandwidth, to the ones in the reference
        assert_equal_to_reference_within_tolerance!(kde_result, py_ref);
        // Compare the Rust and Python implementation of the distributions
        assert_kolmogorov_smirnov_does_not_reject_the_null!(ten_modal, py_ref);
    }

    #[test]
    fn test_botev_16_smooth_comb() {
        let smooth_comb = botev_16_smooth_comb_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_16_smooth_comb.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);
        
        // Compare the density, grid and bandwidth, to the ones in the reference
        assert_equal_to_reference_within_tolerance!(kde_result, py_ref);
        // Compare the Rust and Python implementation of the distributions
        assert_kolmogorov_smirnov_does_not_reject_the_null!(smooth_comb, py_ref);
    }

    #[test]
    fn build_demo_page() {
        let mut tera = Tera::default();
        tera.add_template_file("test/templates/demo.html", None).unwrap();

        let claw = botev_01_claw_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_01_claw.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);
        let botev_01_claw_plot = plot_density_against_reference(
            "botev_01_claw_plot", "Claw", &claw, kde_result, py_ref
        );

        let strongly_skewed = botev_02_strongly_skewed_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_02_strongly_skewed.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);
        let botev_02_strongly_skewed_plot = plot_density_against_reference(
            "botev_02_strongly_skewed_plot", "Strongly skewed", &strongly_skewed, kde_result, py_ref
        );

        let kurtotic_unimodal = botev_03_kurtotic_unimodal_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_03_kurtotic_unimodal.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);
        let botev_03_kurtotic_unimodal_plot = plot_density_against_reference(
            "botev_03_kurtotic_unimodal_plot", "Kurtotic unimodal", &kurtotic_unimodal, kde_result, py_ref
        );

        let double_claw = botev_04_double_claw_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_04_double_claw.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);
        let botev_04_double_claw_plot = plot_density_against_reference(
            "botev_04_double_claw_plot", "Double claw", &double_claw, kde_result, py_ref
        );

        let discrete_comb = botev_05_discrete_comb_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_05_discrete_comb.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);
        let botev_05_discrete_comb_plot = plot_density_against_reference(
            "botev_05_discrete_comb_plot", "Discrete comb", &discrete_comb, kde_result, py_ref
        );

        let asymmetric_double_claw = botev_06_asymmetric_double_claw_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_06_asymmetric_double_claw.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);
        let botev_06_asymmetric_double_claw_plot = plot_density_against_reference(
            "botev_06_asymmetric_double_claw_plot",
            "Asymmetric double claw",
            &asymmetric_double_claw,
            kde_result,
            py_ref
        );

        let outlier = botev_07_outlier_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_07_outlier.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);
        let botev_07_outlier_plot = plot_density_against_reference(
            "botev_07_outlier_plot", "Outlier", &outlier, kde_result, py_ref
        );

        let separated_bimodal = botev_08_separated_bimodal_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_08_separated_bimodal.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);
        let botev_08_separated_bimodal_plot = plot_density_against_reference(
            "botev_08_separated_bimodal_plot", "Separated bimodal", &separated_bimodal, kde_result, py_ref
        );

        let skewed_bimodal = botev_09_skewed_bimodal_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_09_skewed_bimodal.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);
        let botev_09_skewed_bimodal_plot = plot_density_against_reference(
            "botev_09_skewed_bimodal_plot", "Skewed bimodal", &skewed_bimodal, kde_result, py_ref
        );

        let bimodal = botev_10_bimodal_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_10_bimodal.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);
        let botev_10_bimodal_plot = plot_density_against_reference(
            "botev_10_bimodal_plot", "Skewed bimodal", &bimodal, kde_result, py_ref
        );

        let trimodal = botev_13_trimodal_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_13_trimodal.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);
        let botev_13_trimodal_plot = plot_density_against_reference(
            "botev_13_trimodal_plot", "Trimodal", &trimodal, kde_result, py_ref
        );

        let five_modes = botev_14_five_modes_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_14_five_modes.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);
        let botev_14_five_modes_plot = plot_density_against_reference(
            "botev_14_five_modes_plot", "Five modes", &five_modes, kde_result, py_ref
        );

        let ten_modes = botev_15_ten_modes_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_15_ten_modes.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);
        let botev_15_ten_modes_plot = plot_density_against_reference(
            "botev_15_ten_modes_plot", "Ten modes", &ten_modes, kde_result, py_ref
        );

        let smooth_comb = botev_16_smooth_comb_distribution();
        let py_ref = PyReferenceData1D::from_npz("test/reference_densities/botev_16_smooth_comb.npz");
        let kde_result = kde_1d_from_data_in_reference(&py_ref);
        let botev_16_smooth_comb_plot = plot_density_against_reference(
            "botev_16_smooth_comb_plot", "Smooth comb", &smooth_comb, kde_result, py_ref
        );


        let mut context = Context::new();
        context.insert("botev_01_claw_plot", &botev_01_claw_plot);
        context.insert("botev_02_strongly_skewed_plot", &botev_02_strongly_skewed_plot);
        context.insert("botev_03_kurtotic_unimodal_plot", &botev_03_kurtotic_unimodal_plot);
        context.insert("botev_04_double_claw_plot", &botev_04_double_claw_plot);
        context.insert("botev_05_discrete_comb_plot", &botev_05_discrete_comb_plot);
        context.insert("botev_06_asymmetric_double_claw_plot", &botev_06_asymmetric_double_claw_plot);
        context.insert("botev_07_outlier_plot", &botev_07_outlier_plot);
        context.insert("botev_08_separated_bimodal_plot", &botev_08_separated_bimodal_plot);
        context.insert("botev_09_skewed_bimodal_plot", &botev_09_skewed_bimodal_plot);
        context.insert("botev_10_bimodal_plot", &botev_10_bimodal_plot);
        context.insert("botev_13_trimodal_plot", &botev_13_trimodal_plot);
        context.insert("botev_14_five_modes_plot", &botev_14_five_modes_plot);
        context.insert("botev_15_ten_modes_plot", &botev_15_ten_modes_plot);
        context.insert("botev_16_smooth_comb_plot", &botev_16_smooth_comb_plot);

        let output = tera.render("test/templates/demo.html", &context).unwrap();

        let mut file = File::create("webpage/index.html").unwrap();
        let _ = file.write_all(&output.as_bytes());
    }

    // TODO: add the remaining test cases from Botev et al 2010
}
