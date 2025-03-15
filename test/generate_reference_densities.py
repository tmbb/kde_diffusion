import os
from tqdm import tqdm
import numpy

from scipy.stats import norm
from kde_diffusion import kde1d



class Mixture:
    """
    Very slow but gets the job done
    """
    
    def __init__(self, components):
        self.weights = [w for (w, d) in components]
        self.distributions = [d for (w, d) in components]
        
    def rvs(self):
        """
        Sample a single random variable
        """
        
        # Although it's a straightforward translation of the rust code,
        # this function is super inefficient (~2-3s per 100_000 inputs).
        # Because we only want to use it as a reference, we don't care
        # about efficiency as much.
        r = numpy.random.uniform(0.0, 1.0)

        cummulative_weight = self.weights[0]
        i = 0
        while r > cummulative_weight:
            cummulative_weight += self.weights[i + 1]
            i += 1

        dist = self.distributions[i]
        return dist.rvs()
    
    def rvs_many(self, n):
        """Sample random variables in bulk into an array"""
        return numpy.array([self.rvs() for _ in range(0, n)])

def reference_for_dist(name, dist, n):
    # Ensure deterministic output
    numpy.random.seed(seed=31415)

    x = dist.rvs_many(n)

    # Pass the grid size and the limits explicitly because
    # we don't want to guarantee compatibility for those parameters.
    # We only want to guarantee compatibility of density calculations
    # for the same grid size and limits.
    xmax0 = x.max()
    xmin0 = x.min()
    delta = xmax0 - xmin0

    # This is the default formula that kde1d uses to evaluate
    # the limits. We merely copy it here.
    xmax = xmax0 + delta * 0.1
    xmin = xmin0 - delta * 0.1

    (density, grid, bandwidth) = kde1d(x, n=1024, limits=(xmin, xmax))

    # Save the data as a `.npz` file which Rust can read.
    directory = os.path.dirname(__file__)
    path = os.path.join(directory, "reference_densities/botev_" + name + ".npz")
    with open(path, 'wb') as file:
        numpy.savez(file,
            allow_pickle=False,
            x=x,
            N=len(x),
            n=len(grid),
            xmax=xmax,
            xmin=xmin,
            density=density,
            grid=grid,
            bandwidth=bandwidth
        )

def reference_for_botev_01_claw():
    claw = Mixture([(0.5, norm(0.0, 1.0))] + [
        (0.1, norm(k/2 - 1, 0.1)) for k in range(0, 4+1)
    ])

    return reference_for_dist(
        "01_claw",
        claw,
        100_000
    )

def reference_for_botev_02_strongly_skewed():
    strongly_skewed = Mixture([
        (1/8, norm(3*((2/3)**k - 1), (2/3)**k)) for k in range(0, 7+1)
    ])

    return reference_for_dist(
        "02_strongly_skewed",
        strongly_skewed,
        100_000
    )

def reference_for_botev_03_kurtotic_unimodal():
    kurtotic_unimodal = Mixture([
        (2/3, norm(0, 1)),
        (1/3, norm(0, 1/10))
    ])

    return reference_for_dist(
        "03_kurtotic_unimodal",
        kurtotic_unimodal,
        100_000
    )

def reference_for_botev_04_double_claw():
    double_claw = Mixture([
        (49/100, norm(-1, 2/3)),
        (49/100, norm(1, 2/3))
    ] + [
        (1/350, norm((k-3)/2, 0.01)) for k in range(0, 6+1)
    ])

    return reference_for_dist(
        "04_double_claw",
        double_claw,
        100_000
    )


def reference_for_botev_05_discrete_comb():
    discrete_comb = Mixture([
        (2/7, norm((12*k - 15)/7, 2/7)) for k in [0, 1, 2]
    ] + [
        (1/21, norm(2*k/7, 1/21)) for k in [8, 9, 10]
    ])

    return reference_for_dist(
        "05_discrete_comb",
        discrete_comb,
        100_000
    )

def reference_for_botev_06_asymmetric_double_claw():
    asymmetric_double_claw = Mixture([
        (46/100, norm(2*k - 1, 2/3)) for k in [0, 1]
    ] + [
        (1/300, norm(-k/2, 1/100)) for k in [1, 2, 3]
    ] + [
        (7/300, norm(k/2, 7/100)) for k in [1, 2, 3]
    ])

    return reference_for_dist(
        "06_asymmetric_double_claw",
        asymmetric_double_claw,
        100_000
    )

def reference_for_botev_07_outlier():
    outlier = Mixture([
        (1/10, norm(0, 1)),
        (9/10, norm(0, 1/10))
    ])

    return reference_for_dist(
        "07_outlier",
        outlier,
        100_000
    )


def reference_for_botev_08_separated_bimodal():
    separated_bimodal = Mixture([
        (1/2, norm(-12, 1/2)),
        (1/2, norm(12, 1/2))
    ])

    return reference_for_dist(
        "08_separated_bimodal",
        separated_bimodal,
        100_000
    )

def reference_for_botev_09_skewed_bimodal():
    skewed_bimodal = Mixture([
        (3/4, norm(0, 1)),
        (1/4, norm(3/2, 1/3))
    ])

    return reference_for_dist(
        "09_skewed_bimodal",
        skewed_bimodal,
        100_000
    )

def reference_for_botev_10_bimodal():
    bimodal = Mixture([
        (1/2, norm(0, 1/10)),
        (1/2, norm(5, 1))
    ])

    return reference_for_dist(
        "10_bimodal",
        bimodal,
        100_000
    )


def main():
    densities = [
        reference_for_botev_01_claw,
        reference_for_botev_02_strongly_skewed,
        reference_for_botev_03_kurtotic_unimodal,
        reference_for_botev_04_double_claw,
        reference_for_botev_05_discrete_comb,
        reference_for_botev_06_asymmetric_double_claw,
        reference_for_botev_07_outlier,
        reference_for_botev_08_separated_bimodal,
        reference_for_botev_09_skewed_bimodal,
        reference_for_botev_10_bimodal
    ]
    
    for density in tqdm(densities):
        density()

main()