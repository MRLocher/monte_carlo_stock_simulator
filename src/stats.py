#contains the statistical function used throughout.
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit


def montecarlo_mean(simulations):
    """Calculate the mean stock price across all Monte Carlo simulations for each day.

    Args:
        simulations (numpy array): simulated stock price paths

    Returns:
        numpy array: mean stock price for each day
    """
    return np.mean(simulations, axis=1)

def montecarlo_median(simulations):
    """Calculate the median stock price across all Monte Carlo simulations for each day.

    Args:
        simulations (numpy array): simulated stock price paths

    Returns:
        numpy array: median stock price for each day
    """
    return np.median(simulations, axis=1)


def montecarlo_sd(simulations):
    """Calculate the standard deviation of stock prices across all Monte Carlo simulations for each day.

    Args:
        simulations (numpy array): simulated stock price paths

    Returns:
        numpy array: standard deviation of stock prices for each day
    """
    return np.std(simulations, axis=1)

def montecarlo_mean_confidence_intervals(simulations, confidence_level):
    """Calculate confidence intervals for the simulated stock prices at a given confidence level.

    Args:
        simulations (numpy array): simulated stock price paths
        confidence_level (float): confidence level for the intervals (e.g., 0.95 for 95% confidence)

    Returns:
        tuple: lower and upper bounds of the confidence intervals
    """
    mean = montecarlo_mean(simulations)
    sd = montecarlo_sd(simulations)
    z_score = norm.ppf((1 + confidence_level) / 2)
    margin_error = z_score * sd / np.sqrt(simulations.shape[1])
    return mean - margin_error, mean + margin_error

def montecarlo_prediction_intervals(simulations, confidence_level):
    
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100

    lower = np.percentile(simulations, lower_percentile, axis=1)
    upper = np.percentile(simulations, upper_percentile, axis=1)

    return lower, upper

def montecarlo_percentiles(simulations, percentiles):
    """Calculate specified percentiles for the simulated stock prices.
        Args:
        simulations (numpy array): simulated stock price paths
        percentiles (list): list of percentiles to calculate (e.g., [25, 50, 75])

    Returns:
        numpy array: calculated percentiles for each day
    """
    return np.percentile(simulations, percentiles, axis=1)

def gaussian_params(data, bin_width):
    """Calculate the parameters of a Gaussian distribution fitted to the given data.

    Args:
        data (pandas Series): data to be fitted
        bin_width (float): width of each bin in the histogram

    Returns:
        A_fit (float): amplitude of the Gaussian fit
        mu_fit (float): mean of the Gaussian fit
        sigma_fit (float): standard deviation of the Gaussian fit
    """
    
    bins = np.arange(data.min()-bin_width, data.max()+bin_width, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    counts, _ = np.histogram(data, bins=bins, density=True)

    # Gaussian function
    def gaussian(x, A, mu, sigma):
        return A * np.exp(-(x-mu)**2 / (2*sigma**2))

    # Fit the histogram counts
    params, _ = curve_fit(gaussian, bin_centers, counts, p0=[1, data.mean(), data.std()])

    A_fit, mu_fit, sigma_fit = params

    return A_fit, mu_fit, sigma_fit