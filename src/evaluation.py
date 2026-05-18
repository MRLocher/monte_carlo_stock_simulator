#functions used to evaluate the simulation performance by comparing the simulated data to actual stock price data. This includes calculating various metrics and visualizations to assess the accuracy of the Monte Carlo simulations.
import numpy as np
from src.stats import montecarlo_mean, montecarlo_median, montecarlo_sd, montecarlo_mean_confidence_intervals, montecarlo_prediction_intervals
from src.data import clean_stock_data

def prediction_interval_coverage(simulations, actual_data, confidence_level):
    """Calculate the coverage of the prediction intervals by checking how many actual data points fall within the predicted intervals. Ideally the coverage is the same as the confidence level.

    Args:
        simulations (numpy array): simulated stock price paths
        actual_data (numpy array or pandas Series): actual future stock price data to compare against the prediction intervals
        confidence_level (float): confidence level for the prediction intervals (e.g., 0.95 for 95% confidence)

    Returns:
        float: percentage of actual data points that fall within the prediction intervals
    """
    prediction_lower, prediction_upper = montecarlo_prediction_intervals(simulations, confidence_level)
    
    actual_data_array = np.asarray(actual_data).flatten()

    comparison_length = min( len(actual_data_array), len(prediction_lower) ) #we assume that upper and lower are equal here

    within_interval = np.logical_and(actual_data_array[:comparison_length] >= prediction_lower[:comparison_length], 
                                     actual_data_array[:comparison_length] <= prediction_upper[:comparison_length])

    coverage_percentage = np.mean(within_interval) * 100

    return coverage_percentage

def mean_absolute_percentage_error(simulations, actual_data, use_median=False):
    """Calculate the absolute percentage error of the mean (or median) forecast compared to the actual data.

    Args:
        simulations (numpy array): simulated stock price paths
        actual_data (numpy array or pandas Series): actual future stock price data to compare against the mean forecast
        use_median (bool): whether to use the median forecast instead of the mean

    Returns:
        float: mean absolute percentage error (MAPE) between the mean forecast and actual data
    """
    if use_median:
        mean_forecast = montecarlo_median(simulations)
    else:
        mean_forecast = montecarlo_mean(simulations)

    actual_data_array = np.asarray(actual_data).flatten()

    
    comparison_length = min( len(actual_data_array), len(mean_forecast) )

    mape = np.mean(np.abs((actual_data_array[:comparison_length] - mean_forecast[:comparison_length]) / actual_data_array[:comparison_length])) * 100

    return mape

def root_mean_squared_error(simulations, actual_data, use_median=False):
    """Calculate the root mean squared error of the mean (or median) forecast compared to the actual data.

    Args:
        simulations (numpy array): simulated stock price paths
        actual_data (numpy array or pandas Series): actual future stock price data to compare against the mean forecast
        use_median (bool): whether to use the median forecast instead of the mean

    Returns:
        float: root mean squared error (RMSE) between the mean (or median) forecast and actual data
    """
    if use_median:
        mean_forecast = montecarlo_median(simulations)
    else:
        mean_forecast = montecarlo_mean(simulations)

    actual_data_array = np.asarray(actual_data).flatten()
    
    comparison_length = min( len(actual_data_array), len(mean_forecast) )

    rmse = np.sqrt(np.mean((actual_data_array[:comparison_length] - mean_forecast[:comparison_length]) ** 2))

    return rmse

def rolling_root_mean_squared_error(simulations, actual_data, use_median=False, window_size=20):
    """
    Calculate the rolling root mean squared error (RMSE)
    between the forecast and actual stock prices.

    A sliding window is used to evaluate how forecast
    quality changes over time.

    Args:
        simulations (numpy array):
            Simulated stock price paths

        actual_data (numpy array or pandas Series):
            Actual future stock price data

        use_median (bool):
            Whether to use the median forecast
            instead of the mean forecast

        window_size (int):
            Number of days used in each rolling window

    Returns:
        numpy array:
            Rolling RMSE values
    """

    # -----------------------------------------
    # Select forecast type
    # -----------------------------------------

    if use_median:
        forecast = montecarlo_median(simulations)
    else:
        forecast = montecarlo_mean(simulations)

    # -----------------------------------------
    # Prepare actual data
    # -----------------------------------------

    actual_data_array = np.asarray(actual_data).flatten()

    comparison_length = min(
        len(actual_data_array),
        len(forecast)
    )

    actual_data_array = (
        actual_data_array[:comparison_length]
    )

    forecast = forecast[:comparison_length]

    # -----------------------------------------
    # Calculate squared errors
    # -----------------------------------------

    squared_errors = (
        actual_data_array - forecast
    ) ** 2

    # -----------------------------------------
    # Rolling mean squared error
    # -----------------------------------------

    rolling_mse = np.convolve(
        squared_errors,
        np.ones(window_size) / window_size,
        mode="valid"
    )

    rolling_rmse = np.sqrt(rolling_mse)

    return rolling_rmse


def maximum_absolute_percentage_error(simulations, actual_data, use_median=False):
    """Calculate the maximum absolute percentage error of the mean (or median) forecast compared to the actual data.

    Args:
        simulations (numpy array): simulated stock price paths
        actual_data (numpy array or pandas Series): actual future stock price data to compare against the mean forecast
        use_median (bool): whether to use the median forecast instead of the mean

    Returns:
        float: maximum absolute percentage error between the mean (or median) forecast and actual data
    """
    if use_median:
        mean_forecast = montecarlo_median(simulations)
    else:
        mean_forecast = montecarlo_mean(simulations)

    actual_data_array = np.asarray(actual_data).flatten()

    comparison_length = min(len(actual_data_array), len(mean_forecast))

    mape = np.max(np.abs((actual_data_array[:comparison_length] - mean_forecast[:comparison_length]) / actual_data_array[:comparison_length])) * 100

    return mape

def forecast_dispersion_width(simulations, confidence_level):
    """Calculate the width of the forecast dispersion by measuring the distance between the upper and lower prediction intervals.

    Args:
        simulations (numpy array): simulated stock price paths
        confidence_level (float): the confidence level for the prediction intervals

    Returns:
        numpy array: width of the forecast dispersion for each day
    """
    prediction_lower, prediction_upper = montecarlo_prediction_intervals(simulations, confidence_level)

    dispersion_width = prediction_upper - prediction_lower

    return dispersion_width

def average_forecast_dispersion_growth_rate(simulations, confidence_level):
    """Calculate the average growth rate of the forecast dispersion by measuring how the width of the prediction intervals changes over time.
    Args:
        simulations (numpy array): simulated stock price paths
        confidence_level (float): the confidence level for the prediction intervals
    Returns:
        float: average growth rate of the forecast dispersion   (positive value indicates increasing dispersion, negative value indicates decreasing dispersion)
    """ 

    dispersion_width = forecast_dispersion_width( simulations, confidence_level ) # day-to-day width increases 

    width_changes = np.diff(dispersion_width) # average slope

    average_growth_rate = np.mean(width_changes) 

    return average_growth_rate


def jump_comparison(simulations, actual_data, training_data, jump_threshold_multiplier=2):
    """
    Compare jump behavior between real stock data and Monte Carlo simulations.

    Jumps are defined using the training-data return distribution:
        |return - mu| > threshold_multiplier * sigma

    Args:
        simulations (numpy array):
            Monte Carlo price paths with shape (days, iterations)

        training_data (pandas DataFrame):
            Historical training stock data

        actual_data (pandas DataFrame):
            Future/validation stock data

        jump_threshold_multiplier (float):
            Number of standard deviations used to define a jump

    Returns:
        dict: {
            "real_data": {
                "positive_jump_count": int,
                "negative_jump_count": int,
                "average_positive_jump": float,
                "average_negative_jump": float
            },
            "simulation_data": {
                "average_positive_jump_count": float,
                "average_negative_jump_count": float,
                "average_positive_jump": float,
                "average_negative_jump": float
            }
        }
    """

    # -----------------------------------------
    # Prepare training return data
    # -----------------------------------------

    pct_training_data = clean_stock_data(training_data)
    pct_training_data["pct_change"] = (
        pct_training_data.iloc[:, 0].pct_change() #to get it as a decimal
    )

    training_returns = (
        pct_training_data["pct_change"]
        .dropna()
        .to_numpy()
    )

    mu = np.mean(training_returns)
    sigma = np.std(training_returns)

    jump_threshold = jump_threshold_multiplier * sigma

    # -----------------------------------------
    # Prepare actual return data
    # -----------------------------------------

    pct_actual_data = clean_stock_data(actual_data)
    pct_actual_data["pct_change"] = (
        pct_actual_data.iloc[:, 0].pct_change() #to get it as a decimal
    )

    actual_returns = (
        pct_actual_data["pct_change"]
        .dropna()
        .to_numpy()
    )

    # simulation returns have length days-1
    comparison_length = min(
        len(actual_returns),
        simulations.shape[0] - 1
    )

    actual_returns = actual_returns[:comparison_length]

    # -----------------------------------------
    # Detect real jumps
    # -----------------------------------------

    real_jump_mask = (
        np.abs(actual_returns - mu)
        > jump_threshold
    )

    real_jumps = actual_returns[real_jump_mask]

    real_positive_jumps = real_jumps[real_jumps > 0]
    real_negative_jumps = real_jumps[real_jumps < 0]

    real_positive_count = len(real_positive_jumps)
    real_negative_count = len(real_negative_jumps)

    real_positive_mean = (
        np.mean(real_positive_jumps)
        if real_positive_count > 0
        else 0
    )

    real_negative_mean = (
        np.mean(real_negative_jumps)
        if real_negative_count > 0
        else 0
    )

    # -----------------------------------------
    # Containers for simulation statistics
    # -----------------------------------------

    sim_positive_jump_counts = []
    sim_negative_jump_counts = []

    sim_positive_jump_means = []
    sim_negative_jump_means = []

    # -----------------------------------------
    # Analyze each simulation path
    # -----------------------------------------

    for i in range(simulations.shape[1]):

        sim_prices = simulations[:, i]

        # decimal returns
        sim_returns = (
            np.diff(sim_prices)
            / sim_prices[:-1]
        )

        sim_returns = sim_returns[:comparison_length]

        sim_jump_mask = (
            np.abs(sim_returns - mu)
            > jump_threshold
        )

        sim_jumps = sim_returns[sim_jump_mask]

        sim_positive_jumps = sim_jumps[sim_jumps > 0]
        sim_negative_jumps = sim_jumps[sim_jumps < 0]

        # counts
        sim_positive_jump_counts.append(
            len(sim_positive_jumps)
        )

        sim_negative_jump_counts.append(
            len(sim_negative_jumps)
        )

        # mean jump magnitudes
        if len(sim_positive_jumps) > 0:
            sim_positive_jump_means.append(
                np.mean(sim_positive_jumps)
            )

        if len(sim_negative_jumps) > 0:
            sim_negative_jump_means.append(
                np.mean(sim_negative_jumps)
            )

    # -----------------------------------------
    # Aggregate simulation statistics
    # -----------------------------------------

    average_sim_positive_count = np.mean(
        sim_positive_jump_counts
    )

    average_sim_negative_count = np.mean(
        sim_negative_jump_counts
    )

    average_sim_positive_jump = (
        np.mean(sim_positive_jump_means)
        if len(sim_positive_jump_means) > 0
        else 0
    )

    average_sim_negative_jump = (
        np.mean(sim_negative_jump_means)
        if len(sim_negative_jump_means) > 0
        else 0
    )

    # -----------------------------------------
    # Return results
    # -----------------------------------------

    return {

        "real_data": {

            "positive_jump_count":
                real_positive_count,

            "negative_jump_count":
                real_negative_count,

            "average_positive_jump":
                real_positive_mean,

            "average_negative_jump":
                real_negative_mean
        },

        "simulation_data": {

            "average_positive_jump_count":
                average_sim_positive_count,

            "average_negative_jump_count":
                average_sim_negative_count,

            "average_positive_jump":
                average_sim_positive_jump,

            "average_negative_jump":
                average_sim_negative_jump
        }
    }


#A function to print all of our essential evaluation out nicely for the case study.

def show_evaluation(
    simulations,
    actual_data,
    training_data,
    stock_name,
    jumps_enabled=False,
    confidence_level=0.95
):
    """
    Display a compact evaluation summary for Monte Carlo
    stock price simulations.

    Args:
        simulations (numpy array):
            Monte Carlo simulation paths

        actual_data (pandas DataFrame or array):
            Real future stock price data

        training_data (pandas DataFrame):
            Historical stock data used for training

        stock_name (str):
            Stock ticker/name displayed in the output

        jumps_enabled (bool):
            Whether jump diffusion was enabled

        confidence_level (float):
            Confidence level used for interval coverage

    Returns:
        None
    """

    # --------------------------------------------------
    # Core Metrics
    # --------------------------------------------------

    rmse = root_mean_squared_error(
        simulations,
        actual_data
    )

    mape = mean_absolute_percentage_error(
        simulations,
        actual_data
    )

    coverage = prediction_interval_coverage(
        simulations,
        actual_data,
        confidence_level
    )

    dispersion = (
        average_forecast_dispersion_growth_rate(
            simulations,
            confidence_level
        )
    )

    jumps = jump_comparison(
        simulations,
        actual_data,
        training_data
    )

    # --------------------------------------------------
    # Model Label
    # --------------------------------------------------

    model = (
        "With Jumps"
        if jumps_enabled
        else "Standard Model"
    )

    # --------------------------------------------------
    # Display Output
    # --------------------------------------------------

    print(
        f"\n{'='*72}"
        f"\n{stock_name} | {model}"
        f"\n{'='*72}"
    )

    print(
        f"{'RMSE':<18}{rmse:>10.4f}   "
        f"{'MAPE (%)':<18}{mape:>10.4f}"
    )

    print(
        f"{'Coverage (%)':<18}{coverage:>10.2f}   "
        f"{'Dispersion':<18}{dispersion:>10.6f}"
    )

    print("-"*72)

    print(
        f"{'':<24}"
        f"{'Real':>10}"
        f"{'Simulation':>16}"
    )

    print(
        f"{'Positive Jumps':<24}"
        f"{jumps['real_data']['positive_jump_count']:>10}"
        f"{jumps['simulation_data']['average_positive_jump_count']:>16.2f}"
    )

    print(
        f"{'Negative Jumps':<24}"
        f"{jumps['real_data']['negative_jump_count']:>10}"
        f"{jumps['simulation_data']['average_negative_jump_count']:>16.2f}"
    )

    print(
        f"{'Avg Positive Jump':<24}"
        f"{jumps['real_data']['average_positive_jump']:>10.4f}"
        f"{jumps['simulation_data']['average_positive_jump']:>16.4f}"
    )

    print(
        f"{'Avg Negative Jump':<24}"
        f"{jumps['real_data']['average_negative_jump']:>10.4f}"
        f"{jumps['simulation_data']['average_negative_jump']:>16.4f}"
    )

