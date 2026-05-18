#function which will bve the core of the montecarlo simulation. It will take in the historical data, the number of days to simulate, and the number of iterations to perform. It will return a list of simulated price paths.
import numpy as np
from src.data import clean_stock_data
from src.stats import gaussian_params


def montecarlo_iteration(data, bin_width, sigma_multiplier, days, iterations, allow_jumps):
    """Perform Monte Carlo simulations to predict future stock prices based on historical data.

    Args:
        data (pandas Series): historical stock price data
        bin_width (float): width of each bin in the histogram used to calculate Gaussian parameters
        sigma_multiplier (float): multiplier for the standard deviation to determine jump thresholds
        days (int): number of days to simulate into the future
        iterations (int): number of Monte Carlo iterations to perform
        allow_jumps (bool): whether to allow for sudden price jumps in the simulation
    Returns:
        simulations (numpy array): simulated stock price paths
    """

    #adjust the data to ignore outliers that are more than 2sd from the mean, as these can skew the Gaussian fit and are not representative of the typical daily price changes. We will handle these outliers separately if allow_jumps is True.
    
    pct_data = clean_stock_data(data)
    pct_data["pct_change"] = pct_data.iloc[:, 0].pct_change()


    inital_sigma = pct_data["pct_change"].std()
    inital_mu = pct_data["pct_change"].mean()

    jump_mask = np.abs(pct_data["pct_change"] - inital_mu) > sigma_multiplier * inital_sigma

    #take all values but the ones that are jumps.
    normal_data = pct_data["pct_change"][~jump_mask].dropna()

    if len(normal_data) == 0:
        raise ValueError(
            "No normal data remaining after jump filtering."
        )

    #Now using the normal data to calculate the Gaussian parameters for the Monte Carlo simulation, as this will give us a more accurate representation of the typical daily price changes without being skewed by outliers.
    A_monte, mu_monte, sigma_monte = gaussian_params(normal_data, bin_width)
    
    sigma_monte = abs(sigma_monte) #mathematically the same for our fit but will break if not abs

    #adding the sigma correction factor here obtained using our optimization notebook i.e. V3
    optimization_factor = 1.388

    if(allow_jumps):

        # look for any values outside 2 standard deviations and add them to a jump array 
        jump_threshold = sigma_multiplier * sigma_monte *optimization_factor

        jumps = pct_data["pct_change"].loc[np.abs(pct_data["pct_change"])> jump_threshold]

        #get the probability of a jump occurring based on the historical data
        jump_probability = len(jumps) / len(pct_data["pct_change"].dropna())

        #create a jump array for the simulation, where each day has a chance of being a jump day based on bernoulli trials
        jump_occurs = np.random.rand(days, iterations) < jump_probability

        jumps_array = np.zeros((days, iterations))

        # Get all True indices
        indices = np.where(jump_occurs)

        # Randomly pick a jump magnitude for each True
        jumps_array[indices] = np.random.choice(jumps, size=len(indices[0])) #/100 to make it a decimal once again.

        #this will now output a 0 array at every iteration and day unless a jump occurs, in which case it will output a random jump magnitude from the historical data
        #we can then add this jump array to the random shocks in the simulation to allow for sudden price jumps based on historical data.


    simulations = np.zeros((days, iterations))
    simulations[0] = pct_data["Close"].iloc[-1]

#populating the array a day at a time for each iteration
    for t in range(1, days): 

        random_shocks = np.random.normal(mu_monte, sigma_monte, iterations)
        
        #Added changes for V2 to avoid over fitting jumps
        #=====================================================
        jump_cutoff = sigma_multiplier * sigma_monte * optimization_factor
        # redraw any shocks outside cutoff
        while np.any(np.abs(random_shocks - mu_monte) > jump_cutoff):

            mask = np.abs(random_shocks - mu_monte) > jump_cutoff

            random_shocks[mask] = np.random.normal(
                mu_monte,
                sigma_monte,
                np.sum(mask)
            )
        #=====================================================

        if(allow_jumps):
            simulations[t] = simulations[t-1] * (1 + random_shocks + jumps_array[t])
        else:
            simulations[t] = simulations[t-1] * (1 + random_shocks)

    return simulations