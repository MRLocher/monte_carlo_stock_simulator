#functions containing the plotting tools used for visualizing the stock data and the results of the Monte Carlo simulations.
import matplotlib.pyplot as plt
import numpy as np
from src.stats import montecarlo_mean, montecarlo_median, montecarlo_sd, montecarlo_mean_confidence_intervals, montecarlo_prediction_intervals, gaussian_params
import plotly.graph_objects as go

def line_graph(data, title, xlabel, ylabel, show_graph, save_graph):
    """Create a line graph for the given data.

    Args:
        data (pandas Series): data to be plotted
        title (string): title for the graph
        xlabel (string): label for the x-axis
        ylabel (string): label for the y-axis
    """
    plt.plot(data, color='red')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if(save_graph):
        #plt.savefig(f"{title}.pdf") # Save the graph as a pdf file
        plt.savefig(f"{title}.pdf")
    if(show_graph):
        plt.show()


def multi_line_graph(data, title, xlabel, ylabel, show_graph, save_graph):
    """Create a multi-line graph for the given data.

    Args:
        data (pandas DataFrame): data to be plotted, with each column representing a line
        title (string): title for the graph
        xlabel (string): label for the x-axis
        ylabel (string): label for the y-axis
    """
    plt.plot(data, color='blue', alpha=0.3)  # Plot all lines with low opacity
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if(save_graph):
        #plt.savefig(f"{title}.pdf") # Save the graph as a pdf file
        plt.savefig(f"{title}.pdf")
    if(show_graph):
        plt.show()

def histogram(data, bin_width, gaussian_fit, title, xlabel, ylabel, show_graph, save_graph):
    """Create a histogram for the given data.

    Args:
        data (pandas Series): data to be plotted
        bin_width (float): width of each bin in the histogram
        gaussian_fit (bool): whether to overlay a Gaussian fit
        title (string): title for the graph
        xlabel (string): label for the x-axis
        ylabel (string): label for the y-axis
    """

    bins = np.arange(data.min()-bin_width, data.max()+bin_width, bin_width)

    if(gaussian_fit):
        params = gaussian_params(data, bin_width)
        A_fit, mu_fit, sigma_fit = params

          # Gaussian function
        def gaussian(x, A, mu, sigma):
            return A * np.exp(-(x-mu)**2 / (2*sigma**2))


        x = np.linspace(data.min(), data.max(), 500)
        p = gaussian(x, A_fit, mu_fit, sigma_fit)
        plt.plot(x, p, 'r', linewidth=2, label=f'Gaussian Fit\n$y = {A_fit:.2f} e^{{-(x-{mu_fit:.3f})^2/(2*{sigma_fit:.3f}^2)}}$')
        plt.legend()


    plt.hist(data, bins=bins, density=True, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if(save_graph):
        plt.savefig(f"{title}.pdf") # Save the graph as a pdf file
    if(show_graph):
        plt.show()

def boxplot(data, title, ylabel, show_graph, save_graph):
    """Create a boxplot for the given data.

    Args:
        data (pandas Series): data to be plotted
        title (string): title for the graph
        ylabel (string): label for the y-axis
    """
    plt.boxplot(data)
    plt.title(title)
    plt.ylabel(ylabel)
    if(save_graph):
        plt.savefig(f"{title}.pdf") # Save the graph as a pdf file
    if(show_graph):
        plt.show()


import numpy as np
import plotly.graph_objects as go



def plot_results(
    simulations,
    confidence_level=0.95,
    sd_multiplier=1,
    stock_name=None,
    historical_data=None,
    historical_plot_days=75,
    actual_data=None,
    allow_jumps=False,
    show_paths=True,
    show_mean=True,
    show_median=False,
    show_prediction_interval=True,
    show_sd_band=False,
    show_mean_confidence_interval=False,
    show_actual=True,
    max_paths=100,
    iterations=1000,
    show_figure = False
):
        """Create a comprehensive plot of Monte Carlo simulation results, including mean, median, prediction intervals, and historical/actual data. 
    Args:
        simulations (numpy array): simulated stock price paths
        confidence_level (float): confidence level for prediction intervals and mean confidence intervals (e.g., 0.95 for 95% confidence)
        sd_multiplier (float): multiplier for standard deviation in calculating prediction intervals
        stock_name (string): name of the stock being simulated (used for title)
        historical_data (numpy array or pandas Series): historical stock price data to be plotted alongside simulations
        historical_plot_days (int): number of days of historical data to plot (default is 252 trading days, approximately 1 year)
        actual_data (numpy array or pandas Series): actual future stock price data to be plotted alongside simulations
        allow_jumps (bool): whether the Monte Carlo simulation included jump processes (used for model description in the plot)
        show_paths (bool): whether to show individual Monte Carlo paths
        show_mean (bool): whether to show the mean forecast line
        show_median (bool): whether to show the median forecast line
        show_prediction_interval (bool): whether to show the prediction interval band
        show_sd_band (bool): whether to show the standard deviation band around the mean
        show_mean_confidence_interval (bool): whether to show confidence intervals for the mean forecast
        show_actual (bool): whether to show actual future data if provided
        max_paths (int): maximum number of individual Monte Carlo paths to display for clarity
        iterations (int): total number of Monte Carlo iterations that were run (used for statistics textbox in the plot)    
        show_figure (bool): should the figure be automatically shown.    
    """

        # ==================================================
        # Statistics
        # ==================================================

        mean = montecarlo_mean(simulations)

        median = montecarlo_median(simulations)

        sd = montecarlo_sd(simulations)

        prediction_lower, prediction_upper = (
            montecarlo_prediction_intervals(
                simulations,
                confidence_level
            )
        )        
        prediction_lower = np.maximum(prediction_lower, 0) #cant be a negative value 0 is already bankruptcy!

        mean_ci_lower, mean_ci_upper = (
            montecarlo_mean_confidence_intervals(
                simulations,
                confidence_level
            )
        )

        # ==================================================
        # Figure setup
        # ==================================================

        fig = go.Figure()

        # ==================================================
        # X-axis alignment
        # ==================================================

        if historical_data is not None:
            historical_data_array = np.asarray(historical_data.iloc[-int(historical_plot_days):]).flatten()

            historical_days = np.arange(len(historical_data_array))

            simulation_days = np.arange(
                len(historical_data_array)-1,
                len(historical_data_array) -1 + simulations.shape[0]
            )

        else:

            simulation_days = np.arange(simulations.shape[0])

        # ==================================================
        # Monte Carlo paths
        # ==================================================

        for i in range(min(max_paths, simulations.shape[1])):

            fig.add_trace(
                go.Scatter(

                    x=simulation_days,
                    y=simulations[:, i],

                    mode="lines",

                    line=dict(width=1, color="rgba(129,140,248,0.05)"),

                    #opacity=0.075,

                    showlegend=False,

                    hoverinfo="skip",
                    
                    visible=True if show_paths else "legendonly"
                )
            )

        # ==================================================
        # Mean forecast
        # ==================================================

        fig.add_trace(
            go.Scatter(

                x=simulation_days,
                y=mean,

                mode="lines",

                name="Mean Forecast",

                line=dict(width=2, color = "#818CF8"),
                visible=True if show_mean else "legendonly"
            )
        )

        # ==================================================
        # Median forecast
        # ==================================================

        fig.add_trace(
            go.Scatter(

                x=simulation_days,
                y=median,

                mode="lines",

                name="Median Forecast",

                line=dict(
                    width=1.5,
                    dash="dot",
                    color = "#FB7185"
                ),
                
                visible=True if show_median else "legendonly"
            )
        )

        # ==================================================
        # Prediction interval
        # ==================================================

        fig.add_trace(
            go.Scatter(

                x=simulation_days,
                y=prediction_upper,

                mode="lines",

                line=dict(width=0),

                hoverinfo="skip",

                showlegend=False,

                opacity=0
            )
        )

        fig.add_trace(
            go.Scatter(

                x=simulation_days,
                y=prediction_lower,

                mode="lines",

                fill="tonexty",

                fillcolor = "rgba(168,85,247,0.28)",

                name=f"{int(confidence_level * 100)}% Prediction Interval",

                line=dict(width=0),

                #opacity=0.20,

                hoverinfo="skip",
                
                visible=True if show_prediction_interval else "legendonly"
            )
        )

        # ==================================================
        # Standard deviation band
        # ==================================================

        upper_sd = mean + sd *sd_multiplier
        lower_sd = mean - sd *sd_multiplier
        lower_sd = np.maximum(lower_sd, 0)

        fig.add_trace(
            go.Scatter(

                x=simulation_days,
                y=upper_sd,

                mode="lines",

                line=dict(width=0),

                hoverinfo="skip",

                showlegend=False,

                opacity=0
            )
        )

        fig.add_trace(
            go.Scatter(

                x=simulation_days,
                y=lower_sd,

                mode="lines",

                fill="tonexty",

                fillcolor = "rgba(34,211,238,0.16)",

                name=f"±{sd_multiplier} Standard Deviations",

                line=dict(width=0),

                #opacity=0.10,

                hoverinfo="skip",

                visible=True if show_sd_band else "legendonly"
            )
        )

        # ==================================================
        # Mean confidence interval
        # ==================================================          

        fig.add_trace(
            go.Scatter(

                x=simulation_days,
                y=mean_ci_upper,

                mode="lines",

                line=dict(width=0),

                hoverinfo="skip",

                showlegend=False,

                opacity=0
            )
        )

        fig.add_trace(
            go.Scatter(

                x=simulation_days,
                y=mean_ci_lower,

                mode="lines",

                fill="tonexty",

                fillcolor = "rgba(163,230,53,0.12)",

                name="Mean Confidence Interval",

                line=dict(width=0),

                #opacity=0.08,

                hoverinfo="skip",
                
                visible=True if show_mean_confidence_interval else "legendonly"
            )
        )

        # ==================================================
        # Historical data
        # ==================================================

        if historical_data is not None:

            fig.add_trace(
                go.Scatter(

                    x=historical_days,
                    y=historical_data_array,

                    mode="lines",

                    name="Historical Data",

                    line=dict(width=2.5, color="#F472B6")
                )
            )

            # ----------------------------------------------
            # Forecast start divider
            # ----------------------------------------------

            fig.add_vline(

                x=len(historical_data.iloc[-int(historical_plot_days):]) - 1,

                line_dash="dash",

                line_color="rgba(148,163,184,0.5)"

               #opacity=0.6

            )

        # ==================================================
        # Actual future data
        # ==================================================

        if show_actual and actual_data is not None:

            actual_data_array = np.concatenate([
                [historical_data_array[-1]],
                np.asarray(actual_data).flatten()
            ])

            actual_days = np.arange(
                simulation_days[0],
                simulation_days[0] + len(actual_data_array)
            )

            fig.add_trace(
                go.Scatter(

                    x=actual_days,
                    y=actual_data_array,

                    mode="lines",

                    name="Actual Future Data",

                    line=dict(
                        width=2.5,
                        color="#FBBF24"
                    )
                )
            )

        # ==================================================
        # Title
        # ==================================================

        if stock_name is not None:

            title = f"{stock_name} Monte Carlo Forecast"

        else:

            title = "Monte Carlo Forecast"

        # ==================================================
        # Model description
        # ==================================================

        if allow_jumps:

            model_text = (
                "Model: Gaussian Monte Carlo with Jump Process"
            )

        else:

            model_text = (
                "Model: Gaussian Monte Carlo"
            )

        # ==================================================
        # Layout
        # ==================================================

        fig.update_layout(

            title=title,

            xaxis_title="Trading Days",

            yaxis_title="Stock Price",

            template="plotly_dark",

            hovermode="x unified",

            width=1100,
            height=650,

            legend=dict(orientation="h", yanchor="top", y=1.10, xanchor="center", x=0.5 ),

            margin=dict(l=60, r=40, t=120, b=120 )
            
        )

        fig.update_yaxes( range=[0, None] )

        # ==================================================
        # Crosshair hover lines
        # ==================================================

        fig.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="grey",
            spikethickness=1
        )

        fig.update_yaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="grey",
            spikethickness=1
        )

        # ==================================================
        # Statistics textbox
        # ==================================================


        stats_text = (

            f"<b>Forecast Summary</b><br><br>"

            f"Simulations: {iterations}<br>"
            f"Forecast Days: {len(simulations)}<br>"
            f"Confidence Level: {int(confidence_level * 100)}%<br><br>"

            #we comment thes eout as it messed up the html export with plotly :(
            #f"<b>Final Day Statistics</b><br><br>"

            #f"Mean Price: ${mean[-1]:.2f}<br>"
            #f"Median Price: ${median[-1]:.2f}<br>"
            #f"Standard Deviation: ${sd[-1]:.2f}<br><br>"

            #f"Prediction Interval:<br>"
            #f"${prediction_lower[-1]:.2f}"
            #f" → "
            #f"${prediction_upper[-1]:.2f}"
        )

        fig.add_annotation(

            text=stats_text,

            xref="paper",
            yref="paper",

            x=0.02,
            y=0.98,

            xanchor="left",
            yanchor="top",

            align="left",

            showarrow=False,

            bordercolor="rgba(180,180,180,0.25)",
            borderwidth=1,

            borderpad=10,

            bgcolor="rgba(15,20,30,0.85)",

            font=dict(
                size=13,
                color="white"
            )
        )

        # ==================================================
        # Footer model text
        # ==================================================

        fig.add_annotation(

            text=model_text,

            xref="paper",
            yref="paper",

            x=0.5,
            y=-0.2,

            showarrow=False
        )
        if(show_figure):
            fig.show()
        else:
            return fig
