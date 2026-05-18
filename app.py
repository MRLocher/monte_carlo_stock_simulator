import streamlit as st

# =====================================================
# Imports
# =====================================================

from datetime import date

from src.data import (
    download_stock_data,
    clean_stock_data
)

from src.simulation import montecarlo_iteration

from src.visualization import plot_results

from src.evaluation import (
    root_mean_squared_error,
    mean_absolute_percentage_error,
    prediction_interval_coverage,
    jump_comparison
)

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Monte Carlo Stock Forecast",
    layout="wide"
)

# =====================================================
# Session State Storage
# Keeps results between reruns.
# =====================================================

if "fig" not in st.session_state:
    st.session_state.fig = None

if "metrics" not in st.session_state:
    st.session_state.metrics = None

if "jump_stats" not in st.session_state:
    st.session_state.jump_stats = None

# =====================================================
# Sidebar
# =====================================================

with st.sidebar:

    st.title("Simulation Settings")

    # -------------------------------------------------
    # Stock / Dates
    # -------------------------------------------------

    st.subheader("Data")

    stock_list = [

        # =================================================
        # Technology
        # =================================================

        "AAPL",   # Apple
        "MSFT",   # Microsoft
        "NVDA",   # Nvidia
        "META",   # Meta
        "GOOGL",  # Alphabet
        "INTC",   #Intel

        # =================================================
        # AI / High Growth
        # =================================================

        "PLTR",   # Palantir
        "AMD",    # AMD
        "TSLA",   # Tesla
        "SNOW",   # Snowflake
        "CRM",    # Salesforce

        # =================================================
        # Finance
        # =================================================

        "JPM",    # JPMorgan
        "BAC",    # Bank of America
        "GS",     # Goldman Sachs
        "V",      # Visa
        "MA",     # Mastercard

        # =================================================
        # Consumer
        # =================================================

        "AMZN",   # Amazon
        "WMT",    # Walmart
        "COST",   # Costco
        "KO",     # Coca-Cola
        "MCD",    # McDonalds

        # =================================================
        # Healthcare
        # =================================================

        "JNJ",    # Johnson & Johnson
        "PFE",    # Pfizer
        "LLY",    # Eli Lilly
        "UNH",    # UnitedHealth
        "MRK",    # Merck

        # =================================================
        # Energy / Industry
        # =================================================

        "XOM",    # ExxonMobil
        "CVX",    # Chevron
        "CAT",    # Caterpillar
        "BA",     # Boeing
        "GE",     # General Electric

        # =================================================
        # Market Indices / ETFs
        # =================================================

        "SPY",    # S&P 500 ETF
        "QQQ",    # Nasdaq ETF
        "DIA",    # Dow Jones ETF
        "IWM",    # Russell 2000 ETF
    ]



    stock = st.selectbox(
        "Stock",
        stock_list
    )

    training_start_date = st.date_input(
        "Training Start Date",
        value=date(2015, 1, 1)
    )

    evaluation_start_date = st.date_input(
        "Evaluation Start Date",
        value=date(2023, 1, 1)
    )

    evaluation_end_date = date.today()

    # -------------------------------------------------
    # Simulation Controls
    # -------------------------------------------------

    st.subheader("Simulation")

    col1, col2 = st.columns(2)

    with col1:

        montecarlo_days = st.number_input(
            "Forecast Days",
            min_value=1,
            value=126
        )

    with col2:

        montecarlo_iterations = st.number_input(
            "Iterations",
            min_value=100,
            value=100,
            step=10,
            help="""
            Higher iteration counts improve statistical accuracy
            but increase computation time.
            """
        )

    # -------------------------------------------------
    # Jump Controls
    # -------------------------------------------------

    st.subheader("Jump Process")

    col1, col2 = st.columns(2)

    with col1:

        allow_jumps = st.checkbox(
            "Enable Jumps",
            value=True
        )

    with col2:

        sigma_multiplier = st.number_input(
            "Sigma Threshold",
            min_value=1.0,
            max_value=5.0,
            value=2.5,
            step=0.1,
            help="""
            Defines the jump cutoff threshold in standard deviations.
            Larger values classify fewer events as jumps.
            """
        )

    # -------------------------------------------------
    # Evaluation
    # -------------------------------------------------

    st.subheader("Evaluation")

    confidence_level = st.number_input(
        "Confidence Level",
        min_value=0.01,
        max_value=0.99,
        value=0.95,
        step=0.01,
        help="""
        Confidence level for prediction intervals.
        Higher values produce wider intervals.
        """
    )

    # -------------------------------------------------
    # Run Button
    # -------------------------------------------------

    run_simulation = st.button(
        "Run Simulation",
        use_container_width=True
    )

# =====================================================
# Main Title
# =====================================================

st.title("Monte Carlo Stock Forecast")

# =====================================================
# RUN SIMULATION
# Only recomputes when button is pressed
# =====================================================

if run_simulation:

    # -------------------------------------------------
    # Download Data
    # -------------------------------------------------

    training_data = download_stock_data(
        stock,
        training_start_date,
        evaluation_start_date
    )

    evaluation_data = download_stock_data(
        stock,
        evaluation_start_date,
        evaluation_end_date
    )

    # -------------------------------------------------
    # Clean Data
    # -------------------------------------------------

    clean_training_data = clean_stock_data(
        training_data
    )

    clean_evaluation_data = clean_stock_data(
        evaluation_data
    )

    # -------------------------------------------------
    # Run Monte Carlo Simulation
    # -------------------------------------------------

    simulations = montecarlo_iteration(
        data=clean_training_data,
        bin_width=0.001,
        sigma_multiplier=sigma_multiplier,
        days=montecarlo_days,
        iterations=montecarlo_iterations,
        allow_jumps=allow_jumps
    )

    # -------------------------------------------------
    # Store Plotly Figure
    # -------------------------------------------------

    st.session_state.fig = plot_results(

        simulations,

        confidence_level=confidence_level,

        stock_name=stock,

        historical_data=clean_training_data["Close"],

        actual_data=clean_evaluation_data["Close"].iloc[
            0:min(montecarlo_days, len(clean_evaluation_data))
        ],

        allow_jumps=allow_jumps
    )

    # -------------------------------------------------
    # Evaluation Metrics
    # -------------------------------------------------

    rmse = root_mean_squared_error(
        simulations,
        clean_evaluation_data["Close"]
    )

    mape = mean_absolute_percentage_error(
        simulations,
        clean_evaluation_data["Close"]
    )

    coverage = prediction_interval_coverage(
        simulations,
        clean_evaluation_data["Close"],
        confidence_level
    )

    # -------------------------------------------------
    # Store Metrics
    # -------------------------------------------------

    st.session_state.metrics = {

        "rmse": rmse,

        "mape": mape,

        "coverage": coverage
    }

    # -------------------------------------------------
    # Store Jump Statistics
    # -------------------------------------------------

    st.session_state.jump_stats = jump_comparison(
        simulations,
        clean_evaluation_data,
        training_data,
        sigma_multiplier
    )

# =====================================================
# DISPLAY SAVED RESULTS
# =====================================================

if st.session_state.fig is not None:

    # =================================================
    # Forecast Plot
    # =================================================

    st.subheader("Forecast Visualization")

    st.plotly_chart(
        st.session_state.fig,
        use_container_width=True
    )

# =====================================================
# DISPLAY METRICS
# =====================================================

if st.session_state.metrics is not None:

    metrics = st.session_state.metrics

    st.subheader("Forecast Evaluation")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "RMSE",
            f'{metrics["rmse"]:.2f}'
        )

    with col2:

        st.metric(
            "MAPE",
            f'{metrics["mape"]:.2f}%'
        )

    with col3:

        st.metric(
            "Coverage",
            f'{metrics["coverage"]:.1f}%'
        )

# =====================================================
# DISPLAY JUMP STATISTICS
# =====================================================

if st.session_state.jump_stats is not None:

    jump_stats = st.session_state.jump_stats

    with st.expander("Advanced Statistics"):

        st.subheader("Jump Statistics")

        # -------------------------------------------------
        # Real Data
        # -------------------------------------------------

        st.markdown("### Real Data")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Positive Jump Count",
                jump_stats["real_data"][
                    "positive_jump_count"
                ]
            )

            st.metric(
                "Average Positive Jump",
                f'{jump_stats["real_data"]["average_positive_jump"]:.4f}'
            )

        with col2:

            st.metric(
                "Negative Jump Count",
                jump_stats["real_data"][
                    "negative_jump_count"
                ]
            )

            st.metric(
                "Average Negative Jump",
                f'{jump_stats["real_data"]["average_negative_jump"]:.4f}'
            )

        # -------------------------------------------------
        # Simulation Data
        # -------------------------------------------------

        st.markdown("### Simulation Data")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Average Positive Jump Count",
                f'{jump_stats["simulation_data"]["average_positive_jump_count"]:.2f}'
            )

            st.metric(
                "Average Positive Jump",
                f'{jump_stats["simulation_data"]["average_positive_jump"]:.4f}'
            )

        with col2:

            st.metric(
                "Average Negative Jump Count",
                f'{jump_stats["simulation_data"]["average_negative_jump_count"]:.2f}'
            )

            st.metric(
                "Average Negative Jump",
                f'{jump_stats["simulation_data"]["average_negative_jump"]:.4f}'
            )