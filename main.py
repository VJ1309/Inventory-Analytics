import streamlit as st
import pandas as pd
import numpy as np
import random
from scipy.stats import norm
from datetime import datetime
from geneticalgorithm import geneticalgorithm as ga
import plotly.express as px
from pygwalker.api.streamlit import StreamlitRenderer
#from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
from pandasai import SmartDataframe

#load_dotenv()

llm = ChatGroq(
    model_name = "mixtral-8x7b-32768",
    api_key = "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",
    temperature=0.2
)

# Configure the page
st.set_page_config(page_title="Inventory Analytics Data App", layout="wide")

# Main header
st.title("Inventory Analytics Data App")

# Sidebar: File uploader for inventory dataj
st.sidebar.header("Upload Your Inventory Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Process uploaded file
data = None
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

# Create tabs for different dashboard sections
tabs = st.tabs(["Data Exploration", "Chat with your Data", "Simulation"])

# Data Exploration Tab
with tabs[0]:
    st.header("Data Exploration")
    if data is not None:
        pyg_app = StreamlitRenderer(data)
        pyg_app.explorer()
    else:
        st.info("Please upload a CSV file from the sidebar to explore your data.")

# Spoilage Alert Tab
with tabs[1]:
    st.header("Chat with your Data")
    if data is not None:
        with st.expander("🔎 Dataframe Preview"):
            st.write(data.head(5))
        query = st.text_area("🗣️ Chat with Dataframe")
        st.write(query)
        if query:
            df = SmartDataframe(data, config = {"llm":llm})
            result = df.chat(query)
            st.write(result)
    else:
        st.info("Please upload a CSV file from the sidebar to chat with your data.")

# Simulation Tab
with tabs[2]:
    st.header("Simulation")
    st.write("Use this tab to simulate different inventory optimization algorithms.")
    
    # Add a selectbox for choosing the simulation model
    sim_model = st.selectbox("Select Simulation Model", options=["Choose the model from the drop down","Genetic Algorithm", "EOQ", "Monte Carlo Simulation"])

    if sim_model == "Genetic Algorithm":
        st.write("Configure Genetic Algorithm parameters below.")
        simulation_days = st.number_input("For how many days do you want to run the simulation?", min_value=1, value=30, step=1)
        demand_mean = st.number_input("Average Daily Demand", min_value=0.0, value=50.0, step=1.0)
        demand_sd = st.number_input("Standard Deviation of Daily Demand", min_value=0.0, value=10.0, step=1.0)
        lead_time_min = st.number_input("Minimum Lead Time (days)", min_value=1, value=2, step=1)
        lead_time_max = st.number_input("Maximum Lead Time (days)", min_value=1, value=5, step=1)
        asl = st.number_input("Required Service Level (%)", min_value=0.0, max_value=100.0, value=95.0, step=1.0)
        
        if not (0 < asl < 100):
            st.error("Service level must be between 0 and 100 (exclusive).")
            st.stop()

        # Adjust simulation days to allow stabilization as in original model
        days = simulation_days + 2 * lead_time_max
        days2 = days * 2

        # Define the stochastic inventory simulation function
        def stoch_inv_sim(X):
            order_qty = X[0]
            reorder_pt = X[1]
            obj = 0
            for k in range(10):
                tot_demand = 0
                tot_sales = 0
                a = [max(0, np.random.normal(demand_mean, demand_sd)) for _ in range(days)]
                inv = []
                in_qty = [0 for _ in range(days2)]
                for i in range(days):
                    if i == 0:
                        beg_inv = reorder_pt
                        in_inv = 0
                        stock_open = beg_inv + in_inv
                    else:
                        beg_inv = end_inv
                        in_inv = in_qty[i]
                        stock_open = beg_inv + in_inv
                    demand = a[i]
                    lead_time = random.randint(lead_time_min, lead_time_max)
                    if demand < stock_open:
                        end_inv = stock_open - demand
                    else:
                        end_inv = 0
                    inv.append(0.5 * stock_open + 0.5 * end_inv)
                    if i == 0:
                        pipeline_inv = 0
                    else:
                        pipeline_inv = sum(in_qty[j] for j in range(i+1, days2))
                    if pipeline_inv + end_inv <= reorder_pt:
                        if i + lead_time < days:
                            in_qty[i + lead_time] += order_qty
                    if i >= 2 * lead_time_max:
                        tot_sales += (stock_open - end_inv)
                        tot_demand += demand
                cycle_inv = sum(inv) / len(inv)
                achieved_service = tot_sales * 100 / (tot_demand + 1e-6)
                if achieved_service < asl:
                    penalty = 10000000 * demand_mean * (tot_demand - tot_sales)
                    aa = cycle_inv + penalty
                else:
                    aa = cycle_inv
                obj += aa
            return obj / 10

        varbound = np.array([[0, demand_mean * lead_time_max * 5]] * 2)
        algorithm_param = {
            'max_num_iteration': 1000,
            'population_size': 15,
            'mutation_probability': 0.1,
            'elit_ratio': 0.01,
            'crossover_probability': 0.5,
            'parents_portion': 0.3,
            'crossover_type': 'uniform',
            'max_iteration_without_improv': 200
        }

        if st.button("Run Genetic Algorithm Optimization"):
            with st.spinner("Running optimization... This may take a few moments."):
                model = ga(function=stoch_inv_sim, dimension=2, variable_type='real',
                           variable_boundaries=varbound, algorithm_parameters=algorithm_param)
                model.run()
                result = model.output_dict
            st.success("Optimization completed!")
            st.markdown("### Optimal Inventory Parameters")
            st.write("**Optimal Order Quantity:**", result['variable'][0])
            st.write("**Optimal Reorder Point:**", result['variable'][1])
            st.write("**Estimated Cycle Inventory Level (Objective Value):**", result['function'])
    
    elif sim_model == "EOQ":
        st.write("Configure EOQ model parameters below.")
        # Add input fields and simulation code for EOQ model here
        initial_inventory = st.number_input("Initial Inventory Level", min_value=0, value=100, step=1)
        ordering_cost = st.number_input("Ordering Cost", min_value=0.0, value=100.0, step=1.0)
        holding_cost = st.number_input("Holding Cost per Unit", min_value=0.0, value=2.0, step=0.5)
        demand_rate = st.number_input("Annual Demand", min_value=0, value=1000, step=10)
        
        if st.button("Calculate EOQ"):
            # EOQ formula: sqrt((2 * Demand * Ordering Cost) / Holding Cost)
            eoq = np.sqrt((2 * demand_rate * ordering_cost) / holding_cost)
            st.success("EOQ calculation completed!")
            st.markdown("### EOQ Results")
            st.write("**Economic Order Quantity (EOQ):**", round(eoq, 2))

    elif sim_model == "Monte Carlo Simulation":
        st.write("### Monte Carlo Simulation for Inventory Optimization")
        st.markdown("""
This simulation integrates several key parameters:
1. **Demand Variability:** Choose the probability distribution and specify its parameters.
2. **Lead Time Variability:** Modeled using a triangular distribution (min, most likely, max).
3. **Inventory Costs:** Specify holding, ordering, and shortage costs.
4. **Reorder Point (ROP) & Order Quantity (Q):** For Normal demand, these are calculated using safety stock and EOQ formulas.
5. **Service Level Targets:** Used to compute safety stock via the Z-score.
6. **Delay in Order Arrival:** Simulated by tracking pending orders to generate realistic stockouts.
        """)
        
        # --- Simulation Settings ---
        simulation_days_mc = st.number_input("Simulation Days", min_value=1, value=30, step=1)
        num_simulations = st.number_input("Number of Simulation Runs", min_value=100, value=1000, step=100) 
        
        # --- Demand Variability ---
        demand_dist_option = st.selectbox("Select Demand Distribution", options=["Normal", "Poisson", "Lognormal"])
        if demand_dist_option == "Normal":
            demand_mean_mc = st.number_input("Average Daily Demand (Mean)", min_value=0.0, value=50.0, step=1.0)
            demand_sd_mc = st.number_input("Daily Demand Standard Deviation", min_value=0.0, value=10.0, step=1.0)
        elif demand_dist_option == "Poisson":
            demand_lambda = st.number_input("Daily Demand Lambda (mean)", min_value=0.0, value=50.0, step=1.0)
        elif demand_dist_option == "Lognormal":
            demand_mean_mc = st.number_input("Lognormal Demand Mean", min_value=0.0, value=3.5, step=0.1)
            demand_sigma = st.number_input("Lognormal Demand Sigma", min_value=0.0, value=0.4, step=0.1)
        
        # --- Lead Time Variability (Triangular Distribution) ---
        lead_time_min_mc = st.number_input("Minimum Lead Time (days)", min_value=1.0, value=2.0, step=0.5)
        lead_time_mode_mc = st.number_input("Most Likely Lead Time (days)", min_value=1.0, value=3.0, step=0.5)
        lead_time_max_mc = st.number_input("Maximum Lead Time (days)", min_value=1.0, value=5.0, step=0.5)
        
        # Validate lead time inputs
        if not (lead_time_min_mc <= lead_time_mode_mc <= lead_time_max_mc):
            st.error("Lead time values must satisfy: Minimum ≤ Most Likely ≤ Maximum.")
            st.stop()
        
        # Compute the mean lead time from the triangular distribution
        lead_time_mean_mc = (lead_time_min_mc + lead_time_mode_mc + lead_time_max_mc) / 3.0
        
        # --- Inventory Costs ---
        holding_cost = st.number_input("Holding Cost per Unit per Day", min_value=0.0, value=2.0, step=0.1)
        ordering_cost = st.number_input("Ordering Cost per Order", min_value=0.0, value=100.0, step=1.0)
        shortage_cost = st.number_input("Shortage Cost per Unit", min_value=0.0, value=20.0, step=1.0)
        
        # --- Service Level Target ---
        service_level_target = st.number_input("Desired Service Level (%)", min_value=0.0, max_value=100.0, value=95.0, step=1.0)
        
        # --- Safety Stock, ROP, and Order Quantity Calculation (for Normal demand) ---
        if demand_dist_option == "Normal":
            # Compute the Z-score for the desired service level
            z_score = norm.ppf(service_level_target / 100)
            # Calculate safety stock based on lead time demand variability
            safety_stock = z_score * demand_sd_mc * np.sqrt(lead_time_mean_mc)
            # Reorder Point (ROP): Average demand during lead time plus safety stock
            reorder_point = demand_mean_mc * lead_time_mean_mc + safety_stock
            # Calculate EOQ (Economic Order Quantity) based on annual demand approximated as 365 * daily mean
            annual_demand = demand_mean_mc * 365
            order_quantity = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        else:
            st.info("For non-Normal distributions, please manually input ROP and Order Quantity.")
            reorder_point = st.number_input("Reorder Point (ROP)", min_value=0.0, value=100.0, step=1.0)
            order_quantity = st.number_input("Order Quantity (Q)", min_value=0.0, value=100.0, step=1.0)
        
        if demand_dist_option == "Normal":
            st.markdown("#### Calculated Inventory Parameters:")
            st.write("**Lead Time Mean (days):**", round(lead_time_mean_mc, 2))
            st.write("**Safety Stock:**", round(safety_stock, 2))
            st.write("**Reorder Point (ROP):**", round(reorder_point, 2))
            st.write("**Order Quantity (Q) [EOQ]:**", round(order_quantity, 2))
        
        # --- Monte Carlo Simulation Function with Order Delay ---
        def run_monte_carlo_simulation_with_delay(sim_days, num_runs, demand_dist, **kwargs):
            total_costs = []
            service_levels = []
            
            for _ in range(num_runs):
                inventory = reorder_point  # starting inventory
                order_pipeline = []  # track pending orders as (arrival_day, quantity)
                daily_costs = 0
                total_demand = 0
                total_sales = 0
                
                for day in range(sim_days):
                    # Process any orders arriving today
                    arriving_orders = [qty for (arrival_day, qty) in order_pipeline if arrival_day == day]
                    if arriving_orders:
                        inventory += sum(arriving_orders)
                    
                    # Remove orders that have arrived
                    order_pipeline = [(arrival_day, qty) for (arrival_day, qty) in order_pipeline if arrival_day > day]
                    
                    # Generate daily demand based on the selected distribution
                    if demand_dist == "Normal":
                        daily_demand = max(0, np.random.normal(kwargs["demand_mean"], kwargs["demand_sd"]))
                    elif demand_dist == "Poisson":
                        daily_demand = np.random.poisson(kwargs["demand_lambda"])
                    elif demand_dist == "Lognormal":
                        daily_demand = np.random.lognormal(mean=kwargs["demand_mean"], sigma=kwargs["demand_sigma"])
                    else:
                        daily_demand = 0
                    
                    total_demand += daily_demand
                    
                    # Sales are limited by available inventory
                    sales = min(inventory, daily_demand)
                    total_sales += sales
                    shortage = daily_demand - sales
                    
                    # Daily cost components: holding, shortage, and ordering (if applicable)
                    holding_cost_day = holding_cost * inventory
                    shortage_cost_day = shortage_cost * shortage
                    ordering_cost_day = 0
                    
                    # Check ordering policy: if inventory falls below ROP and no order is pending, place an order
                    if inventory < reorder_point and not any(True for order in order_pipeline):
                        # Sample lead time from a triangular distribution
                        lead_time = np.random.triangular(kwargs["lead_time_min"], kwargs["lead_time_mode"], kwargs["lead_time_max"])
                        arrival_day = day + int(round(lead_time))
                        order_pipeline.append((arrival_day, order_quantity))
                        ordering_cost_day = ordering_cost
                    
                    # Update inventory by subtracting sales
                    inventory -= sales
                    daily_costs += holding_cost_day + ordering_cost_day + shortage_cost_day
                
                # Calculate fill rate (service level) for the run
                service_level_run = (total_sales / total_demand) * 100 if total_demand > 0 else 100
                total_costs.append(daily_costs)
                service_levels.append(service_level_run)
            
            avg_cost = np.mean(total_costs)
            avg_service_level = np.mean(service_levels)
            return avg_cost, avg_service_level
        
        # --- Set up keyword arguments based on the selected demand distribution ---
        if demand_dist_option == "Normal":
            kwargs = {
                "demand_mean": demand_mean_mc,
                "demand_sd": demand_sd_mc,
                "lead_time_min": lead_time_min_mc,
                "lead_time_mode": lead_time_mode_mc,
                "lead_time_max": lead_time_max_mc
            }
        elif demand_dist_option == "Poisson":
            kwargs = {
                "demand_lambda": demand_lambda,
                "lead_time_min": lead_time_min_mc,
                "lead_time_mode": lead_time_mode_mc,
                "lead_time_max": lead_time_max_mc
            }
        else:  # Lognormal
            kwargs = {
                "demand_mean": demand_mean_mc,
                "demand_sigma": demand_sigma,
                "lead_time_min": lead_time_min_mc,
                "lead_time_mode": lead_time_mode_mc,
                "lead_time_max": lead_time_max_mc
            }
        
        # --- Run the Monte Carlo Simulation with Delay ---
        if st.button("Run Monte Carlo Simulation with Delay"):
            with st.spinner("Running Monte Carlo simulation..."):
                avg_cost, avg_service = run_monte_carlo_simulation_with_delay(
                    simulation_days_mc, num_simulations, demand_dist_option, **kwargs
                )
            st.success("Monte Carlo Simulation completed!")
            st.markdown("#### Monte Carlo Simulation Results (With Delay)")
            st.write("**Average Total Cost over Simulation Period:**", round(avg_cost, 2))
            st.write("**Average Service Level (%):**", round(avg_service, 2))
