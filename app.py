import streamlit as st
import matplotlib.pyplot as plt
from structured_simulation import run_simulation
import pandas as pd
from datetime import datetime
import numpy as np

st.title("Retirement Monte Carlo Simulation")

st.header("Input Parameters")

# Organize some inputs in the sidebar
with st.sidebar:
    st.header("Investment & Contribution")
    current_retirement_balance = st.number_input("Current Retirement Balance", value=524562.0, step=1000.0)
    current_base_salary = st.number_input("Current Base Salary", value=238100.0, step=1000.0)
    vip_bonus_rate = st.number_input("VIP Bonus Rate", value=0.3, step=0.01, format="%.2f")
    retirement_contribution_rate = st.number_input("Retirement Contribution Rate", value=0.05, step=0.01, format="%.2f")
    company_match = st.number_input("Company Match", value=0.05, step=0.01, format="%.2f")
    company_contribution_rate = st.number_input("Company Contribution Rate", value=0.05, step=0.01, format="%.2f")

    st.header("Return Assumptions")
    stock_monthly_ret = st.number_input("Stock Monthly Return", value=0.0101, step=0.0001, format="%.4f")
    stock_monthly_vol = st.number_input("Stock Monthly Volatility", value=0.0489, step=0.0001, format="%.4f")
    bnd_monthly_ret = st.number_input("Bond Monthly Return", value=0.0058, step=0.0001, format="%.4f")
    bnd_monthly_vol = st.number_input("Bond Monthly Volatility", value=0.0219, step=0.0001, format="%.4f")
    bnd_stk_correlations = st.number_input("Bond Stock Correlations", value=0.155603146, step=0.001, format="%.6f")
    pre_retirement_stock_mix = st.slider("Pre-Retirement Stock Mix", 0.0, 1.0, 0.9, step=0.01)
    post_retirement_stock_mix = st.slider("Post-Retirement Stock Mix", 0.0, 1.0, 0.6, step=0.01)

    st.header("Mortality & Retirement Age")
    j_death_age = st.number_input("J Death Age", value=93, step=1)
    j_death_stdev = st.number_input("J Death Stdev", value=3.0, step=0.1)
    e_death_age = st.number_input("E Death Age", value=90, step=1)
    e_death_stdev = st.number_input("E Death Stdev", value=3.33, step=0.01)
    desired_ret_age = st.number_input("Desired Retirement Age", value=62, step=1)
    age_of_first_ss = st.number_input("Age of First SS Payment", value=65, step=1)
    fractional_ss_realized = st.number_input("Fractional SS Realized", value=0.75, step=0.01, format="%.2f")


# Main area inputs
st.header("Other Assumptions")
col1, col2 = st.columns(2)
with col1:
    annual_inflation = st.number_input("Annual Inflation", value=0.025, step=0.001, format="%.3f")
    annual_raise_assumption = st.number_input("Annual Raise Assumption", value=0.03, step=0.001, format="%.3f")
    e_inheritance_pv = st.number_input("E Inheritance PV", value=420000.0, step=1000.0)
    e_age_of_inheritance = st.number_input("E Age of Inheritance", value=62, step=1)
    j_inheritance_pv = st.number_input("J Inheritance PV", value=47714.0, step=100.0)
    j_age_of_inheritance = st.number_input("J Age of Inheritance", value=62, step=1)


with col2:
    age_house_paidoff = st.number_input("Age House Paid Off", value=67, step=1)
    retirement_tax_rate = st.number_input("Retirement Tax Rate", value=0.24, step=0.01, format="%.2f")
    retirment_income_pct = st.number_input("Retirement Income Pct", value=0.7, step=0.01, format="%.2f")
    current_mortgage = st.number_input("Current Mortgage", value=3690.0, step=10.0)
    current_home_value = st.number_input("Current Home Value", value=2400000.0, step=10000.0)
    dynamic_withdrawal_decrease = st.number_input("Dynamic Withdrawal Decrease", value=-0.5, step=0.01, format="%.2f")
    dynamic_withdrawal_increase = st.number_input("Dynamic Withdrawal Increase", value=0.2, step=0.01, format="%.2f")
    dynamic_decrease_sensitivity = st.number_input("Dynamic Decrease Sensitivity", value=-0.2, step=0.01, format="%.2f")
    dynamic_increase_sensitivity = st.number_input("Dynamic Increase Sensitivity", value=2.0, step=0.1)

trials = st.slider("Number of Trials", 100, 10000, 5000, step=100)

# Add sliders for zoomed plot age range
st.header("Zoomed Plot Age Range")
zoom_start_age, zoom_end_age = st.slider(
    "Select Age Range for Zoomed Plot",
    40, 100, (58, 85)
)

# Add checkboxes to toggle plot visibility
show_box_plot = st.checkbox("Show Box Plot", value=True)
show_percentile_plot = st.checkbox("Show Full Percentile Plot", value=True)


# Add a calculate button
if st.button("Run Simulation"):
    st.write("Running simulation...")
    # Capture user inputs
    results = run_simulation(
        current_retirement_balance=current_retirement_balance,
        current_base_salary=current_base_salary,
        vip_bonus_rate=vip_bonus_rate,
        retirement_contribution_rate=retirement_contribution_rate,
        company_contribution_rate=company_contribution_rate,
        company_match=company_match,
        annual_inflation=annual_inflation,
        annual_raise_assumption=annual_raise_assumption,
        age_of_first_ss=age_of_first_ss,
        fractional_ss_realized=fractional_ss_realized,
        e_inheritance_pv=e_inheritance_pv,
        e_age_of_inheritance=e_age_of_inheritance,
        j_inheritance_pv=j_inheritance_pv,
        j_age_of_inheritance=j_age_of_inheritance,
        stock_monthly_ret=stock_monthly_ret,
        stock_monthly_vol=stock_monthly_vol,
        bnd_monthly_ret=bnd_monthly_ret,
        bnd_monthly_vol=bnd_monthly_vol,
        bnd_stk_correlations=bnd_stk_correlations,
        pre_retirement_stock_mix=pre_retirement_stock_mix,
        post_retirement_stock_mix=post_retirement_stock_mix,
        j_death_age=j_death_age,
        j_death_stdev=j_death_stdev,
        e_death_age=e_death_age,
        e_death_stdev=e_death_stdev,
        desired_ret_age=desired_ret_age,
        trials=trials,
        age_house_paidoff=age_house_paidoff,
        retirement_tax_rate=retirement_tax_rate,
        retirment_income_pct=retirment_income_pct,
        current_mortgage=current_mortgage,
        current_home_value=current_home_value,
        dynamic_withdrawal_decrease=dynamic_withdrawal_decrease,
        dynamic_withdrawal_increase=dynamic_withdrawal_increase,
        dynamic_decrease_sensitivity=dynamic_decrease_sensitivity,
        dynamic_increase_sensitivity=dynamic_increase_sensitivity,
    )

    st.header("Simulation Results")

    # Display End of Life Stats
    st.subheader("End of Life Statistics")
    end_of_life_stats = results["end_of_life_stats"]
    for key, value in end_of_life_stats.items():
        if isinstance(value, (int, float)):
            if "Percent" in key:
                 st.write(f"{key}: {value:.2%}")
            else:
                st.write(f"{key}: ${value:,.0f}")
        else:
            st.write(f"{key}: {value}")


    # Display Retirement Stats
    st.subheader("Retirement Statistics (at Desired Retirement Age)")
    retirement_stats = results["retirement_stats"]
    for key, value in retirement_stats.items():
        st.write(f"{key}: ${value:,.0f}")

    # Display Plots
    st.subheader("Simulation Plots")
    if show_box_plot:
        st.pyplot(results["plots"]["box_plot"])
    if show_percentile_plot:
        st.pyplot(results["plots"]["percentile_plot"])


    # Regenerate zoomed plot with selected age range
    average_balance = results["dataframes"]["investment_values_df"].median(axis=1)
    percentile_15 = results["dataframes"]["investment_values_df"].quantile(0.15, axis=1)
    percentile_85 = results["dataframes"]["investment_values_df"].quantile(0.85, axis=1)
    percentile_25 = results["dataframes"]["investment_values_df"].quantile(0.25, axis=1)
    percentile_75 = results["dataframes"]["investment_values_df"].quantile(0.75, axis=1)
    percentile_35 = results["dataframes"]["investment_values_df"].quantile(0.35, axis=1)
    percentile_65 = results["dataframes"]["investment_values_df"].quantile(0.65, axis=1)

    today = datetime.today()
    j_birthday = datetime(1982, 8, 15) # Assuming a fixed birthday for calculation
    current_age = today - j_birthday
    current_age_years = current_age.days / 365.25
    months = len(average_balance)
    years = np.arange((current_age_years) * 12, months + ((current_age_years) * 12)) / 12

    filtered_years = years[(years >= zoom_start_age) & (years <= zoom_end_age)]
    filtered_average_balance = average_balance[(years >= zoom_start_age) & (years <= zoom_end_age)]
    filtered_percentile_15 = percentile_15[(years >= zoom_start_age) & (years <= zoom_end_age)]
    filtered_percentile_85 = percentile_85[(years >= zoom_start_age) & (years <= zoom_end_age)]
    filtered_percentile_25 = percentile_25[(years >= zoom_start_age) & (years <= zoom_end_age)]
    filtered_percentile_75 = percentile_75[(years >= zoom_start_age) & (years <= zoom_end_age)]
    filtered_percentile_35 = percentile_35[(years >= zoom_start_age) & (years <= zoom_end_age)]
    filtered_percentile_65 = percentile_65[(years >= zoom_start_age) & (years <= zoom_end_age)]


    fig3, ax5 = plt.subplots(figsize=(12, 6))
    ax5.plot(filtered_years, filtered_average_balance, label='Median Balance', color='blue')
    ax5.fill_between(filtered_years, filtered_percentile_15, filtered_percentile_85, color='lightgray', alpha=0.5, label='15th-85th Percentile Range')
    ax5.fill_between(filtered_years, filtered_percentile_25, filtered_percentile_75, color='gray', alpha=0.5, label='25th-75th Percentile Range')
    ax5.fill_between(filtered_years, filtered_percentile_35, filtered_percentile_65, color='dimgrey', alpha=0.5, label='35th-65th Percentile Range')
    ax5.set_xlabel('Years')
    ax5.set_ylabel('Balance')
    ax5.set_title(f'Monte Carlo Simulation of Investment Account Growth (Zoomed In: Ages {zoom_start_age}-{zoom_end_age})')
    ax5.legend()
    ax5.grid(True)
    st.pyplot(fig3)


    # Add download buttons for dataframes
    st.subheader("Download Simulation Data")
    investment_values_df = results["dataframes"]["investment_values_df"]
    dynamic_withdrawals_df = results["dataframes"]["dynamic_withdrawals_df"]

    st.download_button(
        label="Download Investment Values CSV",
        data=investment_values_df.to_csv().encode('utf-8'),
        file_name='investment_values.csv',
        mime='text/csv',
    )

    st.download_button(
        label="Download Dynamic Withdrawals CSV",
        data=dynamic_withdrawals_df.to_csv().encode('utf-8'),
        file_name='dynamic_withdrawals.csv',
        mime='text/csv',
    )

    st.write("Simulation complete!")
