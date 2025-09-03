import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

def create_ss_benefits_df():
    """Creates a DataFrame for Social Security benefits."""
    ss_data = {
        'Age of Retirement': [62, 63, 64, 65, 66, 67, 68, 69, 70],
        'Primary SS Monthly Benefit': [2753, 2999, 3245, 3492, 3738, 3984, 4303, 4622, 4941]
    }
    df_ss_benefits = pd.DataFrame(ss_data)
    df_ss_benefits['Spousal Benefit'] = df_ss_benefits['Primary SS Monthly Benefit'] * 0.5
    df_ss_benefits['Total Monthly Benefit (2024 Dollars)'] = df_ss_benefits['Primary SS Monthly Benefit'] + df_ss_benefits['Spousal Benefit']
    return df_ss_benefits

def combine_prepost_retirement_returns(stk_return, bnd_return, retirement_age, current_age_years, stock_weight_pre, stock_weight_post):
    """Combines stock and bond returns based on pre/post retirement weights."""
    random_portfolio_ret = []
    for i, (a1, a2) in enumerate(zip(stk_return, bnd_return)):
        if i < ((retirement_age - current_age_years) * 12):
            combined_value = stock_weight_pre * a1 + (1 - stock_weight_pre) * a2
        else:
            combined_value = stock_weight_post * a1 + (1 - stock_weight_post) * a2
        random_portfolio_ret.append(combined_value)
    return np.array(random_portfolio_ret)

def create_decrease_coefficient(random_portfolio_ret, dynamic_decrease_sensitivity, post_retirement_stock_mix, stock_monthly_ret, bnd_monthly_ret):
    """Generates dynamic decrease coefficients based on portfolio returns."""
    binary_returns = []
    for row in random_portfolio_ret.T: # Transpose to iterate through trials
        binary_row = [1 if value < ((1 + dynamic_decrease_sensitivity) * ((post_retirement_stock_mix * stock_monthly_ret) + ((1 - post_retirement_stock_mix) * bnd_monthly_ret))) else 0 for value in row]
        binary_returns.append(binary_row)
    return np.array(binary_returns).T # Transpose back

def create_increase_coefficient(random_portfolio_ret, dynamic_increase_sensitivity, post_retirement_stock_mix, stock_monthly_ret, bnd_monthly_ret):
    """Generates dynamic increase coefficients based on portfolio returns."""
    binary_returns = []
    for row in random_portfolio_ret.T: # Transpose to iterate through trials
        binary_row = [1 if value > ((1 + dynamic_increase_sensitivity) * ((post_retirement_stock_mix * stock_monthly_ret) + ((1 - post_retirement_stock_mix) * bnd_monthly_ret))) else 0 for value in row]
        binary_returns.append(binary_row)
    return np.array(binary_returns).T # Transpose back


def retirement_monthly_withdrawals(max_investment_horizon, retirement_age, current_age_years, house_paid_age, retirement_tax_rate, mortgage_cost, desired_after_tax_income_pre_mortgage, annual_inflation):
    """Calculates monthly withdrawal needs."""
    monthly_withdrawals = []
    for i in range(int(round(max_investment_horizon))):
        if i < ((retirement_age - current_age_years) * 12):
            withdrawals = 0
        elif i < ((house_paid_age - current_age_years) * 12):
            withdrawals = (desired_after_tax_income_pre_mortgage / (1 - retirement_tax_rate)) * ((1 + (annual_inflation / 12)) ** i)
        else:
            withdrawals = ((desired_after_tax_income_pre_mortgage - mortgage_cost) / (1 - retirement_tax_rate)) * ((1 + (annual_inflation / 12)) ** i)
        monthly_withdrawals.append(withdrawals)
    return np.array(monthly_withdrawals)

def ss_income(max_investment_horizon, current_age_years, beg_ss_withdrawal_age, initial_ss_pymt, annual_inflation):
    """Calculates monthly Social Security income."""
    ss_monthly_income = []
    today = datetime.today()
    date_of_initial_ss_payment = datetime(int(1982 + beg_ss_withdrawal_age), 8, 15)
    months_til_first_ss_check = ((date_of_initial_ss_payment.year - today.year) * 12) + (date_of_initial_ss_payment.month - today.month)

    for i in range(int(round(max_investment_horizon))):
        if i < months_til_first_ss_check:
            ss_income_val = 0
        else:
            ss_income_val = initial_ss_pymt * ((1 + (annual_inflation / 12)) ** (i - months_til_first_ss_check))
        ss_monthly_income.append(ss_income_val)
    return np.array(ss_monthly_income)


def run_simulation(
    current_retirement_balance,
    current_base_salary,
    vip_bonus_rate,
    retirement_contribution_rate,
    company_contribution_rate,
    company_match,
    annual_inflation,
    annual_raise_assumption,
    age_of_first_ss,
    fractional_ss_realized,
    e_inheritance_pv,
    e_age_of_inheritance,
    j_inheritance_pv,
    j_age_of_inheritance,
    stock_monthly_ret,
    stock_monthly_vol,
    bnd_monthly_ret,
    bnd_monthly_vol,
    bnd_stk_correlations,
    pre_retirement_stock_mix,
    post_retirement_stock_mix,
    j_death_age,
    j_death_stdev,
    e_death_age,
    e_death_stdev,
    desired_ret_age,
    trials,
    age_house_paidoff,
    retirement_tax_rate,
    retirment_income_pct,
    current_mortgage,
    current_home_value,
    dynamic_withdrawal_decrease,
    dynamic_withdrawal_increase,
    dynamic_decrease_sensitivity,
    dynamic_increase_sensitivity,
):
    """Runs the Monte Carlo retirement simulation."""

    # Data Setup
    df_ss_benefits = create_ss_benefits_df()
    first_ss_pmt_2024 = df_ss_benefits.loc[df_ss_benefits['Age of Retirement'] == age_of_first_ss, 'Total Monthly Benefit (2024 Dollars)'].values[0]
    initial_ss_pymt = first_ss_pmt_2024 * fractional_ss_realized

    today = datetime.today()
    j_birthday = datetime(1982, 8, 15) # Assuming a fixed birthday for calculation
    current_age = today - j_birthday
    current_age_years = current_age.days / 365.25

    slope_bond_stock_ret = (bnd_monthly_vol / stock_monthly_vol) * bnd_stk_correlations
    y_intercept_bond_ret = bnd_monthly_ret - (slope_bond_stock_ret * stock_monthly_ret)
    stdev_corr_bnd_ret = bnd_monthly_vol * np.sqrt(1 - bnd_stk_correlations**2) # Corrected calculation

    desired_after_tax_monthly_income = retirment_income_pct * (((current_base_salary * (1 + vip_bonus_rate)) / 12) * (1 - retirement_tax_rate))
    home_value_at_death = ((1 + (annual_inflation / 12)) ** ((100 - current_age_years) * 12)) * current_home_value

    # Simulate Random Ages of Death
    random_j_death_age = (np.random.normal(j_death_age, j_death_stdev, trials)).astype(int)
    random_e_death_age = (np.random.normal(e_death_age, e_death_stdev, trials)).astype(int)

    # Generate an array of investment Horizon from 1 to x in months from now until death
    investment_horizon = np.maximum(random_e_death_age, random_j_death_age) * 12 - (12 * current_age_years)
    max_investment_horizon = int(round(np.max(investment_horizon)))

    # Generate Random Stock Returns
    random_stk_returns = np.random.normal(stock_monthly_ret, stock_monthly_vol, (max_investment_horizon, trials))

    # Generate Random Bond Returns
    stage1 = random_stk_returns * slope_bond_stock_ret
    predicted_bnd_return_initial = stage1 + y_intercept_bond_ret
    predicted_error_term = np.random.normal(0, stdev_corr_bnd_ret, (max_investment_horizon, trials))
    random_bnd_returns = predicted_bnd_return_initial + predicted_error_term

    # Generate Random Portfolio Returns across the investment horizon
    random_portfolio_ret = combine_prepost_retirement_returns(random_stk_returns, random_bnd_returns, desired_ret_age, current_age_years, pre_retirement_stock_mix, post_retirement_stock_mix)

    # Generate corresponding dynamic decrease coefficients
    dynamic_decreased_withdrawal_coefficient = create_decrease_coefficient(random_portfolio_ret, dynamic_decrease_sensitivity, post_retirement_stock_mix, stock_monthly_ret, bnd_monthly_ret)

    # Generate corresponding dynamic increase coefficients
    dynamic_increased_withdrawal_coefficient = create_increase_coefficient(random_portfolio_ret, dynamic_increase_sensitivity, post_retirement_stock_mix, stock_monthly_ret, bnd_monthly_ret)

    # Monthly Withdrawal Needs
    monthly_withdrawals_final = retirement_monthly_withdrawals(max_investment_horizon, desired_ret_age, current_age_years, age_house_paidoff, retirement_tax_rate, current_mortgage, desired_after_tax_monthly_income, annual_inflation)

    # Final Monthly Social Security Income
    final_ss_monthly_income = ss_income(max_investment_horizon, current_age_years, age_of_first_ss, initial_ss_pymt, annual_inflation)

    # Run Simulation
    investment_values = np.zeros((max_investment_horizon, trials))
    investment_values[0] = current_retirement_balance
    dynamic_withdrawals_values = np.zeros((max_investment_horizon, trials))

    # periodic contribution (monthly) from salary + employer contributions/match
    periodic_contribution = (
        ((current_base_salary * (1 + vip_bonus_rate))
         * (retirement_contribution_rate + company_contribution_rate + company_match))
        / 12.0
    )


    # Determine safe iteration length T (months)
    T = int(min(
        max_investment_horizon,
        len(random_portfolio_ret),
        len(final_ss_monthly_income),
        len(monthly_withdrawals_final),
        len(dynamic_increased_withdrawal_coefficient),
        len(dynamic_decreased_withdrawal_coefficient)
    ))
    
    # Prepare arrays sized to T
    investment_values = np.zeros((T, trials))
    dynamic_withdrawals_values = np.zeros((T, trials))
    investment_values[0] = current_retirement_balance  # seed initial balance
    
    for t in range(1, T):
        e_inheritance_variable = 1 if (t / 12) >= (e_age_of_inheritance - current_age_years) and (t-1)/12 < (e_age_of_inheritance - current_age_years) else 0
        j_inheritance_variable = 1 if (t / 12) >= (j_age_of_inheritance - current_age_years) and (t-1)/12 < (j_age_of_inheritance - current_age_years) else 0
        contribution_variable = 0 if (t / 12) >= (desired_ret_age - current_age_years) else 1
    
        prev_investment_value = investment_values[t-1]
        contribution = periodic_contribution * contribution_variable * ((1 + (annual_raise_assumption / 12)) ** (t-1))
        e_inheritance_value = e_inheritance_variable * (e_inheritance_pv * ((1 + (annual_inflation / 12)) ** (t-1)))
        j_inheritance_value = j_inheritance_variable * (j_inheritance_pv * ((1 + (annual_inflation / 12)) ** (t-1)))
    
        # use values directly (we already sized arrays to T)
        portfolio_return = random_portfolio_ret[t]
        ss_income_t = final_ss_monthly_income[t]
        withdrawal_amount = monthly_withdrawals_final[t] \
                            * (1 + (dynamic_withdrawal_increase * dynamic_increased_withdrawal_coefficient[t])) \
                            * (1 + (dynamic_withdrawal_decrease * dynamic_decreased_withdrawal_coefficient[t]))
    
        investment_values[t] = (prev_investment_value + contribution + e_inheritance_value + j_inheritance_value) * (1 + portfolio_return) + ss_income_t - withdrawal_amount
        dynamic_withdrawals_values[t] = withdrawal_amount
    
    # convert to DataFrames once loop finishes
    df_investment_values = pd.DataFrame(investment_values)
    df_dynamic_withdrawals = pd.DataFrame(dynamic_withdrawals_values)



    # Dataframe Stats
    ninety_percentile_endoflife = df_investment_values.iloc[-1].quantile(.9)
    average_end_of_life_balance = df_investment_values.iloc[-1].median()
    ten_percentile_endoflife = df_investment_values.iloc[-1].quantile(.1)
    min_end_of_life_balance = df_investment_values.iloc[-1].min()
    max_end_of_life_balance = df_investment_values.iloc[-1].max()
    last_row = df_investment_values.iloc[-1]
    less_than_zero = (last_row < 0).sum()
    less_than_house = (last_row < -home_value_at_death).sum()
    period_retired = int(round((desired_ret_age - current_age_years) * 12, 0))

    percent_simulation_greater_than_0 = ((trials - less_than_zero) / trials)
    percent_simulation_greater_than_0_after_house = ((trials - less_than_house) / trials)

    ninety_percentile_retirement = df_investment_values.iloc[period_retired].quantile(.9)
    median_balance_retirement = df_investment_values.iloc[period_retired].median()
    ten_percentile_retirement = df_investment_values.iloc[period_retired].quantile(.1)
    min_balance_retirement = df_investment_values.iloc[period_retired].min()
    max_balance_retirement = df_investment_values.iloc[period_retired].max()

    # Plots
    # Box Plot
    percentile_95_last_row = df_investment_values.iloc[-1].quantile(0.95)
    percentile_05_last_row = df_investment_values.iloc[-1].quantile(0.05)
    filtered_df = df_investment_values.loc[:, df_investment_values.iloc[-1] < percentile_95_last_row]
    filtered2_df = filtered_df.loc[:, filtered_df.iloc[-1] > percentile_05_last_row]
    df_every_30_rows = filtered2_df.iloc[::30, :]
    df_clipped = df_every_30_rows.T


    fig1 = go.Figure()
    for col in df_clipped.columns:
        fig1.add_trace(go.Box(
            y=df_clipped[col],
            name=str(col),
            boxpoints=False,
            whiskerwidth=0.5
        ))
    fig1.update_layout(
        title='Probabilistic Growth of Account Value Over Time',
        xaxis_title='Time Period',
        yaxis_title='Account Value'
    )



    # Percentile Plot
    # Compute percentile series
    average_balance = df_investment_values.median(axis=1)
    percentile_5 = df_investment_values.quantile(0.05, axis=1)
    percentile_95 = df_investment_values.quantile(0.95, axis=1)
    percentile_25 = df_investment_values.quantile(0.25, axis=1)
    percentile_75 = df_investment_values.quantile(0.75, axis=1)
    percentile_35 = df_investment_values.quantile(0.35, axis=1)
    percentile_65 = df_investment_values.quantile(0.65, axis=1)
    percentile_15 = df_investment_values.quantile(0.15, axis=1)
    percentile_85 = df_investment_values.quantile(0.85, axis=1)
    
    # Build a years axis that aligns with the number of months we actually simulated
    months = len(average_balance)
    start_month = int(np.floor(current_age_years * 12))
    months_range = np.arange(start_month, start_month + months)
    years = months_range / 12.0  # float years (e.g., 42.5)
    
    # Align all series to the same minimum length (defensive)
    min_len = min(
        len(years),
        len(average_balance),
        len(percentile_5),
        len(percentile_95),
        len(percentile_15),
        len(percentile_85),
        len(percentile_25),
        len(percentile_75),
        len(percentile_35),
        len(percentile_65)
    )
    
    # Clip each to min_len
    years = years[:min_len]
    years_clipped = years  # now safely defined and iterable
    average_balance = average_balance.iloc[:min_len]
    percentile_5 = percentile_5.iloc[:min_len]
    percentile_95 = percentile_95.iloc[:min_len]
    percentile_15 = percentile_15.iloc[:min_len]
    percentile_85 = percentile_85.iloc[:min_len]
    percentile_25 = percentile_25.iloc[:min_len]
    percentile_75 = percentile_75.iloc[:min_len]
    percentile_35 = percentile_35.iloc[:min_len]
    percentile_65 = percentile_65.iloc[:min_len]



    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=years, y=average_balance, mode='lines', name='Median Balance', line=dict(color='blue')))
    fig2.add_trace(go.Scatter(x=years, y=percentile_5, mode='lines', name='5th Percentile', line=dict(color='lightgray')))
    fig2.add_trace(go.Scatter(x=years, y=percentile_95, mode='lines', name='95th Percentile', line=dict(color='lightgray')))
    # Add fill between for percentiles
    fig2.add_trace(go.Scatter(
        x=np.concatenate([years, years[::-1]]),
        y=np.concatenate([percentile_95, percentile_5[::-1]]),
        fill='toself',
        fillcolor='whitesmoke',
        line=dict(color='whitesmoke'),
        name='5th-95th Percentile Range',
        showlegend=True
    ))
    fig2.update_layout(title='Monte Carlo Simulation of Investment Account Growth', xaxis_title='Years', yaxis_title='Balance')



    # --- Zoomed-in Percentile Plot with safe wrapper ---
    mask = (years >= 58) & (years <= 85)

    filtered_years = years[mask]
    filtered_average_balance = average_balance[mask]
    filtered_percentile_15 = percentile_15[mask]
    filtered_percentile_85 = percentile_85[mask]
    filtered_percentile_25 = percentile_25[mask]
    filtered_percentile_75 = percentile_75[mask]
    filtered_percentile_35 = percentile_35[mask]
    filtered_percentile_65 = percentile_65[mask]


    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=filtered_years, y=filtered_average_balance, mode='lines', name='Median Balance', line=dict(color='blue')))
    fig3.add_trace(go.Scatter(
        x=np.concatenate([filtered_years, filtered_years[::-1]]),
        y=np.concatenate([filtered_percentile_85, filtered_percentile_15[::-1]]),
        fill='toself',
        fillcolor='lightgray',
        line=dict(color='lightgray'),
        name='15th-85th Percentile Range',
        showlegend=True
    ))
    fig3.update_layout(title='Monte Carlo Simulation of Investment Account Growth (Zoomed In)', xaxis_title='Years', yaxis_title='Balance')



    # Return outputs
    return {
        "end_of_life_stats": {
            "90th Percentile": ninety_percentile_endoflife,
            "Average": average_end_of_life_balance,
            "10th Percentile": ten_percentile_endoflife,
            "Minimum": min_end_of_life_balance,
            "Maximum": max_end_of_life_balance,
            "Simulations < 0": less_than_zero,
            "Simulations < 0 (with house)": less_than_house,
            "Percent > 0": percent_simulation_greater_than_0,
            "Percent > 0 (with house)": percent_simulation_greater_than_0_after_house,
        },
        "retirement_stats": {
            "90th Percentile": ninety_percentile_retirement,
            "Median Balance": median_balance_retirement,
            "10th Percentile": ten_percentile_retirement,
            "Minimum": min_balance_retirement,
            "Maximum": max_balance_retirement,
        },
        "plots": {
            "box_plot": fig1,
            "percentile_plot": fig2,
            "zoomed_percentile_plot": fig3,
        },
        "dataframes":{
            "investment_values_df":df_investment_values,
            "dynamic_withdrawals_df":df_dynamic_withdrawals,
        }

    }
