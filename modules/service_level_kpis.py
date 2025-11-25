import pandas as pd

def calculate_otif(df):
    """
    Approximation of OTIF (On-Time In-Full).
    Since we don't have explicit order delivery dates, we use Stockout_Flag as a proxy for failure.
    OTIF % = (1 - Stockout Rate) * 100
    """
    stockout_rate = df['Stockout_Flag'].mean()
    return (1 - stockout_rate) * 100

def calculate_fill_rate(df):
    """
    Fill Rate: (Total Demand - Lost Sales) / Total Demand
    Assuming Stockout_Flag=1 implies some lost sales. 
    Without explicit 'Lost Sales' column, we approximate:
    If Stockout=1, we assume we missed the 'Units_Sold' amount (or a fraction of it).
    For this dataset, let's use the inverse of stockout frequency as a high-level proxy, 
    or if we assume Units_Sold is actual sales, Fill Rate is 100% of captured sales, 
    but we want to measure potential.
    
    Better proxy: 
    Fill Rate = Total Units Sold / (Total Units Sold + Estimated Lost Demand)
    Estimated Lost Demand = Avg Demand when Stockout=1
    """
    total_sales = df['Units_Sold'].sum()
    
    # Estimate lost demand: Avg sales of non-stockout days * number of stockout days
    avg_daily_demand = df[df['Stockout_Flag'] == 0]['Units_Sold'].mean()
    stockout_days = df['Stockout_Flag'].sum()
    estimated_lost_sales = avg_daily_demand * stockout_days
    
    if total_sales + estimated_lost_sales == 0:
        return 0
        
    return (total_sales / (total_sales + estimated_lost_sales)) * 100

def calculate_inventory_turnover(df):
    """
    Inventory Turnover = Cost of Goods Sold / Avg Inventory Value
    COGS = Units_Sold * Unit_Cost
    Avg Inv Value = Inventory_Level * Unit_Cost
    """
    df['COGS'] = df['Units_Sold'] * df['Unit_Cost']
    df['Inv_Value'] = df['Inventory_Level'] * df['Unit_Cost']
    
    total_cogs = df['COGS'].sum()
    avg_inv_value = df['Inv_Value'].mean()
    
    if avg_inv_value == 0:
        return 0
        
    # Annualized turnover if data is less than a year? 
    # For now, return the turnover for the period.
    return total_cogs / avg_inv_value

def calculate_perfect_order_index(otif, reliability_score):
    """
    Composite metric.
    POI = OTIF * (Reliability / 100)
    """
    return (otif / 100) * (reliability_score / 100) * 100
