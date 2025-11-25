import numpy as np

def calculate_safety_stock(std_dev_demand, avg_lead_time, std_dev_lead_time, avg_demand, service_level_z=1.65):
    """
    Calculates Safety Stock.
    Formula: Z * sqrt((Avg LT * sigma_D^2) + (Avg D^2 * sigma_LT^2))
    If lead time variance is zero, simplifies to Z * sqrt(Avg LT) * sigma_D
    """
    variance_demand = std_dev_demand ** 2
    variance_lead_time = std_dev_lead_time ** 2
    
    term1 = avg_lead_time * variance_demand
    term2 = (avg_demand ** 2) * variance_lead_time
    
    combined_std_dev = np.sqrt(term1 + term2)
    return service_level_z * combined_std_dev

def calculate_reorder_point(avg_daily_demand, avg_lead_time, safety_stock):
    """
    Calculates Reorder Point (ROP).
    ROP = (Avg Daily Demand * Avg Lead Time) + Safety Stock
    """
    return (avg_daily_demand * avg_lead_time) + safety_stock

def calculate_eoq(annual_demand, ordering_cost, holding_cost_per_unit):
    """
    Calculates Economic Order Quantity (EOQ).
    EOQ = sqrt((2 * D * S) / H)
    """
    if holding_cost_per_unit == 0:
        return 0
    return np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)

def calculate_days_of_supply(current_inventory, avg_daily_demand):
    """
    Calculates Days of Supply.
    DoS = Inventory / Avg Daily Demand
    """
    if avg_daily_demand == 0:
        return 999 # Infinite/High
    return current_inventory / avg_daily_demand
