import pandas as pd
import numpy as np

# Constants for Simulation
MODES = {
    "Air": {"speed_kmh": 500, "base_days": 1, "cost_per_km_unit": 0.5, "co2_per_km_unit": 0.5},
    "Road": {"speed_kmh": 60, "base_days": 0, "cost_per_km_unit": 0.1, "co2_per_km_unit": 0.1},
    "Sea": {"speed_kmh": 30, "base_days": 5, "cost_per_km_unit": 0.01, "co2_per_km_unit": 0.02},
    "Rail": {"speed_kmh": 50, "base_days": 2, "cost_per_km_unit": 0.05, "co2_per_km_unit": 0.05}
}

def calculate_lead_time(distance_km, mode, traffic_factor=1.0):
    """
    Calculate estimated lead time in days.
    """
    if mode not in MODES:
        return np.nan
    
    params = MODES[mode]
    # Travel time in hours / 24 for days
    travel_days = (distance_km / params["speed_kmh"]) / 24
    
    # Total time = Base handling time + Travel time * Traffic/Delay Factor
    total_days = params["base_days"] + (travel_days * traffic_factor)
    return max(1, round(total_days, 1))

def calculate_shipping_cost(units, distance_km, mode):
    """
    Calculate estimated shipping cost.
    """
    if mode not in MODES:
        return 0.0
    
    params = MODES[mode]
    # Simple linear cost model
    cost = units * distance_km * params["cost_per_km_unit"]
    return round(cost, 2)

def compare_transport_modes(sku_id, quantity, distance_km, current_mode="Road"):
    """
    Compare all available modes for a specific shipment.
    Returns a DataFrame with Cost, Time, and Savings.
    """
    results = []
    
    # Baseline (Current Mode)
    base_time = calculate_lead_time(distance_km, current_mode)
    base_cost = calculate_shipping_cost(quantity, distance_km, current_mode)
    
    for mode in MODES.keys():
        time = calculate_lead_time(distance_km, mode)
        cost = calculate_shipping_cost(quantity, distance_km, mode)
        
        time_diff = time - base_time
        cost_diff = cost - base_cost
        
        results.append({
            "Mode": mode,
            "Est. Days": time,
            "Est. Cost ($)": cost,
            "Time Savings (Days)": -time_diff, # Positive means faster
            "Cost Impact ($)": cost_diff # Positive means more expensive
        })
        
    return pd.DataFrame(results).sort_values("Est. Days")
