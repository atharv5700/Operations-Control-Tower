"""
Monte Carlo Simulation Engine
------------------------------
Probabilistic what-if scenario analysis for supply chain resilience testing.
Simulates thousands of scenarios to provide confidence intervals.
100% offline - all simulations run locally.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ScenarioParameters:
    """Parameters for what-if simulation"""
    demand_change_percent: float = 0.0  # e.g., +20% = 0.20
    lead_time_change_days: float = 0.0  # e.g., +5 days delay
    supplier_reliability_change: float = 0.0  # -0.1 = 10% worse
    cost_change_percent: float = 0.0  # e.g., +15% cost increase
    num_simulations: int = 10000


class MonteCarloSimulator:
    """
    Monte Carlo simulation for supply chain risk analysis.
    Generates probabilistic forecasts with confidence intervals.
    """
    
    def __init__(self, historical_data: pd.DataFrame, scenario: ScenarioParameters):
        """
        Args:
            historical_data: DataFrame with historical supply chain data
            scenario: Scenario parameters to test
        """
        self.data = historical_data
        self.scenario = scenario
        self.results = None
    
    def simulate_demand(self, base_demand: float, demand_std: float) -> np.ndarray:
        """
        Simulate demand as a stochastic process.
        Uses negative binomial distribution (more realistic for demand than normal).
        """
        # Apply scenario change
        adjusted_mean = base_demand * (1 + self.scenario.demand_change_percent)
        
        # Negative binomial parameters
        # Mean = r * p / (1-p), Var = r * p / (1-p)^2
        # Solving for r and p given mean and variance
        variance = demand_std ** 2
        if variance > adjusted_mean:
            # Overdispersed (common in demand data)
            p = adjusted_mean / variance
            r = adjusted_mean * p / (1 - p)
        else:
            # Use Poisson as fallback
            return np.random.poisson(adjusted_mean, self.scenario.num_simulations)
        
        # Generate samples
        samples = np.random.negative_binomial(n=max(r, 1), p=min(p, 0.99), 
                                              size=self.scenario.num_simulations)
        
        return samples
    
    def simulate_lead_time(self, base_lt: float, lt_std: float) -> np.ndarray:
        """
        Simulate lead times using log-normal distribution (skewed, always positive).
        """
        # Apply scenario change
        adjusted_mean = base_lt + self.scenario.lead_time_change_days
        
        # Log-normal parameters
        if lt_std > 0:
            sigma = np.sqrt(np.log(1 + (lt_std / adjusted_mean) ** 2))
            mu = np.log(adjusted_mean) - 0.5 * sigma ** 2
            samples = np.random.lognormal(mu, sigma, self.scenario.num_simulations)
        else:
            # Deterministic if no variation
            samples = np.full(self.scenario.num_simulations, adjusted_mean)
        
        return np.maximum(samples, 0.5)  # Min 0.5 days
    
    def simulate_supplier_delays(self, reliability: float) -> np.ndarray:
        """
        Simulate supplier delay events (binary: on-time vs delayed).
        
        Returns:
            Array of booleans (True = delayed)
        """
        # Apply scenario change
        adjusted_reliability = max(0, min(1, reliability + self.scenario.supplier_reliability_change))
        delay_probability = 1 - adjusted_reliability
        
        # Bernoulli trials
        delays = np.random.random(self.scenario.num_simulations) < delay_probability
        
        return delays
    
    def simulate_stockouts(
        self,
        current_inventory: float,
        demand_samples: np.ndarray,
        lead_time_samples: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate stockout events.
        
        Returns:
            (stockout_flags, stockout_quantities)
        """
        # Demand during lead time
        demand_during_lt = demand_samples * lead_time_samples
        
        # Stockout if demand > inventory
        stockout_flags = demand_during_lt > current_inventory
        stockout_qty = np.maximum(0, demand_during_lt - current_inventory)
        
        return stockout_flags, stockout_qty
    
    def simulate_costs(
        self,
        stockout_qty: np.ndarray,
        excess_inventory: np.ndarray,
        unit_cost: float,
        stockout_cost_multiplier: float = 2.0
    ) -> Dict[str, np.ndarray]:
        """
        Simulate financial impact.
        
        Returns:
            Dict with holding_cost, stockout_cost, total_cost arrays
        """
        # Apply scenario cost change
        adjusted_unit_cost = unit_cost * (1 + self.scenario.cost_change_percent)
        
        # Holding cost (for excess inventory)
        holding_cost_rate = 0.25  # 25% annual holding cost
        holding_cost_daily = adjusted_unit_cost * holding_cost_rate / 365
        holding_cost = excess_inventory * holding_cost_daily
        
        # Stockout cost (lost sales + expediting)
        stockout_cost_per_unit = adjusted_unit_cost * stockout_cost_multiplier
        stockout_cost = stockout_qty * stockout_cost_per_unit
        
        total_cost = holding_cost + stockout_cost
        
        return {
            'holding_cost': holding_cost,
            'stockout_cost': stockout_cost,
            'total_cost': total_cost
        }
    
    def run_simulation(
        self,
        sku_id: str,
        avg_demand: float,
        std_demand: float,
        avg_lead_time: float,
        std_lead_time: float,
        current_inventory: float,
        unit_cost: float,
        supplier_reliability: float = 0.9
    ) -> Dict[str, any]:
        """
        Run complete Monte Carlo simulation for a single SKU.
        
        Returns:
            Dict with simulation results and statistics
        """
        # Generate samples
        demand_samples = self.simulate_demand(avg_demand, std_demand)
        lead_time_samples = self.simulate_lead_time(avg_lead_time, std_lead_time)
        supplier_delays = self.simulate_supplier_delays(supplier_reliability)
        
        # Adjust lead times for supplier delays
        # If supplier delays, add extra days
        delay_days = np.random.uniform(3, 10, self.scenario.num_simulations)
        lead_time_samples = np.where(supplier_delays, 
                                     lead_time_samples + delay_days,
                                     lead_time_samples)
        
        # Simulate stockouts
        stockout_flags, stockout_qty = self.simulate_stockouts(
            current_inventory, demand_samples, lead_time_samples
        )
        
        # Excess inventory
        demand_during_lt = demand_samples * lead_time_samples
        excess_inventory = np.maximum(0, current_inventory - demand_during_lt)
        
        # Costs
        costs = self.simulate_costs(stockout_qty, excess_inventory, unit_cost)
        
        # Calculate statistics
        stockout_probability = np.mean(stockout_flags)
        
        results = {
            'sku_id': sku_id,
            'scenario': {
                'demand_change': f"{self.scenario.demand_change_percent*100:+.0f}%",
                'lead_time_change': f"{self.scenario.lead_time_change_days:+.1f} days",
                'cost_change': f"{self.scenario.cost_change_percent*100:+.0f}%"
            },
            'current_state': {
                'inventory': current_inventory,
                'avg_demand': avg_demand,
                'avg_lead_time': avg_lead_time
            },
            'simulation_results': {
                'stockout_probability': round(stockout_probability * 100, 1),
                'expected_stockout_qty': round(np.mean(stockout_qty), 2),
                'max_stockout_qty': round(np.percentile(stockout_qty, 95), 2),  # 95th percentile
                'expected_total_cost': round(np.mean(costs['total_cost']), 2),
                'cost_5th_percentile': round(np.percentile(costs['total_cost'], 5), 2),
                'cost_95th_percentile': round(np.percentile(costs['total_cost'], 95), 2),
                'expected_holding_cost': round(np.mean(costs['holding_cost']), 2),
                'expected_stockout_cost': round(np.mean(costs['stockout_cost']), 2)
            },
            'risk_level': self._classify_risk(stockout_probability)
        }
        
        # Store raw results for plotting
        self.results = {
            'demand': demand_samples,
            'lead_time': lead_time_samples,
            'stockout_qty': stockout_qty,
            'total_cost': costs['total_cost']
        }
        
        return results
    
    def _classify_risk(self, stockout_prob: float) -> str:
        """Classify risk level based on stockout probability"""
        if stockout_prob < 0.05:
            return "LOW"
        elif stockout_prob < 0.15:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def compare_scenarios(
        self,
        sku_id: str,
        base_params: Dict,
        scenarios: List[ScenarioParameters]
    ) -> pd.DataFrame:
        """
        Compare multiple what-if scenarios side by side.
        
        Returns:
            DataFrame with comparison results
        """
        comparison = []
        
        for i, scenario in enumerate(scenarios):
            self.scenario = scenario
            results = self.run_simulation(sku_id, **base_params)
            
            comparison.append({
                'Scenario': f"Scenario {i+1}",
                'Description': f"Demand {scenario.demand_change_percent*100:+.0f}%, LT {scenario.lead_time_change_days:+.1f}d",
                'Stockout Risk (%)': results['simulation_results']['stockout_probability'],
                'Expected Cost ($)': results['simulation_results']['expected_total_cost'],
                'Cost Range ($)': f"${results['simulation_results']['cost_5th_percentile']:.0f} - ${results['simulation_results']['cost_95th_percentile']:.0f}",
                'Risk Level': results['risk_level']
            })
        
        return pd.DataFrame(comparison)
    
    def get_distribution_data(self) -> Dict[str, np.ndarray]:
        """
        Get raw simulation data for plotting distributions.
        
        Returns:
            Dict with arrays for histograms/violin plots
        """
        if self.results is None:
            raise ValueError("No simulation run yet. Call run_simulation() first.")
        
        return self.results


def quick_scenario_test(
    df: pd.DataFrame,
    sku_id: str,
    demand_spike_percent: float = 20,
    supplier_delay_days: float = 5,
    num_simulations: int = 10000
) -> Dict[str, any]:
    """
    Quick scenario test for a specific SKU.
    
    Args:
        df: Historical data
        sku_id: SKU to analyze
        demand_spike_percent: % increase in demand (e.g., 20 for +20%)
        supplier_delay_days: Additional delay days (e.g., 5)
        num_simulations: Number of Monte Carlo runs
    
    Returns:
        Simulation results dict
    """
    # Filter data for this SKU
    sku_data = df[df['SKU_ID'] == sku_id]
    
    if sku_data.empty:
        return {'error': f'SKU {sku_id} not found'}
    
    # Calculate parameters
    avg_demand = sku_data['Units_Sold'].mean()
    std_demand = sku_data['Units_Sold'].std()
    avg_lead_time = sku_data['Supplier_Lead_Time_Days'].mean()
    std_lead_time = sku_data['Supplier_Lead_Time_Days'].std()
    current_inventory = sku_data['Inventory_Level'].iloc[-1]
    unit_cost = sku_data['Unit_Cost'].iloc[0] if 'Unit_Cost' in sku_data.columns else 10
    
    # Create scenario
    scenario = ScenarioParameters(
        demand_change_percent=demand_spike_percent / 100,
        lead_time_change_days=supplier_delay_days,
        num_simulations=num_simulations
    )
    
    # Run simulation
    simulator = MonteCarloSimulator(sku_data, scenario)
    results = simulator.run_simulation(
        sku_id, avg_demand, std_demand, avg_lead_time, std_lead_time,
        current_inventory, unit_cost
    )
    
    return results
