"""
Intelligent Inventory Optimization Module
------------------------------------------
Multi-objective optimization for safety stock, reorder points, and EOQ.
Replaces hardcoded "magic numbers" with proper inventory theory formulas.
100% offline - no external API calls.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, differential_evolution
from typing import Dict, Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')


class InventoryOptimizer:
    """
    Intelligent inventory optimizer that balances service level targets
    with inventory holding costs and order costs.
    """
    
    def __init__(self, service_level: float = 0.95):
        """
        Args:
            service_level: Target service level (0-1), e.g., 0.95 = 95% fill rate
        """
        self.service_level = service_level
        self.z_score = stats.norm.ppf(service_level)  # Z-score for normal distribution
    
    def calculate_safety_stock(
        self,
        avg_demand: float,
        std_demand: float,
        avg_lead_time: float,
        std_lead_time: float = 0.0
    ) -> float:
        """
        Calculate safety stock using proper inventory theory formula.
        
        Formula: SS = Z * sqrt((LT * σ_D²) + (D² * σ_LT²))
        
        Where:
            Z = Z-score for target service level
            LT = Average lead time
            σ_D = Standard deviation of demand
            D = Average demand
            σ_LT = Standard deviation of lead time
        
        This is the industry-standard formula that accounts for both
        demand variability and lead time variability.
        """
        if avg_demand <= 0 or avg_lead_time <= 0:
            return 0.0
        
        # Term 1: Demand variability during lead time
        term1 = avg_lead_time * (std_demand ** 2)
        
        # Term 2: Lead time variability effect on demand
        term2 = (avg_demand ** 2) * (std_lead_time ** 2)
        
        # Combined standard deviation
        combined_std = np.sqrt(term1 + term2)
        
        # Safety stock
        safety_stock = self.z_score * combined_std
        
        return max(safety_stock, 0)
    
    def calculate_reorder_point(
        self,
        avg_daily_demand: float,
        avg_lead_time: float,
        safety_stock: float
    ) -> float:
        """
        Calculate Reorder Point (ROP).
        
        Formula: ROP = (Average Daily Demand * Average Lead Time) + Safety Stock
        
        This is when you should place an order to avoid stockouts during lead time.
        """
        expected_demand_during_lt = avg_daily_demand * avg_lead_time
        rop = expected_demand_during_lt + safety_stock
        
        return max(rop, 0)
    
    def calculate_eoq(
        self,
        annual_demand: float,
        ordering_cost: float,
        holding_cost_per_unit: float
    ) -> float:
        """
        Calculate Economic Order Quantity (EOQ).
        
        Formula: EOQ = sqrt((2 * D * S) / H)
        
        Where:
            D = Annual demand
            S = Fixed ordering cost per order
            H = Holding cost per unit per year
        
        EOQ minimizes total inventory cost (ordering cost + holding cost).
        """
        if annual_demand <= 0 or ordering_cost <= 0 or holding_cost_per_unit <= 0:
            return 0.0
        
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)
        
        return max(eoq, 1)
    
    def calculate_eoq_with_discounts(
        self,
        annual_demand: float,
        ordering_cost: float,
        holding_cost_percent: float,
        price_breaks: List[Tuple[int, float]]
    ) -> Tuple[float, float, float]:
        """
        Calculate EOQ with quantity discounts (realistic supplier pricing).
        
        Args:
            price_breaks: List of (quantity, unit_price) tuples, sorted by quantity
                         e.g., [(0, 10.0), (100, 9.5), (500, 9.0)]
        
        Returns:
            (optimal_order_qty, unit_price, total_annual_cost)
        """
        best_qty = 0
        best_cost = float('inf')
        best_price = 0
        
        for min_qty, unit_price in price_breaks:
            # Holding cost for this price level
            H = holding_cost_percent * unit_price
            
            # Calculate EOQ for this price
            eoq = self.calculate_eoq(annual_demand, ordering_cost, H)
            
            # Adjust EOQ to meet minimum quantity requirement
            order_qty = max(eoq, min_qty)
            
            # Calculate total annual cost
            # TC = Purchase Cost + Ordering Cost + Holding Cost
            purchase_cost = annual_demand * unit_price
            ordering_cost_annual = (annual_demand / order_qty) * ordering_cost
            holding_cost_annual = (order_qty / 2) * H
            
            total_cost = purchase_cost + ordering_cost_annual + holding_cost_annual
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_qty = order_qty
                best_price = unit_price
        
        return best_qty, best_price, best_cost
    
    def newsvendor_model(
        self,
        demand_mean: float,
        demand_std: float,
        unit_cost: float,
        unit_price: float,
        salvage_value: float = 0.0
    ) -> Tuple[float, float]:
        """
        Newsvendor model for perishable/seasonal goods.
        
        Determines optimal order quantity balancing:
        - Cost of overstocking (leftover inventory)
        - Cost of understocking (lost sales)
        
        Returns:
            (optimal_order_qty, expected_profit)
        """
        # Critical ratio
        overstocking_cost = unit_cost - salvage_value
        understocking_cost = unit_price - unit_cost
        critical_ratio = understocking_cost / (overstocking_cost + understocking_cost)
        
        # Find optimal quantity using inverse CDF (assuming normal distribution)
        z = stats.norm.ppf(critical_ratio)
        optimal_qty = demand_mean + z * demand_std
        
        # Calculate expected profit
        # Simplified - assumes normal demand distribution
        expected_sales = demand_mean
        expected_revenue = min(optimal_qty, expected_sales) * unit_price
        ordering_cost = optimal_qty * unit_cost
        expected_salvage = max(0, optimal_qty - expected_sales) * salvage_value
        expected_profit = expected_revenue - ordering_cost + expected_salvage
        
        return max(optimal_qty, 0), expected_profit
    
    def multi_objective_optimization(
        self,
        avg_demand: float,
        std_demand: float,
        avg_lead_time: float,
        std_lead_time: float,
        unit_cost: float,
        holding_cost_percent: float = 0.25,
        stockout_cost_per_unit: float = None
    ) -> Dict[str, float]:
        """
        Multi-objective optimization balancing:
        1. Minimize holding cost
        2. Minimize stockout cost
        3. Achieve target service level
        
        Uses weighted objective function to find optimal safety stock.
        """
        # Default stockout cost = lost margin
        if stockout_cost_per_unit is None:
            stockout_cost_per_unit = unit_cost * 0.5
        
        def objective(ss):
            """Total cost as function of safety stock"""
            safety_stock = ss[0]
            
            # Holding cost (annual)
            annual_holding_cost = safety_stock * unit_cost * holding_cost_percent
            
            # Stockout cost (expected annual stockouts)
            # Simplified model: higher SS → lower stockout probability
            rop = self.calculate_reorder_point(avg_demand, avg_lead_time, safety_stock)
            
            # Probability of stockout during lead time
            demand_during_lt_std = np.sqrt(
                avg_lead_time * (std_demand ** 2) + (avg_demand ** 2) * (std_lead_time ** 2)
            )
            if demand_during_lt_std > 0:
                z_actual = safety_stock / demand_during_lt_std
                stockout_prob = 1 - stats.norm.cdf(z_actual)
            else:
                stockout_prob = 0
            
            # Expected stockouts per year
            orders_per_year = 365 / avg_lead_time if avg_lead_time > 0 else 52
            expected_stockouts = stockout_prob * orders_per_year * avg_demand
            annual_stockout_cost = expected_stockouts * stockout_cost_per_unit
            
            total_cost = annual_holding_cost + annual_stockout_cost
            return total_cost
        
        # Optimize
        result = minimize(
            objective,
            x0=[avg_demand * avg_lead_time * 0.5],  # Initial guess
            bounds=[(0, avg_demand * avg_lead_time * 3)],  # Reasonable bounds
            method='L-BFGS-B'
        )
        
        optimal_ss = result.x[0]
        optimal_rop = self.calculate_reorder_point(avg_demand, avg_lead_time, optimal_ss)
        
        return {
            'safety_stock': round(optimal_ss, 2),
            'reorder_point': round(optimal_rop, 2),
            'annual_cost': round(result.fun, 2)
        }
    
    def calculate_days_of_supply(
        self,
        current_inventory: float,
        avg_daily_demand: float
    ) -> float:
        """Calculate how many days current inventory will last"""
        if avg_daily_demand <= 0:
            return 999  # Infinite
        
        dos = current_inventory / avg_daily_demand
        return max(dos, 0)
    
    def calculate_inventory_turnover(
        self,
        cogs: float,
        avg_inventory_value: float
    ) -> float:
        """
        Calculate inventory turnover ratio.
        
        Higher is better (inventory is moving quickly).
        Industry benchmark varies: 5-10 for retail, 15-25 for grocery.
        """
        if avg_inventory_value <= 0:
            return 0
        
        return cogs / avg_inventory_value


def optimize_inventory_policy(
    demand_history: pd.Series,
    lead_time_history: pd.Series,
    unit_cost: float,
    service_level: float = 0.95,
    holding_cost_percent: float = 0.25,
    ordering_cost: float = 100.0
) -> Dict[str, any]:
    """
    End-to-end inventory policy optimization.
    
    Args:
        demand_history: Historical daily demand
        lead_time_history: Historical lead times
        unit_cost: Cost per unit
        service_level: Target fill rate (0-1)
        holding_cost_percent: Annual holding cost as % of unit cost
        ordering_cost: Fixed cost per order
    
    Returns:
        Dict with optimal safety_stock, ROP, EOQ, and cost analysis
    """
    optimizer = InventoryOptimizer(service_level)
    
    # Calculate demand statistics
    avg_demand = demand_history.mean()
    std_demand = demand_history.std()
    
    # Calculate lead time statistics
    avg_lead_time = lead_time_history.mean()
    std_lead_time = lead_time_history.std()
    
    # Safety stock (proper formula, not magic number!)
    safety_stock = optimizer.calculate_safety_stock(
        avg_demand, std_demand, avg_lead_time, std_lead_time
    )
    
    # Reorder point
    rop = optimizer.calculate_reorder_point(avg_demand, avg_lead_time, safety_stock)
    
    # EOQ
    annual_demand = avg_demand * 365
    holding_cost_per_unit = unit_cost * holding_cost_percent
    eoq = optimizer.calculate_eoq(annual_demand, ordering_cost, holding_cost_per_unit)
    
    # Cost analysis
    annual_ordering_cost = (annual_demand / eoq) * ordering_cost if eoq > 0 else 0
    annual_holding_cost = (eoq / 2 + safety_stock) * holding_cost_per_unit
    total_annual_cost = annual_ordering_cost + annual_holding_cost
    
    return {
        'service_level': service_level,
        'avg_daily_demand': round(avg_demand, 2),
        'std_demand': round(std_demand, 2),
        'avg_lead_time': round(avg_lead_time, 2),
        'safety_stock': round(safety_stock, 2),
        'reorder_point': round(rop, 2),
        'eoq': round(eoq, 2),
        'annual_ordering_cost': round(annual_ordering_cost, 2),
        'annual_holding_cost': round(annual_holding_cost, 2),
        'total_annual_cost': round(total_annual_cost, 2),
        'z_score': round(optimizer.z_score, 2)
    }


def quick_policy_recommendation(
    avg_demand: float,
    std_demand: float,
    avg_lead_time: float,
    std_lead_time: float = 0.0,
    current_inventory: float = 0.0,
    service_level: float = 0.95
) -> Dict[str, any]:
    """
    Quick inventory policy recommendation for a single SKU.
    
    Returns status and action recommendations.
    """
    optimizer = InventoryOptimizer(service_level)
    
    ss = optimizer.calculate_safety_stock(avg_demand, std_demand, avg_lead_time, std_lead_time)
    rop = optimizer.calculate_reorder_point(avg_demand, avg_lead_time, ss)
    dos = optimizer.calculate_days_of_supply(current_inventory, avg_demand)
    
    # Determine status
    if current_inventory < ss:
        status = "CRITICAL"
        action = "⚡ EXPEDITE ORDER (Air Freight)"
        urgency = "Immediate"
    elif current_inventory < rop:
        status = "REORDER NOW"
        action = "📦 Place Standard Order"
        urgency = "Within 24 hours"
    elif current_inventory < rop * 1.2:
        status = "WATCH"
        action = "👀 Monitor Closely"
        urgency = "Normal"
    else:
        status = "OK"
        action = "✅ No Action Needed"
        urgency = "None"
    
    return {
        'status': status,
        'action': action,
        'urgency': urgency,
        'current_inventory': round(current_inventory, 2),
        'safety_stock': round(ss, 2),
        'reorder_point': round(rop, 2),
        'days_of_supply': round(dos, 1),
        'service_level': f"{service_level * 100:.0f}%"
    }
