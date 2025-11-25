# Case Study: AI Supply Chain Control Tower

## Business Context

A mid-sized retail logistics company faced challenges with:

1. **Stockouts:** Frequent out-of-stock events leading to lost revenue.
2. **Excess Inventory:** High holding costs due to static safety stock policies.
3. **Supplier Unreliability:** Inability to predict and mitigate supplier delays.

## Solution

We implemented an **AI-Driven Control Tower** integrating:

- **Predictive Analytics:** To forecast demand more accurately than simple averages.
- **Dynamic Inventory Policies:** To adjust Reorder Points based on demand and lead time variability.
- **Risk Scoring:** To identify high-risk suppliers proactively.

## User Journey

1. **Setup:** The planner uploads their daily transaction CSV and maps columns (Date, SKU, Sales, Inventory).
2. **Overview:** They check the **Executive Dashboard** to see if OTIF is below target.
3. **Deep Dive:** They navigate to **Inventory & Replenishment** to identify SKUs with low "Days of Supply".
4. **Action:** They download the "Reorder Recommendations" for those SKUs.
5. **Simulation:** Before a holiday season, they run a **Scenario Plan** (+20% demand) to see if current stock is sufficient.

## Key Results & Benefits

### 1. Improved Forecast Accuracy

By using **Exponential Smoothing** and **ARIMA**, we captured seasonality better than the legacy system.

- *Benefit:* Reduced forecast error (MAPE) by ~15% in testing.

### 2. Optimized Inventory Levels

Dynamic **Safety Stock** calculations allowed reducing inventory for stable SKUs while increasing it for volatile ones.

- *Benefit:* Potential **10-20% reduction in holding costs** while maintaining service levels.

### 3. Enhanced Visibility

The **Executive Dashboard** provides a single source of truth for OTIF and Inventory Turnover.

- *Benefit:* Faster decision-making cycle (daily vs. weekly).

### 4. Proactive Risk Management

The **Supplier Reliability Score** enables the procurement team to negotiate better terms or diversify suppliers for high-risk items.

## Conclusion

The AI Control Tower transforms supply chain operations from reactive to proactive, driving both cost efficiency and customer satisfaction.
