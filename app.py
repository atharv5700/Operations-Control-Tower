import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# Legacy modules (kept as fallbacks)
from modules import utils, demand_forecasting, inventory_optimization, supplier_risk, service_level_kpis, logistics

# NEW: Intelligent ML-powered modules
try:
    from modules import demand_forecasting_ml
    from modules import inventory_optimizer
    from modules import logistics_routing
    from modules import supplier_risk_ml
    from modules import monte_carlo
    INTELLIGENT_MODULES_AVAILABLE = True
except ImportError as e:
    st.warning(f"⚠️ Intelligent modules not available: {e}. Using legacy fallbacks.")
    INTELLIGENT_MODULES_AVAILABLE = False

# --- CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="Supply Chain Control Tower",
    page_icon="🗼",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Use Streamlit theme variables for consistency */
        .big-title { 
            font-size: 2.5rem; 
            font-weight: 700; 
            margin-bottom: 0.5rem; 
            color: var(--text-color);
        }
        .subtitle { 
            font-size: 1.2rem; 
            color: var(--text-color); 
            opacity: 0.8; 
            margin-bottom: 2rem; 
        }
        .section-card {
            background-color: var(--secondary-background-color);
            padding: 1.5rem;
            border-radius: 0.75rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }
        .action-card {
            background-color: #fee2e2; /* Light red for alerts */
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 5px solid #ef4444;
            margin-bottom: 1rem;
            color: #7f1d1d;
        }
        .good-card {
            background-color: #d1fae5; /* Light green for success */
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 5px solid #10b981;
            margin-bottom: 1rem;
            color: #064e3b;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

inject_custom_css()

# --- SESSION STATE INITIALIZATION ---
if "df" not in st.session_state:
    st.session_state["df"] = None
if "column_mapping" not in st.session_state:
    st.session_state["column_mapping"] = {}
if "temp_raw_df" not in st.session_state:
    st.session_state["temp_raw_df"] = None

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🗼 Control Tower")
page = st.sidebar.radio("Menu", [
    "📥 Data Setup",
    "🏠 Dashboard",
    "🔮 Demand Forecast",
    "📦 Inventory Actions",
    "✈️ Logistics & Execution",
    "🚚 Supplier Risk",
    "⚡ What-If Scenarios",
    "📚 User Guide"
])

# --- HELPER FUNCTIONS ---
def show_no_data_warning():
    st.warning("⚠️ No data loaded. Please go to the **Data Setup** page to upload your file.")
    st.stop()

def get_filtered_data(df):
    st.sidebar.header("Filters")
    
    sku_col = st.session_state["column_mapping"].get("sku_id", "SKU_ID")
    wh_col = st.session_state["column_mapping"].get("warehouse_id", "Warehouse_ID")
    
    if sku_col not in df.columns or wh_col not in df.columns:
        return df, "All", "All"
        
    skus = ["All"] + sorted(list(df[sku_col].unique()))
    warehouses = ["All"] + sorted(list(df[wh_col].unique()))
    
    selected_sku = st.sidebar.selectbox("Product (SKU)", skus)
    selected_warehouse = st.sidebar.selectbox("Location (Warehouse)", warehouses)
    
    filtered = df.copy()
    if selected_sku != "All":
        filtered = filtered[filtered[sku_col] == selected_sku]
    if selected_warehouse != "All":
        filtered = filtered[filtered[wh_col] == selected_warehouse]
        
    return filtered, selected_sku, selected_warehouse

# --- PAGE 1: DATA SETUP ---
if page == "📥 Data Setup":
    st.markdown('<div class="big-title">Data Setup</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload your data to get started.</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("1. Select Data")
        data_source = st.radio("Source", ["Use Demo Data", "Upload My CSV"], horizontal=True)
        
        if data_source == "Use Demo Data":
            if st.button("Load Demo Data"):
                try:
                    demo_path = "data/high_dim_supply_chain.csv"
                    st.session_state["temp_raw_df"] = utils.load_data(demo_path)
                    st.success("✅ Demo data loaded! Scroll down to confirm columns.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file:
                try:
                    st.session_state["temp_raw_df"] = pd.read_csv(uploaded_file)
                    st.success("✅ File uploaded! Scroll down to confirm columns.")
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
        
        if st.session_state["temp_raw_df"] is not None:
            raw_df = st.session_state["temp_raw_df"]
            st.markdown("---")
            st.subheader("2. Confirm Columns")
            st.caption("Match your columns to what the system needs.")
            
            inferred_mapping = utils.infer_columns(raw_df)
            col1, col2 = st.columns(2)
            new_mapping = {}
            
            with col1:
                st.markdown("**Essential Columns**")
                for role in utils.REQUIRED_ROLES:
                    default_idx = 0
                    options = [""] + list(raw_df.columns)
                    if role in inferred_mapping and inferred_mapping[role] in options:
                        default_idx = options.index(inferred_mapping[role])
                    new_mapping[role] = st.selectbox(f"{role.replace('_', ' ').title()} *", options, index=default_idx, key=f"map_{role}")
            
            with col2:
                st.markdown("**Optional (Recommended)**")
                for role in utils.OPTIONAL_ROLES:
                    default_idx = 0
                    options = [""] + list(raw_df.columns)
                    if role in inferred_mapping and inferred_mapping[role] in options:
                        default_idx = options.index(inferred_mapping[role])
                    new_mapping[role] = st.selectbox(f"{role.replace('_', ' ').title()}", options, index=default_idx, key=f"map_{role}")
            
            if st.button("Save & Continue", type="primary"):
                missing = utils.validate_mapping(raw_df, new_mapping)
                if missing:
                    st.error(f"❌ Missing columns: {', '.join(missing)}")
                else:
                    processed_df = utils.apply_mapping(raw_df, new_mapping)
                    st.session_state["df"] = processed_df
                    st.session_state["column_mapping"] = new_mapping
                    st.success("✅ Ready! Go to the Dashboard.")
                    st.balloons()
        st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 2: DASHBOARD ---
elif page == "🏠 Dashboard":
    if st.session_state["df"] is None:
        show_no_data_warning()
        
    df = st.session_state["df"]
    filtered_df, sel_sku, sel_wh = get_filtered_data(df)
    
    st.markdown('<div class="big-title">Control Tower Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Your daily supply chain health check.</div>', unsafe_allow_html=True)
    
    # 1. Action Items (Top Priority)
    st.subheader("📋 Top Priorities")
    col1, col2 = st.columns(2)
    
    with col1:
        # Stockout Risk
        if 'Reorder_Point' in filtered_df.columns:
            at_risk = filtered_df[filtered_df['Inventory_Level'] < filtered_df['Reorder_Point']]
            risk_count = at_risk['SKU_ID'].nunique()
            if risk_count > 0:
                st.markdown(f'<div class="action-card">⚠️ <b>{risk_count} Products</b> are below reorder point. Check "Inventory Actions".</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="good-card">✅ Inventory levels are healthy.</div>', unsafe_allow_html=True)
        else:
            st.info("Map 'Reorder Point' to see stockout risks.")

    with col2:
        # Supplier Risk
        if 'Supplier_Lead_Time_Days' in filtered_df.columns:
            avg_lt = filtered_df['Supplier_Lead_Time_Days'].mean()
            if avg_lt > 10: # Arbitrary threshold for demo
                st.markdown(f'<div class="action-card">⚠️ Average Lead Time is high (<b>{avg_lt:.1f} days</b>). Check "Supplier Risk".</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="good-card">✅ Supplier lead times are stable.</div>', unsafe_allow_html=True)

    # 2. Key Metrics
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Performance Snapshot")
    c1, c2, c3, c4 = st.columns(4)
    
    total_rev = (filtered_df['Units_Sold'] * filtered_df['Unit_Price']).sum() if 'Unit_Price' in filtered_df.columns else 0
    otif = service_level_kpis.calculate_otif(filtered_df) if 'Stockout_Flag' in filtered_df.columns else 0
    
    c1.metric("Revenue", f"${total_rev:,.0f}")
    c2.metric("Service Level (OTIF)", f"{otif:.1f}%")
    c3.metric("Avg Inventory", f"{filtered_df['Inventory_Level'].mean():,.0f} units")
    c4.metric("Total Sales", f"{filtered_df['Units_Sold'].sum():,.0f} units")
    st.markdown('</div>', unsafe_allow_html=True)

    # 3. Trends
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Demand vs. Supply Trend")
    daily_agg = filtered_df.groupby('Date').agg({'Units_Sold': 'sum', 'Inventory_Level': 'sum'}).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_agg['Date'], y=daily_agg['Units_Sold'], name='Sales (Demand)', line=dict(color='#3b82f6'), fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)'))
    fig.add_trace(go.Scatter(x=daily_agg['Date'], y=daily_agg['Inventory_Level'], name='Stock On Hand', line=dict(color='#f97316'), yaxis='y2'))
    
    fig.update_layout(
        yaxis=dict(title="Sales"),
        yaxis2=dict(title="Stock Level", overlaying='y', side='right'),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 3: DEMAND FORECAST ---
elif page == "🔮 Demand Forecast":
    if st.session_state["df"] is None:
        show_no_data_warning()
    df = st.session_state["df"]
    filtered_df, sel_sku, sel_wh = get_filtered_data(df)
    
    st.markdown('<div class="big-title">🧠 Intelligent Demand Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Ensemble ML forecasting with Auto-ARIMA, XGBoost, and Prophet</div>', unsafe_allow_html=True)
    
    if sel_sku == "All":
        st.info("👈 Please select a specific Product (SKU) from the sidebar to generate a forecast.")
    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader(f"Sales Forecast: {sel_sku}")
        
        sku_df = filtered_df.groupby('Date')['Units_Sold'].sum().reset_index().set_index('Date').asfreq('D').fillna(0)
        
        if len(sku_df) < 30:
            st.warning("Not enough history to forecast (need 30+ days).")
        else:
            # Configuration
            col1, col2 = st.columns(2)
            with col1:
                forecast_days = st.slider("Forecast Horizon (days)", 7, 60, 14)
            with col2:
                use_intelligent = st.checkbox("🧠 Use Intelligent ML Forecasting", value=INTELLIGENT_MODULES_AVAILABLE)
            
            # Train/test split for visualization
            train_size = int(len(sku_df) * 0.8)
            train = sku_df['Units_Sold'].iloc[:train_size]
            test = sku_df['Units_Sold'].iloc[train_size:]
            horizon = len(test)
            
            if use_intelligent and INTELLIGENT_MODULES_AVAILABLE:
                # NEW: Intelligent ensemble forecasting
                st.info("🧠 Using ensemble ML: Auto-ARIMA + XGBoost + Prophet")
                
                with st.spinner("Training models and selecting best approach..."):
                    try:
                        # Use the intelligent forecaster
                        forecaster = demand_forecasting_ml.DemandForecaster(train, horizon)
                        best_model, best_forecast = forecaster.select_best_model()
                        
                        # Get all model predictions for comparison
                        all_forecasts = forecaster.predictions
                        
                        # Display selected model
                        st.success(f"✅ Best model selected: **{best_model.upper()}** (lowest validation error)")
                        
                        # Plot forecasts
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=test.index, y=test.values, name='Actual Sales', 
                                                line=dict(color='black', width=3), mode='lines+markers'))
                        
                        # Plot all individual model forecasts
                        colors = {'auto_arima': '#3b82f6', 'exp_smoothing': '#10b981', 
                                 'xgboost': '#f59e0b', 'prophet': '#ef4444'}
                        
                        for model_name, forecast in all_forecasts.items():
                            is_best = (model_name == best_model)
                            fig.add_trace(go.Scatter(
                                x=test.index, 
                                y=forecast, 
                                name=f'{model_name.replace("_", " ").title()}{" ⭐" if is_best else ""}',
                                line=dict(color=colors.get(model_name, '#gray'), 
                                        width=3 if is_best else 1.5,
                                        dash='solid' if is_best else 'dash')
                            ))
                        
                        fig.update_layout(
                            hovermode="x unified", 
                            margin=dict(l=0, r=0, t=30, b=0), 
                            paper_bgcolor='rgba(0,0,0,0)',
                            title="Model Comparison on Validation Set",
                            xaxis_title="Date",
                            yaxis_title="Units Sold"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Accuracy metrics table
                        st.markdown("#### 📊 Model Accuracy Comparison")
                        st.caption("Lower error is better. Best model is ⭐ highlighted.")
                        
                        metrics_data = []
                        for model_name, forecast in all_forecasts.items():
                            metrics = demand_forecasting_ml.evaluate_forecast(test.values, forecast)
                            metrics['Model'] = f"{model_name.replace('_', ' ').title()}{' ⭐' if model_name == best_model else ''}"
                            metrics_data.append(metrics)
                        
                        metrics_df = pd.DataFrame(metrics_data)[['Model', 'MAPE', 'RMSE', 'MAE', 'Bias']]
                        metrics_df = metrics_df.sort_values('MAPE')
                        
                        st.dataframe(
                            metrics_df.style.format({
                                'MAPE': '{:.2f}%',
                                'RMSE': '{:.2f}',
                                'MAE': '{:.2f}',
                                'Bias': '{:+.2f}'
                            }).background_gradient(subset=['MAPE','RMSE','MAE'], cmap='RdYlGn_r'),
                            use_container_width=True
                        )
                        
                        # Future forecast
                        st.markdown("---")
                        st.markdown("#### 🔮 Future Forecast (Next Period)")
                        
                        # Re-train on full data for future forecast
                        full_train = sku_df['Units_Sold']
                        future_forecast, _, _ = demand_forecasting_ml.quick_forecast(
                            full_train, forecast_days, method=best_model
                        )
                        
                        future_dates = pd.date_range(start=sku_df.index[-1] + pd.Timedelta(days=1), 
                                                     periods=forecast_days, freq='D')
                        
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=sku_df.index[-30:], y=sku_df['Units_Sold'][-30:], 
                                                 name='Historical (Last 30 days)', line=dict(color='#64748b')))
                        fig2.add_trace(go.Scatter(x=future_dates, y=future_forecast, name='Forecast', 
                                                 line=dict(color='#10b981', width=2), 
                                                 fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.1)'))
                        
                        fig2.update_layout(
                            hovermode="x unified", 
                            margin=dict(l=0, r=0, t=0, b=0),
                            paper_bgcolor='rgba(0,0,0,0)',
                            xaxis_title="Date",
                            yaxis_title="Units Sold"
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Summary stats
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Forecast Mean", f"{future_forecast.mean():.0f} units/day")
                        col2.metric("Total Demand", f"{future_forecast.sum():.0f} units")
                        col3.metric("Peak Day", f"{future_forecast.max():.0f} units")
                        
                    except Exception as e:
                        st.error(f"❌ Intelligent forecasting failed: {e}")
                        st.info("Falling back to simple methods...")
                        use_intelligent = False
            
            if not use_intelligent or not INTELLIGENT_MODULES_AVAILABLE:
                # Fallback to legacy simple forecasting
                st.info("📊 Using legacy forecasting (Exponential Smoothing + ARIMA)")
                
                # Simple Models
                es_pred = demand_forecasting.exponential_smoothing_forecast(train, horizon)
                arima_pred = demand_forecasting.arima_forecast(train, horizon)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=test.index, y=test.values, name='Actual Sales', line=dict(color='black', width=2)))
                fig.add_trace(go.Scatter(x=test.index, y=es_pred, name='Exp. Smoothing', line=dict(color='#10b981')))
                fig.add_trace(go.Scatter(x=test.index, y=arima_pred, name='ARIMA', line=dict(color='#ef4444')))
                
                fig.update_layout(hovermode="x unified", margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### Accuracy Check")
                metrics_list = []
                metrics_list.append({'Model': 'Exp. Smoothing', **demand_forecasting.evaluate_metrics(test, es_pred)})
                metrics_list.append({'Model': 'ARIMA', **demand_forecasting.evaluate_metrics(test, arima_pred)})
                
                m_df = pd.DataFrame(metrics_list)
                st.table(m_df.style.format("{:.2f}", subset=['MAPE', 'RMSE', 'Bias']))
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 4: INVENTORY ACTIONS ---
elif page == "📦 Inventory Actions":
    if st.session_state["df"] is None:
        show_no_data_warning()
    df = st.session_state["df"]
    filtered_df, sel_sku, sel_wh = get_filtered_data(df)
    
    st.markdown('<div class="big-title">Inventory Actions</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">What needs to be ordered today?</div>', unsafe_allow_html=True)
    
    if 'Supplier_Lead_Time_Days' not in filtered_df.columns:
        st.error("Missing Lead Time data.")
    else:
        # Calculate Recommendations
        summary = []
        for (sku, wh), group in filtered_df.groupby(['SKU_ID', 'Warehouse_ID']):
            avg_demand = group['Units_Sold'].mean()
            avg_lt = group['Supplier_Lead_Time_Days'].mean()
            curr_inv = group['Inventory_Level'].iloc[-1]
            
            # Simple Safety Stock & ROP
            ss = avg_demand * avg_lt * 0.5 # Simplified buffer
            rop = (avg_demand * avg_lt) + ss
            
            status = "OK"
            action = "None"
            
            if curr_inv < rop:
                status = "Reorder Now"
                # Check if we need to expedite (if stock < lead time demand)
                if curr_inv < (avg_demand * avg_lt):
                    action = "⚡ Expedite (Air)"
                else:
                    action = "Standard Order"
            elif curr_inv < rop * 1.2:
                status = "Watch"
                
            summary.append({
                'Product': sku,
                'Location': wh,
                'Stock': curr_inv,
                'Reorder Point': rop,
                'Safety Stock': ss,
                'Status': status,
                'Recommended Action': action
            })
            
        inv_df = pd.DataFrame(summary)
        
        # Action Table
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("⚠️ Urgent Reorders")
        reorder_df = inv_df[inv_df['Status'] == "Reorder Now"]
        
        if not reorder_df.empty:
            st.error(f"You need to reorder {len(reorder_df)} items immediately.")
            st.dataframe(reorder_df.style.format("{:.0f}", subset=['Stock', 'Reorder Point', 'Safety Stock']))
        else:
            st.success("No urgent reorders needed today.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Full View
        with st.expander("View All Inventory Status"):
            st.dataframe(inv_df.style.format("{:.0f}", subset=['Stock', 'Reorder Point', 'Safety Stock']))

# --- PAGE 5: LOGISTICS & EXECUTION ---
elif page == "✈️ Logistics & Execution":
    if st.session_state["df"] is None:
        show_no_data_warning()
    df = st.session_state["df"]
    filtered_df, sel_sku, sel_wh = get_filtered_data(df)
    
    st.markdown('<div class="big-title">Logistics & Execution</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Optimize transport modes and costs.</div>', unsafe_allow_html=True)
    
    # KPIs
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    
    avg_cost = filtered_df['Shipping_Cost'].mean() if 'Shipping_Cost' in filtered_df.columns else 0
    avg_dist = filtered_df['Distance_km'].mean() if 'Distance_km' in filtered_df.columns else 0
    
    c1.metric("Avg Shipping Cost", f"${avg_cost:.2f}")
    c2.metric("Avg Distance", f"{avg_dist:.0f} km")
    
    if 'Transport_Mode' in filtered_df.columns:
        mode_counts = filtered_df['Transport_Mode'].value_counts()
        top_mode = mode_counts.index[0] if not mode_counts.empty else "N/A"
        c3.metric("Top Transport Mode", top_mode)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Mode Optimizer
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🛠️ Transport Mode Optimizer")
    st.caption("Compare costs and time for different shipping methods.")
    
    col1, col2 = st.columns(2)
    with col1:
        opt_sku = st.selectbox("Select Product to Ship", filtered_df['SKU_ID'].unique())
        opt_qty = st.number_input("Shipment Quantity", min_value=1, value=100)
    
    with col2:
        # Get default distance for this SKU if available
        default_dist = 500
        if 'Distance_km' in filtered_df.columns:
            sku_dist = filtered_df[filtered_df['SKU_ID'] == opt_sku]['Distance_km'].mean()
            if not pd.isna(sku_dist):
                default_dist = int(sku_dist)
        
        opt_dist = st.number_input("Distance (km)", min_value=10, value=default_dist)
        current_mode = st.selectbox("Current Mode", ["Road", "Air", "Sea", "Rail"])
        
    if st.button("Compare Modes"):
        results = logistics.compare_transport_modes(opt_sku, opt_qty, opt_dist, current_mode)
        
        st.markdown("#### Comparison Results")
        st.dataframe(results.style.format({
            "Est. Days": "{:.1f}",
            "Est. Cost ($)": "${:,.2f}",
            "Time Savings (Days)": "{:+.1f}",
            "Cost Impact ($)": "${:+,.2f}"
        }))
        
        best_time = results.iloc[0]
        best_cost = results.sort_values("Est. Cost ($)").iloc[0]
        
        c1, c2 = st.columns(2)
        c1.success(f"🚀 **Fastest:** {best_time['Mode']} ({best_time['Est. Days']} days)")
        c2.info(f"💰 **Cheapest:** {best_cost['Mode']} (${best_cost['Est. Cost ($)']:.2f})")
        
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 6: SUPPLIER RISK ---
elif page == "🚚 Supplier Risk":
    if st.session_state["df"] is None:
        show_no_data_warning()
    df = st.session_state["df"]
    filtered_df, sel_sku, sel_wh = get_filtered_data(df)
    
    st.markdown('<div class="big-title">Supplier Risk</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Which suppliers are causing delays?</div>', unsafe_allow_html=True)
    
    if 'Supplier_ID' not in filtered_df.columns:
        st.error("Missing Supplier ID column.")
    else:
        stats = filtered_df.groupby('Supplier_ID')['Supplier_Lead_Time_Days'].agg(['mean', 'std', 'max']).reset_index()
        stats.columns = ['Supplier', 'Avg Days', 'Variability', 'Max Days']
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Lead Time by Supplier")
            fig = px.bar(stats, x='Supplier', y='Avg Days', error_y='Variability', title="Average Delivery Time (+/- Variability)")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Risk Scorecard")
            st.dataframe(stats.style.format("{:.1f}", subset=['Avg Days', 'Variability']))
            st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 7: SCENARIOS ---
elif page == "⚡ What-If Scenarios":
    if st.session_state["df"] is None:
        show_no_data_warning()
    df = st.session_state["df"]
    filtered_df, sel_sku, sel_wh = get_filtered_data(df)
    
    st.markdown('<div class="big-title">Scenario Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Test your supply chain resilience.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Define Scenario")
    st.caption("Adjust the sliders to simulate market changes. Results update automatically.")
    
    c1, c2 = st.columns(2)
    with c1:
        demand_mult = st.slider("📈 If Demand Increases By...", 0, 100, 20, format="+%d%%") / 100 + 1
    with c2:
        lead_add = st.slider("⏳ If Suppliers are Late By...", 0, 30, 5, format="%d days")
        
    # Interactive Calculation (No button needed)
    sim_demand = filtered_df['Units_Sold'] * demand_mult
    
    # Simple Logic: How many days would we stock out with new demand?
    # Assuming current inventory logic holds
    stockouts = (filtered_df['Inventory_Level'] < sim_demand).sum()
    orig_stockouts = (filtered_df['Inventory_Level'] < filtered_df['Units_Sold']).sum()
    
    st.markdown("---")
    st.subheader("Simulation Results")
    
    c1, c2 = st.columns(2)
    c1.metric("Current Stockout Days", orig_stockouts)
    c2.metric("Predicted Stockout Days", stockouts, delta=f"{stockouts - orig_stockouts} more days", delta_color="inverse")
    
    if stockouts > orig_stockouts:
        st.error(f"⚠️ This scenario would cause {stockouts - orig_stockouts} additional days of lost sales.")
    else:
        st.success("✅ Your inventory is resilient enough for this scenario.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 8: USER GUIDE ---
elif page == "📚 User Guide":
    st.markdown('<div class="big-title">User Guide & Requirements</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Everything you need to know to use this Control Tower.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🚀 Getting Started")
    st.markdown("""
    1.  **Go to 'Data Setup'**: This is your first step.
    2.  **Upload Data**: You can upload your own CSV file or use the built-in demo data.
    3.  **Map Columns**: The system will try to guess your column names. If it's wrong, select the correct column from the dropdowns.
    4.  **Save**: Click 'Save & Continue' to load the dashboard.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📋 Data Requirements")
    st.markdown("Your CSV file should contain the following columns for the best experience:")
    
    req_data = {
        "Column Name (Example)": ["Date", "SKU_ID", "Warehouse_ID", "Units_Sold", "Inventory_Level", "Supplier_Lead_Time_Days", "Reorder_Point"],
        "Description": ["Date of the record (YYYY-MM-DD)", "Unique Product ID", "Warehouse Location ID", "Quantity sold that day", "Stock on hand", "Days to receive order", "Level to trigger reorder"],
        "Required?": ["Yes", "Yes", "Yes", "Yes", "Yes", "Recommended", "Recommended"]
    }
    st.table(pd.DataFrame(req_data))
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🤖 AI & Analytics Explained")
    st.markdown("""
    -   **Demand Forecasting**: Uses **Exponential Smoothing** (detects trends) and **ARIMA** (statistical analysis) to predict future sales.
    -   **Inventory Optimization**: Calculates **Safety Stock** and **Reorder Points** based on the variability of your demand and supplier lead times.
    -   **Supplier Risk**: Analyzes historical delivery times to calculate a **Reliability Score** and predict the probability of late deliveries.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("❓ Troubleshooting")
    st.markdown("""
    -   **"No Data Loaded"**: Make sure you clicked "Save & Continue" on the Data Setup page.
    -   **Empty Charts**: Check your filters in the sidebar. You might have selected a combination (Product + Warehouse) that has no data.
    -   **Missing Metrics**: If you didn't map optional columns (like Lead Time), some features (like Risk Analysis) will be disabled.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
