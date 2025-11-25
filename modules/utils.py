"""
Utility functions for data loading, validation, column mapping, and filtering.
Complete implementation with all required functions for app.py
"""
import pandas as pd
import os

# Column role definitions
REQUIRED_ROLES = ["date", "sku_id", "warehouse_id", "units_sold", "inventory_level"]
OPTIONAL_ROLES = ["supplier_id", "supplier_lead_time_days", "unit_cost", "unit_price", 
                  "distance_km", "transport_mode", "shipping_cost", "stockout_flag"]


def load_data(file_path):
    """Load CSV file into pandas DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def infer_columns(df):
    """
    Automatically infer column mappings based on column names.
    Returns a dict mapping roles to actual column names.
    """
    mapping = {}
    columns_lower = {col: col.lower().replace(" ", "_") for col in df.columns}
    
    # Role keywords for automatic detection
    role_keywords = {
        "date": ["date", "day", "time", "dt"],
        "sku_id": ["sku", "product", "item", "sku_id", "product_id"],
        "warehouse_id": ["warehouse", "location", "wh", "warehouse_id"],
        "units_sold": ["units", "sold", "quantity", "qty", "demand", "sales"],
        "inventory_level": ["inventory", "stock", "on_hand", "level"],
        "supplier_id": ["supplier", "vendor", "supplier_id"],
        "supplier_lead_time_days": ["lead_time", "leadtime", "lt", "days"],
        "unit_cost": ["cost", "unit_cost", "price_cost"],
        "unit_price": ["price", "unit_price", "selling_price"],
        "distance_km": ["distance", "km", "miles"],
        "transport_mode": ["transport", "mode", "shipping_mode"],
        "shipping_cost": ["shipping", "freight", "delivery_cost"],
        "stockout_flag": ["stockout","out_of_stock", "shortage"]
    }
    
    # Try to match each role
    for role, keywords in role_keywords.items():
        for col, col_norm in columns_lower.items():
            if any(keyword in col_norm for keyword in keywords):
                mapping[role] = col
                break
    
    return mapping


def validate_mapping(df, mapping):
    """
    Validate that all required columns are mapped and exist in dataframe.
    Returns list of missing required columns.
    """
    missing = []
    for role in REQUIRED_ROLES:
        if role not in mapping or not mapping[role]:
            missing.append(role)
        elif mapping[role] not in df.columns:
            missing.append(f"{role} (column '{mapping[role]}' not found)")
    
    return missing


def apply_mapping(df, mapping):
    """
    Apply column mapping to dataframe - renames columns to standard names.
    Returns new dataframe with standardized column names.
    """
    # Create reverse mapping (actual column name -> standard name)
    rename_dict = {}
    for role, actual_col in mapping.items():
        if actual_col:  # Only rename if mapped
            # Standard names are uppercase with underscores
            standard_name = role.upper()
            if actual_col in df.columns:
                rename_dict[actual_col] = standard_name
    
    # Rename columns
    new_df = df.rename(columns=rename_dict)
    
    # Convert date column to datetime
    if 'DATE' in new_df.columns:
        new_df['DATE'] = pd.to_datetime(new_df['DATE'], errors='coerce')
    
    # Map back to what app.py expects (Title Case)
    final_rename = {
        'DATE': 'Date',
        'SKU_ID': 'SKU_ID',
        'WAREHOUSE_ID': 'Warehouse_ID',
        'UNITS_SOLD': 'Units_Sold',
        'INVENTORY_LEVEL': 'Inventory_Level',
        'SUPPLIER_ID': 'Supplier_ID',
        'SUPPLIER_LEAD_TIME_DAYS': 'Supplier_Lead_Time_Days',
        'UNIT_COST': 'Unit_Cost',
        'UNIT_PRICE': 'Unit_Price',
        'DISTANCE_KM': 'Distance_km',
        'TRANSPORT_MODE': 'Transport_Mode',
        'SHIPPING_COST': 'Shipping_Cost',
        'STOCKOUT_FLAG': 'Stockout_Flag'
    }
    
    new_df = new_df.rename(columns=final_rename)
    
    return new_df


def validate_required_columns(df, required_columns):
    """
    Check if all required columns are present in the dataframe.
    Returns missing columns as a list.
    """
    missing = [col for col in required_columns if col not in df.columns]
    return missing


def filter_data(df, sku_id=None, warehouse_id=None):
    """
    Filters the dataframe by SKU and Warehouse.
    """
    temp_df = df.copy()
    if sku_id and sku_id != "All" and 'SKU_ID' in temp_df.columns:
        temp_df = temp_df[temp_df['SKU_ID'] == sku_id]
    if warehouse_id and warehouse_id != "All" and 'Warehouse_ID' in temp_df.columns:
        temp_df = temp_df[temp_df['Warehouse_ID'] == warehouse_id]
    return temp_df
