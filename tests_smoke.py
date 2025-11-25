import unittest
import pandas as pd

from modules import utils, logistics

class TestLogistics(unittest.TestCase):
    def setUp(self):
        # Create a dummy dataframe with required columns
        self.df = pd.DataFrame({
            'SKU_ID': ['A', 'B'],
            'Warehouse_ID': ['W1', 'W2'],
            'Shipping_Cost': [100, 200],
            'Distance_km': [500, 1000],
            'Transport_Mode': ['Road', 'Air']
        })

    def test_filter_data(self):
        filtered = utils.filter_data(self.df, sku_id='A')
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]['SKU_ID'], 'A')

    def test_compare_transport_modes(self):
        results = logistics.compare_transport_modes('A', 10, 500, current_mode='Road')
        expected_cols = {'Mode', 'Est. Days', 'Est. Cost ($)', 'Time Savings (Days)', 'Cost Impact ($)'}
        self.assertTrue(expected_cols.issubset(set(results.columns)))
        self.assertTrue((results['Est. Days'] >= 1).all())

if __name__ == '__main__':
    unittest.main()
