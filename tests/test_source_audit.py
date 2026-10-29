"""Source reconciliation must not hide units, missing sessions or duplicate keys."""

import copy
import unittest

import numpy as np
import pandas as pd

from src.data.source_audit import compare_pair, compare_returns, parse_kbs, parse_vci


def prices():
    return pd.DataFrame({"Date": pd.bdate_range("2020-01-01", periods=5),
                         "Open": [100., 101., 103., 102., 104.],
                         "High": [101., 102., 104., 103., 105.],
                         "Low": [99., 100., 102., 101., 103.],
                         "Close": [100., 101., 103., 102., 104.], "Volume": [1000] * 5})


class SourceAuditTests(unittest.TestCase):
    def test_vci_string_epoch_and_vnd_units(self):
        payload = [{"symbol": "FPT", "t": ["1577836800"], "o": [100000],
                    "h": [102000], "l": [99000], "c": [101000], "v": [3210]}]
        result = parse_vci(payload)
        self.assertEqual(result.Date.iloc[0], pd.Timestamp("2020-01-01"))
        self.assertEqual(result.Close.iloc[0], 101000)

    def test_vci_rejects_symbol_array_length_and_duplicate_date(self):
        payload = [{"symbol": "OTHER", "t": ["1577836800"], "o": [100],
                    "h": [102], "l": [99], "c": [101], "v": [3210]}]
        with self.assertRaises(ValueError):
            parse_vci(payload)
        payload[0]["symbol"] = "FPT"
        uneven = copy.deepcopy(payload)
        uneven[0]["v"].append(10)
        with self.assertRaises(ValueError):
            parse_vci(uneven)
        for key in "tohlcv":
            payload[0][key] *= 2
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_vci(payload)

    def test_kbs_seven_am_daily_label_and_reverse_order(self):
        source = prices().iloc[::-1]
        records = []
        for _, row in source.iterrows():
            records.append({"t": row.Date.strftime("%Y-%m-%d 07:00"),
                            **{k: row[c] for k, c in zip("ohlcv", ["Open", "High", "Low", "Close", "Volume"])}})
        result = parse_kbs({"symbol": "FPT", "data_day": records})
        self.assertTrue(result.Date.is_monotonic_increasing)
        self.assertEqual(result.Close.iloc[0], 100.)

    def test_duplicate_key_is_not_a_many_to_many_join(self):
        data = prices()
        with self.assertRaisesRegex(ValueError, "unique"):
            compare_pair(pd.concat([data, data.iloc[[0]]]), data)

    def test_missing_values_never_count_as_price_agreement(self):
        left, right = prices(), prices()
        left.loc[1:2, "Close"] = np.nan
        right.loc[2, "Close"] = np.nan
        joined, stats = compare_pair(left, right)
        self.assertEqual(stats["Close_missing_mismatch_rows"], 1)
        self.assertFalse(joined.loc[1:2, "Close_comparable"].any())
        self.assertEqual(stats["comparable_positive_close_rows"], 3)

    def test_no_automatic_thousand_vnd_rescaling(self):
        left, right = prices(), prices()
        right.loc[:, ["Open", "High", "Low", "Close"]] *= 1000
        _, stats = compare_pair(left, right)
        self.assertEqual(stats["close_material_difference_rows"], 5)

    def test_gap_invalidates_return_at_gap_and_next_date(self):
        left, right = prices(), prices().drop(index=2)
        result, stats = compare_returns(left, right)
        self.assertEqual(stats["comparable_adjacent_pairs"], 2)
        self.assertTrue(result.loc[result.Date.isin(left.Date.iloc[2:4]), "difference_bps"].isna().all())

    def test_constant_adjustment_factor_cancels_in_returns(self):
        left, right = prices(), prices()
        right.loc[:, ["Open", "High", "Low", "Close"]] *= .87
        _, stats = compare_returns(left, right)
        self.assertEqual(stats["comparable_adjacent_pairs"], 4)
        self.assertLess(stats["max_abs_difference_bps"], 1e-9)

    def test_zero_volume_is_not_a_verified_session(self):
        left, right = prices(), prices()
        left.loc[2, "Volume"] = 0
        _, stats = compare_returns(left, right)
        self.assertEqual(stats["comparable_adjacent_pairs"], 2)


if __name__ == "__main__":
    unittest.main()
