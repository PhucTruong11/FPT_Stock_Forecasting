"""Independent checks against saved raw evidence; run from any working directory."""

import json
import sys
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
sys.path.insert(0, str(ROOT))

from src.data.source_audit import read_evidence, guarded_hashes  # noqa: E402


def main():
    sources = read_evidence(RUN)  # Verify both immutable ZIPs and every member hash.
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    vci, kbs, yahoo = [sources[name].set_index("Date") for name in ["vci", "kbs", "refreshed"]]
    vci_dates, kbs_dates = [data.index[data.Volume.gt(0)] for data in [vci, kbs]]
    assert set(vci_dates) == set(kbs_dates)
    consensus = vci_dates.intersection(kbs_dates)
    assert int(yahoo.loc[consensus].Volume.eq(0).sum()) == 56
    result = {"immutable_source_archives_verified": True, "corroborated_yahoo_zero_days": 56, "return_pairs": {}}
    for left, right in [("refreshed", "vci"), ("vci", "kbs")]:
        # Independent indexed differencing, instead of production merge/log-ratio code.
        a, b = [sources[name].query("Volume>0").set_index("Date").Close for name in [left, right]]
        dates = a.index.union(b.index).sort_values()
        difference = (np.log(a.reindex(dates)).diff() - np.log(b.reindex(dates)).diff()).dropna() * 10000
        expected = summary["return_pairs"][f"{left}_vs_{right}"]
        assert len(difference) == expected["comparable_adjacent_pairs"]
        assert int(difference.abs().gt(10).sum()) == expected["abs_difference_gt_10bps"]
        assert np.isclose(difference.abs().max(), expected["max_abs_difference_bps"], atol=1e-8)
        result["return_pairs"][f"{left}_vs_{right}"] = {"pairs": len(difference), "over_10bps": int(difference.abs().gt(10).sum())}
    before = json.loads((RUN / "protected_before.json").read_text(encoding="utf-8"))
    assert before == json.loads((RUN / "protected_after.json").read_text(encoding="utf-8"))
    # A clone may not yet have restored canonical CSVs; still validate the recorded guards.
    if all((ROOT / name).exists() for name in before):
        assert guarded_hashes() == before
        result["current_protected_files_match"] = True
    else:
        result["current_protected_files_match"] = "not_checked_missing_local_files"
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
