import math
from typing import Any, Dict, List


def records_safe(df) -> List[Dict[str, Any]]:
    """
    DataFrame -> JSON-safe records, with NaN turned into null.

    df.where(df.notna(), None) looks right but isn't: pandas silently
    casts None back to NaN when assigning into a float64 column, so NaN
    survives and later blows up json.dumps ("Out of range float values
    are not JSON compliant"). Scrub after converting to native dicts
    instead, where the values are plain Python floats.
    """
    if df.empty:
        return []
    records = df.to_dict(orient="records")
    for row in records:
        for key, value in row.items():
            if isinstance(value, float) and math.isnan(value):
                row[key] = None
    return records
