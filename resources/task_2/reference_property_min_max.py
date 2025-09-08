import re
import pandas as pd


# This file creates a reference property file which parses and splits min max values 
# from reference sheet for easier enrichment of RFQs. It also strips units and converts percentages to decimal format.

# Utility: resolve column by name or 0-based index
def _resolve_col(df: pd.DataFrame, col_or_idx):
    if isinstance(col_or_idx, int):
        return df.columns[col_or_idx]
    return col_or_idx


# Strip a trailing unit at the END of the cell (keeps ranges intact)
def strip_trailing_unit(df: pd.DataFrame, col_or_idx, out_col=None, unit_pattern=None) -> pd.Series:
    """
    Removes a trailing unit token at the very end of the string.
    Defaults to removing alphabetic/µ/° (and /) unit tokens at the end.
    """
    col = _resolve_col(df, col_or_idx)
    if out_col is None:
        out_col = col

    # default pattern: spaces + letters/µ/° + optional /letters at end
    if unit_pattern is None:
        unit_pattern = r"\s*[A-Za-zµ°]+[A-Za-zµ°/]*$"

    def _strip_unit(x):
        if pd.isna(x):
            return x
        s = str(x).strip()
        return re.sub(unit_pattern, "", s).strip()

    df[out_col] = df[col].map(_strip_unit)
    return df[out_col]


# Convert percentages to float (remove trailing % and parse as float)
def percent_to_float(df: pd.DataFrame, col_or_idx, out_col=None) -> pd.Series:
    """
    Convert percentage strings to decimal fractions while preserving operators/ranges.
    Examples:
      "26%"   -> "0.26"
      "≥26%"  -> "≥0.26"
      "5-10%" -> "0.05-0.10"
    Non-percent values are returned unchanged.
    """
    col = _resolve_col(df, col_or_idx)
    if out_col is None:
        out_col = col

    def _fmt(v: float) -> str:
        # compact formatting without trailing zeros
        return f"{v:.12g}"

    def _conv_number(token: str) -> str:
        t = token.strip()
        if t.endswith("%"):
            t = t[:-1]
        t = t.replace(",", ".")
        try:
            v = float(t) / 100.0
            return _fmt(v)
        except:
            return token.strip()  # leave as-is if not parseable

    def _pct_to_fraction(x):
        if pd.isna(x):
            return ""
        s = str(x).strip()
        if s == "" or "%" not in s:
            return s  # nothing to do

        s = s.replace("–", "-")  # normalize en-dash

        # capture optional leading comparison operator
        m = re.match(r'^(?P<op>(?:≤|≥|<=|>=|<|>)?)\s*(?P<body>.*)$', s)
        if not m:
            return s
        op = m.group("op")
        body = m.group("body").strip()

        # single value (possibly with operator)
        return f"{op}{_conv_number(body)}"

    df[out_col] = df[col].map(_pct_to_fraction)
    return df[out_col]



# Split "min-max" (or single/≤/≥) into two numeric columns
# Assumes the column already has units stripped if needed.

def split_min_max(df: pd.DataFrame, col_or_idx, out_prefix=None) -> pd.DataFrame:
    col = _resolve_col(df, col_or_idx)
    if out_prefix is None:
        out_prefix = col

    min_col = f"{out_prefix}_min"
    max_col = f"{out_prefix}_max"

    def _parse_range(val):
        if pd.isna(val) or str(val).strip() == "":
            return 0.0, 0.0

        text = str(val).strip()
        text = text.replace(",", ".")  # handle decimal commas
        text = text.replace("–", "-")  # normalize en-dash to hyphen

        # ≤ / >= / <= / ≥ handling
        if text.startswith(("≤", "<=")):
            num = text.replace("≤", "").replace("<=", "").strip()
            try:
                x = float(num)
                return 0.0, x
            except:
                return 0.0, 0.0

        if text.startswith(("≥", ">=")):
            num = text.replace("≥", "").replace(">=", "").strip()
            try:
                x = float(num)
                return x, x
            except:
                return 0.0, 0.0

        # Standard range "A-B"
        if "-" in text:
            parts = [p.strip() for p in text.split("-", 1)]
            try:
                a = float(parts[0])
            except:
                a = 0.0
            try:
                b = float(parts[1])
            except:
                b = 0.0
            return a, b

        # Single value
        try:
            x = float(text)
            return x, x
        except:
            return 0.0, 0.0

    mins, maxs = [], []
    for v in df[col]:
        mn, mx = _parse_range(v)
        mins.append(mn)
        maxs.append(mx)

    df[min_col] = mins
    df[max_col] = maxs
    return df[[min_col, max_col]]





# Process reference_properties
if __name__ == "__main__":
    ref_path = "reference_properties.tsv"
    ref = pd.read_csv(ref_path, sep="\t")


    strip_trailing_unit(ref, "Yield strength (Re or Rp0.2)")
    strip_trailing_unit(ref, "Tensile strength (Rm)")
    percent_to_float(ref, "Elongation (A%)")

    ref.rename(
        columns={
            "Tensile strength (Rm)": "tensile_strength",
            "Yield strength (Re or Rp0.2)": "yield_strength",
        },
        inplace=True
    )

    # split columns "Carbon (C)" to "Elongation (A%)" and "Nb + V + Ti (Others)" to min/max values
    target_cols = list(range(4, 24)) + [32]
    for c in target_cols:
        split_min_max(ref, c)

    # once converted, delete processed columns
    ref = ref.drop(ref.columns[target_cols], axis=1)

    # drop empty columns
    ref.dropna(axis=1, how="all", inplace=True)

    # rename to be consistent with rfq
    ref.columns = ref.columns.str.strip()  # remove leading/trailing spaces


    # TODO: figure out how to best handle Hardness due to varying scales - tempoary solution: treat as category variable
    # Save
    ref.to_csv("task_2_output/reference_properties_split.tsv", sep="\t", index=False)
    print("Saved to reference_properties_split.tsv")


