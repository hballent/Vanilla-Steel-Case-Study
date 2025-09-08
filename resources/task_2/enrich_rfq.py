import pandas as pd
from pathlib import Path

# How missing values are handled:

# Flag any missing grades and if grade is not present in reference_properties.tsv.
# Input yeild strenght, tensile strength, and finish from reference_proterties.tsv.
# The rest of missing values, keep null
#      Here I am not sure this is the correct approach to keep as null
#      would it be better to fill with 0's?
# Delete fully empty columns.


def check_unmatched_grades(rfq_df: pd.DataFrame, ref_df: pd.DataFrame, key_column: str = "_grade_key") -> bool:
    """
    Check for grades in RFQ that don't have matches in reference data.
    
    Args:
        rfq_df: DataFrame with RFQ data
        ref_df: DataFrame with reference properties
        key_column: Column name containing the normalized grade keys
        
    Returns:
        bool: True if all grades matched, False if there are unmatched grades
    """
    # Find grades that are in RFQ but not in reference
    unmatched = rfq_df[~rfq_df[key_column].isin(ref_df[key_column])]
    
    if len(unmatched) == 0:
        print("All grades have matching reference properties")
        return True
    
    print("WARNING: Found RFQ grades without reference properties:")
    for _, row in unmatched.iterrows():
        print(f"  - {row['grade']} (normalized: {row[key_column]})")
    return False



rfq_path = Path("rfq.csv")
ref_path = Path("task_2_output/reference_properties_split.tsv")
out_path = Path("task_2_output/rfq_enriched.csv")

# Read input files
rfq = pd.read_csv(rfq_path)
ref = pd.read_csv(ref_path, sep="\t")

# Create matching keys (normalized to uppercase)
rfq["_grade_key"] = rfq["grade"].astype(str).str.strip().str.upper()
ref["_grade_key"] = ref["Grade/Material"].astype(str).str.strip().str.upper()

# Here I noticed that all values in rfq.csv have a reference in reference_properties.tsv 
# if present in rfq.csv. However, some have missing grade values.
check_unmatched_grades(rfq, ref)

# Merge: left join (keep all RFQ rows), append ref columns
merged = rfq.merge(
    ref,
    how="left",
    on="_grade_key",
    suffixes=("", "_ref")  # RFQ keeps original names; overlapping ref cols get _ref
)

# Handle redundant values - keep original if present, else use reference values
# Assumption: "finish" in RFQ equivalent with "Coating" in reference
merged["yield_strength_min"] = merged["yield_strength_min"].fillna(merged["yield_strength_min_ref"])
merged["yield_strength_max"] = merged["yield_strength_max"].fillna(merged["yield_strength_max_ref"])
merged["tensile_strength_min"] = merged["tensile_strength_min"].fillna(merged["tensile_strength_min_ref"])
merged["tensile_strength_max"] = merged["tensile_strength_max"].fillna(merged["tensile_strength_max_ref"])
# Handle finish - keep original if present, else use reference coating
merged["finish"] = merged["finish"].fillna(merged["Coating"])

# Clean up: remove temporary matching key and duplicate ref columns
cols_to_drop = [
    "_grade_key",
    "yield_strength_min_ref",
    "yield_strength_max_ref",
    "tensile_strength_min_ref",
    "tensile_strength_max_ref",
    "Coating",  # Original finish column from reference
    "Grade/Material"  # Original grade column from reference
]
merged.drop(columns=cols_to_drop, inplace=True)

# Save enriched RFQ
merged.to_csv(out_path, index=False)
print(f"Saved enriched RFQ to: {out_path}")