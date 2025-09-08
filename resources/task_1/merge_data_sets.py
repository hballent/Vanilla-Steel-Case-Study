__requires__ = ['pandas', 'openpyxl']

import pandas as pd


# Assumptions and Normalization Notes

# - 'grade' and 'material' refer to the same concept and are unified under 'grade'.
#
# - 'finish' represents surface treatment (e.g., pickled, oiled, painted) and is
#   normalized across both datasets using a controlled vocabulary.
#
# - 'description' contains defect or quality-related notes
#   and is translated from German to English. It is kept separate from 'finish' to
#   preserve semantic clarity between surface treatment and defect information.
#
# - Columns not present in one file are added as NaN in the other to align schemas.
#   Without knowing how the dataset will be used, it is safer to keep missing data as NaN
#   rather than a more sophistaced imputation such as average or zeros. For example, with 
#   the thickness column, 0 could be a missleading interpretation if its not acutally 0, just missing. 
#
# - Numeric fields like thickness, width, weight, and quantity are coerced to float for consistency




# Load Data
df1 = pd.read_excel("supplier_data1.xlsx")
df2 = pd.read_excel("supplier_data2.xlsx")

# Translation and normalization maps
finish_translation_map = {
    "gebeizt": "pickled",
    "ungebeizt": "unpickled",
    "gebeizt und geglüht": "pickled and annealed"
}

description1_translation_map = {
    "Längs- oder Querisse": "longitudinal or transverse cracks",
    "Kantenfehler - FS-Kantenrisse": "edge defects – FS edge cracks",
    "Sollmasse (Gewicht) unterschritten": "target weight not met"
}

description2_normalization_map = {
    "Material is Oiled": "oiled",
    "Material is not Oiled": "not oiled",
    "Material is Painted": "painted"
}

# Clean df1
df1 = df1.rename(columns={
    "Quality/Choice": "quality",
    "Grade": "grade",
    "Finish": "finish",
    "Thickness (mm)": "thickness_mm",
    "Width (mm)": "width_mm",
    "Description": "description",
    "Gross weight (kg)": "weight_kg",
    "Quantity": "quantity"
})

# Translate finish
df1["finish"] = df1["finish"].map(finish_translation_map).fillna(df1["finish"])

# Translate description and combine into finish
df1["description"] = df1["description"].map(description1_translation_map).fillna(df1["description"])


# Add placeholder fields to df1
df1["material"] = df1["grade"]
df1["article_id"] = None
df1["reserved"] = None
df1["source_file"] = "supplier_data1"

# Clean df2 
df2 = df2.rename(columns={
    "Material": "material",
    "Description": "description",
    "Article ID": "article_id",
    "Weight (kg)": "weight_kg",
    "Quantity": "quantity",
    "Reserved": "reserved"
})

# Normalize description into finish
df2["description"] = df2["description"].map(description2_normalization_map).fillna(df2["description"])
df2["finish"] = df2["description"]
df2.drop(columns=["description"], inplace=True)

# Add placeholder fields to df2
df2["grade"] = df2["material"]
df2["quality"] = None
df2["thickness_mm"] = None
df2["width_mm"] = None
df2["source_file"] = "supplier_data2"


# Combine and Normalize
combined = pd.concat([df1, df2], ignore_index=True)

# Convert numerics
for col in ["weight_kg", "thickness_mm", "width_mm", "quantity"]:
    combined[col] = pd.to_numeric(combined[col], errors="coerce")

# Drop "RP02","RM","AG","AI" because not mentioned in task description
combined.drop(columns=["material","RP02","RM","AG","AI"], inplace=True)

# Save Output 
combined.to_csv("task_1_output/inventory_dataset_cleaned.csv", index=False)
print("inventory_dataset_cleaned.csv saved.")
