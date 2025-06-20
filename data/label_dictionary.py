import pandas as pd
import json

# Assuming df contains string columns: 'country', 'region', 'subregion'
df = pd.read_csv("data/processed/labeledtest.csv")

# Factorize into label indices
df["country_index"], country_labels = pd.factorize(df["country"])
df["region_index"], region_labels = pd.factorize(df["region"])
df["subregion_index"], subregion_labels = pd.factorize(df["subregion"])

# Build lookup dictionaries
country_to_idx = {label: int(idx) for idx, label in enumerate(country_labels)}
region_to_idx = {label: int(idx) for idx, label in enumerate(region_labels)}
subregion_to_idx = {label: int(idx) for idx, label in enumerate(subregion_labels)}

# You may also want the reverse mappings:
idx_to_country = {v: k for k, v in country_to_idx.items()}
idx_to_region = {v: k for k, v in region_to_idx.items()}
idx_to_subregion = {v: k for k, v in subregion_to_idx.items()}

label_map = {
    "country_to_idx": country_to_idx,
    "region_to_idx": region_to_idx,
    "subregion_to_idx": subregion_to_idx,
    "idx_to_country": idx_to_country,
    "idx_to_region": idx_to_region,
    "idx_to_subregion": idx_to_subregion,
}

with open("label_mappings.json", "w") as f:
    json.dump(label_map, f)
