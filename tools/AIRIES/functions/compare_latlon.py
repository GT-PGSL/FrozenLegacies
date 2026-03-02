import pandas as pd
import matplotlib.pyplot as plt

# Load the files
df_csv = pd.read_csv("103.csv")
df_txt = pd.read_csv("sprinsftud_1974_103_nav.txt", sep="\t")

# Ensure merge columns are integer
df_csv["CBD"] = df_csv["CBD"].astype(int)
df_txt["PRED_CBD"] = df_txt["PRED_CBD"].astype(int)

# Merge on CBD
merged = pd.merge(
    df_csv, df_txt, left_on="CBD", right_on="PRED_CBD", suffixes=("_csv", "_txt")
)

# Plotting
plt.figure(figsize=(12, 7))
plt.plot(merged["LON"], merged["LAT"], "o", label="103.csv", alpha=0.7)
plt.plot(
    merged["LONGITUDE"],
    merged["LATITUDE"],
    "x",
    label="sprinsftud_1974_103_nav.txt",
    alpha=0.7,
)

# Draw a line between the two sources for each CBD
for _, row in merged.iterrows():
    plt.plot(
        [row["LON"], row["LONGITUDE"]], [row["LAT"], row["LATITUDE"]], "k-", alpha=0.2
    )

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Lat/Lon at Each CBD (PRED_CBD/CBD) for Flight 103")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Select and rename columns for output
output = merged[["CBD", "LAT", "LON", "LATITUDE", "LONGITUDE"]].copy()
output = output.rename(
    columns={
        "LAT": "LAT (stanford)",
        "LON": "LON (stanford)",
        "LATITUDE": "LAT (bingham)",
        "LONGITUDE": "LON (bingham)",
    }
)

output.to_csv("merged_103_nav.csv", index=False)
