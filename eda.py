"""
eda.py  –  Exploratory Data Analysis
=====================================
Standalone EDA script that mirrors the visual style of model.py.
Run this before model.py to understand the dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# =========================================================
# GLOBAL STYLE  (identical to model.py)
# =========================================================
THESIS_BLUE = "#1A3A5C"
ACCENT      = "#2E86AB"
ACCENT2     = "#E84855"
ACCENT3     = "#F4A261"
ACCENT4     = "#3BB273"
BG          = "#F8F9FA"
GRID_COLOR  = "#DDE3EA"
TEXT_COLOR  = "#1A3A5C"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#CCCCCC",
    "axes.labelcolor":   TEXT_COLOR,
    "axes.titlesize":    15,
    "axes.titleweight":  "bold",
    "axes.titlepad":     14,
    "axes.labelsize":    11,
    "axes.labelpad":     8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    0.8,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 0.9,
    "figure.dpi":        130,
    "font.family":       "DejaVu Sans",
})

# =========================================================
# LOAD & BASIC CLEAN
# =========================================================
df = pd.read_csv("sylhet_real_estate.csv")
df.columns = df.columns.str.strip()

# Clean location
df["Location"] = (
    df["Location"].astype(str).str.strip()
    .str.replace(r"\s+", " ", regex=True)
)
df["Location"] = df["Location"].replace({
    "ZindaBazar, Sylhet":          "Zindabazar, Sylhet",
    "Shahjalal Upashahar, Sylhet": "Shahjalal Uposhahar, Sylhet",
    "Mendibagh, Sylhete":          "Mendibagh, Sylhet",
    "Kazirbazar, Sylhete":         "Kazirbazar, Sylhet",
})

# Clean price
df["Selling price (BDT)"] = pd.to_numeric(
    df["Selling price (BDT)"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("BDT", "", regex=False)
    .str.strip(),
    errors="coerce",
)

df = df.dropna(subset=["Selling price (BDT)"])
df["Price_Crore"]    = df["Selling price (BDT)"] / 1e7
df["price_per_sqft"] = df["Selling price (BDT)"] / df["Size (sqft)"]

# =========================================================
# 1. DATASET OVERVIEW
# =========================================================
print("=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)
print(f"Rows          : {len(df)}")
print(f"Columns       : {df.shape[1]}")
print(f"\nColumn types:\n{df.dtypes}")

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDESCRIPTIVE STATISTICS")
print(df.describe())

# =========================================================
# 2. PRICE DISTRIBUTION
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)

axes[0].hist(df["Price_Crore"], bins=25, color=THESIS_BLUE,
             edgecolor="white", zorder=3)
axes[0].set_xlabel("Selling Price (BDT Crore)")
axes[0].set_ylabel("Number of Properties")
axes[0].set_title("Property Price Distribution")
axes[0].yaxis.grid(True, zorder=0)
axes[0].set_axisbelow(True)

axes[1].hist(np.log(df["Selling price (BDT)"]), bins=25,
             color=ACCENT, edgecolor="white", zorder=3)
axes[1].set_xlabel("log(Selling Price)")
axes[1].set_ylabel("Number of Properties")
axes[1].set_title("Log-Transformed Price Distribution")
axes[1].yaxis.grid(True, zorder=0)
axes[1].set_axisbelow(True)

fig.suptitle("Price Distributions", fontsize=14, fontweight="bold",
             color=TEXT_COLOR, y=1.02)
fig.tight_layout()
plt.show()

# =========================================================
# 3. PROPERTY TYPE BREAKDOWN
# =========================================================
pt_counts = df["Property Type"].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(BG)
colors_pt = [THESIS_BLUE, ACCENT, ACCENT3]

# Count bar
axes[0].bar(pt_counts.index, pt_counts.values,
            color=colors_pt, edgecolor="white", width=0.55)
for i, (idx, val) in enumerate(pt_counts.items()):
    axes[0].text(i, val + 0.5, str(val), ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color=TEXT_COLOR)
axes[0].set_ylabel("Count")
axes[0].set_title("Property Type — Count")
axes[0].yaxis.grid(True, zorder=0)
axes[0].set_axisbelow(True)

# Median price by type
pt_med = df.groupby("Property Type")["Price_Crore"].median().reindex(pt_counts.index)
axes[1].bar(pt_med.index, pt_med.values,
            color=colors_pt, edgecolor="white", width=0.55)
for i, val in enumerate(pt_med.values):
    axes[1].text(i, val + 0.05, f"{val:.2f}Cr", ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color=TEXT_COLOR)
axes[1].set_ylabel("Median Selling Price (BDT Crore)")
axes[1].set_title("Property Type — Median Price")
axes[1].yaxis.grid(True, zorder=0)
axes[1].set_axisbelow(True)

fig.tight_layout()
plt.show()

# =========================================================
# 4. SIZE vs PRICE SCATTER
# =========================================================
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG)

prop_colors = {"Apartment": THESIS_BLUE, "House": ACCENT2, "Duplex": ACCENT3}
for ptype, grp in df.groupby("Property Type"):
    ax.scatter(grp["Size (sqft)"], grp["Price_Crore"],
               label=ptype, color=prop_colors.get(ptype, ACCENT),
               edgecolors="white", s=55, linewidths=0.4, alpha=0.80, zorder=3)

ax.set_xlabel("Size (sqft)")
ax.set_ylabel("Selling Price (BDT Crore)")
ax.set_title("Property Size vs Selling Price (by Type)")
ax.legend(title="Property Type")
ax.xaxis.grid(True, zorder=0)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
plt.show()

# =========================================================
# 5. BEDROOMS vs PRICE BOXPLOT
# =========================================================
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG)

bedroom_groups = [df[df["Bedrooms"] == b]["Price_Crore"].values
                  for b in sorted(df["Bedrooms"].unique())]
bp = ax.boxplot(bedroom_groups, patch_artist=True,
                medianprops=dict(color=ACCENT2, linewidth=2))
for patch in bp["boxes"]:
    patch.set_facecolor(ACCENT)
    patch.set_alpha(0.7)

ax.set_xticklabels(sorted(df["Bedrooms"].unique()))
ax.set_xlabel("Number of Bedrooms")
ax.set_ylabel("Selling Price (BDT Crore)")
ax.set_title("Bedrooms vs Selling Price")
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
plt.show()

# =========================================================
# 6. PRICE PER SQFT BY LOCATION
# =========================================================
loc_stats = (df.groupby("Location")["price_per_sqft"]
             .agg(["mean", "median", "count"])
             .sort_values("mean", ascending=False))

# Shorten labels
loc_stats.index = loc_stats.index.str.replace(", Sylhet", "", regex=False)

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(BG)
colors_loc = [THESIS_BLUE if i == 0 else ACCENT for i in range(len(loc_stats))]
bars = ax.barh(loc_stats.index[::-1], loc_stats["mean"][::-1],
               color=colors_loc[::-1], edgecolor="white", height=0.65)
for bar, (_, row) in zip(bars, loc_stats[::-1].iterrows()):
    w = bar.get_width()
    ax.text(w + 200, bar.get_y() + bar.get_height() / 2,
            f"n={int(row['count'])}", va="center", ha="left",
            fontsize=7.5, color="#666666")
ax.set_xlabel("Average Price per sqft (BDT)")
ax.set_title("Average Price per sqft by Location (n = sample count)")
ax.xaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0, labelsize=8.5)
ax.legend(handles=[
    mpatches.Patch(color=THESIS_BLUE, label="Most expensive"),
    mpatches.Patch(color=ACCENT,      label="Other locations"),
])
fig.tight_layout()
plt.show()

# =========================================================
# 7. FLOOR NUMBER vs PRICE  (Apartments only)
# =========================================================
apts = df[df["Property Type"] == "Apartment"].copy()

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG)
ax.scatter(apts["Floor Number"], apts["Price_Crore"],
           color=THESIS_BLUE, edgecolors="white", s=55, alpha=0.80, zorder=3)
# trend line
z = np.polyfit(apts["Floor Number"], apts["Price_Crore"], 1)
p = np.poly1d(z)
xr = np.linspace(apts["Floor Number"].min(), apts["Floor Number"].max(), 100)
ax.plot(xr, p(xr), color=ACCENT2, linewidth=2, linestyle="--",
        label=f"Trend (slope={z[0]:.3f} Cr/floor)", zorder=4)
ax.set_xlabel("Floor Number")
ax.set_ylabel("Selling Price (BDT Crore)")
ax.set_title("Floor Number vs Selling Price — Apartments")
ax.legend()
ax.xaxis.grid(True, zorder=0)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
plt.show()

# =========================================================
# 8. AMENITY PRESENCE vs PRICE
# =========================================================
amenities = ["Lift", "CCTV", "Generator", "Parking", "Balcony"]

fig, axes = plt.subplots(1, len(amenities), figsize=(16, 5), sharey=True)
fig.patch.set_facecolor(BG)

for ax, amenity in zip(axes, amenities):
    groups = [df[df[amenity] == 0]["Price_Crore"].values,
              df[df[amenity] == 1]["Price_Crore"].values]
    bp = ax.boxplot(groups, patch_artist=True,
                    medianprops=dict(color=ACCENT2, linewidth=2))
    bp["boxes"][0].set_facecolor(ACCENT)
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor(THESIS_BLUE)
    bp["boxes"][1].set_alpha(0.7)
    ax.set_xticklabels(["No", "Yes"])
    ax.set_title(amenity, fontsize=11)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    if ax == axes[0]:
        ax.set_ylabel("Selling Price (BDT Crore)")

fig.suptitle("Amenity Presence vs Selling Price", fontsize=14,
             fontweight="bold", color=TEXT_COLOR, y=1.02)
fig.tight_layout()
plt.show()

# =========================================================
# 9. CORRELATION HEATMAP
# =========================================================
numeric_df = df.select_dtypes(include=np.number).drop(
    columns=["Price_Crore", "price_per_sqft"], errors="ignore"
)
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(11, 8))
fig.patch.set_facecolor(BG)
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="RdYlBu_r", center=0, linewidths=0.5, linecolor="white",
    annot_kws={"size": 8.5}, cbar_kws={"shrink": 0.75, "label": "Pearson r"},
    square=True, ax=ax,
)
ax.set_title("Feature Correlation Heatmap")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
plt.show()

# =========================================================
# 10. DATA SOURCE BREAKDOWN
# =========================================================
df["source_type"] = df["Source"].apply(
    lambda s: "Self-collected" if str(s).strip().lower() == "self"
    else ("Bikroy.com" if "bikroy" in str(s).lower()
    else ("Facebook" if "facebook" in str(s).lower()
    else ("bdhousing.com" if "bdhousing" in str(s).lower()
    else "Other")))
)

src_counts = df["source_type"].value_counts()
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor(BG)
colors_src = [THESIS_BLUE, ACCENT, ACCENT2, ACCENT3, ACCENT4]
bars = ax.bar(src_counts.index, src_counts.values,
              color=colors_src[:len(src_counts)], edgecolor="white", width=0.55)
for bar, val in zip(bars, src_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
            str(val), ha="center", va="bottom", fontsize=9,
            fontweight="bold", color=TEXT_COLOR)
ax.set_ylabel("Count")
ax.set_title("Data Collection Sources")
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis="x", rotation=15)
fig.tight_layout()
plt.show()

print("\nEDA complete.")
