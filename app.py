# AI CAMPAIGN RECOMMENDATION ENGINE
# ============================================================

import pandas as pd
import numpy as np

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("nykaa_campaign.csv")

print("="*80)
print("DATASET LOADED SUCCESSFULLY")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")
print("="*80)

# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_cols = [
    "Campaign_Type",
    "Target_Audience",
    "Language",
    "Customer_Segment",
    "Channel_Used",
    "ROI",
    "Revenue",
    "Acquisition_Cost"
]

missing_cols = [c for c in required_cols if c not in df.columns]

if len(missing_cols) > 0:
    raise ValueError(f"Missing Columns: {missing_cols}")

# ============================================================
# CLEAN DATA
# ============================================================

df.columns = df.columns.str.strip()

for col in [
    "Campaign_Type",
    "Target_Audience",
    "Language",
    "Customer_Segment",
    "Channel_Used"
]:
    df[col] = df[col].astype(str).str.strip()

# ============================================================
# SUCCESS SCORE
# ============================================================

df["ROI_Norm"] = (
    (df["ROI"] - df["ROI"].min())
    /
    (df["ROI"].max() - df["ROI"].min())
)

df["Revenue_Norm"] = (
    (df["Revenue"] - df["Revenue"].min())
    /
    (df["Revenue"].max() - df["Revenue"].min())
)

df["Success_Score"] = (
    0.7 * df["ROI_Norm"]
    +
    0.3 * df["Revenue_Norm"]
)

# ============================================================
# BUILD RECOMMENDATION TABLE
# ============================================================

recommendations = (

    df.groupby(
        [
            "Campaign_Type",
            "Target_Audience",
            "Language",
            "Customer_Segment",
            "Channel_Used"
        ]
    )

    .agg(
        Avg_ROI=("ROI","mean"),
        Avg_Revenue=("Revenue","mean"),
        Avg_Cost=("Acquisition_Cost","mean"),
        Avg_Success=("Success_Score","mean"),
        Campaign_Count=("Campaign_Type","count")
    )

    .reset_index()

)

recommendations["Frequency_Score"] = (

    recommendations["Campaign_Count"]
    /
    recommendations["Campaign_Count"].max()

)

recommendations["Final_Score"] = (

    0.50 * recommendations["Avg_Success"]

    +

    0.30 * recommendations["Frequency_Score"]

    +

    0.20 * (
        recommendations["Avg_ROI"]
        /
        recommendations["Avg_ROI"].max()
    )

)

recommendations = recommendations.sort_values(
    "Final_Score",
    ascending=False
)

print(f"\nUnique Campaign Patterns: {len(recommendations):,}")

# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def recommend_campaign(
    audience=None,
    language=None,
    budget=None,
    top_n=10
):

    temp = recommendations.copy()

    # Exact Match

    filtered = temp.copy()

    if audience:
        filtered = filtered[
            filtered["Target_Audience"]
            .str.lower()
            ==
            audience.lower()
        ]

    if language:
        filtered = filtered[
            filtered["Language"]
            .str.lower()
            ==
            language.lower()
        ]

    if budget:
        filtered = filtered[
            filtered["Avg_Cost"] <= budget
        ]

    if len(filtered) > 0:
        return filtered.head(top_n)

    # Relax Budget

    filtered = temp.copy()

    if audience:
        filtered = filtered[
            filtered["Target_Audience"]
            .str.lower()
            ==
            audience.lower()
        ]

    if language:
        filtered = filtered[
            filtered["Language"]
            .str.lower()
            ==
            language.lower()
        ]

    if len(filtered) > 0:
        print("\nNo budget match found. Showing best campaigns for selected audience and language.")
        return filtered.head(top_n)

    # Relax Language

    filtered = temp.copy()

    if audience:
        filtered = filtered[
            filtered["Target_Audience"]
            .str.lower()
            ==
            audience.lower()
        ]

    if len(filtered) > 0:
        print("\nNo language match found. Showing best campaigns for selected audience.")
        return filtered.head(top_n)

    # Show Overall Best

    print("\nShowing overall best campaigns from historical data.")

    return temp.head(top_n)

# ============================================================
# NLP BUSINESS INSIGHT
# ============================================================

def generate_recommendation(result):

    if len(result) == 0:
        print("No recommendation available.")
        return

    best = result.iloc[0]

    campaign = best["Campaign_Type"]
    audience = best["Target_Audience"]
    language = best["Language"]
    segment = best["Customer_Segment"]
    channel = best["Channel_Used"]

    roi = best["Avg_ROI"]
    revenue = best["Avg_Revenue"]
    cost = best["Avg_Cost"]

    if roi >= 8:
        roi_text = "exceptional"
    elif roi >= 5:
        roi_text = "very strong"
    elif roi >= 3:
        roi_text = "strong"
    else:
        roi_text = "moderate"

    print("\n")
    print("="*80)
    print("AI GENERATED CAMPAIGN INSIGHT")
    print("="*80)

    print(f"""

EXECUTIVE SUMMARY
-----------------

Based on historical campaign performance, the most promising marketing
strategy is a {campaign} campaign targeting {audience} customers.

The recommended communication language is {language}
and the ideal customer segment is {segment}.

The most effective marketing channel combination observed in the data is:

{channel}

PERFORMANCE EXPECTATION
-----------------------

Expected ROI            : {roi:.2f}

Expected Revenue        : Rs {revenue:,.0f}

Expected Campaign Cost  : Rs {cost:,.0f}

BUSINESS INTERPRETATION
-----------------------

This campaign configuration demonstrates {roi_text} ROI performance
compared with other campaign combinations in the dataset.

Historical results suggest that customers belonging to the
{audience} audience group engage positively with
{campaign} campaigns delivered through {channel}.

RECOMMENDED ACTIONS
-------------------

1. Prioritize this campaign setup for future campaigns.

2. Allocate approximately Rs {cost:,.0f}
   as campaign budget.

3. Use {language} creatives and messaging.

4. Focus primarily on the {segment} customer segment.

5. Monitor ROI and revenue after launch to
   validate expected performance.

CONCLUSION
----------

If a campaign must be launched today,
this configuration represents one of the strongest
opportunities for revenue generation and ROI
based on historical performance.

""")

# ============================================================
# TOP 3 EXPLANATIONS
# ============================================================

def explain_top3(result):

    if len(result) == 0:
        return

    print("\n")
    print("="*80)
    print("TOP 3 RECOMMENDED CAMPAIGNS")
    print("="*80)

    top3 = result.head(3)

    for rank, row in enumerate(top3.itertuples(), start=1):

        print(f"""

Recommendation #{rank}

Campaign Type     : {row.Campaign_Type}
Audience          : {row.Target_Audience}
Language          : {row.Language}
Segment           : {row.Customer_Segment}
Channel           : {row.Channel_Used}

Expected ROI      : {row.Avg_ROI:.2f}
Expected Revenue  : Rs {row.Avg_Revenue:,.0f}
Average Cost      : Rs {row.Avg_Cost:,.0f}

Why Recommended?
----------------

This campaign configuration consistently ranks among
the strongest performers in the historical dataset.

It combines strong ROI, revenue generation,
and campaign consistency.

--------------------------------------------------------
""")

# ============================================================
# USER INPUTS
# ============================================================

AUDIENCE = "Youth"
LANGUAGE = "Hindi"
BUDGET = 500

# Examples:
# AUDIENCE = None
# LANGUAGE = None
# BUDGET = None

# ============================================================
# RUN ENGINE
# ============================================================

result = recommend_campaign(
    audience=AUDIENCE,
    language=LANGUAGE,
    budget=BUDGET,
    top_n=10
)

print("\n")
print("="*80)
print("TOP RECOMMENDATIONS")
print("="*80)

print(result)

generate_recommendation(result)



# SMART VISUAL INSIGHTS
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("default")

# ------------------------------------------------------------
# 1. TOP 10 CAMPAIGN TYPES BY ROI
# ------------------------------------------------------------

plt.figure(figsize=(10,6))

campaign_roi = (
    df.groupby("Campaign_Type")["ROI"]
      .mean()
      .sort_values(ascending=False)
)

campaign_roi.plot(kind="bar")

plt.title("Average ROI by Campaign Type")
plt.ylabel("Average ROI")
plt.xlabel("Campaign Type")
plt.xticks(rotation=45)

plt.show()

# ------------------------------------------------------------
# 2. TOP 10 CHANNELS BY REVENUE
# ------------------------------------------------------------

plt.figure(figsize=(12,6))

channel_revenue = (
    df.groupby("Channel_Used")["Revenue"]
      .mean()
      .sort_values(ascending=False)
      .head(10)
)

channel_revenue.plot(kind="bar")

plt.title("Top Marketing Channels by Revenue")
plt.ylabel("Average Revenue")
plt.xlabel("Channel")
plt.xticks(rotation=45)

plt.show()

# ------------------------------------------------------------
# 3. CUSTOMER SEGMENT PERFORMANCE
# ------------------------------------------------------------

plt.figure(figsize=(10,6))

segment_roi = (
    df.groupby("Customer_Segment")["ROI"]
      .mean()
      .sort_values(ascending=False)
)

segment_roi.plot(kind="bar")

plt.title("ROI by Customer Segment")
plt.ylabel("Average ROI")
plt.xlabel("Customer Segment")

plt.xticks(rotation=45)

plt.show()

# ------------------------------------------------------------
# 4. AUDIENCE VS ROI HEATMAP
# ------------------------------------------------------------

pivot_table = pd.pivot_table(
    df,
    values="ROI",
    index="Target_Audience",
    columns="Campaign_Type",
    aggfunc="mean"
)

plt.figure(figsize=(12,6))

sns.heatmap(
    pivot_table,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu"
)

plt.title("Audience vs Campaign Type ROI Heatmap")

plt.show()

# ------------------------------------------------------------
# 5. COST VS REVENUE RELATIONSHIP
# ------------------------------------------------------------

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df,
    x="Acquisition_Cost",
    y="Revenue"
)

plt.title("Campaign Cost vs Revenue")

plt.show()

# ------------------------------------------------------------
# 6. ROI DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(10,6))

sns.histplot(
    df["ROI"],
    kde=True
)

plt.title("ROI Distribution")

plt.show()

# ------------------------------------------------------------
# 7. TOP 10 RECOMMENDED CAMPAIGNS
# ------------------------------------------------------------

top10 = recommendations.head(10)

plt.figure(figsize=(12,6))

sns.barplot(
    data=top10,
    x="Final_Score",
    y="Campaign_Type"
)

plt.title("Top Recommended Campaigns")
plt.xlabel("Recommendation Score")

plt.show()

# ------------------------------------------------------------
# 8. REVENUE BY LANGUAGE
# ------------------------------------------------------------

plt.figure(figsize=(10,6))

language_rev = (
    df.groupby("Language")["Revenue"]
      .mean()
      .sort_values(ascending=False)
)

language_rev.plot(kind="bar")

plt.title("Average Revenue by Language")
plt.ylabel("Revenue")

plt.show()

# ------------------------------------------------------------
# 9. ROI VS REVENUE OF TOP RECOMMENDATIONS
# ------------------------------------------------------------

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=top10,
    x="Avg_ROI",
    y="Avg_Revenue",
    size="Final_Score",
    legend=False
)

plt.title("ROI vs Revenue (Top Recommendations)")

plt.show()

# ------------------------------------------------------------
# 10. EXECUTIVE DASHBOARD SUMMARY
# ------------------------------------------------------------

print("\n")
print("="*80)
print("EXECUTIVE INSIGHTS")
print("="*80)

best_campaign = (
    df.groupby("Campaign_Type")["ROI"]
      .mean()
      .idxmax()
)

best_channel = (
    df.groupby("Channel_Used")["Revenue"]
      .mean()
      .idxmax()
)

best_segment = (
    df.groupby("Customer_Segment")["ROI"]
      .mean()
      .idxmax()
)

best_language = (
    df.groupby("Language")["Revenue"]
      .mean()
      .idxmax()
)

print(f"""
BEST CAMPAIGN TYPE
------------------
{best_campaign}

BEST MARKETING CHANNEL
----------------------
{best_channel}

BEST CUSTOMER SEGMENT
---------------------
{best_segment}

BEST LANGUAGE
-------------
{best_language}

BUSINESS INSIGHT
----------------
Historical campaign data suggests that
'{best_campaign}' campaigns delivered the
highest ROI.

The channel '{best_channel}' generated the
highest average revenue.

The customer segment '{best_segment}'
responded most positively to campaigns.

Using '{best_language}' language creatives
resulted in the strongest revenue performance.
""")







explain_top3(result)
