# ==========================================================
# Market Basket Analysis Dashboard
# Author : Rohit Kumar Prajapati
# ==========================================================

# ==============================
# Import Libraries
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# ==============================
# Page Configuration
# ==============================

st.set_page_config(
    page_title="Market Basket Analysis Dashboard",
    page_icon="🛒",
    layout="wide"
)

# ==============================
# Title
# ==============================

st.title("🛒 Market Basket Analysis Dashboard")
st.markdown("---")

# ==============================
# Load Dataset
# ==============================

@st.cache_data
def load_data():

    instacart = pd.read_csv("instacart_processed.csv")
    feature_store = pd.read_csv("feature_store.csv")

    return instacart, feature_store

instacart, feature_store = load_data()

# ==============================
# Sidebar
# ==============================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "EDA",
        "Market Basket Analysis",
        "Recommendation System",
        "Business Insights"
    ]
)

# ==============================
# HOME PAGE
# ==============================

if page == "Home":

    st.header("Dashboard Overview")

    total_orders = instacart["order_id"].nunique()
    total_users = instacart["user_id"].nunique()
    total_products = instacart["product_name"].nunique()
    total_departments = instacart["department"].nunique()

    col1,col2,col3,col4 = st.columns(4)

    col1.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

    col2.metric(
        "Total Users",
        f"{total_users:,}"
    )

    col3.metric(
        "Total Products",
        f"{total_products:,}"
    )

    col4.metric(
        "Departments",
        f"{total_departments:,}"
    )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(instacart.head(20), use_container_width=True)

    st.markdown("---")

    st.subheader("Dataset Information")

    info = pd.DataFrame({
        "Column":instacart.columns,
        "Data Type":instacart.dtypes.astype(str)
    })

    st.dataframe(info,use_container_width=True)

    st.markdown("---")

    st.subheader("Missing Values")

    missing = instacart.isnull().sum()

    missing = missing[missing>0]

    if len(missing)==0:

        st.success("No Missing Values Found")

    else:

        st.dataframe(
            missing.reset_index(),
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("Statistical Summary")

    st.dataframe(
        instacart.describe(),
        use_container_width=True
    )

# ==========================================================
# EDA PAGE
# ==========================================================

elif page == "EDA":

    st.header("📊 Exploratory Data Analysis")

    st.markdown("### Dataset Filters")

    col1, col2 = st.columns(2)

    with col1:
        department = st.selectbox(
            "Select Department",
            ["All"] + sorted(instacart["department"].dropna().unique().tolist())
        )

    with col2:
        aisle = st.selectbox(
            "Select Aisle",
            ["All"] + sorted(instacart["aisle"].dropna().unique().tolist())
        )

    # -------------------------
    # Apply Filters
    # -------------------------

    filtered_df = instacart.copy()

    if department != "All":
        filtered_df = filtered_df[
            filtered_df["department"] == department
        ]

    if aisle != "All":
        filtered_df = filtered_df[
            filtered_df["aisle"] == aisle
        ]

    # -------------------------
    # KPI Cards
    # -------------------------

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Orders",
        filtered_df["order_id"].nunique()
    )

    c2.metric(
        "Users",
        filtered_df["user_id"].nunique()
    )

    c3.metric(
        "Products",
        filtered_df["product_name"].nunique()
    )

    c4.metric(
        "Departments",
        filtered_df["department"].nunique()
    )

    # -------------------------
    # Top Products
    # -------------------------

    st.markdown("---")

    st.subheader("Top 20 Products")

    top_products = (
        filtered_df["product_name"]
        .value_counts()
        .head(20)
    )

    fig, ax = plt.subplots(figsize=(10,6))

    top_products.sort_values().plot(
        kind="barh",
        ax=ax
    )

    ax.set_xlabel("Orders")
    ax.set_ylabel("Products")

    st.pyplot(fig)

    # -------------------------
    # Top Departments
    # -------------------------

    st.markdown("---")

    st.subheader("Top Departments")

    fig, ax = plt.subplots(figsize=(8,5))

    filtered_df["department"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Orders")

    st.pyplot(fig)

    # -------------------------
    # Top Aisles
    # -------------------------

    st.markdown("---")

    st.subheader("Top 20 Aisles")

    fig, ax = plt.subplots(figsize=(10,6))

    filtered_df["aisle"].value_counts().head(20).sort_values().plot(
        kind="barh",
        ax=ax
    )

    st.pyplot(fig)

    # -------------------------
    # Orders by Hour
    # -------------------------

    if "order_hour_of_day" in filtered_df.columns:

        st.markdown("---")

        st.subheader("Orders by Hour")

        fig, ax = plt.subplots(figsize=(10,4))

        sns.countplot(
            data=filtered_df,
            x="order_hour_of_day",
            ax=ax
        )

        st.pyplot(fig)

    # -------------------------
    # Orders by Day
    # -------------------------

    if "order_dow" in filtered_df.columns:

        st.markdown("---")

        st.subheader("Orders by Day")

        fig, ax = plt.subplots(figsize=(8,4))

        sns.countplot(
            data=filtered_df,
            x="order_dow",
            ax=ax
        )

        st.pyplot(fig)

    # -------------------------
    # Basket Size
    # -------------------------

    st.markdown("---")

    st.subheader("Basket Size Distribution")

    basket = filtered_df.groupby("order_id")["product_name"].count()

    fig, ax = plt.subplots(figsize=(10,5))

    sns.histplot(
        basket,
        bins=30,
        ax=ax
    )

    ax.set_xlabel("Products per Order")

    st.pyplot(fig)

    # -------------------------
    # Reordered Products
    # -------------------------

    if "reordered" in filtered_df.columns:

        st.markdown("---")

        st.subheader("Reordered Products")

        fig, ax = plt.subplots(figsize=(6,4))

        sns.countplot(
            data=filtered_df,
            x="reordered",
            ax=ax
        )

        st.pyplot(fig)

    # -------------------------
    # Feature Store
    # -------------------------

    st.markdown("---")

    st.subheader("Feature Store Preview")

    st.dataframe(
        feature_store.head(20),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Correlation Heatmap")

    numeric = feature_store.select_dtypes(include="number")

    if numeric.shape[1] > 1:

        fig, ax = plt.subplots(figsize=(10,8))

        sns.heatmap(
            numeric.corr(),
            cmap="coolwarm",
            annot=False,
            ax=ax
        )

        st.pyplot(fig)

    # -------------------------
    # Download Dataset
    # -------------------------

    st.markdown("---")

    st.download_button(
        "📥 Download Filtered Dataset",
        filtered_df.to_csv(index=False),
        file_name="filtered_dataset.csv",
        mime="text/csv"
    )

# ==========================================================
# MARKET BASKET ANALYSIS PAGE
# ==========================================================

elif page == "Market Basket Analysis":

    st.header("🛒 Market Basket Analysis")

    st.markdown("### Apriori Algorithm")

    top_products = (
        instacart["product_name"]
        .value_counts()
        .head(200)
        .index
    )

    filtered = instacart[
        instacart["product_name"].isin(top_products)
    ]

    basket = pd.crosstab(
        filtered["order_id"],
        filtered["product_name"]
    ).astype(bool)

    st.success("Transaction Matrix Created Successfully")

    filtered = instacart[
        instacart["product_name"].isin(top_products)
    ]

    basket = pd.crosstab(
        filtered["order_id"],
        filtered["product_name"]
    ).astype(bool)

    st.success("Transaction Matrix Created Successfully")

    st.write("Shape :", basket.shape)

    st.dataframe(
        basket.head(),
        width="stretch"
    )

    st.write("Shape :", basket.shape)

    st.dataframe(
        basket.head(),
        use_container_width=True
    )

    # ----------------------------------------
    # Frequent Itemsets
    # ----------------------------------------

    st.markdown("---")

    st.subheader("Frequent Itemsets")

    with st.spinner("Running Apriori..."):

        frequent_itemsets = apriori(
            basket,
            min_support=min_support,
            use_colnames=True
        )

        frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)

    st.success(f"{len(frequent_itemsets)} Frequent Itemsets Found")

    st.dataframe(
        frequent_itemsets.sort_values(
            "support",
            ascending=False
        ).head(30),
        use_container_width=True
    )

    # ----------------------------------------
    # Frequent Itemset Size
    # ----------------------------------------

    st.markdown("---")

    st.subheader("Itemset Size Distribution")

    fig, ax = plt.subplots(figsize=(8,4))

    sns.countplot(
        data=frequent_itemsets,
        x="length",
        ax=ax
    )

    ax.set_xlabel("Itemset Length")

    st.pyplot(fig)

    # ----------------------------------------
    # Association Rules
    # ----------------------------------------

    st.markdown("---")

    st.subheader("Association Rules")

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    rules = rules[
        rules["lift"] >= min_lift
    ]

    if len(rules)==0:

        st.warning("No Rules Found.")

    else:

        rules["Antecedent"] = rules["antecedents"].apply(
            lambda x:", ".join(list(x))
        )

        rules["Consequent"] = rules["consequents"].apply(
            lambda x:", ".join(list(x))
        )

        st.success(f"{len(rules)} Rules Generated")

        st.dataframe(

            rules[
                [
                    "Antecedent",
                    "Consequent",
                    "support",
                    "confidence",
                    "lift"
                ]
            ].sort_values(
                "lift",
                ascending=False
            ),

            use_container_width=True

        )

    # ----------------------------------------
    # Support Distribution
    # ----------------------------------------

    st.markdown("---")

    st.subheader("Support Distribution")

    fig, ax = plt.subplots(figsize=(8,4))

    sns.histplot(
        rules["support"],
        bins=20,
        ax=ax
    )

    st.pyplot(fig)

    # ----------------------------------------
    # Confidence Distribution
    # ----------------------------------------

    st.markdown("---")

    st.subheader("Confidence Distribution")

    fig, ax = plt.subplots(figsize=(8,4))

    sns.histplot(
        rules["confidence"],
        bins=20,
        ax=ax
    )

    st.pyplot(fig)

    # ----------------------------------------
    # Lift Distribution
    # ----------------------------------------

    st.markdown("---")

    st.subheader("Lift Distribution")

    fig, ax = plt.subplots(figsize=(8,4))

    sns.histplot(
        rules["lift"],
        bins=20,
        ax=ax
    )

    st.pyplot(fig)

    # ----------------------------------------
    # Scatter Plot
    # ----------------------------------------

    st.markdown("---")

    st.subheader("Support vs Confidence")

    fig, ax = plt.subplots(figsize=(10,6))

    sns.scatterplot(
        data=rules,
        x="support",
        y="confidence",
        size="lift",
        hue="lift",
        ax=ax
    )

    st.pyplot(fig)

    # ----------------------------------------
    # Strong Rules
    # ----------------------------------------

    st.markdown("---")

    st.subheader("Strong Association Rules")

    strong_rules = rules[
    (rules["confidence"] >= 0.60) &
    (rules["lift"] >= 2)
    ]

    if strong_rules.empty:
        st.warning("No Strong Association Rules Found")
    else:
        st.dataframe(
        strong_rules[
            [
                "Antecedent",
                "Consequent",
                "support",
                "confidence",
                "lift"
            ]
        ],
        use_container_width=True
    )
    # ----------------------------------------
    # Download Rules
    # ----------------------------------------

    st.markdown("---")

    csv = rules.to_csv(index=False)

    st.download_button(
        label="📥 Download Association Rules",
        data=csv,
        file_name="association_rules.csv",
        mime="text/csv"
    )

    # ----------------------------------------
    # Summary
    # ----------------------------------------

    st.markdown("---")

    col1,col2,col3 = st.columns(3)

    col1.metric(
        "Itemsets",
        len(frequent_itemsets)
    )

    col2.metric(
        "Rules",
        len(rules)
    )

    col3.metric(
        "Strong Rules",
        len(strong_rules)
    )

# ==========================================================
# RECOMMENDATION SYSTEM PAGE
# ==========================================================

elif page == "Recommendation System":

    st.header("🎯 Product Recommendation System")

    top_products = (
        instacart["product_name"]
        .value_counts()
        .head(200)
        .index
    )

    filtered = instacart[
        instacart["product_name"].isin(top_products)
    ]

    basket = pd.crosstab(
        filtered["order_id"],
        filtered["product_name"]
    ).astype(bool)

    # Frequent Itemsets
    frequent_itemsets = apriori(
        basket,
        min_support=0.01,
        use_colnames=True
    )

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.30
    )

    rules["Antecedent"] = rules["antecedents"].apply(
        lambda x: ", ".join(list(x))
    )

    rules["Consequent"] = rules["consequents"].apply(
        lambda x: ", ".join(list(x))
    )

    # Product Selection
    product = st.selectbox(
        "Select Product",
        sorted(instacart["product_name"].unique())
    )

    recommendation = rules[
        rules["Antecedent"].str.contains(product, case=False)
    ]

    if recommendation.empty:

        st.warning("No Recommendation Found")

    else:

        recommendation = recommendation.sort_values(
            by=["lift","confidence"],
            ascending=False
        )

        st.success(f"Top Recommendations for '{product}'")

        st.dataframe(

            recommendation[
                [
                    "Consequent",
                    "support",
                    "confidence",
                    "lift"
                ]
            ].head(10),

            use_container_width=True

        )

        # Recommendation Chart
        fig, ax = plt.subplots(figsize=(10,5))

        top10 = recommendation.head(10)

        ax.barh(
            top10["Consequent"],
            top10["lift"]
        )

        ax.set_xlabel("Lift")

        st.pyplot(fig)

# ==========================================================
# BUSINESS INSIGHTS PAGE
# ==========================================================

elif page == "Business Insights":

    st.header("📈 Business Insights")

    top_products = (
        instacart["product_name"]
        .value_counts()
        .head(200)
        .index
    )

    filtered = instacart[
        instacart["product_name"].isin(top_products)
    ]

    basket = pd.crosstab(
        filtered["order_id"],
        filtered["product_name"]
    ).astype(bool)
    frequent_itemsets = apriori(
        basket,
        min_support=0.01,
        use_colnames=True
    )

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.30
    )

    rules["Antecedent"] = rules["antecedents"].apply(
        lambda x:", ".join(list(x))
    )

    rules["Consequent"] = rules["consequents"].apply(
        lambda x:", ".join(list(x))
    )

    st.subheader("Top 10 Cross Selling Opportunities")

    top_rules = rules.sort_values(
        by="lift",
        ascending=False
    )

    st.dataframe(

        top_rules[
            [
                "Antecedent",
                "Consequent",
                "support",
                "confidence",
                "lift"
            ]
        ].head(10),

        use_container_width=True

    )

    st.markdown("---")

    st.subheader("Business Recommendations")

    st.success("""
✔ Place products with high Lift near each other.

✔ Create combo offers using frequently bought products.

✔ Recommend associated products during checkout.

✔ Keep high-support products well stocked.

✔ Use recommendation engine for online shopping.

✔ Improve store layout using association rules.

✔ Increase Average Order Value through bundles.

✔ Design personalized marketing campaigns.
""")

    st.markdown("---")

    st.subheader("Project Summary")

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "Transactions",
        instacart["order_id"].nunique()
    )

    c2.metric(
        "Products",
        instacart["product_name"].nunique()
    )

    c3.metric(
        "Association Rules",
        len(rules)
    )

    st.markdown("---")

    st.info("""
This dashboard analyzes customer purchasing behavior using the Apriori Algorithm.
The generated association rules help identify products that are frequently bought together,
supporting product recommendations, cross-selling, inventory optimization, and shelf placement.
""")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
<center>

### 🛒 Market Basket Analysis Dashboard

Developed using **Python | Pandas | Streamlit | Mlxtend | Matplotlib**

**Author:** Rohit Kumar Prajapati

</center>
""",
unsafe_allow_html=True
)









