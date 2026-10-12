"""
Amazon Canada Big Data Dashboard
Run:  streamlit run app.py
Reads amazon_mapreduce_results.txt (same folder) or an uploaded copy.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import re

# ---------------------------------------------------------------- parser
def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _rows(lines, pattern, cols, sub=None):
    out = []
    rx = re.compile(pattern)
    for s, line in lines:
        if sub is not None and s != sub:
            continue
        m = rx.match(line)
        if m:
            out.append([m.group(1).strip()] + [_num(g) for g in m.groups()[1:]])
    return pd.DataFrame(out, columns=cols)


def parse_results(text: str) -> dict:
    sections, cur, sub = {}, None, None
    for line in text.splitlines():
        m = re.match(r"^\s*([1-5]) · ", line)
        if m:
            cur, sub = int(m.group(1)), None
            sections[cur] = []
            continue
        if cur is None:
            continue
        if cur == 3:
            if "By Category" in line:
                sub = "category"
                continue
            if "By Price Range" in line:
                sub = "price"
                continue
        sections[cur].append((sub, line))

    ratings = _rows(sections.get(1, []), r"^\s*(.+?)\s{2,}(\d+\.\d+)\s*★*\s*$",
                    ["category", "avg_rating"])
    price = _rows(sections.get(2, []), r"^\s*(\S+)\s+([\d,]+)\s+([\d.]+)%",
                  ["segment", "count", "pct"])
    bs_cat = _rows(sections.get(3, []), r"^\s*(.+?)\s{2,}(\d[\d,]*)\s*$",
                   ["category", "bestsellers"], sub="category")
    bs_price = _rows(sections.get(3, []), r"^\s*(\S+)\s+([\d,]+)\s*$",
                     ["segment", "bestsellers"], sub="price")
    demand = _rows(sections.get(4, []), r"^\s*(.+?)\s{2,}(\d[\d,]*)\s*$",
                   ["category", "demand"])
    pop = _rows(sections.get(5, []),
                r"^\s*(.+?)\s{2,}(\d[\d,]*\.?\d*)\s+\$\s*(-?[\d,]+\.?\d*)\s+(\d[\d,]*)\s*$",
                ["category", "avg_popularity", "avg_price_diff", "products"])

    cats = pop.copy()
    for part in (ratings, bs_cat, demand):
        cats = cats.merge(part, on="category", how="outer")
    cats["bestsellers"] = cats["bestsellers"].fillna(0)
    cats["demand"] = cats["demand"].fillna(0)
    cats["bestseller_rate"] = cats["bestsellers"] / cats["products"] * 100
    cats["demand_per_product"] = cats["demand"] / cats["products"]

    return {"categories": cats, "price": price, "bs_price": bs_price}


RESULTS_FILE = "amazon_mapreduce_results.txt"
TEAL, ORANGE = "#2a9d8f", "#e76f51"

st.set_page_config(page_title="Amazon Canada · Big Data Dashboard", page_icon="📦", layout="wide")


@st.cache_data(show_spinner=False)
def load(text: str):
    return parse_results(text)


def hbar(d, x, title, color=TEAL, fmt=None):
    d = d.sort_values(x)
    fig = px.bar(d, x=x, y="category", orientation="h", title=title,
                 color_discrete_sequence=[color], height=max(380, 26 * len(d) + 120))
    fig.update_layout(yaxis_title=None, xaxis_title=None, margin=dict(l=0, r=0, t=50, b=0))
    if fmt:
        fig.update_traces(hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>")
    return fig


# ---------------------------------------------------------------- load data
st.sidebar.title("📦 Data")
up = st.sidebar.file_uploader("MapReduce results (.txt)", type=["txt"])
if up:
    text = up.getvalue().decode("utf-8", errors="ignore")
elif Path(RESULTS_FILE).exists():
    text = Path(RESULTS_FILE).read_text(encoding="utf-8", errors="ignore")
else:
    st.title("Amazon Canada Product Analytics")
    st.info(f"Put `{RESULTS_FILE}` next to `app.py`, or upload it in the sidebar.")
    st.stop()

data = load(text)
cats, price, bs_price = data["categories"], data["price"], data["bs_price"]
if cats.empty or price.empty:
    st.error("Couldn't parse the results file. Is it the full 5-section report?")
    st.stop()

st.sidebar.title("🔎 Filters")
min_products = st.sidebar.slider(
    "Min products per category", 0, 5000, 500, 100,
    help="Hides tiny categories so averages and rates aren't driven by a handful of items.")
big = cats[cats["products"] >= min_products]

# ---------------------------------------------------------------- derived
total_products = int(price["count"].sum())
total_units = int(cats["demand"].sum())
total_bs = int(cats["bestsellers"].sum())
bs_share = total_bs / total_products * 100

price = price.copy()
price["segment"] = pd.Categorical(price["segment"], price["segment"].tolist(), ordered=True)
ps = price.merge(bs_price, on="segment", how="left")
ps["bestseller_rate"] = ps["bestsellers"] / ps["count"] * 100

dem = cats.sort_values("demand", ascending=False).reset_index(drop=True)
dem["cum_share"] = dem["demand"].cumsum() / dem["demand"].sum() * 100
dem["share"] = dem["demand"] / dem["demand"].sum() * 100

# ---------------------------------------------------------------- header
st.title("📦 Amazon Canada Product Analytics")
st.caption("HDFS → MapReduce → Hive → PySpark MLlib → Python dashboard · "
           f"{total_products:,} products across {len(cats)} categories")

tabs = st.tabs(["Overview", "Ratings", "Price", "Bestsellers", "Demand",
                "Popularity vs price diff", "ML model", "Insights"])

# ---------------------------------------------------------------- Overview
with tabs[0]:
    k = st.columns(5)
    k[0].metric("Products", f"{total_products:,}")
    k[1].metric("Categories", f"{len(cats)}")
    k[2].metric("Units bought (last month)", f"{total_units:,}")
    k[3].metric("Bestsellers", f"{total_bs:,}")
    k[4].metric("Bestseller share", f"{bs_share:.2f}%")
    c1, c2 = st.columns(2)
    c1.plotly_chart(hbar(dem.head(10), "demand", "Top 10 categories by demand"),
                    use_container_width=True)
    fig = px.bar(price, x="segment", y="count", title="Products by price segment",
                 color_discrete_sequence=[TEAL], text="pct")
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(xaxis_title=None, yaxis_title=None, margin=dict(t=50, b=0))
    c2.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- Ratings
with tabs[1]:
    st.subheader("Category-wise average rating (Hive query 1)")
    st.info("Averages include products with 0 stars (unrated), so categories with very low "
            "averages are mostly *unrated* rather than *badly rated*.")
    c1, c2 = st.columns([1, 1])
    side = c1.radio("Show", ["Highest rated", "Lowest rated"], horizontal=True)
    n = c2.slider("Categories", 5, 40, 15, key="n_rating")
    r = big.dropna(subset=["avg_rating"])
    r = r.nlargest(n, "avg_rating") if side == "Highest rated" else r.nsmallest(n, "avg_rating")
    st.plotly_chart(hbar(r, "avg_rating", side), use_container_width=True)
    st.plotly_chart(px.histogram(cats.dropna(subset=["avg_rating"]), x="avg_rating", nbins=40,
                                 title="How category averages are spread",
                                 color_discrete_sequence=[TEAL]).update_layout(
                        yaxis_title="categories", xaxis_title="average rating"),
                    use_container_width=True)

# ---------------------------------------------------------------- Price
with tabs[2]:
    st.subheader("Price distribution & bestseller likelihood (Hive query 2)")
    c1, c2 = st.columns(2)
    fig = px.bar(ps, x="segment", y="count", title="Products per price segment",
                 color_discrete_sequence=[TEAL])
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    c1.plotly_chart(fig, use_container_width=True)
    fig = px.bar(ps, x="segment", y="bestseller_rate",
                 title="Bestseller rate by price segment (% of products)",
                 color_discrete_sequence=[ORANGE], text="bestseller_rate")
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    c2.plotly_chart(fig, use_container_width=True)
    best = ps.loc[ps["bestseller_rate"].idxmax()]
    st.markdown(f"Products priced **{best['segment']}** are most likely to be bestsellers "
                f"({best['bestseller_rate']:.2f}%); the rate falls off sharply above $100.")
    st.dataframe(ps.rename(columns={"count": "products"}).round(2),
                 hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- Bestsellers
with tabs[3]:
    st.subheader("Bestseller trends (Hive query 3)")
    mode = st.radio("Rank categories by", ["Bestseller count", "Bestseller rate (% of category)"],
                    horizontal=True)
    n = st.slider("Categories", 5, 40, 15, key="n_bs")
    if mode == "Bestseller count":
        st.plotly_chart(hbar(cats.nlargest(n, "bestsellers"), "bestsellers", mode),
                        use_container_width=True)
    else:
        st.plotly_chart(hbar(big.nlargest(n, "bestseller_rate"), "bestseller_rate", mode, ORANGE),
                        use_container_width=True)
        st.caption("Rate = bestsellers ÷ products in the category (filtered by the sidebar minimum).")

# ---------------------------------------------------------------- Demand
with tabs[4]:
    st.subheader("Demand analysis (MapReduce: total boughtInLastMonth per category)")
    n = st.slider("Categories", 5, 40, 15, key="n_dem")
    c1, c2 = st.columns(2)
    c1.plotly_chart(hbar(dem.head(n), "demand", "Units bought, last month"),
                    use_container_width=True)
    pareto = px.line(dem.reset_index(), x="index", y="cum_share", markers=False,
                     title="Cumulative share of demand (categories ranked by demand)")
    pareto.update_layout(xaxis_title="category rank", yaxis_title="% of all units")
    c2.plotly_chart(pareto, use_container_width=True)
    top5 = dem.head(5)["share"].sum()
    top10 = dem.head(10)["share"].sum()
    st.markdown(f"The **top 5 categories drive {top5:.0f}%** of demand and the **top 10 drive "
                f"{top10:.0f}%**; {int((cats['demand'] == 0).sum())} categories recorded no purchases.")
    st.plotly_chart(hbar(big.nlargest(n, "demand_per_product"), "demand_per_product",
                         "Demand per product (units ÷ products)", ORANGE),
                    use_container_width=True)

# ---------------------------------------------------------------- Popularity vs price diff
with tabs[5]:
    st.subheader("Popularity score vs price difference (Hive query 5)")
    sc = big[big["avg_popularity"] > 0]
    fig = px.scatter(sc, x="avg_price_diff", y="avg_popularity", size="products",
                     hover_name="category", log_y=True, opacity=0.65,
                     color_discrete_sequence=[TEAL], size_max=30,
                     title="Each bubble is a category (size = number of products)")
    fig.update_layout(xaxis_title="avg price diff (listPrice − price)",
                      yaxis_title="avg popularity score (log)")
    st.plotly_chart(fig, use_container_width=True)
    st.warning("Average price differences are **negative** in nearly every category, which "
               "shouldn't happen if `listPrice` was filled with `price` as in `clean.py` "
               "(that gives a difference of 0). It suggests the MapReduce job read `listPrice = 0` "
               "for most products. Check this before drawing conclusions from this chart.")
    st.dataframe(big.nlargest(15, "avg_popularity")[
        ["category", "avg_popularity", "avg_price_diff", "products"]].round(2),
        hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- ML
with tabs[6]:
    st.subheader("Bestseller prediction (PySpark MLlib Random Forest)")
    c1, c2 = st.columns(2)
    c1.markdown("""**Pipeline** (from `pyspark.py`)
- Imputer (median) → VectorAssembler → StandardScaler → RandomForestClassifier
- 100 trees, max depth 6, seed 42
- Train / test split: 80 / 20
- Label: `isBestSeller`""")
    c2.markdown("""**Features**
- `stars`
- `reviews`
- `price`
- `boughtInLastMonth`
- `discount_percent`""")
    st.warning(
        f"**Class imbalance:** only {bs_share:.2f}% of products are bestsellers, so always "
        f"predicting \"not a bestseller\" scores about {100 - bs_share:.2f}% accuracy. Report "
        "**AUC, and precision/recall/F1 for the bestseller class**, not accuracy alone. "
        "The model has no class weighting, so recall on bestsellers may be low.")
    st.caption("Your README lists category as a feature, but `pyspark.py` doesn't use it. "
               "Either add it (StringIndexer + OneHotEncoder) or update the README.")
    notes = st.text_area(
        "Paste your PySpark output here (AUC, accuracy, F1, feature importances)",
        height=200, placeholder="AUC-ROC : 0.xxxx\nAccuracy : 0.xxxx\nF1 Score : 0.xxxx")
    if notes:
        st.code(notes)

# ---------------------------------------------------------------- Insights
with tabs[7]:
    st.subheader("What the data says")
    r = big.dropna(subset=["avg_rating"])
    top_r = r.loc[r["avg_rating"].idxmax()]
    top_d = dem.iloc[0]
    top_bs = cats.loc[cats["bestsellers"].idxmax()]
    rate_pool = cats[cats["products"] >= 1000]
    top_rate = rate_pool.loc[rate_pool["bestseller_rate"].idxmax()]
    best_seg = ps.loc[ps["bestseller_rate"].idxmax()]
    worst_seg = ps.loc[ps["bestseller_rate"].idxmin()]
    st.markdown(f"""
- **Demand is concentrated.** {top_d['category']} alone accounts for {top_d['share']:.0f}% of units
  bought ({top_d['demand']:,.0f}); the top 5 categories account for {dem.head(5)['share'].sum():.0f}%.
- **Most products are cheap.** {price.loc[price['segment'].isin(price['segment'].cat.categories[:3]), 'pct'].sum():.0f}%
  of products cost under $50, and the biggest segment is {price.loc[price['count'].idxmax(), 'segment']}.
- **Bestsellers are rare** ({bs_share:.2f}% of products). {top_bs['category']} has the most
  ({top_bs['bestsellers']:.0f}), but {top_rate['category']} has the highest *rate*
  ({top_rate['bestseller_rate']:.1f}% of its products, categories with 1,000+ products).
- **Price sweet spot:** {best_seg['segment']} has the highest bestseller rate
  ({best_seg['bestseller_rate']:.2f}%); {worst_seg['segment']} the lowest ({worst_seg['bestseller_rate']:.2f}%).
- **Highest-rated category** (min {min_products} products): {top_r['category']} ({top_r['avg_rating']:.2f}★).
""")
    st.markdown("#### Data caveats worth mentioning in your write-up")
    n_low = int((cats["avg_rating"] < 2).sum())
    n_capped = int(cats["products"].between(8000, 10000).sum())
    n_neg = int((cats["avg_price_diff"] < 0).sum())
    st.markdown(f"""
- **Unrated products pull ratings down.** {n_low} categories have an average rating below 2★, which
  most likely reflects products with 0 stars (no reviews yet). Excluding zero-star products from the
  Hive average would give a fairer ranking.
- **Product counts look capped.** {n_capped} of {len(cats)} categories have between 8,000 and 10,000
  products, so product counts reflect how the data was collected more than market size.
- **Price difference looks wrong.** {n_neg} of {len(cats)} categories have a negative average
  `price_diff`; see the warning on the *Popularity vs price diff* tab.
- **Popularity score is dominated by review count** (`reviews × stars`), so it mostly measures how
  many reviews a category has.
""")
    st.markdown("#### Your own conclusions")
    st.text_area("Takeaways", height=180, label_visibility="collapsed",
                 placeholder="What did you learn from the project?")
