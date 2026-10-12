"""
Amazon Canada · Big Data Dashboard
Run:  streamlit run app.py
Reads amazon_mapreduce_results.txt from the same folder.
"""
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

RESULTS_FILE = "amazon_mapreduce_results.txt"
TEAL, ORANGE, PURPLE = "#2a9d8f", "#e76f51", "#6a4c93"
MIN_PRODUCTS = 500        # ignore tiny categories in ratings / per-product rankings
MIN_PRODUCTS_RATE = 1000  # ignore small categories when ranking bestseller rate

st.set_page_config(page_title="Amazon Canada · Big Data Dashboard", page_icon="📦", layout="wide")


# =============================================================================
# Parser for the 5-section MapReduce report
# =============================================================================
def _num(s):
    return float(s.replace(",", ""))


def _rows(lines, pattern, cols, sub=None):
    out, rx = [], re.compile(pattern)
    for s, line in lines:
        if sub is not None and s != sub:
            continue
        m = rx.match(line)
        if m:
            out.append([m.group(1).strip()] + [_num(g) for g in m.groups()[1:]])
    return pd.DataFrame(out, columns=cols)


def parse_results(text):
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


@st.cache_data(show_spinner=False)
def load():
    path = Path(__file__).parent / RESULTS_FILE
    return parse_results(path.read_text(encoding="utf-8", errors="ignore"))


# =============================================================================
# Small UI helpers
# =============================================================================
def hbar(d, x, color=TEAL, fmt=",.0f"):
    d = d.sort_values(x)
    fig = px.bar(d, x=x, y="category", orientation="h", text=x,
                 color_discrete_sequence=[color], height=max(340, 30 * len(d) + 60))
    fig.update_traces(texttemplate="%{text:" + fmt + "}", textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis_title=None, xaxis_title=None, showlegend=False,
                      margin=dict(l=0, r=50, t=10, b=0))
    fig.update_xaxes(visible=False)
    return fig


def title(text):
    st.markdown(f"##### {text}")


def note(text):
    st.info(text, icon="💡")


# =============================================================================
# Load + derive everything
# =============================================================================
try:
    data = load()
except FileNotFoundError:
    st.error(f"`{RESULTS_FILE}` was not found next to app.py. Add it to the repository and reboot.")
    st.stop()

cats, price, bs_price = data["categories"], data["price"], data["bs_price"]
if cats.empty or price.empty:
    st.error("The results file couldn't be parsed. Is it the full 5-section report?")
    st.stop()

total_products = int(price["count"].sum())
total_units = int(cats["demand"].sum())
total_bs = int(cats["bestsellers"].sum())
bs_share = total_bs / total_products * 100
avg_units_per_product = total_units / total_products

price = price.copy()
price["segment"] = pd.Categorical(price["segment"], price["segment"].tolist(), ordered=True)
ps = price.merge(bs_price, on="segment", how="left")
ps["bestseller_rate"] = ps["bestsellers"] / ps["count"] * 100

dem = cats.sort_values("demand", ascending=False).reset_index(drop=True)
dem["share"] = dem["demand"] / dem["demand"].sum() * 100
dem["cum_share"] = dem["share"].cumsum()
dem["rank"] = dem.index + 1
top_d = dem.iloc[0]
top5_share = dem.head(5)["share"].sum()
top10_share = dem.head(10)["share"].sum()
n_zero_demand = int((cats["demand"] == 0).sum())

best_seg = ps.loc[ps["bestseller_rate"].idxmax()]
worst_seg = ps.loc[ps["bestseller_rate"].idxmin()]

rate_pool = cats[cats["products"] >= MIN_PRODUCTS_RATE]
if rate_pool.empty:
    rate_pool = cats
count_leader = cats.loc[cats["bestsellers"].idxmax()]
rate_leader = rate_pool.loc[rate_pool["bestseller_rate"].idxmax()]
top10_bs_share = cats.nlargest(10, "bestsellers")["bestsellers"].sum() / total_bs * 100

rated = cats[(cats["products"] >= MIN_PRODUCTS)].dropna(subset=["avg_rating"])
n_low_rating = int((cats["avg_rating"] < 2).sum())
n_rated = int(cats["avg_rating"].notna().sum())

pp = cats[(cats["demand"] > 0) & (cats["avg_popularity"] > 0)]
corr = pp["avg_popularity"].rank().corr(pp["demand"].rank()) if len(pp) > 2 else float("nan")
pop_leader = cats.loc[cats["avg_popularity"].idxmax()]
pop_leader_demand_rank = int(dem.loc[dem["category"] == pop_leader["category"], "rank"].iloc[0])
corr_word = ("strong" if corr > 0.6 else "moderate" if corr > 0.35 else "weak")
pop_tracks = corr > 0.6

# =============================================================================
# Header + KPIs
# =============================================================================
st.title("📦 Amazon Canada Product Analytics")
st.caption("Built on a HDFS → MapReduce → Hive → PySpark pipeline · "
           f"{total_products:,} products · {len(cats)} categories")

k = st.columns(5)
k[0].metric("Products analysed", f"{total_products:,}")
k[1].metric("Categories", f"{len(cats)}")
k[2].metric("Units bought (last month)", f"{total_units / 1e6:.1f}M")
k[3].metric("Bestsellers", f"{total_bs:,}", f"{bs_share:.2f}% of products", delta_color="off")
k[4].metric("Biggest category", top_d["category"], f"{top_d['share']:.0f}% of demand", delta_color="off")

st.markdown("### Key findings")
h = st.columns(4)
with h[0].container(border=True):
    st.markdown(f"**🔥 Demand is concentrated**\n\nThe top 5 categories make up "
                f"**{top5_share:.0f}%** of units bought. {top_d['category']} alone is "
                f"**{top_d['share']:.0f}%**.")
with h[1].container(border=True):
    st.markdown(f"**💰 Price sweet spot**\n\nProducts at **{best_seg['segment']}** are "
                f"{best_seg['bestseller_rate'] / worst_seg['bestseller_rate']:.0f}x more likely to be "
                f"bestsellers than at {worst_seg['segment']}.")
with h[2].container(border=True):
    st.markdown(f"**🏆 Bestsellers are rare**\n\nOnly **{bs_share:.2f}%** of products have the "
                f"badge, and {rate_leader['category']} has the highest share "
                f"(**{rate_leader['bestseller_rate']:.1f}%**).")
with h[3].container(border=True):
    st.markdown(f"**⭐ {'Popularity tracks sales' if pop_tracks else 'Popularity ≠ sales'}**\n\n"
                f"Review-based popularity has a **{corr_word}** link to units bought "
                f"(rank correlation {corr:.2f}).")

# =============================================================================
# 1. Demand
# =============================================================================
st.divider()
st.header("Demand")
c1, c2 = st.columns(2)
with c1:
    title("Units bought last month, top 15 categories")
    st.plotly_chart(hbar(dem.head(15), "demand"), use_container_width=True)
with c2:
    title("How quickly demand accumulates (categories ranked by demand)")
    fig = px.area(dem, x="rank", y="cum_share", color_discrete_sequence=[ORANGE])
    fig.update_layout(xaxis_title="Number of categories (largest first)",
                      yaxis_title="Cumulative % of all units", margin=dict(l=0, r=0, t=10, b=0),
                      height=480)
    fig.add_hline(y=80, line_dash="dot", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)
n80 = int((dem["cum_share"] < 80).sum() + 1)
note(f"**{top_d['category']} sells {top_d['demand']:,.0f} units, about "
     f"{top_d['demand'] / dem.iloc[1]['demand']:.1f}x the runner-up ({dem.iloc[1]['category']}).** "
     f"Just {n80} of {len(cats)} categories account for 80% of demand, and {n_zero_demand} "
     f"categories recorded no purchases at all. Values come in steps of 50, so treat them as "
     f"rounded estimates.")

c1, c2 = st.columns(2)
per = cats[cats["products"] >= MIN_PRODUCTS].nlargest(12, "demand_per_product")
with c1:
    title(f"Units sold per product (categories with {MIN_PRODUCTS}+ products)")
    st.plotly_chart(hbar(per, "demand_per_product", ORANGE, ",.0f"), use_container_width=True)
with c2:
    st.write("")
    st.write("")
    note(f"**Big isn't the same as efficient.** The average product sells "
         f"{avg_units_per_product:.1f} units a month, but a typical {per.iloc[0]['category']} "
         f"product sells {per.iloc[0]['demand_per_product']:.0f}. Categories with few products and "
         f"high demand per product are the least crowded opportunities.")

# =============================================================================
# 2. Price
# =============================================================================
st.divider()
st.header("Price")
c1, c2 = st.columns(2)
with c1:
    title("Products by price segment")
    fig = px.bar(price, x="segment", y="count", text="pct", color_discrete_sequence=[TEAL])
    fig.update_traces(texttemplate="%{text}%", textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title=None, yaxis_title=None, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
with c2:
    title("Chance a product is a bestseller, by price segment")
    fig = px.bar(ps, x="segment", y="bestseller_rate", text="bestseller_rate",
                 color_discrete_sequence=[ORANGE])
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title=None, yaxis_title=None, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
under50 = price.loc[price["segment"].isin(price["segment"].cat.categories[:3]), "pct"].sum()
note(f"**{under50:.0f}% of products cost under $50**, and that is also where bestsellers cluster: "
     f"{best_seg['segment']} has the best odds ({best_seg['bestseller_rate']:.2f}%). "
     f"Chances fall steadily as prices rise, down to {worst_seg['bestseller_rate']:.2f}% at "
     f"{worst_seg['segment']}, so premium products rarely earn the badge.")

# =============================================================================
# 3. Bestsellers
# =============================================================================
st.divider()
st.header("Bestsellers")
c1, c2 = st.columns(2)
with c1:
    title("Most bestsellers (count)")
    st.plotly_chart(hbar(cats.nlargest(15, "bestsellers"), "bestsellers"), use_container_width=True)
with c2:
    title(f"Highest bestseller rate (% of products, {MIN_PRODUCTS_RATE:,}+ products)")
    st.plotly_chart(hbar(rate_pool.nlargest(15, "bestseller_rate"), "bestseller_rate", ORANGE, ".1f"),
                    use_container_width=True)
if count_leader["category"] != rate_leader["category"]:
    lead = (f"**Count and rate tell different stories.** {count_leader['category']} has the most "
            f"bestsellers ({count_leader['bestsellers']:.0f}) but only "
            f"{count_leader['bestseller_rate']:.1f}% of its products hold the badge, while "
            f"{rate_leader['category']} reaches {rate_leader['bestseller_rate']:.1f}%.")
else:
    lead = (f"**{count_leader['category']} leads on both count and rate** "
            f"({count_leader['bestsellers']:.0f} bestsellers, {count_leader['bestseller_rate']:.1f}%).")
note(f"{lead} The top 10 categories hold {top10_bs_share:.0f}% of all bestsellers, and "
     f"{int((cats['bestsellers'] == 0).sum())} categories have none.")

# =============================================================================
# 4. Ratings
# =============================================================================
st.divider()
st.header("Ratings")
c1, c2, c3 = st.columns([2, 2, 2])
with c1:
    title(f"Highest rated ({MIN_PRODUCTS}+ products)")
    st.plotly_chart(hbar(rated.nlargest(12, "avg_rating"), "avg_rating", TEAL, ".2f"),
                    use_container_width=True)
with c2:
    title(f"Lowest rated ({MIN_PRODUCTS}+ products)")
    st.plotly_chart(hbar(rated.nsmallest(12, "avg_rating"), "avg_rating", ORANGE, ".2f"),
                    use_container_width=True)
with c3:
    title("Spread of category averages")
    fig = px.histogram(cats.dropna(subset=["avg_rating"]), x="avg_rating", nbins=30,
                       color_discrete_sequence=[PURPLE])
    fig.update_layout(xaxis_title="Average rating", yaxis_title="Categories",
                      margin=dict(l=0, r=0, t=10, b=0), height=420)
    st.plotly_chart(fig, use_container_width=True)
best_r = rated.loc[rated["avg_rating"].idxmax()]
note(f"**Averages are pulled down by unrated products.** {n_low_rating} of {n_rated} categories "
     f"average below 2 stars, which almost certainly reflects products with 0 stars (no reviews yet) "
     f"rather than bad products. Among well-populated categories, {best_r['category']} tops the "
     f"list at {best_r['avg_rating']:.2f}.")

# =============================================================================
# 5. Popularity
# =============================================================================
st.divider()
st.header("Popularity vs sales")
c1, c2 = st.columns(2)
with c1:
    title("Highest average popularity score (reviews × stars)")
    st.plotly_chart(hbar(cats.nlargest(15, "avg_popularity"), "avg_popularity", PURPLE),
                    use_container_width=True)
with c2:
    title("Popularity vs units bought (each bubble is a category)")
    fig = px.scatter(pp, x="avg_popularity", y="demand", size="products", hover_name="category",
                     log_x=True, log_y=True, opacity=0.65, size_max=28,
                     color_discrete_sequence=[TEAL])
    fig.update_layout(xaxis_title="Avg popularity (log)", yaxis_title="Units bought (log)",
                      margin=dict(l=0, r=0, t=10, b=0), height=520)
    st.plotly_chart(fig, use_container_width=True)
note(f"**{'Popularity and sales broadly agree' if pop_tracks else 'Popular is not the same as selling now'}.** "
     f"{pop_leader['category']} has the highest "
     f"popularity score ({pop_leader['avg_popularity']:,.0f}) but ranks #{pop_leader_demand_rank} "
     f"in units bought. The score multiplies lifetime review counts, so it favours older, "
     f"established categories, while `boughtInLastMonth` captures current buying. Overall the "
     f"relationship is {corr_word} (rank correlation {corr:.2f}).")

# =============================================================================
# Data notes
# =============================================================================
st.divider()
n_capped = int(cats["products"].between(8000, 10000).sum())
with st.expander("About this data"):
    st.markdown(f"""
- **Source:** Amazon Canada product data (Kaggle), {total_products:,} products after cleaning,
  processed with MapReduce and Hive.
- **Unrated products count as 0 stars,** so category rating averages are conservative.
- **Product counts look capped:** {n_capped} of {len(cats)} categories have between 8,000 and
  10,000 products, so category size reflects data collection more than real market size.
- **Price difference was left out:** the average `listPrice − price` is negative in almost every
  category, which points to missing list prices rather than real pricing behaviour.
- **Demand is approximate:** `boughtInLastMonth` is reported in steps of 50 and is 0 for most products.
""")
