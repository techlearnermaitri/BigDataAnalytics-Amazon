"""Parses amazon_mapreduce_results.txt (the 5-section report) into DataFrames."""
import re

import pandas as pd


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
