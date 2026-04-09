"""
PySpark RDD Operations — Amazon.ca Dataset
===========================================
Covers: SparkContext, parallelize, map, filter, flatMap,
        distinct, union, reduceByKey, groupByKey, lazy evaluation

ROOT CAUSE OF ALL ERRORS
-------------------------
The CSV contains 132,908+ product titles with doubled-quote escaping, e.g.:
    "3"" Overall Length, 0.500"" Cutting Diameter"
Python's csv module handles this by default (RFC 4180 standard).
Spark's CSV reader does NOT by default — it misreads the closing quote,
causing the remainder of the title to spill into the numeric columns,
producing values like ' 0.500"" Cutting Diameter' where a Double is expected.

Additionally, numeric columns contain literal "-inf" strings (from prior
discount calculations) which also cannot be cast to Double.

FIXES APPLIED
-------------
1. spark.read.csv: escape='"' tells Spark to use doubled-quote escaping
   (RFC 4180) instead of backslash escaping. This is the single most
   important fix — it stops title text bleeding into numeric columns.
2. inferSchema=False: read everything as StringType — no type guessing.
3. Safe cast loop: nullifies inf/-inf/nan/empty strings, then uses
   F.col().cast(DoubleType()) which returns NULL (not an exception) for
   any remaining unparseable value (equivalent to SQL TRY_CAST).
4. isBestSeller is '0'/'1' integers — cast via IntegerType → boolean.
5. safe_float() helper in all RDD lambdas prevents TypeError on None.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

# ── SparkSession + SparkContext ───────────────────────────────────────────────

spark = SparkSession.builder \
    .appName("Amazon_RDD_Operations") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
#
# Critical CSV options:
#   inferSchema=False  — all columns read as StringType; no type guessing
#   quote='"'          — double-quote is the quoting character (explicit)
#   escape='"'         — "" inside a quoted field means a literal "
#                        This is RFC 4180 and matches Python's csv module
#   ignoreLeadingWhiteSpace / ignoreTrailingWhiteSpace — trim stray spaces
# ══════════════════════════════════════════════════════════════════════════════

df_raw = spark.read.csv(
    "/cleaned_amazon_data.csv",
    header=True,
    inferSchema=False,
    quote='"',
    escape='"',
    multiLine=False,
    ignoreLeadingWhiteSpace=True,
    ignoreTrailingWhiteSpace=True,
)

# ── Safe numeric cast ─────────────────────────────────────────════════────────
# Step 1 — replace inf/-inf/nan/empty literals with NULL
# Step 2 — cast to DoubleType (returns NULL silently on any bad value)

NUMERIC_COLS = [
    "stars", "reviews", "price", "listPrice",
    "boughtInLastMonth", "discount_percent",
    "popularity_score", "price_diff",
]

INF_STRINGS = ["inf", "-inf", "infinity", "-infinity", "nan", "null", "none", ""]

df = df_raw
for c in NUMERIC_COLS:
    if c in df.columns:
        df = df.withColumn(
            c,
            F.when(
                F.lower(F.trim(F.col(c))).isin(INF_STRINGS),
                F.lit(None).cast(DoubleType())
            ).otherwise(
                F.col(c).cast(DoubleType())
            )
        )

# ── Safe boolean cast ─────────────────────────────────────────────────────────
# isBestSeller is stored as '0' / '1' in this dataset

if "isBestSeller" in df.columns:
    df = df.withColumn(
        "isBestSeller",
        F.col("isBestSeller").cast(IntegerType()).cast("boolean")
    )

if "isAmazonChoice" in df.columns:
    df = df.withColumn(
        "isAmazonChoice",
        F.col("isAmazonChoice").cast(IntegerType()).cast("boolean")
    )

# Convert DataFrame → RDD of Row objects
rdd_raw = df.rdd

print(f"Total products loaded: {rdd_raw.count()}")
print(f"Sample row: {rdd_raw.first()}\n")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — safe float conversion for all RDD lambdas
# Prevents TypeError when a field is None/null
# ══════════════════════════════════════════════════════════════════════════════

def safe_float(val, default=0.0):
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


# ══════════════════════════════════════════════════════════════════════════════
# CREATING AN RDD WITH parallelize()
# ══════════════════════════════════════════════════════════════════════════════

sample_prices = sc.parallelize([9.99, 24.99, 49.99, 99.99, 199.99, 499.99])
print("Parallelized price list:", sample_prices.collect())


# ══════════════════════════════════════════════════════════════════════════════
# 1. map()
#    Apply a function to EVERY element — returns same number of elements
# ══════════════════════════════════════════════════════════════════════════════

title_price_rdd = rdd_raw.map(
    lambda row: (row["title"], safe_float(row["price"]))
)

print("── map() — Extract (title, price) ──")
for title, price in title_price_rdd.take(5):
    print(f"  ${price:.2f}  |  {str(title)[:60]}")

discount_rdd = rdd_raw.map(
    lambda row: (
        row["asin"],
        round(safe_float(row["listPrice"]) - safe_float(row["price"]), 2)
    )
)
print("\n── map() — Compute discount per product ──")
for asin, discount in discount_rdd.take(5):
    print(f"  {asin}  →  discount = ${discount}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. filter()
#    Keep only elements that satisfy a condition
# ══════════════════════════════════════════════════════════════════════════════

premium_rdd = rdd_raw.filter(
    lambda row: row["price"] is not None and safe_float(row["price"]) > 100
)
print(f"\n── filter() — Products above $100 ──")
print(f"  Count: {premium_rdd.count()}")

top_rated_rdd = rdd_raw.filter(
    lambda row: row["stars"] is not None
             and row["reviews"] is not None
             and safe_float(row["stars"]) >= 4.5
             and safe_float(row["reviews"]) >= 100
)
print(f"\n── filter() — Highly rated (≥4.5 stars, ≥100 reviews) ──")
print(f"  Count: {top_rated_rdd.count()}")
for row in top_rated_rdd.take(3):
    print(f"  {safe_float(row['stars']):.1f}★  "
          f"{int(safe_float(row['reviews']))} reviews  "
          f"${safe_float(row['price']):.2f}  "
          f"{str(row['title'])[:50]}")

bestseller_rdd = rdd_raw.filter(lambda row: row["isBestSeller"] == True)
print(f"\n── filter() — Best Sellers only ──")
print(f"  Count: {bestseller_rdd.count()}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. flatMap()
#    Like map() but flattens — one input can produce many outputs
# ══════════════════════════════════════════════════════════════════════════════

title_words_rdd = rdd_raw.flatMap(
    lambda row: str(row["title"]).lower().split() if row["title"] else []
)
print(f"\n── flatMap() — All words across all product titles ──")
print(f"  Total words: {title_words_rdd.count()}")
print(f"  Sample words: {title_words_rdd.take(10)}")

word_freq = (
    title_words_rdd
    .map(lambda word: (word, 1))
    .reduceByKey(lambda a, b: a + b)
    .sortBy(lambda x: x[1], ascending=False)
)

print("\n  Top 10 most common words in product titles:")
for word, count in word_freq.take(10):
    print(f"    {word:<20} {count}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. distinct()
#    Remove duplicates — returns only unique values
# ══════════════════════════════════════════════════════════════════════════════

categories_rdd        = rdd_raw.map(lambda row: row["categoryName"])
unique_categories_rdd = categories_rdd.distinct()

print(f"\n── distinct() — Unique product categories ──")
print(f"  Total rows (with duplicates): {categories_rdd.count()}")
print(f"  Unique categories:            {unique_categories_rdd.count()}")
print(f"  Category list: {sorted(unique_categories_rdd.collect())}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. union()
#    Combine two RDDs into one (duplicates kept)
# ══════════════════════════════════════════════════════════════════════════════

budget_rdd = (
    rdd_raw
    .filter(lambda row: row["price"] is not None and safe_float(row["price"]) < 20)
    .map(lambda row: (row["asin"], safe_float(row["price"]), "budget"))
)

luxury_rdd = (
    rdd_raw
    .filter(lambda row: row["price"] is not None and safe_float(row["price"]) > 200)
    .map(lambda row: (row["asin"], safe_float(row["price"]), "luxury"))
)

combined_rdd = budget_rdd.union(luxury_rdd)

print(f"\n── union() — Budget + Luxury products combined ──")
print(f"  Budget   products : {budget_rdd.count()}")
print(f"  Luxury   products : {luxury_rdd.count()}")
print(f"  Combined total    : {combined_rdd.count()}")
print("  Sample:")
for asin, price, tier in combined_rdd.take(5):
    print(f"    {tier:<8}  ${price:.2f}  {asin}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. reduceByKey()
#    Group by key and aggregate — more efficient than groupByKey for sums
# ══════════════════════════════════════════════════════════════════════════════

reviews_by_category = (
    rdd_raw
    .filter(lambda row: row["categoryName"] and row["reviews"] is not None)
    .map(lambda row: (row["categoryName"], int(safe_float(row["reviews"]))))
    .reduceByKey(lambda a, b: a + b)
    .sortBy(lambda x: x[1], ascending=False)
)

print("\n── reduceByKey() — Total reviews per category ──")
for category, total in reviews_by_category.take(8):
    print(f"  {category:<30}  {total:,} reviews")

products_per_category = (
    rdd_raw
    .filter(lambda row: row["categoryName"] is not None)
    .map(lambda row: (row["categoryName"], 1))
    .reduceByKey(lambda a, b: a + b)
    .sortBy(lambda x: x[1], ascending=False)
)

print("\n── reduceByKey() — Product count per category ──")
for category, count in products_per_category.take(8):
    print(f"  {category:<30}  {count} products")


# ══════════════════════════════════════════════════════════════════════════════
# 7. groupByKey()
#    Group all values under each key into a list (no aggregation)
# ══════════════════════════════════════════════════════════════════════════════

category_prices_rdd = (
    rdd_raw
    .filter(lambda row: row["categoryName"] and row["price"] is not None)
    .map(lambda row: (row["categoryName"], round(safe_float(row["price"]), 2)))
    .groupByKey()
)

print("\n── groupByKey() — All prices grouped by category ──")
for category, prices in category_prices_rdd.take(4):
    price_list = sorted(list(prices))
    print(f"  {category:<30}  prices: {price_list[:5]}{'...' if len(price_list) > 5 else ''}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. LAZY EVALUATION
#    Transformations build a DAG; execution only happens on an action call
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Lazy Evaluation Demo ──")

step1 = rdd_raw.filter(
    lambda row: row["stars"] is not None and safe_float(row["stars"]) >= 4.0
)
step2 = step1.map(
    lambda row: (row["categoryName"], safe_float(row["price"]) if row["price"] is not None else None)
)
step3 = step2.filter(lambda x: x[1] is not None and x[1] < 50)

print("  step1, step2, step3 defined — NO computation has run yet")
print("  Calling count() now triggers the full execution chain...")

result_count = step3.count()
print(f"  Products rated ≥4.0 stars AND priced under $50: {result_count}")

print("\n  Execution plan (DAG) for step3:")
print(step3.toDebugString().decode("utf-8"))


# ══════════════════════════════════════════════════════════════════════════════
# 9. MACHINE LEARNING — Random Forest Classifier (PySpark MLlib)
#    Predict whether a product is a Best Seller (binary classification)
# ══════════════════════════════════════════════════════════════════════════════

from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

print("\n" + "═" * 55)
print("ML — Best Seller Prediction (Random Forest)")
print("═" * 55)

ml_df = df.select(
    "stars", "reviews", "price", "listPrice",
    "boughtInLastMonth", "isBestSeller"
).dropna(subset=["isBestSeller"])

# Engineer discount_percent
ml_df = ml_df.withColumn(
    "discount_percent",
    F.when(
        F.col("listPrice").isNotNull() & (F.col("listPrice") > 0),
        ((F.col("listPrice") - F.col("price")) / F.col("listPrice")) * 100
    ).otherwise(F.lit(0.0))
)

# Label: Boolean → Double
ml_df = ml_df.withColumn("label", F.col("isBestSeller").cast(DoubleType()))

print(f"\n  ML dataset size  : {ml_df.count()} rows")
print(f"  Best sellers     : {ml_df.filter(F.col('label') == 1).count()}")
print(f"  Non-best sellers : {ml_df.filter(F.col('label') == 0).count()}")

train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)
print(f"\n  Train : {train_df.count()} rows")
print(f"  Test  : {test_df.count()} rows")

feature_cols = ["stars", "reviews", "price", "boughtInLastMonth", "discount_percent"]

imputer = Imputer(
    inputCols=feature_cols,
    outputCols=[c + "_imp" for c in feature_cols],
    strategy="median"
)

assembler = VectorAssembler(
    inputCols=[c + "_imp" for c in feature_cols],
    outputCol="raw_features",
    handleInvalid="skip"
)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withMean=True,
    withStd=True
)

rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=100,
    maxDepth=6,
    seed=42
)

pipeline = Pipeline(stages=[imputer, assembler, scaler, rf])

print("\n  Training Random Forest...")
model = pipeline.fit(train_df)

predictions = model.transform(test_df)

auc = BinaryClassificationEvaluator(
    labelCol="label", metricName="areaUnderROC"
).evaluate(predictions)

accuracy = MulticlassClassificationEvaluator(
    labelCol="label", metricName="accuracy"
).evaluate(predictions)

f1 = MulticlassClassificationEvaluator(
    labelCol="label", metricName="f1"
).evaluate(predictions)

print(f"\n  ── Evaluation Results ──")
print(f"  AUC-ROC  : {auc:.4f}  (1.0 = perfect, 0.5 = random)")
print(f"  Accuracy : {accuracy:.4f}")
print(f"  F1 Score : {f1:.4f}")

rf_stage    = model.stages[-1]
importances = rf_stage.featureImportances.toArray()

print("\n  ── Feature Importances ──")
for name, score in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
    bar = "█" * int(score * 40)
    print(f"  {name:<22} {score:.4f}  {bar}")

print("\n  ── Sample Predictions ──")
predictions.select(
    F.round("stars",      1).alias("stars"),
    F.round("price",      2).alias("price"),
    "reviews",
    F.round("label",      0).alias("actual"),
    F.round("prediction", 0).alias("predicted"),
).show(10)


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 55)
print("RDD Operations Summary")
print("═" * 55)
print("  parallelize()  create RDD from local Python list")
print("  map()          transform every element 1-to-1")
print("  filter()       keep elements matching a condition")
print("  flatMap()      transform + flatten (1-to-many)")
print("  distinct()     remove duplicate elements")
print("  union()        merge two RDDs (keeps duplicates)")
print("  reduceByKey()  aggregate values per key")
print("  groupByKey()   group values into list per key")
print("  collect()      action — brings data to driver")
print("  count()        action — triggers lazy evaluation")
print("  MLlib Pipeline → Imputer + VectorAssembler + Scaler + RF")
print("═" * 55)

spark.stop()
