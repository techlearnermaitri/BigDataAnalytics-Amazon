## Project Overview

This project analyzes Amazon Canada product data using Big Data tools and techniques. It demonstrates how distributed systems can process and analyze large-scale e-commerce datasets efficiently.

The workflow includes data preprocessing, MapReduce-based aggregation, Hive-based querying, machine learning using PySpark, and data visualization.

---

## Objectives

* Understand distributed data processing concepts
* Perform data cleaning and preprocessing
* Apply MapReduce for large-scale aggregation
* Use Hive for structured querying
* Implement machine learning using PySpark MLlib
* Generate insights through data visualization

---

## Dataset

Kaggle dataset used:
https://www.kaggle.com/datasets/asaniczka/amazon-canada-products-2023-2-1m-products

The dataset consists of Amazon Canada product data with the following attributes:

* Product ID (asin)
* Title and category
* Price and list price
* Ratings and reviews
* Bestseller status
* Purchase activity (boughtInLastMonth)

---

## Tech Stack

### Big Data Technologies

* Hadoop Distributed File System (HDFS)
* MapReduce
* Apache Hive

### Data Processing and Machine Learning

* PySpark
* PySpark MLlib

### Programming

* Python
* Pandas
* NumPy

### Visualization

* Power BI or Tableau

---

## Data Preprocessing

* Converted data types for compatibility
* Handled missing and invalid values
* Standardized text fields
* Removed duplicate records
* Created derived features such as discount percentage, popularity score, and price difference
* Dropped non-essential columns

---

## MapReduce Task

Objective: Compute total product demand per category

* Map Phase: Emit (categoryName, boughtInLastMonth)
* Reduce Phase: Aggregate total purchases per category

---

## Hive Analysis

The following analytical queries were performed using Hive:

* Average rating per category
* Average price per category
* Bestseller count per category
* Top reviewed products

---

## Machine Learning

A machine learning model was implemented using PySpark MLlib:

* Task: Predict whether a product is a bestseller
* Features: price, reviews, ratings, category
* Model: Logistic Regression or Random Forest

---

## Data Visualization

Insights were visualized using Power BI or Tableau, including:

* Category-wise ratings
* Price distribution
* Bestseller trends
* Demand analysis

---

## How to Run

1. Clone the repository
2. Install required Python libraries
3. Run the data preprocessing script
4. Load the dataset into HDFS or Hive
5. Execute MapReduce and Hive queries
6. Run the PySpark machine learning model
7. Visualize results using Power BI or Tableau

---

## Conclusion

This project demonstrates the use of Big Data tools for processing and analyzing large-scale retail data. It highlights how distributed systems enable efficient computation and support data-driven decision-making.
