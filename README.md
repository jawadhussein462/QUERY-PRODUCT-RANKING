# Project still under development

# Multi-language query product ranking

This project aims to rank products in an e-commerce setting based on their relevance to a given query. The input for this task is a list of queries with their corresponding identifiers, and the system outputs a CSV file with the ranked products for each query. The input data for each query will be sorted based on four degrees of relevance: Exacts, Substitutes, Compliments, and Irrelevants.

## Data

### Input

The input for this task is a CSV file with the following columns:

* query_id: a unique identifier for the query
* query: the text of the query
* query_locale: the locale of the query
* product_id: a unique identifier for the product

An example of the input file is shown below:

| query_id | query | query_locale | product_id | esci_label | 
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Query_1 | "Query_1" | us | product_23 | exact
| Query_2 |	"Query_2" | us | product_24 | substitute

### Output

The system will output a CSV file with the following columns:

* query_id: the unique identifier for the query
* product_id: the unique identifier for the product, ranked based on relevance

An example of the output file is shown below:

| query_id | product_id | 
| ------------- | ------------- |
| Query_1 | product_50 | 
| Query_1 | product_900 | 
| Query_1 | product_80 | 
| Query_2 | product_32 | 
| Query_2 | product_987 | 
| Query_2 | product_105 | 

## Getting Started

### 🛠 Installation

Install Requirements :

```
    $ sh install.sh
```

Download Data : 
```
     $ sh download_data.sh
```

## ⚡️ Run

### Training

### Prediction



