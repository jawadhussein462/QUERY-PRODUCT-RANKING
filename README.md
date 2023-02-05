# Project still under development

# Multi-language query product ranking

This project aims to rank products in an e-commerce setting based on their relevance to a given query. The input for this task is a list of queries with their corresponding identifiers, and the system outputs a CSV file with the ranked products for each query. The input data for each query will be sorted based on four degrees of relevance: Exacts, Substitutes, Compliments, and Irrelevants.

## Data Format

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

### Product Catalogue

This is a table of product catalog information. The table contains information about different products.

The columns are:

* product_id - a unique identifier for each product
* product_title - the title of the product
* product_description - a brief description of the product
* product_bullet_point - bullet points that highlight the key features of the product
* product_brand - the brand name of the product
* product_color_name - the color of the product
* product_locale - the locale where the product is available

Here is an example of a single row in the table:

| product_id | product_title | product_description | product_bullet_point | product_brand |  product_color_name | product_locale
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| B075VXJ9VG | "BAZIC Pencil #2 HB Pencils, Latex Free Eraser" | Our goal is to provide each customer with long-lasting supplies at an affordable cost... | UN-SHARPENED #2 PREMIUM PENCILS. | BAZIC Products | Yellow | us


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

```
    $ python cli/learning.py --train_data [PATH_TO_TRAIN_DATA] --product_catalogue [PATH_TO_PRODUCT_CATALOGUE]
```

### Prediction

```
    $ python cli/prediction.py --test_data [PATH_TO_TEST_DATA] --product_catalogue [PATH_TO_PRODUCT_CATALOGUE]
```

