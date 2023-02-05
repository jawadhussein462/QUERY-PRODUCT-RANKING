# Project still under development

# Multi-language Query Product Ranking

## 🚀 Overview 
This project aims to rank products in an e-commerce setting based on their relevance to a given query. The input for this task is a list of queries with their corresponding identifiers, and the system outputs a CSV file with the ranked products for each query. The input data for each query will be sorted based on four degrees of relevance: Exacts, Substitutes, Compliments, and Irrelevants.

## 📚 Data Format

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

This is a table of product catalog information. It is  csv file that contains information about different products.

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

### ⚡️ Training

The cli/learning.py script provides a convenient way to train a machine learning model on the provided data. The trained model will be saved to the specified location, ready for use in the [Your Project Name] system.

#### Usage

To train a model, use the following command in your terminal:

```
    $ python cli/learning.py \
    --train_data [PATH_TO_TRAIN_DATA] \
    --product_catalogue [PATH_TO_PRODUCT_CATALOGUE]
    --model_save_dir [PATH_TO_SAVE_MODELS]
```

#### Arguments

The following arguments must be passed to the script:

* `--train_data`: The path to the training data file, in [Your preferred file format].

* `--product_catalogue`: The path to the product catalogue file, in [Your preferred file format].

* `--model_save_dir`: The directory where the trained models will be saved, in [Your preferred model format].

#### Output

The script will output the training results to the console, including the final model's accuracy. The trained models will be saved to the specified model_save_dir for future use.


### ⚡️ Prediction

```
    $ python cli/prediction.py \
    --test_data [PATH_TO_TEST_DATA] \
    --product_catalogue [PATH_TO_PRODUCT_CATALOGUE]
```

