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

## 💻 Content

```
├── cli                                            <- Entry code to the project
│   ├──learning.py                                 <- Training code
│   └──prediction                                  <- Prediction code
├── src                                            <- Python package - scripts of the project
│   ├── configuration                              <- Setup configs
│   │   ├── resources                              <- configuration files
│   │   │   ├── data_structure.json                <- data configuration file
│   │   │   ├── model.json                         <- model configuration file
│   │   │   └── path.json                          <- path configuration file
│   │   ├── app.py                                 <- Environment configuration
│   │   └── data.py                                <- Data configuration
│   ├── models                                     <- Modeling
│   │   ├── bert_model.py                          <- Bert model class (used in cross encoder)
│   │   ├── bm25_model.py                          <- BM25 model class
│   │   ├── cross_encoder_model.py                 <- Cross Encoder model class
│   │   ├── cross_encoder_dataset.py               <- Cross Encoder Dataset class used to import data
│   │   └── cross_encoder_factory.py               <- Cross Encoder Factory class used to get models
│   ├── pipeline                                   <- Pipelines of the application
│   │   ├── catalogue_preprocessing_pipeline.py    <- Pre-process the product catalogue
│   │   ├── create_features_pipeline.py            <- Create features for final training
│   │   ├── data_preprocessing_pipeline.py         <- Pre-porcess query product data
│   │   ├── get_data_pipeline.py                   <- import data
│   │   ├── save_pipeline.py                       <- save trained models
│   │   ├── train_bm25_pipeline.py                 <- Train BM25 model
│   │   ├── train_cross_encoder_pipeline.py        <- Train cross encoder
│   │   └── train_ensemble_pipeline.py             <- Train the ranking model (stacking of previous models)
│   └── utils                                      <- Useful transversal functions
│       ├── constant.py                            <- Useful constant values                           
│       └── support.py                             <- System functions
│
│
├── .gitignore                                <- Files that should be ignored by git
├── README.md                                 <- The top-level README of the repository
└── requirements.txt                          <- Python libraries used in the project
```

## 🛠 Getting Started

### Installation

To get started, first make sure you have the necessary requirements installed. Run the following command in your terminal:

```
    $ sh install.sh
```

### Donwaload data 

Next, download the data for the project by running the following command:

```
     $ sh download_data.sh
```

With these two simple steps, you're all set to start working on the project!

## ⚡️ Usage 

### Training

The cli/learning.py script trains a machine learning model on the provided data. The trained model will be saved to the specified location, ready for use in the query product ranking system.

#### Command

To train a model, use the following command in your terminal:

```
    $ python cli/learning.py \
    --train_data [PATH_TO_TRAIN_DATA] \
    --product_catalogue [PATH_TO_PRODUCT_CATALOGUE]
    --model_save_dir [PATH_TO_SAVE_MODELS]
```

#### Arguments

The following arguments must be passed to the script:

* `--train_data`: The path to the training data file, in CSV format.

* `--product_catalogue`: The path to the product catalogue file, in CSV format.

* `--model_save_dir`: The directory where the trained models will be saved.

#### Output

The script will output the training results to the console, including the final model's accuracy. The trained models will be saved to the specified model_save_dir for future use.


### Prediction

The cli/prediction.py script makes the prediction using a trained machine learning model and the provided test data. The predictions will be saved to the specified output location.

#### Command

To make predictions, use the following command in your terminal:

```
    $ python cli/prediction.py \
    --test_data [PATH_TO_TEST_DATA] \
    --product_catalogue [PATH_TO_PRODUCT_CATALOGUE]
    --model_load_dir [PATH_TO_LOADED_MODEL] 
    --output [PATH_TO_OUTPUT]
```

#### Arguments

The following arguments must be passed to the script:

* `--test_data`: The path to the test data file, in CSV format.

* `--product_catalogue`: The path to the product catalogue file, in CSV format. This file should be the same as the one used for training.

* `--model_load_dir`: The directory where the trained model is located, in [Your preferred model format].

* `--output`: The path to the output file where the predictions will be saved, in CSV format.

#### Output

The script will output the predictions to the specified output file. The predictions will be in CSV format.
