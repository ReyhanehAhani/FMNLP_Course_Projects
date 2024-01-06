```markdown
# FlickrStyle Image Captioning

This repository contains code for various tasks related to image captioning using the FlickrStyle dataset. It includes steps for fine-tuning a pre-trained model on the dataset and evaluating its performance. This README provides an overview of the code and explains each line and method used.

## Getting Started

### Prerequisites

Before running the code, make sure you have the following prerequisites:

- Python 3.x
- Dependencies: kaggle, pandas, transformers, nltk, rouge, tabulate

You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Dataset

The code utilizes the following datasets:

1. **FlickrStyle:** A dataset containing image descriptions (funny and romantic) for fine-tuning the model.
2. **Flickr Image:** A dataset of images used for training and testing.

To download and prepare the datasets, follow these steps:

1. Download the FlickrStyle dataset by running the following command:
    ```bash
    !wget https://zhegan27.github.io/Papers/FlickrStyle_v0.9.zip
    ```

2. Extract the contents of the downloaded zip file:
    ```bash
    !unzip /content/FlickrStyle_v0.9.zip
    ```

3. Download the Flickr Image dataset using the Kaggle API. First, install the Kaggle library:
    ```bash
    !pip install -q kaggle
    ```

4. Upload your Kaggle API credentials file and run the following command to download the dataset:
    ```bash
    !kaggle datasets download -d hsankesara/flickr-image-dataset
    ```

5. Extract the Flickr Image dataset:
    ```bash
    !unzip -q /content/flickr-image-dataset.zip
    ```

6. Upload any additional files required when prompted.

## Fine-tuning the Model

The main part of the code is focused on fine-tuning a pre-trained model on the FlickrStyle dataset. This process involves several steps:

### Load Funny and Romantic Descriptions

The code reads the funny descriptions from the `funny_train.txt` file and stores them in a list. It also reads the image names from the `train.p` file, which will be used as keys in the image dictionary.

### Create Image Dictionary

An image dictionary is created to store image names and their corresponding descriptions. Initially, only funny descriptions are added, and romantic descriptions are initialized as None.

### Fill in Romantic Descriptions

The code reads the romantic descriptions from the `romantic_train.txt` file and fills in the corresponding descriptions in the image dictionary.

### Write Data to CSV File

The image names, funny descriptions, and romantic descriptions from the image dictionary are written to a CSV file named `output.csv`.

### Data Preprocessing

The code reads the CSV file using pandas and performs any necessary data preprocessing.

### Copy Images to Destination Folder

Images are copied from a source folder to a destination folder based on the information in the CSV file.

### Count Files in Folder

The code counts the number of files in a specified folder.

## Model Evaluation

The code evaluates the fine-tuned model on the test dataset using various evaluation metrics, including BLEU, ROUGE, and METEOR. The evaluation process includes the following steps:

### Generate Captions for Test Images

Captions are generated for test images using the fine-tuned model. Neutral descriptions and image names are used as input. Predicted captions are compared against ground truth captions, and evaluation metrics are calculated.

### Evaluation Metrics

- **BLEU Scores:** The code calculates the BLEU score for funny, romantic, and neutral captions.
- **ROUGE Scores:** ROUGE-1, ROUGE-2, and ROUGE-L scores are calculated for funny, romantic, and neutral captions.
- **METEOR Scores:** METEOR score is calculated for funny, romantic, and neutral captions.

## Results

The table below provides the evaluation results for the fine-tuned model on the test dataset:

| CATEGORY | METRIC   | VALUE   |
|----------|----------|---------|
| Funny    | BLEU     | 0.512   |
| Romantic | BLEU     | 0.551   |
| Neutral  | BLEU     | 0.577   |
| Funny    | ROUGE-1  | 0.384   |
| Funny    | ROUGE-2  | 0.201   |
| Funny    | ROUGE-L  | 0.342   |
| Romantic | ROUGE-1  | 0.339   |
| Romantic | ROUGE-2  | 0.158   |
| Romantic | ROUGE-L  | 0.323   |
| Neutral  | ROUGE-1  | 0.552   |
| Neutral  | ROUGE-2  | 0.373   |
| Neutral  | ROUGE-L  | 0.521   |
| Funny    | METEOR   | 0.238   |
| Romantic | METEOR   | 0.222   |
| Neutral  | METEOR   | 0.273   |

These evaluation metrics provide insights into the performance of the fine-tuned model on generating captions for funny, romantic, and neutral descriptions.

## Acknowledgments
- Any acknowledgments or credits for external libraries, models, or datasets used in the code.
```

This GitHub README format provides a clear structure, using markdown syntax for headers, code blocks, and tables. Users can easily follow the instructions and understand the overview of the provided code.
