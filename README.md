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

## Image and Description Processing

The following code snippet focuses on processing and organizing the image descriptions from the FlickrStyle dataset. This step involves reading funny and romantic descriptions from their respective files, associating them with image names, and storing the information in a CSV file.

```python
import csv

# Read funny descriptions from the funny file
funny_descriptions = []
with open('/content/FlickrStyle_v0.9/humor/funny_train.txt', 'r', encoding='latin-1') as funny_file:
    funny_descriptions = funny_file.readlines()

# Read image names from the train.p file
image_names = []
with open('/content/FlickrStyle_v0.9/humor/train.p', 'r', encoding='latin-1') as train_file:
    lines = train_file.readlines()
    for line in lines:
        if line.startswith('aV') or line.startswith('V'):
            image_name = line.split('_')[0][1:]
            image_names.append(image_name)

# Create a dictionary to store image names and corresponding descriptions
image_dict = {}
for i, image_name in enumerate(image_names):
    image_dict[image_name] = {
        'funny_description': funny_descriptions[i].strip(),
        'romantic_description': None  # Initialize as None, to be filled later
    }

# Read romantic descriptions from the romantic file
with open('/content/FlickrStyle_v0.9/romantic/romantic_train.txt', 'r', encoding='latin-1') as romantic_file:
    romantic_descriptions = romantic_file.readlines()

# Fill in the romantic descriptions in the dictionary
for i, image_name in enumerate(image_names):
    image_dict[image_name]['romantic_description'] = romantic_descriptions[i].strip()

# Write the data to a CSV file
with open('output.csv', 'w', newline='') as csvfile:
    fieldnames = ['Image Name', 'Funny Description', 'Romantic Description']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for image_name, descriptions in image_dict.items():
        writer.writerow({
            'Image Name': f'{image_name}.jpg',
            'Funny Description': descriptions['funny_description'],
            'Romantic Description': descriptions['romantic_description']
        })

print("CSV file created successfully.")
```



The code counts the number of files in a specified folder.
```python
# Import necessary modules from the transformers library
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the processor for the pre-trained image captioning model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Load the pre-trained image captioning model
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
```

In this code snippet:

- The `AutoProcessor` and `BlipForConditionalGeneration` classes are imported from the transformers library.
- The `AutoProcessor.from_pretrained` method is used to load the processor for the pre-trained image captioning model.
- The `BlipForConditionalGeneration.from_pretrained` method is used to load the pre-trained image captioning model.

```python
# Get the tokenizer from the processor
tokenizer = processor.tokenizer

# Access the BERT model from the text_decoder of the image captioning model
bert = model.text_decoder.bert

# Define special tokens for additional categories (funny, romantic, neutral)
special_tokens_dict = {'additional_special_tokens': [funny_token, romantic_token, neutral_token]}

# Add the special tokens to the tokenizer
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# Resize the BERT model's token embeddings to accommodate the added special tokens
bert.resize_token_embeddings(len(tokenizer))
```

In this code snippet:

- The `tokenizer` is obtained from the `processor`.
- The BERT model is accessed from the `text_decoder` attribute of the image captioning model.
- Special tokens for additional categories (e.g., funny, romantic, neutral) are defined in the `special_tokens_dict`.
- The `tokenizer.add_special_tokens` method is used to add the special tokens, and the number of added tokens is stored in `num_added_toks`.
- The `resize_token_embeddings` method is called on the BERT model to adjust the token embeddings based on the new vocabulary size after adding special tokens.



```python
# Loop through epochs
for e in range(2):
    # Loop through batches using tqdm for progress visualization
    for i in (pbar := tqdm(range(0, len(train_df), BATCH_SIZE))):
        # Define the batch range
        batch = range(i, i + BATCH_SIZE)
        
        # Extract text for funny, romantic, and neutral categories
        text_funny = list(df.iloc[batch, 1])
        text_romantic = list(df.iloc[batch, 2])
        text_neutral = list(df.iloc[batch, 4])
        
        # Load images from the specified folder
        images = [Image.open(os.path.join('/content/FMNLP_dataset/flickerstyle/images', df.iloc[idx, 0])) for idx in batch]
        images = list(chain.from_iterable(repeat(images, 3)))

        # Create input text for the model
        text = [neutral_token + ':' + text + in_ + ':' for in_ in (funny_token, romantic_token, neutral_token) for text in text_neutral]
        
        # Create target text for training
        target = [text[i] + t for i, t in enumerate(text_funny + text_romantic + text_neutral)]

        # Prepare input for the processor
        pre_out = processor(text_target=target, text=text, images=images, return_tensors="pt", truncation=True, padding='max_length').to(DEVICE)
        
        # Forward pass through the model
        output = model(**pre_out)

        # Calculate loss
        loss = output.loss

        # Print and log loss every 10 steps
        if i % 10 == 0:
            pbar.set_description(f'Loss: {loss.item()}')
            writer.add_scalar('training loss', loss.item())

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

In this code snippet:

- The outer loop iterates over epochs (`for e in range(2)`).
- The inner loop iterates over batches using tqdm for progress visualization.
- Batches of text and images are extracted from the dataframe and loaded from the specified image folder.
- Input text for the model is created, including categories (funny, romantic, neutral).
- Target text for training is generated based on the input.
- The input is preprocessed using the `processor`.
- The model is forward-passed with the preprocessed input.
- Loss is calculated based on the model output.
- Loss is printed and logged every 10 steps.
- Backward pass and optimization step are performed to update model parameters.

```markdown
| Category | Metric    | Score   |
|----------|-----------|---------|
| Funny    | BLEU      | 0.512   |
| Romantic | BLEU      | 0.551   |
| Neutral  | BLEU      | 0.577   |
| Funny    | ROUGE-1   | 0.384   |
| Funny    | ROUGE-2   | 0.201   |
| Funny    | ROUGE-L   | 0.342   |
| Romantic | ROUGE-1   | 0.339   |
| Romantic | ROUGE-2   | 0.158   |
| Romantic | ROUGE-L   | 0.323   |
| Neutral  | ROUGE-1   | 0.552   |
| Neutral  | ROUGE-2   | 0.373   |
| Neutral  | ROUGE-L   | 0.521   |
| Funny    | METEOR    | 0.238   |
| Romantic | METEOR    | 0.222   |
| Neutral  | METEOR    | 0.273   |
```

The detailed table above presents the evaluation results for the fine-tuned model on the test dataset, encompassing BLEU scores, ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L), and METEOR scores. The scores are categorized based on the type of descriptions, including funny, romantic, and neutral.


This organized CSV file serves as input for further tasks such as fine-tuning a model on the FlickrStyle dataset.
```
This code accomplishes the following tasks:
1. Loads image information from a CSV file into a DataFrame (`df`).
2. Checks if the destination folder exists; if not, it creates the
