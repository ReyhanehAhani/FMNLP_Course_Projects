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

## Image and Description Processing

The provided code block is responsible for processing and organizing image descriptions from the FlickrStyle dataset. It involves reading funny and romantic descriptions from their respective files, associating them with image names, and storing the information in a CSV file.

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

This code block performs the following steps:

1. **Read Funny Descriptions:** Reads funny descriptions from the `funny_train.txt` file.

2. **Read Image Names:** Reads image names from the `train.p` file.

3. **Create Image Dictionary:** Creates a dictionary (`image_dict`) to store image names along with their funny and romantic descriptions.

4. **Read Romantic Descriptions:** Reads romantic descriptions from the `romantic_train.txt` file.

5. **Fill in Romantic Descriptions:** Fills in the romantic descriptions in the dictionary.

6. **Write Data to CSV File:** Writes the collected data to a CSV file named `output.csv`, including image names, funny descriptions, and romantic descriptions.

This organized CSV file serves as input for further tasks such as fine-tuning a model on the FlickrStyle dataset. 

## Image and Description Processing

The provided code block is responsible for processing and organizing image descriptions from the FlickrStyle dataset. It involves reading funny and romantic descriptions from their respective files, associating them with image names, and storing the information in a CSV file.

```python
import csv  # Import the CSV module for working with CSV files

# Read funny descriptions from the funny file
funny_descriptions = []
with open('/content/FlickrStyle_v0.9/humor/funny_train.txt', 'r', encoding='latin-1') as funny_file:
    funny_descriptions = funny_file.readlines()  # Read lines from the funny file

# Read image names from the train.p file
image_names = []
with open('/content/FlickrStyle_v0.9/humor/train.p', 'r', encoding='latin-1') as train_file:
    lines = train_file.readlines()  # Read lines from the train.p file
    for line in lines:
        if line.startswith('aV') or line.startswith('V'):
            image_name = line.split('_')[0][1:]  # Extract image names from lines
            image_names.append(image_name)  # Append image names to the list

# Create a dictionary to store image names and corresponding descriptions
image_dict = {}
for i, image_name in enumerate(image_names):
    image_dict[image_name] = {
        'funny_description': funny_descriptions[i].strip(),  # Add funny descriptions to the dictionary
        'romantic_description': None  # Initialize romantic descriptions as None, to be filled later
    }

# Read romantic descriptions from the romantic file
with open('/content/FlickrStyle_v0.9/romantic/romantic_train.txt', 'r', encoding='latin-1') as romantic_file:
    romantic_descriptions = romantic_file.readlines()  # Read lines from the romantic file

# Fill in the romantic descriptions in the dictionary
for i, image_name in enumerate(image_names):
    image_dict[image_name]['romantic_description'] = romantic_descriptions[i].strip()  # Add romantic descriptions

# Write the data to a CSV file
with open('output.csv', 'w', newline='') as csvfile:
    fieldnames = ['Image Name', 'Funny Description', 'Romantic Description']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)  # Create a CSV writer with specified field names

    writer.writeheader()  # Write the header row to the CSV file
    for image_name, descriptions in image_dict.items():
        writer.writerow({
            'Image Name': f'{image_name}.jpg',  # Format image name with '.jpg' extension
            'Funny Description': descriptions['funny_description'],  # Write funny descriptions to the CSV file
            'Romantic Description': descriptions['romantic_description']  # Write romantic descriptions
        })

print("CSV file created successfully.")  # Print a success message
```

This code block performs the following steps:

1. **Read Funny Descriptions:** Reads funny descriptions from the `funny_train.txt` file.
2. **Read Image Names:** Reads image names from the `train.p` file.
3. **Create Image Dictionary:** Creates a dictionary (`image_dict`) to store image names along with their funny and romantic descriptions.
4. **Read Romantic Descriptions:** Reads romantic descriptions from the `romantic_train.txt` file.
5. **Fill in Romantic Descriptions:** Fills in the romantic descriptions in the dictionary.
6. **Write Data to CSV File:** Writes the collected data to a CSV file named `output.csv`, including image names, funny descriptions, and romantic descriptions.

This organized CSV file serves as input for further tasks such as fine-tuning a model on the FlickrStyle dataset.
```
This code accomplishes the following tasks:
1. Loads image information from a CSV file into a DataFrame (`df`).
2. Checks if the destination folder exists; if not, it creates the
