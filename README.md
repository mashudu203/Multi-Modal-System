# Multi-Modal Image Retrieval System

# Overview

This project implements a multi-modal image retrieval system using CLIP and FAISS. Users provide a text query, and the system returns the top-5 matching images from a dataset.

#Dataset

You must manually download the dataset from Kaggle before running the system.

Go to this Kaggle dataset:https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset/data?select=test_data_v2

Download and extract the test2_data_v2 folder.

Place it in a directory of your choice.

## Setup & Installation

Clone the repository:

~git clone https://github.com/yourusername/multi-modal-System.git
  <br>cd multi-modal-retrieval

## Install dependencies:

`pip install -r requirements.txt

Run the data preparation script (update the dataset path in prepare_data.py first!):

`python prepare_data.py

## Start the backend:

`uvicorn api:app --reload

## Start the front-end:
front-end includes index.html and style.css for UI styling

`streamlit run Multimodal.py

## Running Tests

To test the system:

`pytest tests/

## Assumptions

1.Users will manually download the dataset from Kaggle.

2.The dataset is placed in the correct directory before running prepare_data.py.

3.The system is tested on Python 3.8+.



