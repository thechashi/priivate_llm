# Deciphering Messy and Poorly Formatted Texts: Private Language Models for Document Summarization and Classification 

## Table of Contents

1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Model Details](#model-details)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Speed and Lightweight Models](#speed-and-lightweight-models)
7. [Privacy Considerations](#privacy-considerations)
8. [Contributing](#contributing)
9. [License](#license)

## Introduction

Welcome to the Deciphering Messy and Poorly Formatted Texts project! This repository contains code and instructions for building a document text summarization and classification system using private language models. The goal of this project is to efficiently process messy, unstructured text data from document files while ensuring data privacy by utilizing locally hosted private language models.

## Project Overview

The primary objectives of this project are as follows:


1. Implement text summarization to generate concise and informative summaries for each document.
2. Perform document text classification to categorize documents into predefined classes or topics.
3. Utilize private language models, namely "Bloom 560M", "Bloomz1b7" and "Red Pjama Incite 3B," to ensure data does not leave our local environment.
4. Utilizing Parameter-Efficient Fine-Tuning (PEFT) approach to efficiently fine-tune model parameters and  decreasing the computational and storage costs.

## Model Details

The three private language models used in this project are:

1. **LLM Bloom 560M**: A large language model optimized for processing unstructured text data with 560 million parameters.
2. **LLM Bloomz1b7**: A large language model optimized for processing unstructured text data with 1.7 billion parameters.
3. **Red Pajama Incite 3B**: Another powerful language model with 3 billion parameters, suitable for text analysis tasks.

## Installation

To set up the environment and install the necessary dependencies for this project, follow the steps below:

1. Clone this repository to your local machine.
2. Install the required Python libraries using `setup.sh`.


## Usage

To use the document classifier and summarizer, you need to follow these steps:

1. Prepare your unstructured and messy document text data in a suitable format.
2. Preprocess the data to clean and organize it, if required.
3. Employ the "Red Pjama Incite 3B" model to generate concise summaries for the classified documents.
4. Run the document classifier using the "LLM Bloom 560M" or "LLM Bloom 1b7" model to categorize each document into specific classes or topics.


## Speed and Lightweight Models

To further optimize the system for speed and efficiency, we have incorporated **PEFT** (Parameter-Efficient Fine-Tuning). PEFT methods tune large language models for new tasks by only updating a small portion of the model parameters, keeping most of the original pretrained parameters frozen. This greatly reduces the computational and storage costs compared to full fine-tuning of all parameters. A key benefit is avoiding catastrophic forgetting, where full fine-tuning can degrade the original capabilities of the LLM. PEFT enables efficient adaptation even with limited data, and results in models that generalize better to out-of-domain data. By selectively tuning just the minimal parameters needed for a new task, PEFT provides a practical and scalable approach to unlocking the versatility of large pretrained models like Claude across diverse applications.

## Privacy Considerations

Data privacy is crucial in our project. By utilizing private language models hosted within our local environment, we can ensure that sensitive data does not leave our system or leak to the outside world. This approach safeguards confidential information and complies with strict data privacy regulations.

