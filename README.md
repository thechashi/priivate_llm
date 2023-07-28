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

## Model Details

The two private language models used in this project are:

1. **LLM Bloom 560M Bloomz1b7**: A large language model optimized for processing unstructured text data with 560 million parameters.
2. **Red Pjama Incite 3B**: Another powerful language model with 3 billion parameters, suitable for text analysis tasks.

## Installation

To set up the environment and install the necessary dependencies for this project, follow the steps below:

1. Clone this repository to your local machine.
2. Install the required Python libraries specified in `requirements.txt`. You can use the following command:



3. Download the private language models, "LLM Bloom 560M Bloomz1b7" and "Red Pjama Incite 3B," and place them in the appropriate directories (if not included in the repository).

## Usage

To use the document classifier and summarizer, you need to follow these steps:

1. Prepare your unstructured and messy document text data in a suitable format.
2. Preprocess the data to clean and organize it, if required.
3. Run the document classifier using the "LLM Bloom 560M Bloomz1b7" model to categorize each document into specific classes or topics.
4. Employ the "Red Pjama Incite 3B" model to generate concise summaries for the classified documents.
5. Save or export the results as needed.

Detailed instructions and example code can be found in the `examples` directory.

## Speed and Lightweight Models

To further optimize the system for speed and efficiency, we have incorporated **PEFT** (Probabilistic Early Exiting Transformer) and **LORA** (Lightweight Online Real-time Attention) techniques. These techniques allow for faster inference times while maintaining accurate results.

## Privacy Considerations

Data privacy is crucial in our project. By utilizing private language models hosted within our local environment, we can ensure that sensitive data does not leave our system or leak to the outside world. This approach safeguards confidential information and complies with strict data privacy regulations.

## Contributing

We welcome contributions to this project! If you find any issues or want to enhance the functionality, feel free to open an issue or submit a pull request. Please ensure that your contributions adhere to the project's coding standards and best practices.

## License

This project is licensed under the [MIT License](LICENSE.md). You are free to use, modify, and distribute the code as long as you comply with the license terms.
