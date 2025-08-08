# Emergency Healthcare RAG Challenge Solution

In this repository, we present our solution for the Emergency Healthcare RAG Challenge. Our mission was to build a fast, accurate, and private system to help emergency department staff validate medical statements and identify their clinical topic.

To tackle this, we designed and implemented a Retrieval-Augmented Generation (RAG) system that runs completely offline. This approach ensures we meet the strict constraints on speed, privacy, and memory usage required in a critical healthcare environment.

## Our Approach: An Overview

At the core of our project is a RAG pipeline that leverages a local Large Language Model (LLM) and a specialized vector database. When a medical statement comes in, our system doesn't just guess the answer. Instead, it first retrieves factually relevant information from a knowledge base of medical articles we prepared. We then provide this context to our LLM, guiding it to make a highly accurate and grounded classification.

Our workflow is divided into two main stages:

1. Data Ingestion (ingest.py): This is a one-time setup step where we process all the provided medical articles. We wrote a script that reads the documents, splits them into manageable chunks, and then converts them into vector embeddings. These embeddings are stored in a local ChromaDB database, creating a searchable knowledge base.

2. Inference (api.py & model.py): This is the real-time part of our solution. We built an API that receives a medical statement, searches our vector database for the most relevant context, and then uses a local LLM to generate the final prediction about the statement's truthfulness and topic.
 
## System Architecture & Key Components

We built our solution using a stack of offline, efficient tools to ensure we met all the competition's requirements. Here’s a breakdown of the key files and the role they play.

### 1. ingest.py — Building the Knowledge Base
This script is the foundation of our RAG system. Its only job is to prepare the data for fast and accurate retrieval.

- Our Process:

    1. Load Topic Map: We start by loading the data/topics.json file to create a map between   topic names (like "Pulmonary Embolism") and their unique IDs (like 63).

    2. Scan & Read: The script scans the data/topics/ directory. For each topic folder, it reads all the .md articles within it.

    3. Chunk & Structure: We use a RecursiveCharacterTextSplitter to break down large articles into smaller, overlapping text chunks. This preserves the semantic meaning across splits. For each chunk, we attach important metadata: the source_file, the topic_name (from the folder name), and the topic_id (from our map).

    4. Embed & Store: Finally, we use the nomic-embed-text model via Ollama to convert each chunk into a vector embedding. We then store these vectors and their rich metadata in a local ChromaDB database.

### 2. model.py — The Core Prediction Logic

This is the brain of our operation. It contains the predict function that orchestrates the entire RAG process during inference.

- Our Process:

    1. Load Resources: The function initializes the LLM (llama3) and connects to the ChromaDB vector store we created earlier.

    2. Retrieve Context: When a statement is received, we first use the vector store to find the most semantically similar document chunks from our knowledge base.

    3. Augment Prompt: We then construct a detailed prompt for the LLM. This prompt includes the original medical statement along with the relevant context we just retrieved.

    4. Generate Prediction: We send this "augmented" prompt to the LLM and ask it to return a structured JSON object containing its prediction for statement_is_true and statement_topic.

    5. Parse & Return: The final step is to parse the LLM's response and return the clean, validated prediction.

### 3. api.py — The Inference Server

To make our model accessible, we wrapped it in a lightweight and fast API using FastAPI.

- Endpoints:

    - GET /: A simple landing page to confirm the service is running.

    - GET /api: Provides service uptime information.

    - POST /predict: This is the main endpoint. It accepts a medical statement, passes it to the predict() function in model.py, validates the output, and returns the final JSON prediction.

### 4. example.py — Testing Our Model

To quickly test our predict function without running the full API, we created this simple script. It loads a sample statement from the training data, gets a prediction from our model, and compares it against the true answer, printing whether the prediction was correct.

## How to Run Our Solution

To run the solution, you first need to set up your environment by installing Python and Ollama, and then downloading the required llama3 and nomic-embed-text models.

After that, the process involves six main steps:

1. Clone the project repository from GitHub.

2. Install the necessary Python packages using the requirements.txt file.

3. Configure the environment by creating and setting up a .env file.

4. Build the local knowledge base by running the ingest.py script (a one-time step).

5. Start the server by running api.py.

6. Test the setup (optional) using the example.py script.

## Our Design Choices & Rationale

- Offline First: We chose Ollama and ChromaDB because they are powerful, open-source tools that run entirely locally. This allowed us to build a solution that is 100% private and requires no internet connection, which is critical for a healthcare setting.

- RAG for Accuracy and Trust: We believe a RAG approach is superior to fine-tuning for this task. By grounding the LLM's reasoning in factual data from the provided articles, we significantly reduce the risk of hallucinations and produce more reliable and trustworthy answers.

- Metadata is Key: During ingestion, we were careful to attach rich metadata (topic_name and topic_id) to every document chunk. This structured approach was a key part of our strategy, as it makes it much easier for the LLM to correctly identify the topic of a given statement.

## Evaluation & Results

To measure the performance of our solution, we used the accuracy metric defined by the competition for both the truthfulness and topic classification tasks.

### Evaluation Metric:

The accuracy is calculated as the number of correct predictions divided by the total number of predictions for each task independently.

Truthfulness Accuracy: Correct statement_is_true predictions / Total predictions

Topic Accuracy: Correct statement_topic predictions / Total predictions

### Our Performance:

We evaluated our model against the provided validation set. The results below demonstrate the effectiveness of our RAG-based approach.

Metric	Validation Set Score
Truthfulness Accuracy	[XX.X]%
Topic Accuracy	[XX.X]%

(Note: These results are based on the validation set. The final performance on the private evaluation set may differ.)

### How to Reproduce:

You can run a quick check on a single sample using our example.py script. For a full evaluation on a dataset, a dedicated evaluation script would be used, following the logic in model.py to process each statement in the validation file.
