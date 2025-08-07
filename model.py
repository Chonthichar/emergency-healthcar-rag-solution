import os
import json
from typing import Tuple

# Import LangChain and related components
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# =====================================================================================
# 1. INITIALIZE MODELS AND DATABASE (This code runs only once when the API starts)
# =====================================================================================

# Load environment variables from .env file
load_dotenv()

# Initialize the LLM for generation
llm = Ollama(model=os.getenv("CHAT_MODEL"))

# Initialize the Embeddings model
embeddings = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"))

# Connect to the existing Chroma vector store
vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"),
)

# Load the topics JSON for the prompt
with open('data/topics.json', 'r', encoding='utf-8') as f:
    topics_map = json.load(f)
topics_list_str = json.dumps(topics_map, indent=2)


# --- CHANGE #1: IMPROVED PROMPT ---
# This prompt is more forceful, telling the model to ignore prior knowledge.
prompt_template = PromptTemplate.from_template(
    """You are a meticulous medical expert system. Your task is to evaluate a given medical statement.
    **You must base your answer EXCLUSIVELY on the provided "Reference Text". Ignore any of your own prior knowledge.**
    
    Your task has two parts:
    1.  Determine if the statement is TRUE or FALSE according to the text.
    2.  Identify the primary medical topic of the statement from the "List of Possible Topics", choosing the one that is the most direct match.
    
    **Reference Text:**
    ---
    {context}
    ---
    
    **Medical Statement to Evaluate:**
    ---
    "{statement}"
    ---
    
    **List of Possible Topics:**
    ---
    {topics_list}
    ---
    
    Provide your answer in a single, valid JSON object. Do not add any other text, comments, or explanations.
    
    {{
        "statement_is_true": <1 for TRUE, 0 for FALSE>,
        "statement_topic": <the integer ID of the most relevant topic>
    }}
    """
)

# =====================================================================================
# 2. DEFINE THE PREDICTION FUNCTION (This code runs for every API call)
# =====================================================================================

def predict(statement: str) -> Tuple[int, int]:
    """
    Predict both binary classification (true/false) and topic classification for a medical statement
    using a Retrieval-Augmented Generation (RAG) approach.

    Args:
        statement (str): The medical statement to classify

    Returns:
        Tuple[int, int]: (statement_is_true, statement_topic)
    """
    # --- CHANGE #2: MMR RETRIEVER ---
    # 1. RETRIEVE: Find relevant and diverse documents using MMR.
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 5, 'fetch_k': 20} # Fetches 20, then selects the best 5 diverse ones.
    )
    retrieved_docs = retriever.invoke(statement)

    # 2. AUGMENT: Combine the content of the retrieved docs into a single string
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # 3. GENERATE: Create the final prompt and invoke the LLM
    final_prompt = prompt_template.format(
        context=context,
        statement=statement,
        topics_list=topics_list_str
    )
    response_str = llm.invoke(final_prompt)

    # 4. PARSE: Extract the structured JSON from the LLM's string response
    try:
        if response_str.startswith("```json"):
            response_str = response_str[7:-4]

        result_json = json.loads(response_str)
        statement_is_true = int(result_json["statement_is_true"])
        statement_topic = int(result_json["statement_topic"])

        return statement_is_true, statement_topic

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Failed to parse LLM output. Error: {e}")
        print(f"Raw LLM output: {response_str}")
        return 0, 0 # Return a default 'safe' prediction on failure