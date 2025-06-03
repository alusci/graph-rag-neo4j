"""
Data ingestion script for creating a knowledge graph from Wikipedia content.

This module loads Wikipedia articles, processes them into chunks, converts them
to graph format using LLM, and stores them in a Neo4j graph database.
"""
import os
from dotenv import load_dotenv
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from tqdm import tqdm

load_dotenv()

QUERY=WIKI_QUERY=os.getenv("WIKI_QUERY")
CHUNK_SIZE=512
OVERLAP=32
MODEL=os.getenv("LLM_MODEL")


def main():
    """
    Main function to orchestrate the data ingestion process.
    
    This function:
    1. Loads Wikipedia documents based on the query
    2. Splits documents into chunks for processing
    3. Converts documents to graph format using LLM
    4. Ingests the graph data into Neo4j database
    """
    # Load raw docs from wikipedia
    wiki_docs = WikipediaLoader(query=QUERY).load()

    # Tokenize documents using TokenTextSplitter
    text_splitter = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
    docs = text_splitter.split_documents(wiki_docs[:5])

    # Ingest data into graph database
    llm = ChatOpenAI(
        temperature=0.0,
        model_name=MODEL
    )
    llm_transformer = LLMGraphTransformer(llm=llm)

    graph = Neo4jGraph()

    print("Converting documents to graph format...")
    graph_docs = []
    for doc in tqdm(docs, desc="Processing documents"):
        graph_doc = llm_transformer.convert_to_graph_documents([doc])
        graph_docs.extend(graph_doc)

    print("Adding graph documents to Neo4j...")
    graph.add_graph_documents(
        graph_docs,
        baseEntityLabel=True,
        include_source=True
    )
    print("Ingestion complete!")


if __name__ == "__main__":
    main()


