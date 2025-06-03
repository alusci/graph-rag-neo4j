"""
Main script for running graph-based question answering queries.

This script demonstrates how to use the QA chain to ask questions
about the ingested knowledge graph data.
"""
from dotenv import load_dotenv
from utils.qa_chain import get_qa_chain
import warnings
import logging
import os

def setup_logging():
    """
    Configure logging and warning settings.
    
    Suppresses Python warnings and sets Neo4j logging to ERROR level
    to reduce verbose output during query execution.
    """
    warnings.filterwarnings("ignore")
    logging.getLogger("neo4j").setLevel(logging.ERROR)


def main():
    """
    Main function to demonstrate question answering capabilities.
    
    Sets up the environment, creates a QA chain, and runs sample
    queries about Eleanor of Arborea to showcase the system's
    question-answering capabilities.
    """
    setup_logging()
    load_dotenv()

    # Get Q&A chain
    chain = get_qa_chain()

    print(chain.invoke(os.getenv("TEST_QUERY_1")))
    print(chain.invoke(os.getenv("TEST_QUERY_2")))


if __name__ == "__main__":
    main()







