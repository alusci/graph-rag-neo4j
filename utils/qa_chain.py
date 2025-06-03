"""
Question-answering chain utilities for graph-based RAG system.

This module provides functionality to create a QA chain that uses
a retriever to get relevant context and an LLM to generate answers.
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .retrievers import retriever


def get_qa_chain():
    """
    Create and return a question-answering chain.
    
    This function constructs a LangChain pipeline that:
    1. Retrieves relevant context using the configured retriever
    2. Formats the context and question using a prompt template
    3. Generates an answer using the OpenAI LLM
    4. Parses the output as a string
    
    Returns:
        Runnable: A LangChain runnable that can be invoked with a question
                 string and returns an answer based on retrieved context.
    
    Environment Variables:
        LLM_MODEL: The OpenAI model to use for generating answers
    
    Example:
        >>> chain = get_qa_chain()
        >>> answer = chain.invoke("Who was Eleanor of Arborea?")
    """
    llm = ChatOpenAI(
        temperature=0.0,
        model=os.getenv("LLM_MODEL")
    )

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    Use natural language and be concise.
    Do not make up facts that are outside the scope of context
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnableParallel(
            {
                "context": RunnablePassthrough() | retriever,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

