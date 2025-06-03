from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Neo4jVector
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
from typing import List
import os

load_dotenv()

# Define unstructured data retriever
vector_index = Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
)

# Define chain to exctract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )

llm = ChatOpenAI(
    temperature=0.0,
    model=os.getenv("LLM_MODEL")
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting organization and person entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)

entity_chain = prompt | llm.with_structured_output(Entities)

# Initialize graph DB connector
graph = Neo4jGraph()
# Create full text index if it doesn't exist
graph.query(
    "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]"
)


def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()


def run_graph_query(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
            """
            CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])

    print(f"Graph query result: {result}")

    return result


def retriever(question: str):

    structured_data = run_graph_query(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    
    combined_data = f"""
    Structured data: 
    {structured_data}
    Unstructured data:
    {"#Document ".join(unstructured_data)}
    """
    
    return combined_data
