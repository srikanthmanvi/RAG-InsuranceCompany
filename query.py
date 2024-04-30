# query.py
from llama_index.core import VectorStoreIndex, QueryBundle, Settings
from llama_index.llms.openai import OpenAI

from index import es_vector_store

# Use Public LLM to send user query and Related Documents
llm = OpenAI()

index = VectorStoreIndex.from_vector_store(es_vector_store)

# This query_engine, for a given user query retrieves top 10 similar documents from
# Elasticsearch vector database and sends the documents along with the user query to the LLM.
# Note that documents are sent as-is. So any PII/Sensitive data is sent to the LLM.
query_engine = index.as_query_engine(llm, similarity_top_k=10)

query="Give me summary of water related claims that customers raised."
bundle = QueryBundle(query, embedding=Settings.embed_model.get_query_embedding(query))
result = query_engine.query(bundle)
print(result)


# response="""Customers have raised various water-related claims, including issues such as water damage in basements, burst pipes, hail damage to roofs, and denial of claims due to reasons like lack of timely notification, maintenance issues, gradual wear and tear, and pre-existing damage. In each case, customers expressed frustration with claim denials and sought fair evaluations and decisions regarding their claims.
# """
