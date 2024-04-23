# query.py
from llama_index.core import VectorStoreIndex, QueryBundle, Response, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from index import es_vector_store

# Local LLM to send user query to
local_llm = Ollama(model="mistral")
Settings.embed_model= OllamaEmbedding("mistral")

index = VectorStoreIndex.from_vector_store(es_vector_store)

# This query_engine, for a given user query retrieves top 10 similar documents from
# Elasticsearch vector database and sends the documents along with the user query to the LLM.
# Note that documents are sent as-is. So any PII/Sensitive data is sent to the LLM.
query_engine = index.as_query_engine(local_llm, similarity_top_k=10)

query="Give me summary of water related claims that customers raised."
bundle = QueryBundle(query, embedding=Settings.embed_model.get_query_embedding(query))
result = query_engine.query(bundle)
print(result)


response="""In various conversations, several customers have expressed dissatisfaction with their insurance company regarding water damage claims. Some claims were denied due to specific exclusions such as lack of timely notification, gradual wear and tear, or acts of nature. Others were denied without any explanation given. Customers have reported stress and financial hardship due to the denial of these claims. The insurance agents involved in these conversations acknowledged the customers' frustration and promised to escalate their concerns for further review. Some agents clarified that certain types of water damage, such as flooding, are covered under specific circumstances but may require separate policies.
"""

#
