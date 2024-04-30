# pip install llama-index-llms-ollama
from llama_index.core import VectorStoreIndex, QueryBundle, Settings
from llama_index.core.postprocessor import PIINodePostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from index import es_vector_store

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Use Public LLM to send user query and Related Documents
public_llm = OpenAI()
local_llm = Ollama(model="mistral")

pii_processor = PIINodePostprocessor(llm=local_llm)
index = VectorStoreIndex.from_vector_store(es_vector_store)

# This query_engine, for a given user query retrieves top 10 similar documents from
# Elasticsearch vector database and sends the documents along with the user query to the LLM.
# Note that documents are masked using the local llm via PIINodePostprocessor
# so that PII/Sensitive data is not sent to the public LLM.
query_engine = index.as_query_engine(public_llm, similarity_top_k=10, node_postprocessors=[pii_processor])

query = "Give me summary of fire related claims that customers raised."
bundle = QueryBundle(query, embedding=Settings.embed_model.get_query_embedding(query))
result = query_engine.query(bundle)
print(result)
