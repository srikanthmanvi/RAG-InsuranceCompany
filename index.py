# index.py
# pip install sentence-transformers
# pip install llama-index-embeddings-openai
# pip install llama-index-embeddings-huggingface
# pip install openai
# pip install llama-index-llms-openai

import json
import os

from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

# Load .env file contents into env
load_dotenv('.env')

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# ElasticsearchStore is a VectorStore that
# takes care of Elasticsearch Index and Data management.
es_vector_store = ElasticsearchStore(index_name="convo_index",
                                     vector_field='conversation_vector',
                                     text_field='conversation',
                                     es_cloud_id=os.getenv("ELASTIC_CLOUD_ID"),
                                     es_api_key=os.getenv("ELASTIC_API_KEY"))


def get_documents_from_file(file):
    """Reads a json file and returns list of Documents"""

    with open(file=file, mode='rt') as f:
        conversations_dict = json.loads(f.read())

    # Build Document objects using fields of interest.
    documents = [Document(text=item['conversation'],
                          metadata={"conversation_id": item['conversation_id']})
                 for
                 item in conversations_dict]
    return documents


def main():
    # LlamaIndex Pipeline configured to take care of chunking, embedding
    # and storing the embeddings in the vector store.
    llamaindex_pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=350, chunk_overlap=50),
            Settings.embed_model
        ],
        vector_store=es_vector_store
    )

    # Load data from a json file into a list of LlamaIndex Documents
    documents = get_documents_from_file(file="conversations.json")

    llamaindex_pipeline.run(documents=documents)
    print(".....Indexing Data Completed.....\n")


if __name__ == "__main__":
    main()
