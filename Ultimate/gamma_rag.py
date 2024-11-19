from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, PromptTemplate
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

# load data
input_dir_path = "RAG_Folder"
loader = SimpleDirectoryReader(
            input_dir = input_dir_path,
            required_exts=[".pdf"],
            recursive=True
        )
docs = loader.load_data()
# print(f"Loaded {len(docs)} documents")


#embedding
embed_model = HuggingFaceEmbedding( model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)


#Vector database
# ====== Create vector store and upload indexed data ======
Settings.embed_model = embed_model # we specify the embedding model to be used
index = VectorStoreIndex.from_documents(docs)

#Query Engine
# setting up the llm
llm = Ollama(model="llama3.5", request_timeout=120.0) 

# ====== Setup a query engine on the index previously created ======
Settings.llm = llm # specifying the llm to be used
query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)


qa_prompt_tmpl_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
            "Query: {query_str}\n"
            "Answer: "
            )

qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

response = query_engine.query('Create a Table of what needs to be fixed in the Home to improve the property value. Have a price range for each item.')
print(response)