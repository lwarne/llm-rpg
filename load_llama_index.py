#https://gpt-index.readthedocs.io/en/latest/guides/tutorials/building_a_chatbot.html

from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, download_loader, GPTRAKEKeywordTableIndex
from llama_index import LLMPredictor, ServiceContext, StorageContext, load_index_from_storage
from langchain import OpenAI
import os

from apikey import oai_key

os.environ['OPENAI_API_KEY'] = oai_key

#set up service context
service_context = ServiceContext.from_defaults(
    llm_predictor=LLMPredictor(
        llm=OpenAI(temperature=0, model_name="text-davinci-003")#, streaming=True)
    )
)
# set up storage context
storage_context = StorageContext.from_defaults(persist_dir=f'./storage/chapter_1_dm_short')

#load index
index = load_index_from_storage(storage_context=storage_context)

#create query engine
query_engine = index.as_query_engine(
    # streaming=True, 
    similarity_top_k=3
)

#ask a question
#what are the most important aspects of world building? Show statements in bullet form and show page reference after each statement'
response = query_engine.query("what are the types of government? and what is best for a dark world building setting?")
print(response)

#https://gpt-index.readthedocs.io/en/latest/reference/response.html response object


#print sources
for node in response.source_nodes:
    print('-----')
    text_fmt = node.node.text.strip().replace('\n', ' ')[:1000]
    print(f"Text:\t {text_fmt} ...")
    print(f'Metadata:\t {node.node.extra_info}')
    print(f'Score:\t {node.score:.3f}')