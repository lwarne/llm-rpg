#https://github.com/jerryjliu/llama_index/blob/main/docs/examples/citation/pdf_page_reference.ipynb
#https://colab.research.google.com/drive/1uL1TdMbR4kqa0Ksrd_Of_jWSxWt1ia7o?usp=sharing


from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, download_loader, GPTRAKEKeywordTableIndex
from llama_index import LLMPredictor, ServiceContext, StorageContext
from langchain import OpenAI
import os

from apikey import oai_key

os.environ['OPENAI_API_KEY'] = oai_key

#set up context
service_context = ServiceContext.from_defaults(
    llm_predictor=LLMPredictor(
        llm=OpenAI(temperature=0, model_name="text-davinci-003", streaming=True)
    )
)
storage_context = StorageContext.from_defaults()

#import pdf
reader = SimpleDirectoryReader(input_files=['chapter_1_dm_short.pdf'])
data = reader.load_data()

#create index
index = GPTVectorStoreIndex.from_documents(data, service_context=service_context, storage_context = storage_context)

#set the summary
index.summary = 'chapter 1 of the dungeon masters guide talking about world building'
#save to disk
storage_context.persist(persist_dir=f'./storage/chapter_1_dm_short')

##---test

#create query engine
query_engine = index.as_query_engine(
    streaming=True, 
    similarity_top_k=3
)

#ask a question
#what are the most important aspects of world building? Show statements in bullet form and show page reference after each statement'
response = query_engine.query("what are the types of government? and what is best for a dark world building setting?")
response.print_response_stream()

#print sources
for node in response.source_nodes:
    print('-----')
    text_fmt = node.node.text.strip().replace('\n', ' ')[:1000]
    print(f"Text:\t {text_fmt} ...")
    print(f'Metadata:\t {node.node.extra_info}')
    print(f'Score:\t {node.score:.3f}')