from llama_index import GPTListIndex,SimpleDirectoryReader, GPTVectorStoreIndex, download_loader, GPTRAKEKeywordTableIndex
from llama_index import LLMPredictor, ServiceContext, StorageContext
from llama_index import load_index_from_storage, load_indices_from_storage
from llama_index.indices.composability import ComposableGraph
from langchain import OpenAI
import os
from pathlib import Path

from apikey import oai_key

os.environ['OPENAI_API_KEY'] = oai_key

#set up context
service_context = ServiceContext.from_defaults(
    llm_predictor=LLMPredictor(
        llm=OpenAI(temperature=0, model_name="text-davinci-003", streaming=True)
    )
)
storage_context = StorageContext.from_defaults()

directory_path = './handbook_split'
index_set = {}

# Get a list of all files in the directory
files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

print(files)

# describe each index to help traversal of composed graph
index_summaries = ['describes the various setting flavor options of a campaign',
                    'describes the scope of the world for specific levels',
                    'describes the types of religions or god patheons',
                    'describes the play style options',
                    'describes world buildings core assumptions with examples and questions', 
                    'describes the process and steps of building a world and campaign', 
                    'describes how to build world shakeing campaign events and types of events', 
                    'describes various factors and orginaizations and activities assoticated', 
                    'describes how to build the settings of settlements or towns including many options', 
                    'describes options for the magic system', 
                    'describes the different scales of maps']

##### CREATE THE VECTOR STORES

# Loop through each file
for i, file in enumerate(files):
    #create storage context, resets each time
    storage_context = StorageContext.from_defaults()
    
    # Read the file content
    reader = SimpleDirectoryReader(input_files=[os.path.join(directory_path, file)])
    data = reader.load_data()

    # Create a GPTSimpleVectorIndex from the document
    cur_index = GPTVectorStoreIndex.from_documents(data, service_context=service_context, storage_context = storage_context)
    
    filename_without_extension, extension = os.path.splitext(file)
    # Store the index
    cur_index.summary = index_summaries[i]
    index_set[filename_without_extension] = cur_index

    # Save the index to disk
    filename_without_extension, extension = os.path.splitext(file)
    storage_context.persist(persist_dir=f'./storage/dm1/{filename_without_extension}')


##### LOAD THE VECTOR STORES
for file in files:
    filename_without_extension, extension = os.path.splitext(file)
    print(filename_without_extension)
    print(f'./storage/dm1/{filename_without_extension}/')
    storage_context = StorageContext.from_defaults(persist_dir=f'./storage/dm1/{filename_without_extension}/')
    cur_index = load_indices_from_storage(storage_context=storage_context)[0]
    index_set[filename_without_extension] = cur_index


# define an LLMPredictor set number of output tokens
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
storage_context = StorageContext.from_defaults()

# define a list index over the vector indices
# allows us to synthesize information across each index
graph = ComposableGraph.from_indices(
    GPTListIndex,
    [index_set[os.path.splitext(file)[0]] for file in files], 
    index_summaries=index_summaries,
    service_context=service_context,
    storage_context = storage_context,
)
root_id = graph.root_id
print(root_id)

# [optional] save to disk
storage_context.persist(persist_dir=f'./storage/dm1/chap1_graph')


