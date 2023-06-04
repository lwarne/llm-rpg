from llama_index import GPTListIndex, GPTVectorStoreIndex, LLMPredictor, ServiceContext, StorageContext
from llama_index import load_graph_from_storage ,load_indices_from_storage
from langchain import OpenAI
from llama_index.indices.composability import ComposableGraph

# do imports
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from llama_index.query_engine.transform_query_engine import TransformQueryEngine

import os

from apikey import oai_key

os.environ['OPENAI_API_KEY'] = oai_key


##### CONTEXTs
#set up service context
llm_predictor = LLMPredictor(llm=OpenAI(temperature=.8, max_tokens=512))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
# set up storage context
storage_context = StorageContext.from_defaults()

##### LOADs
# load from disk, so you don't need to build graph from scratch
root_id = 'fa932aea-71f7-414e-b8d2-4760d6832a48'
graph = load_graph_from_storage(
    root_id=root_id, 
    service_context=service_context,
    storage_context=StorageContext.from_defaults(persist_dir=f'./storage/dm1/chap1_graph/'),
)

# Get a list of all the indices
files = ['dm_1_11_flavors.pdf', 'dm_1_10_tiers.pdf', 'dm1_2_gods.pdf', 'dm_1_9_play_style.pdf', 'dm1_1.pdf', 'dm_1_7_campaign.pdf', 'dm_1_8_events.pdf', 'dm_1_5_factions.pdf', 'dm_1_4_settlements.pdf', 'dm_1_6_magic.pdf', 'dm1_3_map.pdf']
file_split = [ os.path.splitext(file)[0] for file in files ]

#import the Indexes
index_set = {}
for file in file_split:
    storage_context = StorageContext.from_defaults(persist_dir=f'./storage/dm1/{file}/')
    cur_index = load_indices_from_storage(storage_context=storage_context)[0]
    index_set[file] = cur_index
    print(f"{file}: {cur_index.index_struct.summary}")

##### DEFINE TOOLS + toolkit
# define a decompose transform
decompose_transform = DecomposeQueryTransform(
    llm_predictor, verbose=True
)

# define custom retrievers fill it with engines
custom_query_engines = {}
for index in index_set.values():
    query_engine = index.as_query_engine()
    query_engine = TransformQueryEngine(
        query_engine,
        query_transform=decompose_transform,
        transform_extra_info={'index_summary': index.index_struct.summary},
    )
    custom_query_engines[index.index_id] = query_engine

custom_query_engines[graph.root_id] = graph.root_index.as_query_engine(
    response_mode='tree_summarize',
    verbose=True,
)

# tool config
graph_config = IndexToolConfig(
    query_engine=query_engine,
    name=f"Graph Index",
    description="useful when you want to answer questions about world building, dungeon mastering, or story telling",
    tool_kwargs={"return_direct": True}
)

# Create tool configs for each individual index
index_configs = []
for f in file_split:
    query_engine = index_set[f].as_query_engine(
        similarity_top_k=3,
    )
    tool_config = IndexToolConfig(
        query_engine=query_engine, 
        name=f"Vector Index {f}",
        description=f"useful for when you want to answer questions that relate to {index_set[f].index_struct.summary} ",
        tool_kwargs={"return_direct": True}
    )
    index_configs.append(tool_config)

#create toolkit
toolkit = LlamaToolkit(
    index_configs=index_configs + [graph_config],
)

##### WORK WITH AGENT
#Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions.
#Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

agent_prefix = """
Assistant is a language model, able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant will help the user build a tool

Assistant will always use tools when answering question about building a story world

Assistant does not know about building a story world
Assistant does not know about story settings
Assistant will need to use tools to know how to describe

TOOLS:
------

Assistant has access to the following tools:"""


memory = ConversationBufferMemory(memory_key="chat_history")
llm=OpenAI(temperature=0)
agent_chain = create_llama_chat_agent(
    toolkit,
    llm,
    memory=memory,
    verbose=True,
    prefix=agent_prefix,
)

r1 = agent_chain.run(input="What are the play style options for a dungeon and dragons game?")
print(r1)

# cross_query_str = (
#     "if i want to create a story world with a dark setting, what is a good small town setting and what does the government look like?"
# )
# r2 = agent_chain.run(input=cross_query_str)
# print(r2)

# while True:
#     text_input = input("User: ")
#     response = agent_chain.run(input=text_input)
#     print(f'Agent: {response}')
