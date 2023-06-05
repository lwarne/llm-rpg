#set up enviroment
import sys,os
sys.path.insert(0, '..')
from apikey import oai_key
os.environ['OPENAI_API_KEY'] = oai_key
import json


#imports
import streamlit as st
from streamlit_chat import message

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.chains import LLMChain

from typing import List
from pydantic import BaseModel

class Actor:
    """
    A class to represent an Actor in a game.

    Attributes
    ----------
    hit_points : int
        The health points of the actor.
    description : str
        A brief description of the actor.
    player_flag : bool
        A flag indicating if the actor is a player. False if the actor is a Non-Player Character (NPC).

    Methods
    -------
    __init__(self, hit_points: int, description: str, player_flag: bool):
        Initializes an Actor with the given hit points, description, and player flag.
    """

    def __init__(self, hit_points: int, description: str, player_flag: bool):
        """
        Initializes an Actor with the given hit points, description, and player flag.
        """

        self.hit_points = hit_points
        self.description = description
        self.player_flag = player_flag

    def describe_for_dm(self):
        if self.player_flag == True:
            return f"The player is {self.description} with {self.hit_points} out of 100 hit points"
        else:
            return f"The NPC is {self.description} with {self.hit_points} out of 100 hit points"



def init():
    #set keys
    os.environ['OPENAI_API_KEY'] = oai_key
    st.set_page_config(
        page_title = "Dungeon Master LLM",
        page_icon = "🎲"
    )

def main(): 
    init()

    #initalize actors
    player = Actor(hit_points= 100, description="A brave knight with a sword and shield", player_flag=True)
    npc = Actor(hit_points= 100, description="A savage goblin with a jagged knife", player_flag=False)

    # # Describe the siuation + setting
    # description_prompt_template = PromptTemplate(
    #     input_variables = ['player_desc','npc_desc']
    #     , template = """You are an expert dungeon master (DM), and you are describing a combat situation
    
    # The charcters of the situation:
    # {player_desc}
    # {npc_desc}
    
    # The setting is a cave underground lit by a fire

    # As the DM describe the situation, make sure to represent and accurately describe the health of each character realtive to their total in descriptive language
    
    # DM:
    # """
    # )

    #open an LLM
    llm = OpenAI(temperature=0.0)
    
    # #set up description chain
    # description_chain = LLMChain(llm=llm, prompt=description_prompt_template)
    
    # #call description chain
    # response = description_chain({
    #                             'player_desc':player.describe_for_dm(),
    #                             'npc_desc': npc.describe_for_dm()
    #                             })
    # print(response['text'])


    #user action
    user_action = "slash the sword at the goblins face"

    #parse out into target, item, action_type

    # Describe the siuation + setting
    action_checker_template = PromptTemplate(
        input_variables = ['user_input']
        , template = """You are a language parseing machine that extracts information from text and responds in the desired format
    Your goal is to parse a user input string and identify the following information: Action_Type, Target, Item 

    user input:
    {user_input}

    Possible action types are: Attack, Cast
    The Target is the is character that will be affected by or receive the effect of the action
    The Item is the tool or method of the action

    Respond in a json dictionary format following exactly 
    "action_type":Action_Type,"target":Target,"item":Item
    
    DM Response:
    """
    )

    #set up attack parser chain
    checker_chain = LLMChain(llm=llm, prompt=action_checker_template)
    
    #call description chain
    response = checker_chain({'user_input':user_action})
    print(response)
    print(response['text'])

    #parse
    parsed = json.loads(response['text'])
    action_type = parsed['action_type']
    target = parsed['target']
    item = parsed['item']

    print(parsed)




# Run the main function
if __name__ == '__main__':
    main()




    # llm = OpenAI(temperature=0.9)


#     conversation = ConversationChain(llm = chat_model
#                                         , memory = st.session_state.cbmem
#                                         , verbose = True
#                                         )
#     conversation.prompt.template = custom_start_prompt

#     chattab, othertab = st.tabs(["chat", "Other"])
#     last_response = None

#     with chattab:

#         user_input = st.text_input("Your message: ", key="user_input")

#         if st.button('submit'):
#             # Check if user input is empty
#             if user_input.strip() == "":
#                 st.warning("Please enter a message before submitting.")
#             else:
#                 # Store human message
#                 st.session_state.message_history.append(HumanMessage(content=user_input))
#                 # Fetch AI response
#                 with st.spinner("writing"):
#                     response = conversation.predict(input = user_input)
#                     last_response = response
#                 # Append AI response
#                 st.session_state.message_history.append(AIMessage(content=response))
        
#         #write message history to chat window
#         message_history = st.session_state.get('message_history', [])
#         for i, msg in enumerate(reversed(message_history[1:])):
#             if isinstance(msg, SystemMessage):
#                 message(msg.content, is_user = True, key = str(i)+'_system')
#             elif isinstance(msg, HumanMessage):
#                 message(msg.content, is_user = True,key = str(i)+'_human')
#             else:
#                 message(msg.content, is_user = False,  key = str(i)+'_ai')
    
#     with othertab:
        
#         if st.button('summarize history'):
#             sum_hist = summarization_call(st.session_state.cbmem)
#             st.write(sum_hist)

#         if st.button('print raw history'):
#             readble_hist = st.session_state.cbmem.load_memory_variables({})['history']
#             st.write(readble_hist)


# llm = OpenAI(temperature=0.9)
# prompt = PromptTemplate(
#     input_variables=["product"],
#     template="What is a good name for a company that makes {product}?",
# )
