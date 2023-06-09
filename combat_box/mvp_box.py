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
from random import randint


class Action(BaseModel):
    action_type: str
    target: str
    item: str
    description: str

class Actor:

    def __init__(self, hit_points: int, description: str, player_flag: bool, name: str):
        """
        Initializes an Actor with the given hit points, description, and player flag.
        """
        self.hit_points = hit_points
        self.description = description
        self.player_flag = player_flag
        self.name = name

    def describe_for_dm(self):
        if self.player_flag == True:
            return f"The player is {self.description} with {self.hit_points} out of 100 hit points"
        else:
            return f"The NPC is {self.description} with {self.hit_points} out of 100 hit points"
    
    def get_name(self):
        return self.name
    
    def apply_damage(self, damage: int):
        self.hit_points -= damage

    def get_hit_points(self):
        return self.hit_points

    def generate_attack_action_string(self, enemies, llm = None, return_raw_response = False):
        if llm == None:
            llm = OpenAI(temperature=0.5)
            
        # Create prompt template for attack action
        attack_prompt_template = PromptTemplate(
            input_variables=['player_name','player_desc', 'enemy_name'],
            template="""You are the player {player_name}, {player_desc}. You want to attack the enemy {enemy_name}, . 

            Describe your attack action. include the target of the attack, and the item you use

            Attack action: 
            """
        )

        # Create LLMChain for attack action
        attack_chain = LLMChain(llm=llm, prompt=attack_prompt_template)

        # Select a random enemy from the list of enemies
        enemy = enemies[randint(0, len(enemies) - 1)]

        # Call attack chain with the relevant inputs
        response = attack_chain({
            'player_name': self.name,
            'player_desc': self.description,
            'enemy_name': enemy.get_name()
        })

        if return_raw_response:
            return response
        else:
            return response['text']

def extract_action(user_input, player, npc_list, llm = None):
   
    # Set up LLM
    if llm == None:
        llm = OpenAI(temperature=0.1)

    # Define prompt template for action extraction
    action_checker_template = PromptTemplate(
        input_variables=['player_name', 'user_input', 'npc_names'],
        template="""You are a language parsing machine that extracts information from text and responds in the desired format.
    Your goal is to parse a user input string and identify the following information: Action_Type, Target, Item.

    The player is {player_name}.
    User input: {user_input}.

    Other characters' names are {npc_names}.

    Possible action types are: Attack, Cast.
    The Target is the exactName character that will be affected by or receive the effect of the action.
    The Item is the tool or method of the action.

    Respond in a JSON dictionary format following this structure:
    "action_type": Action_Type, "target": Target, "item": Item

    DM Response:
    """
    )

    # Set up action checker chain
    checker_chain = LLMChain(llm=llm, prompt=action_checker_template)

    # Prepare NPC names
    npc_names = ", ".join([npc.get_name() for npc in npc_list])

    # Call description chain
    response = checker_chain({'player_name': player.get_name(), 'user_input': user_input, 'npc_names': npc_names})

    # Parse the response
    parsed = json.loads(response['text'])
    action_type = parsed['action_type']
    target = parsed['target']
    item = parsed['item']

    # return parsed
    return Action(action_type=action_type, target=target, item=item, description = user_input)

def describe_situation(actor_list, llm = None, return_raw_response = False):

    if llm == None:
        llm = OpenAI(temperature=0.2)
        
    # Define prompt template for situation description
    description_prompt_template = PromptTemplate(
        input_variables=['actor_descriptions'],
        template="""You are an expert dungeon master (DM), and you are describing a combat situation.

    The characters of the situation:
    {actor_descriptions}

    The setting is a cave underground lit by a fire.

    As the DM, describe the situation, making sure to represent and accurately describe the health of each character relative to their total in descriptive language.

    DM:
    """
    )

    # Create a description for each actor
    actor_descriptions = "\n".join([actor.describe_for_dm() for actor in actor_list])


    # Set up description chain
    description_chain = LLMChain(llm=llm, prompt=description_prompt_template)

    # Call description chain
    response = description_chain({'actor_descriptions': actor_descriptions})

    if return_raw_response:
        return response
    else:
        return response['text']

def resolve_action(player, actor_list, action, llm = None, return_raw_response = False):
    if llm == None:
        llm = OpenAI(temperature=0.1)
    
    # Get a random number between 1 and 20 for damage
    max_damage = 20
    damage = randint(1, max_damage)
    # damage = 5

    # Deduct damage from the target's health
    target_actor = next(actor for actor in actor_list if actor.name == action.target)
    target_actor.apply_damage(damage)

    # print(f'damage roll {damage}')

    # Create prompt template for action description
    description_prompt_template = PromptTemplate(
        input_variables=['player_name', 'target_name', 'action_description', 'item','damage', 'max_damage','attacker_desc','target_desc'],
        template="""As the Dungeon Master (DM), you describe the action and its impact. 
    Be creative in the description that you give and the description of the damange

    As the DM, make sure the damage of the action is reflected in the textual description you give
    an attack that deals close to max damage should stagger the enemy and leave a large wound
    an attack that deals close to zero damage should leave something like a light scratch on the enemy

    the attacker is {attacker_desc} and the enemy target is {target_desc}
    
    The action you should describe is:
    {player_name} {action_description} {target_name} using {item} and deals {damage} damage out of {max_damage} max damage.
    
    DM:
    """
    )

    # Create LLMChain for action description
    action_resolution_chain = LLMChain(llm=llm, prompt=description_prompt_template)

    # Call description chain with the relevant inputs
    response = action_resolution_chain({
        'player_name': player.name,
        'target_name': target_actor.name,
        'action_description': action.description,
        'item': action.item,
        'damage': damage,
        'max_damage': max_damage,
        'attacker_desc': player.describe_for_dm,
        'target_desc': target_actor.describe_for_dm
    })

    if return_raw_response:
        return response
    else:
        return response['text']

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
    player = Actor(hit_points= 30, description="A brave knight with a sword and shield", player_flag=True, name = 'Knight')
    npc = Actor(hit_points= 30, description="A savage goblin with a jagged knife", player_flag=False, name = 'Goblin')
    actor_list = [player,npc]


    # Define the number of rounds
    num_rounds = 20

    # Loop through multiple rounds
    for round in range(1, num_rounds + 1):
        print(f"--- Round {round} ---")

        # Describe the situation at the start of each round
        situation_description = describe_situation(actor_list)
        print(situation_description)

        # Player attacks
        attack_action_string_player = player.generate_attack_action_string([npc])
        action_player = extract_action(attack_action_string_player, player, [npc])
        resolution_text_player = resolve_action(player, actor_list, action_player)
        print(f'player action string: {attack_action_string_player}')
        print(resolution_text_player)
        print(f"Goblin health: {npc.hit_points}")

        # NPC attacks
        attack_action_string_npc = npc.generate_attack_action_string([player])
        action_npc = extract_action(attack_action_string_npc, npc, [player])
        resolution_text_npc = resolve_action(npc, actor_list, action_npc)
        print(f'goblin action string: {attack_action_string_npc}')
        print(resolution_text_npc)
        print(f"Knight health: {player.hit_points}")

        # Check if any actor's health is 0
        if player.hit_points <= 0 or npc.hit_points <= 0:
            break

    # Describe the final situation
    situation_description = describe_situation(actor_list)
    print(situation_description)    

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
