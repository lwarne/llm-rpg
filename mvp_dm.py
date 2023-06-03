
# Import necessary modules
import streamlit as st
from streamlit_chat import message
import os

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from apikey import oai_key

def init():
    #set keys
    os.environ['OPENAI_API_KEY'] = oai_key
    st.set_page_config(
        page_title = "Dungeon Master LLM",
        page_icon = "🎲"
    )

def main():
    init()

    #initalize chat model
    chat_model = ChatOpenAI(temperature=0.9)

    chattab, othertab = st.tabs(["chat", "Other"])
    last_response = None

    # initalize
    if "message_history" not in st.session_state:
        st.session_state.message_history = []
        st.session_state.cbmem = ConversationBufferMemory()

    # provide the first prompt
    custom_start_prompt = """ You are an expert dungon master (DM), and you are guiding a user through a fantasy LitRPG dungon and dragons world, in a text adventure style.
    Provide the user with a fun, descriptive experience. 
    Give detailed and exciting descriptions of the world, and give options about what the user can do. 
    If they user asks questions answer helpfully 
    
    Current conversation:
    {history}
    User: {input}
    DM:"""
    conversation = ConversationChain(llm = chat_model
                                        , memory = st.session_state.cbmem
                                        , verbose = True
                                        )
    conversation.prompt.template = custom_start_prompt

    with chattab:

        user_input = st.text_input("Your message: ", key="user_input")

        if st.button('submit'):
            #store human message
            st.session_state.message_history.append(HumanMessage(content=user_input))
            #fetch ai resposne
            with st.spinner("thinking"):
                response = conversation.predict(input = user_input)
                last_response = response
            #append ai response
            st.session_state.message_history.append(AIMessage(content=response))
        
        message_history = st.session_state.get('message_history', [])
        
        for i, msg in enumerate(reversed(message_history[1:])):
            if isinstance(msg, SystemMessage):
                message(msg.content, is_user = True, key = str(i)+'_system')
            elif isinstance(msg, HumanMessage):
                message(msg.content, is_user = True,key = str(i)+'_human')
            else:
                message(msg.content, is_user = False,  key = str(i)+'_ai')
    with othertab:
        st.subheader('input')
        st.write(user_input)
        st.subheader('output')
        st.write(last_response)

# Run the main function
if __name__ == '__main__':
    main()
