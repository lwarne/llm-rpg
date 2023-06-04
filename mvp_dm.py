
# Import necessary modules
import streamlit as st
from streamlit_chat import message
import os

from langchain.chat_models import ChatOpenAI
# from langchain.chat_models import OpenAI
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

def summarization_call(mem_input = None):
    
    if mem_input == None:
        return "There is no history to summarize"
    
    # Initialize summarization model
    summarization_model = ChatOpenAI(temperature=0.5)  # You might want to use a lower temperature for summarization

    # Set up summarization chain
    summarization_model = ConversationChain(llm=summarization_model, memory=mem_input, verbose=True)

    # Set up summarization prompt
    summarization_prompt = """
    The following is a converstion between a world building dungeon master AI and human user

    {history}

    Provide a concice summary of this information, keeping key information about the setting and physical items but dropping all flowery language
    The summary should have a high information to word ratio
    Other key information to keep: {input}

    Summary:
    """
    # Set the prompt
    summarization_model.prompt.template = summarization_prompt

    # Call the prompt
    response = summarization_model.predict(input = 'none')

    return response

def main(): 
    init()

    #initalize chat model
    chat_model = ChatOpenAI(temperature=0.9)

    chattab, othertab = st.tabs(["chat", "Other"])
    last_response = None

    # Initialize conversation history and memory if not already done
    if "message_history" not in st.session_state:
        st.session_state.message_history = []
        st.session_state.cbmem = ConversationBufferMemory()

    # Provide the first prompt
    custom_start_prompt = """ You are an expert dungon master (DM), and you are guiding a user through a fantasy LitRPG dungon and dragons world, in a text adventure style.
    Provide the user with a fun, descriptive experience. 
    Give detailed and exciting descriptions of the world, and give options about what the user can do. 
    If the user asks questions answer helpfully 
    
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
            # Check if user input is empty
            if user_input.strip() == "":
                st.warning("Please enter a message before submitting.")
            else:
                # Store human message
                st.session_state.message_history.append(HumanMessage(content=user_input))
                # Fetch AI response
                with st.spinner("writing"):
                    response = conversation.predict(input = user_input)
                    last_response = response
                # Append AI response
                st.session_state.message_history.append(AIMessage(content=response))
        
        #write message history to chat window
        message_history = st.session_state.get('message_history', [])
        for i, msg in enumerate(reversed(message_history[1:])):
            if isinstance(msg, SystemMessage):
                message(msg.content, is_user = True, key = str(i)+'_system')
            elif isinstance(msg, HumanMessage):
                message(msg.content, is_user = True,key = str(i)+'_human')
            else:
                message(msg.content, is_user = False,  key = str(i)+'_ai')
    
    with othertab:
        
        if st.button('summarize history'):
            sum_hist = summarization_call(st.session_state.cbmem)
            st.write(sum_hist)

        if st.button('print raw history'):
            readble_hist = st.session_state.cbmem.load_memory_variables({})['history']
            st.write(readble_hist)

        # if st.button('run summary of conversation'):


        # st.write(user_input)
        # st.subheader('output')
        # st.write(last_response)st.subhead

# Run the main function
if __name__ == '__main__':
    main()
