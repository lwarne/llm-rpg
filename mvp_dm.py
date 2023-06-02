
# Import necessary modules
import streamlit as st
from streamlit_chat import message
import os

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def main():
    st.set_page_config(
        page_title = "Dungeon Master LLM",
        page_icon = "🎲"
    )
    st.header("Welcome to the Dungeon Master LLM!")
    chattab, othertab = st.tabs(["chat", "Other"])
    last_response = None

    # Initialize LLM and define prompts
    llm = OpenAI(temperature=0.9)
    prompt_template = PromptTemplate(input_variables=['command'], template='{command}')

    with st.chattab():

        user_input = st.text_input("Your message: ", key="user_input")

        if user_input:
            # Create the LLMChain
            chain = LLMChain(llm=llm, prompt=prompt_template)

            # Process user command
            response = chain.run(command=user_input)
            
            # Display the response
            st.subheader("Response:")
            st.write(response)

            # Handle specific commands
            if user_input.lower() == "look":
                st.subheader("Exploration:")
                st.write("You are in a dark and mysterious dungeon. You see torches flickering in the distance, and a path leading deeper into the depths. What would you like to do next?")
            elif user_input.lower() == "move":
                st.subheader("Movement:")
                st.write("You take a step forward, following the path deeper into the dungeon. The air grows colder and the sounds of dripping water echo in the distance.")
            elif user_input.lower() == "attack":
                st.subheader("Combat:")
                st.write("You draw your sword and prepare to face the unknown dangers lurking in the shadows. Be careful, adventurer!")

            # Add more commands and corresponding actions as needed
            # For example: if user_input.lower() == "cast spell":

# Run the main function
if __name__ == '__main__':
    main()
