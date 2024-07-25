## importing libraries
import os
import subprocess
import pygame
import speech_recognition as sr
import bs4
import uuid
import streamlit as st

from openai import OpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_fireworks import ChatFireworks
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.merge import MergedDataLoader

# api keys
FIREWORKS_KEY = "<YOUR FIREWORKS API KEY>"
OPENAI_KEY = "<YOUR OPEN_AI API KEY>"

#initiating phone call
def initiate_call():
    """
    function: initiates a phone call

    input: none

    output: telephone ring followed by the AI assistant delivering the intial message.
    """
    pygame.init()
    pygame.mixer.init()
    
    file = 'teleRing.mp3'
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()
    pygame.time.wait(10000)
    pygame.mixer.music.stop()

    text = "Hello, this is the AutoBuddy showroom. How may I help you?"
    tts(text)


# ending the conversation
def end_call():
    """
    function: deliver the good-bye message

    input: none

    output: AI assistant speaks the goodbye message
    """

    text = "Thank you for calling AutoBuddy. Have a wonderful day!"
    tts(text)

# converting speech to text
def speech_to_text():
    """
    function: converts speech to text

    input: none

    output: transcribed speech
    """
    # recording audio
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Recording...")
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.listen(source)
        print("Stopped!")
    
    # converting speech to text
    text = recognizer.recognize_google(audio_data)

    return text

# text to speech
def tts(text):
    """
    function: converts text to speech and saves the speech as a MP3 file

    input: text

    output: AI-Assistant speaks
    """
    client = OpenAI(api_key = OPENAI_KEY)
    
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=f"{text}",
    ) as response:
        response.stream_to_file("speech.mp3")
    
    pygame.mixer.init()
    
    # Load your audio file
    pygame.mixer.music.load("speech.mp3")
    
    # Play the audio
    pygame.mixer.music.play()
    
    # Keep the program running while the audio is playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(5)

    os.remove('speech.mp3')

## driver function
def conversation():
    """
    function: picks up the human call and interacts

    input: none

    output: conversation with human
    """

    # initializing LLM
    llm = ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct", api_key = FIREWORKS_KEY)
    # loading vector database
    vectorstore = FAISS.load_local("knowledgebase", embeddings=HuggingFaceEmbeddings(), allow_dangerous_deserialization = True)
    # initializing retriever
    retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs = {"k": 5})
    
    # system prompt with context of chat history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    # chat template
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # creating a retriever with history
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    system_prompt = (
        "You are a knowledgable car sales assistant at the company AutoBuddy."
        "Do not give information about cars unless asked by the caller."
        "Be nice to the callers, and greet them when they call you before you answer their questions"
        "the question. If you don't know the answer, say that you "
        "don't know. Use one sentences maximum and keep the "
        "answers concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    # question answer chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # main rag chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # store to keep track of sessions
    store = {}
    session_id = str(uuid.uuid4())

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store.keys():
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    # call starts here
    initiate_call()
    while True:
        user_input = speech_to_text()
    
        if "bye" in user_input.lower():
            end_call()
            return
        else:
            response = conversational_rag_chain.invoke(
                {"input": f"{user_input}"},
                config={
                    "configurable": {"session_id": f"{session_id}"}
                },  # constructs a key "abc123" in `store`.
            )["answer"]

            tts(response)

def main():
    conversation()

#if __name__ == "__main__":
#    main()

# streamlit app
st.title('AutoBuddy Car Showroom 🚗')
st.write("""Welcome to the AutoBuddy Showroom. Call our helpline below to get more information about cars on sale and booking an appointment!""")
if st.button('Call AutoBuddy'):
    st.write("Calling AutoBuddy. Please Wait...This could take a while!")
    main()
