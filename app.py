import os
from time import time
import asyncio
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl
from langchain_groq import ChatGroq


# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Directory containing the PDF files
PDF_DIRECTORY = r"C:\Users\sangh\OneDrive\Desktop\Cbt\pdfs"
# Directory to store Chroma embeddings
EMBEDDINGS_DIRECTORY = r"C:\Users\sangh\OneDrive\Desktop\Cbt\embeddings"
# Global variable to hold the model instance
model_instance = None

# Retrieve API keys from environment variables
groq_api_key = os.getenv('GROQ_API_KEY')

# Ensure the API keys are loaded correctly
if not groq_api_key:
    raise ValueError("Groq API key is not set. Please set the GROQ_API_KEY environment variable.")

@cl.on_chat_start
async def on_chat_start():
    # List available PDFs
    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith('.pdf')]
    
    # Create a single message with the greeting and the list of available PDFs with numbers
    greeting_message = "Hi, please choose a PDF from the list below by entering the corresponding number:\n\n"
    pdf_list_message = "\n".join([f"{idx + 1}. {pdf_file}" for idx, pdf_file in enumerate(pdf_files)])
    combined_message = greeting_message + pdf_list_message
    
    # Send the combined message
    await cl.Message(content=combined_message).send()

    # Store the list of available PDFs in user session
    cl.user_session.set("pdf_files", pdf_files)

async def load_vector_store(pdf_name):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    persist_directory = os.path.join(EMBEDDINGS_DIRECTORY, pdf_name)
    
    if os.path.exists(persist_directory):
        print(f"Loading existing embeddings for {pdf_name}...")
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        return None

async def get_model_instance():
    global model_instance
    if model_instance is None:
        print("Loading the model for the first time...")
        model_instance = ChatGroq(
            groq_api_key=groq_api_key,
            model_name='llama3-8b-8192'
        )
    return model_instance

@cl.on_message
async def main(message: cl.Message):
    # Check if the user has selected a PDF
    selected_pdf = cl.user_session.get("selected_pdf")
    if not selected_pdf:
        try:
            # Validate the selected PDF number
            pdf_files = cl.user_session.get("pdf_files", [])
            selected_pdf_index = int(message.content) - 1

            if 0 <= selected_pdf_index < len(pdf_files):
                selected_pdf = pdf_files[selected_pdf_index]
                cl.user_session.set("selected_pdf", selected_pdf)

                await cl.Message(content=f"You have selected {selected_pdf}. Checking for existing embeddings...").send()

                # Measure time taken to load the Chroma vector store
                start_time = time()
                docsearch = await load_vector_store(selected_pdf)
                end_time = time()
                print(f"Time taken to load Chroma vector store: {end_time - start_time} seconds")

                if docsearch is None:
                    await cl.Message(content="Please create embeddings for this file and try again.").send()
                    return

                # Initialize message history for conversation
                message_history = ChatMessageHistory()

                # Memory for conversational context
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    output_key="answer",
                    chat_memory=message_history,
                    return_messages=True,
                )

                try:
                    # Measure time taken to create a ConversationalRetrievalChain
                    start_time = time()
                    model = await get_model_instance()  # Get the model instance
                    chain = ConversationalRetrievalChain.from_llm(
                        model,
                        chain_type="stuff",
                        retriever=docsearch.as_retriever(),
                        memory=memory,
                        return_source_documents=False,  # Disable returning source documents
                    )
                    end_time = time()
                    print(f"Time taken to create ConversationalRetrievalChain: {end_time - start_time} seconds")
                except Exception as e:
                    print(f"Error creating ConversationalRetrievalChain: {e}")
                    await cl.Message(content="Failed to create conversational chain. Please check the model name and try again.").send()
                    return

                # Store the chain in user session
                cl.user_session.set("chain", chain)

                await cl.Message(content="PDF embeddings loaded successfully. You can now ask your questions.").send()
            else:
                await cl.Message(content="Invalid number. Please choose a valid number from the list.").send()
        except ValueError:
            await cl.Message(content="Invalid input. Please enter a number corresponding to the PDF.").send()
    else:
        # Retrieve the chain from user session
        chain = cl.user_session.get("chain")

        if chain is None:
            await cl.Message(content="Chain is not initialized. Please refresh and try again.").send()
            return

        # Callbacks happen asynchronously/parallel
        cb = cl.AsyncLangchainCallbackHandler()

        try:
            # Measure time taken to retrieve response
            start_time = time()
            res = await chain.ainvoke(message.content, callbacks=[cb])
            end_time = time()
            print(f"Time taken to retrieve response: {end_time - start_time} seconds")

            answer = res["answer"]
        except Exception as e:
            print(f"Error invoking chain: {e}")
            await cl.Message(content="Failed to retrieve response. Please try again.").send()
            return

        # Return the answer
        await cl.Message(content=answer).send()
