from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import streamlit as sl
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from tqdm import tqdm
#import comtypes.client
from PIL import  Image
from constants import user_api_key
import numpy as np



# to create a new file named vectorstore in your current directory.
def load_knowledgeBase():
        embeddings=OpenAIEmbeddings(api_key=user_api_key)
        DB_FAISS_PATH = "Model/faiss_model"
        db = FAISS.load_local(DB_FAISS_PATH, embeddings)
        return db

def load_prompt():
        prompt = """ You need to answer the question in the sentence as same as in the  pdf content. . 
        Given below is the context and question of the user.
        context = {context}
        question = {question}
        if the answer is not in the pdf , answer "i donot know what the hell you are asking about"
         """
        prompt = ChatPromptTemplate.from_template(prompt)
        return prompt
        
#to load the OPENAI LLM
def load_llm():
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=user_api_key)
        return llm
        

def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

def save_uploadedfile(uploadedfile): #Fonction appelée pour s'assurer du bon format du fichier audio
    with open(os.path.join("Input/", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())

def split_document(docs):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"Split blog post into {len(all_splits)} sub-documents.")
    embeddings = OpenAIEmbeddings(request_timeout=60)
    vectors = FAISS.from_documents(docs, embeddings)
    vectors.save_local("Model/faiss_model")
    

def load_pdf(file):
   
    loader = PyPDFLoader(file)
    docs = loader.load()
    print(len(docs))
    #print(docs[1].page_content[0:10])
    split_document(docs)
    #print(docs[1])

if __name__=='__main__':
    file_list = []
    
    sl.logo('Img/logo-1.PNG',size="large", link=None, icon_image=None)
    display = Image.open('Img/QA_logo.jpg')
    display = np.array(display)
    sl.title("GulF-AI PDF bot")

    col1, col2 = sl.columns(2)
    col1.image(display, width = 300)
    col2.header("About")
    col2.markdown(
        """ 
        This PDF bot allow user to discuss with theirs PDF documents by upload and query them. You can ask any question and get answer from
        you documents.
        """)
    
    #sl.header("welcome to the pdf bot")
    #Création du système de drop&drag pour upload un fichier
    Files=sl.file_uploader("Upload your File(s) :",accept_multiple_files=True)
        
    if Files is not None:
        for file in Files:
            save_uploadedfile(file)
           
    else:
        sl.text('You have not uploaded any files yet')
    
    
    #comtypes.CoInitialize()
    infolder=os.path.dirname(os.path.abspath(__file__))
    #path = os.path.abspath("..\\Input\\")
    path = "./Input/"
    files = os.listdir(path)
    
        
    col1, col2, col3 = sl.columns(3)

    with col1:
        if sl.button("Run Analysis"):
            if len(os.listdir('./Input/')) == 0:
                sl.write("no input yet")
            else:    
                #sl.write("you uploded")
                for file in files:
                    print(file)
                    load_pdf(path+"\\"+file)
    knowledgeBase=load_knowledgeBase()
    llm=load_llm()
    prompt=load_prompt()
    #with col3:
        #if st.button('Remove files'):
     #   if sl.button('New File'):
      #      for file in tqdm(os.listdir("./input/")):
       #         if file == None:
        #            continue
         #       in_path = os.path.join("./input/", file)
          #      os.remove(in_path)
           # for file in tqdm(os.listdir(OUTPUT_PATH)):
            #    if file == None:
             #       continue
              #  out_path = os.path.join(OUTPUT_PATH, file)
               # os.remove(out_path)"""
    query=sl.text_input('Enter your question')
    if(query):
        #getting only the chunks that are similar to the query for llm to produce the output
        similar_embeddings=knowledgeBase.similarity_search(query)
        similar_embeddings=FAISS.from_documents(documents=similar_embeddings, embedding=OpenAIEmbeddings(api_key=user_api_key))
        
        #creating the chain for integrating llm,prompt,stroutputparser
        retriever = similar_embeddings.as_retriever()
        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
        
        response=rag_chain.invoke(query)
        sl.write(response)
                    
    
    
    
    
    
    
        
