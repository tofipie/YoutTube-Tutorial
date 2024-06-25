import os
import torch
import streamlit as st
from langchain_groq import ChatGroq
#from langchain_community.document_loaders import WebBaseLoader
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
import time
from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings
from utils import get_data_files, reset_conversation
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.chains import LLMChain
from langchain_community.document_loaders.csv_loader import CSVLoader
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

#model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
#tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    logging.warning("GPU not found, using CPU, translation will be very slow.")

st.cache(suppress_st_warning=True, allow_output_mutation=True)
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_model(
    pretrained_model: str = "facebook/m2m100_1.2B",
    cache_dir: str = "models/",
):
    tokenizer = M2M100Tokenizer.from_pretrained(pretrained_model, cache_dir=cache_dir)
    model = M2M100ForConditionalGeneration.from_pretrained(
        pretrained_model, cache_dir=cache_dir
    ).to(device)
    model.eval()
    return tokenizer, model

st.title("Chat with Docs - AWS bedrock and Claude :) ")
st.sidebar.title("App Description")
with st.sidebar:
 st.button('New Chat', on_click=reset_conversation)
 st.write("Files loaded in VectorDB:")
 for file in get_data_files():
  st.markdown("- " + file)
 st.write('Made by Noa Cohen')

#llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",
# model_kwargs={"temperature":0.5, "max_length":512},huggingfacehub_api_token='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'),

embeddings = HuggingFaceEmbeddings(
model_name="sentence-transformers/all-MiniLM-L6-v2",
model_kwargs={"device": "cpu"},
 )
load_dotenv() #
groq_api_key = os.environ['GROQ_API_KEY']
#DB_FAISS_PATH = "vectorstores/db_faiss"
DATA_PATH = "./data/text+translation.csv"

#if "vector" not in st.session_state:
# st.session_state.embeddings = embeddings #OllamaEmbeddings()
 #st.session_state.loader = PyPDFDirectoryLoader("./pdfs/")
 #st.session_state.docs = st.session_state.loader.load()
 #st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
 #st.session_state.documents = st.session_state.text_splitter.split_documents( st.session_state.docs)
 #st.session_state.vector = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)
##########
loader = CSVLoader(file_path=DATA_PATH, metadata_columns=["hebrew"],encoding='cp1255')
docs = loader.load()
db = FAISS.from_documents(docs, embedding=embeddings)
 ##############
#db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True)
# st.session_state.vector = db

llm = ChatGroq(
 groq_api_key=groq_api_key,
 model_name='mixtral-8x7b-32768'
 )
#~~~~~~~~~~~~~~~~~~~~~~`
#translator = pipeline("translation_he_to_en", model="facebook/m2m100_418M",kwargs = ({'max_length ':500}))
#translator2 = pipeline("translation_en_to_he", model="facebook/m2m100_418M",kwargs = ({'max_length ':500}))
#~~~~~~~~~~~~~~~~~~~~~~~
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer.
I will tip you $200 if the user finds the answer helpful.
<context>
{context}
</context>
Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
#retriever = st.session_state.vector.as_retriever()
retriever =db.as_retriever(search_type="similarity")
retrieval_chain = create_retrieval_chain(retriever, document_chain)
prompt = st.text_input("Input your prompt here")

# expose this index in a retriever interface

# If the user hits enter
if prompt:
 tokenizer, model = load_model()
 src_lang = 'he'
 trg_lang = 'en'
 tokenizer.src_lang = src_lang
 with torch.no_grad():
  encoded_input = tokenizer(prompt, return_tensors="pt").to(device)
  generated_tokens = model.generate(**encoded_input, forced_bos_token_id=tokenizer.get_lang_id(trg_lang))
  translated_prompt = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

 
 #translated_prompt = translation_chain.run({"text": prompt}) #using llm
 #translated_prompt = translator(prompt) #using m2m
 #tokenizer.src_lang = "he"
 #encoded_hi = tokenizer(prompt, return_tensors="pt")
 #generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("en"))
 #translated_prompt = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

# Then pass the prompt to the LLM
 start = time.process_time()
 response = retrieval_chain.invoke({"input": translated_prompt[0]})#['translation_text']}) 
 print(f"Response time: {time.process_time() - start}")
 
 st.write('Translation:'+ translated_prompt) #added
 #st.write(response["answer"]) #translate to hebrew
#""" """"""""""""""""""""""""""""""""
 #translated_answer = translator2(response['answer']) #using m2m
 #tokenizer.src_lang = "en"
 #encoded_hi = tokenizer(response['answer'], return_tensors="pt")
 #generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("he"))
 #translated_answer = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
 #st.write(translated_answer)
# """""""""""""""""""""""""""""""
 # With a streamlit expander
 with st.expander("Document Similarity Search"):
  for i, doc in enumerate(response["context"]):
   print(f"Source Document # {i+1} : {doc.metadata['hebrew']}")
 # print(f"Source Document # {i+1} : {doc.metadata['source'].split('/')[-1]}")
   print("------ --------------------------")


