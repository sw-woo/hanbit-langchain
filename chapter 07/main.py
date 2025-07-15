# Streamlit Cloud에 배포하기 위해 requirements.txt 파일에도 pysqlite3-binary 패키지를 추가하고, 그리고 아래 주석을 해제하여서 진행 부탁드립니다. 
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button
from langchain.callbacks.base import BaseCallbackHandler
#from dotenv import load_dotenv
#load_dotenv()

#제목
st.title("ChatPDF")
st.write("---")

#OpenAI 키 입력받기
openai_key = st.text_input('OPEN_AI_API_KEY', type="password")

#파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요!", type=['pdf'])
st.write("---")

#Buy me a coffee
button(username="{계정 ID}", floating=True, width=221)

def pdf_to_document(uploaded_file):
   temp_dir = tempfile.TemporaryDirectory()
   temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
   with open(temp_filepath, "wb") as f:
       f.write(uploaded_file.getvalue())
   loader = PyPDFLoader(temp_filepath)
   pages = loader.load_and_split()
   return pages

#업로드된 파일 처리
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
    
    #Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)

    #Embedding
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=openai_key
        # With the `text-embedding-3` class
        # of models, you can specify the size
        # of the embeddings you want returned.
        # dimensions=1024
    )
    
    # 캐싱 문제 발생시 아래 코드를 2줄을 주석해제 하여서 사용하시면 됩니다.
    # import chromadb
    # chromadb.api.client.SharedSystemClient.clear_system_cache()

    #Chroma DB
    db = Chroma.from_documents(texts, embeddings_model)

    #스트리밍 처리할 Handler 생성
    class StreamHandler(BaseCallbackHandler):
       def __init__(self, container, initial_text=""):
           self.container = container
           self.text=initial_text
       def on_llm_new_token(self, token: str, **kwargs) -> None:
           self.text+=token
           self.container.markdown(self.text)

    #User Input
    st.header("PDF에게 질문해보세요!!")
    question = st.text_input("질문을 입력하세요")

    if st.button("질문하기"):
        with st.spinner('Wait for it...'):
            #Retriever
            llm = ChatOpenAI(temperature=0, openai_api_key=openai_key)
            retriever_from_llm = MultiQueryRetriever.from_llm(
                retriever=db.as_retriever(), llm=llm
            )

            #Prompt Template
            prompt = hub.pull("rlm/rag-prompt")

            #Generate
            chat_box = st.empty()
            stream_handler = StreamHandler(chat_box)
            generate_llm = ChatOpenAI(model="gpt-4o-mini",temperature=0, openai_api_key=openai_key, streaming=True, callbacks=[stream_handler])
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            rag_chain = (
                {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
                | prompt
                | generate_llm
                | StrOutputParser()
            )

            #Question
            result = rag_chain.invoke(question)