# from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()


#Loader
loader = PyPDFLoader("20240814_company_816921000.pdf")
pages = loader.load_and_split()


#Splitter
text_splitter = RecursiveCharacterTextSplitter(
   # Set a really small chunk size, just to show.
   chunk_size=300,
   chunk_overlap=20,
   length_function=len,
   is_separator_regex=False,
)


texts = text_splitter.split_documents(pages)


#Embedding
embeddings_model = OpenAIEmbeddings()


#Chroma DB
db = Chroma.from_documents(texts, embeddings_model)


#Retriever
llm = ChatOpenAI(temperature=0)
retriever_from_llm = MultiQueryRetriever.from_llm(
   retriever=db.as_retriever(), llm=llm
)


#Prompt Template
from langchain import hub
prompt = hub.pull("rlm/rag-prompt")


#Generate
def format_docs(docs):
   return "\n\n".join(doc.page_content for doc in docs)
rag_chain = (
   {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
   | prompt
   | llm
   | StrOutputParser()
)


#Question
result = rag_chain.invoke("아내가 먹고 싶어하는 음식은 무엇이야?")
print(result)