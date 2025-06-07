from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()


#ChatOpenAI 초기화
llm = ChatOpenAI()


#프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_messages([
   ("system", "You are a helpful assistant."),
   ("user", "{input}")
])


#문자열 출력 파서
output_parser = StrOutputParser()


#LLM 체인 구성
chain = prompt | llm | output_parser
result = chain.invoke({"input": "hi"})
print(result)