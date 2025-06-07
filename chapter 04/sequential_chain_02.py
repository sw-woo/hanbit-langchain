import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# .env 파일에서 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ChatOpenAI 모델 초기화
# Temperature 설정 (모델의 응답 다양성 조절)
openai_llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY, temperature=0.7)

# --- 프롬프트 템플릿 정의 ---

# 프롬프트 1: 리뷰 번역
prompt1 = PromptTemplate(
    input_variables=['review'],
    template="다음 숙박 시설 리뷰를 한글로 번역하세요.\n\n{review}"
)

# 프롬프트 2: 번역된 리뷰 요약
prompt2 = PromptTemplate.from_template(
    "다음 숙박 시설 리뷰를 한 문장으로 요약하세요.\n\n{translation}"
)

# 프롬프트 3: 번역된 리뷰 감성 점수 평가
prompt3 = PromptTemplate.from_template(
    "다음 숙박 시설 리뷰를 읽고 0점부터 10점 사이에서 부정/긍정 점수를 매기세요. 숫자만 대답하세요.\n\n{translation}"
)

# 프롬프트 4: 원본 리뷰 언어 식별
prompt4 = PromptTemplate.from_template(
    "다음 숙박 시설 리뷰에 사용된 언어가 무엇인가요? 언어 이름만 답하세요.\n\n{review}"
)

# 프롬프트 5: 요약에 대한 공손한 답변 생성 (원본 언어 사용)
prompt5 = PromptTemplate.from_template(
    "다음 숙박 시설 리뷰 요약에 대해 공손한 답변을 작성하세요.\n답변 언어:{language}\n리뷰 요약:{summary}"
)

# 프롬프트 6: 생성된 답변을 한국어로 번역
prompt6 = PromptTemplate.from_template(
    "다음 숙박 시설 리뷰를 한국어로 번역해주세요. \n 리뷰 번역 {reply1}"
)

# --- LCEL을 사용한 체인 구성 요소 정의 ---
# 각 단계는 '프롬프트 | LLM | 출력 파서'의 형태로 구성됩니다.

# 단계 1: 리뷰 번역 체인
translate_chain_component = prompt1 | openai_llm | StrOutputParser()

# 단계 2: 번역된 리뷰 요약 체인
summarize_chain_component = prompt2 | openai_llm | StrOutputParser()

# 단계 3: 감성 점수 평가 체인
sentiment_score_chain_component = prompt3 | openai_llm | StrOutputParser()

# 단계 4: 언어 식별 체인
language_chain_component = prompt4 | openai_llm | StrOutputParser()

# 단계 5: 첫 번째 답변 생성 체인
reply1_chain_component = prompt5 | openai_llm | StrOutputParser()

# 단계 6: 두 번째 답변 (한국어 번역) 생성 체인
reply2_chain_component = prompt6 | openai_llm | StrOutputParser()


# --- LCEL을 사용하여 전체 체인 결합 ---
# RunnablePassthrough.assign을 사용하여 각 단계의 출력을 다음 단계의 입력으로 전달하고,
# 중간 결과들을 딕셔너리에 누적합니다.

combined_lcel_chain = (
    RunnablePassthrough.assign(
        # 입력: {'review': '원본 리뷰 텍스트'}
        # translate_chain_component는 'review' 키를 사용하여 호출됩니다.
        # 출력: {'review': '...', 'translation': '번역된 텍스트'}
        translation=lambda x: translate_chain_component.invoke({"review": x["review"]})
    )
    | RunnablePassthrough.assign(
        # 입력: {'review': '...', 'translation': '번역된 텍스트'}
        # summarize_chain_component와 sentiment_score_chain_component는 'translation' 키를 사용합니다.
        # language_chain_component는 원본 'review' 키를 사용합니다.
        # 출력: {'review': '...', 'translation': '...', 'summary': '요약된 텍스트', 'sentiment_score': '감성 점수', 'language': '언어'}
        summary=lambda x: summarize_chain_component.invoke({"translation": x["translation"]}),
        sentiment_score=lambda x: sentiment_score_chain_component.invoke({"translation": x["translation"]}),
        language=lambda x: language_chain_component.invoke({"review": x["review"]})
    )
    | RunnablePassthrough.assign(
        reply1=lambda x: reply1_chain_component.invoke({"language": x["language"], "summary": x["summary"]})
    )
    | RunnablePassthrough.assign(
        reply2=lambda x: reply2_chain_component.invoke({"reply1": x["reply1"]})
    )
)

# 숙박 시설 리뷰 입력
review_text = """
The hotel was clean and the staff were very helpful.
The location was convenient, close to many attractions.
However, the room was a bit small and the breakfast options were limited.
Overall, a decent stay but there is room for improvement.
"""

# 체인 실행 및 결과 출력
try:
   # .invoke() 메서드에 초기 입력을 딕셔너리 형태로 전달합니다.
   result = combined_lcel_chain.invoke(input={'review': review_text})

   # 결과 딕셔너리에서 각 키를 사용하여 값을 출력합니다.
   print(f'translation 결과: {result.get("translation", "N/A")} \n')
   print(f'summary 결과: {result.get("summary", "N/A")} \n')
   print(f'sentiment_score 결과: {result.get("sentiment_score", "N/A")} \n')
   print(f'language 결과: {result.get("language", "N/A")} \n')
   print(f'reply1 결과: {result.get("reply1", "N/A")} \n') 
   print(f'reply2 결과: {result.get("reply2", "N/A")} \n')
except Exception as e:
   print(f"Error: {e}")