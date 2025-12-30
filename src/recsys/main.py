import pandas as pd
import re
from transformers import pipeline

# # 1. 감성 분석 모델 로드 (한국어 다국어 모델)
# # 문장의 긍정/부정 확률을 점수로 변환합니다.
# sentiment_model = pipeline("sentiment-analysis", model="whiteknight8/korean-senti-bert")

# def get_sentiment_score(text):
#     """텍스트의 긍정 확률을 0~1 사이 점수로 반환"""
#     if not text or len(text) < 5: return 0.5
#     result = sentiment_model(text[:512])[0] # 모델 입력 제한 512자
#     # 긍정이면 score 그대로, 부정이면 (1 - score)로 계산하여 '긍정 정도' 산출
#     return result['score'] if result['label'] == 'LABEL_1' else (1 - result['score'])

# # 2. 데이터셋 로드 및 속성별 텍스트 추출
# # 원본 데이터(df)에서 '음식', '주차' 관련 문장만 분리하여 감성 분석을 돌립니다.
# def analyze_venue_data(df):
#     results = []
    
#     for _, row in df.iterrows():
#         content = row['content']
        
#         # [정규표현식] 가격 정보 추출
#         f_price = re.findall(r'식대\s*(\d+,?\d*)', content)
#         food_price = int(f_price[0].replace(',', '')) if f_price else 80000
        
#         h_price = re.findall(r'대관료\s*(\d+,?\d*)', content)
#         hall_price = int(h_price[0].replace(',', '')) * 10000 if h_price else 5000000

#         # [감성 분석] 특정 키워드 주변 문장만 추출해서 모델에 입력
#         # 예: '식사'나 '뷔페' 단어가 포함된 문장의 긍정도를 분석
#         food_sentences = " ".join(re.findall(r'[^.]*(?:식사|뷔페|음식|밥)[^.]*\.', content))
#         parking_sentences = " ".join(re.findall(r'[^.]*(?:주차|교통|셔틀|역)[^.]*\.', content))
        
#         food_score = get_sentiment_score(food_sentences)
#         parking_score = get_sentiment_score(parking_sentences)
        
#         results.append({
#             'venue_name': row['title'].split(' ')[0], # 제목에서 첫 단어를 베뉴명으로 가정
#             'food_price': food_price,
#             'hall_price': hall_price,
#             'food_score': food_score,
#             'parking_score': parking_score
#         })
#     return pd.DataFrame(results)

# 3. 랭킹화 및 최종 점수 계산 (Weighting & Ranking)
def rank_venues(analyzed_df, user_prefs):
    df = analyzed_df.copy()
    
    # 가격 정규화 (Min-Max Scaling 후 반전: 쌀수록 높은 점수)
    df['price_norm'] = 1 - (df['food_price'] - df['food_price'].min()) / (df['food_price'].max() - df['food_price'].min() + 1e-5)
    
    # Final Score 계산 (가중치 적용)
    df['final_score'] = (
        df['price_norm'] * user_prefs['w_price'] +
        df['food_score'] * user_prefs['w_food'] +
        df['parking_score'] * user_prefs['w_parking']
    )
    
    return df.sort_values(by='final_score', ascending=False)

# --- 실행부 ---
# 사용자가 입력한 가중치 (합이 1이 되도록 설정)
user_input_weights = {
    'w_price': 0.3, 
    'w_food': 0.5,   # 밥을 가장 중요하게 생각하는 유저
    'w_parking': 0.2
}

# 1) CSV 로드 (수집한 데이터)
# raw_df = pd.read_csv('test.csv') 
# 2) 분석 (BERT 감성 분석 모델 작동)
# analyzed_df = analyze_venue_data(raw_df)
# 3) 랭킹 (가중치 반영)
# final_ranked_list = rank_venues(analyzed_df, user_input_weights)