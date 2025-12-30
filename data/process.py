import pandas as pd
import re

# 1. 추출할 태그 리스트 (영문 컬럼명)
tags = [
    'name', 'hall_vibe', 'catering', 'parking_facility', 'transport_access', 
    'bridal_room', 'guest_flow', 'pricing_detail', 'service_etc', 'disadvantage', 'in_guarantee'
]

# column = ['hall_name_raw', 'full_text', 'gpt-processed']
# crawling_df = pd.read_csv(r'C:\Users\yiuri\OneDrive\GitHub\Recsys-Wedding\data\20251230_141950_gpt_results.csv',header=None, names=column)
crawling_df = pd.read_csv(r'C:\Users\yiuri\OneDrive\GitHub\Recsys-Wedding\data\20251230_141950_gpt_results.csv')


def parse_gpt_tags(text):
    text = str(text)
    extracted_data = {}
    
    for tag in tags:
        # 정규표현식 설명:
        # <tag> : 시작 태그를 찾음
        # (.*?) : 다음 태그(<)가 나오거나 문자열 끝($)이 나올 때까지의 내용을 캡처
        # (?=<|$) : 전방 탐색을 통해 다음 태그 직전까지만 가져옴
        pattern = rf"<{tag}>(.*?)(?=<|$)"
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            extracted_data[tag] = match.group(1).strip()
        else:
            extracted_data[tag] = "None"
            
    return pd.Series(extracted_data)


tag_df = crawling_df['gpt-processed'].apply(parse_gpt_tags)


final_df = pd.concat([crawling_df, tag_df], axis=1)


print(tag_df.head(2))
tag_df.to_csv('final_weddingDatasets.csv', index=True, encoding='utf-8-sig')
