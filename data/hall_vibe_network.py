import pandas as pd
import re
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt

# 1. 무의미한 감성 단어 리스트 (Stopwords)
# 분석 결과에 따라 계속 추가해나가면 정밀도가 올라갑니다.
stopwords = ['느낌', '있는', '가능', '홀에', '느낌의', '진짜', '너무', '조금', '상당히']

def get_meaningful_pairs(text):
    # 한글 단어만 추출
    words = re.findall(r'[가-힣]{2,}', str(text))
    # 불용어 제거
    cleaned = [w for w in words if w not in stopwords]
    
    # 연속된 두 단어 쌍 생성
    pairs = []
    for i in range(len(cleaned) - 1):
        # (단어1, 단어2) 형태로 저장
        pairs.append((cleaned[i], cleaned[i+1]))
    return pairs

# 2. 모든 데이터에서 쌍 추출 및 카운트
all_pairs = []

tag_df = pd.read_csv(r'C:\Users\yiuri\OneDrive\GitHub\Recsys-Wedding\data\final_weddingDatasets.csv')

for vibe in tag_df['hall_vibe']:
    all_pairs.extend(get_meaningful_pairs(vibe))

pair_counts = Counter(all_pairs).most_common(30)

# 3. 네트워크 그래프 시각화 (맥락 파악용)
G = nx.Graph()
for (w1, w2), count in pair_counts:
    G.add_edge(w1, w2, weight=count)

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, k=0.6) # 노드 간의 거리 조절
nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='orange', alpha=0.7)
nx.draw_networkx_edges(G, pos, width=[d['weight']*0.5 for u, v, d in G.edges(data=True)], edge_color='gray')
nx.draw_networkx_labels(G, pos, font_family='Malgun Gothic', font_size=10) # 맥은 AppleGothic

plt.title("Wedding Hall Attributes Connection (Except Emotion Words)")
plt.show()