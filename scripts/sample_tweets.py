import pandas as pd

df = pd.read_csv('data/political_tweets_clean.csv', encoding='utf-8-sig')
sample = df.sample(150, random_state=42)['clean_text'].reset_index(drop=True)
sample.index += 1

for i, t in sample.iloc[100:150].items():
    print(f'{i+1}. {t}')
    print()