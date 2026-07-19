import os
import re
import random
from typing import List, Dict

import numpy as np
import pandas as pd
import pymorphy3
import torch
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Фиксация сидов для воспроизводимости результатов
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)


class Config:
    # Параметры алгоритма и моделей
    BM25_K1: float = 4.0
    BM25_B: float = 0.5
    ALPHA: float = 0.5  # Вес векторной части в гибридном поиске
    MODEL_NAME: str = "intfloat/multilingual-e5-large"
    MAX_TEXT_LENGTH: int = 1500
    
    # Стоп-слова для удаления шума
    STOP_WORDS: set = {
        'здравствуйте', 'подскажите', 'пишете', 'получается', 'пожалуйста',
        'можно', 'нужно', 'просто', 'тоже', 'только', 'очень', 'вроде',
        'добрый', 'день', 'вечер', 'прошу', 'уточнить', 'типа', 'что',
        'вот', 'уже', 'бы', 'же', 'ли', 'будто', 'если', 'то'
    }


def clean_html(text: str) -> str:
    """Очистка текста от HTML-разметки и служебных блоков."""
    if not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.extract()
    return soup.get_text(separator=" ", strip=True)


def get_bm25_tokens(text: str) -> List[str]:
    """Токенизация, лемматизация и генерация биграмм для BM25."""
    if not isinstance(text, str):
        return []
    
    clean_text = re.sub(r'[^a-zа-яё0-9\s]', ' ', text.lower())
    tokens = clean_text.split()
    
    morph = pymorphy3.MorphAnalyzer()
    lemmas = [
        morph.parse(token)[0].normal_form 
        for token in tokens 
        if token not in Config.STOP_WORDS and morph.parse(token)
    ]
    
    final_tokens = list(lemmas)
    # Добавление биграмм для учета устойчивых выражений
    for i in range(len(lemmas) - 1):
        final_tokens.append(f"{lemmas[i]}_{lemmas[i+1]}")
    
    return final_tokens


def get_embedding_text(text: str) -> str:
    """Подготовка чистого текста для векторной модели."""
    if not isinstance(text, str):
        return ""
    clean = clean_html(text)
    return re.sub(r'\s+', ' ', clean)[:Config.MAX_TEXT_LENGTH]


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-Max нормализация массива скоров."""
    min_s, max_s = np.min(scores), np.max(scores)
    if max_s - min_s == 0:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s + 1e-8)


def build_bm25_index(articles_df: pd.DataFrame) -> BM25Okapi:
    """Создание индекса BM25 на основе заголовков и текстов статей."""
    articles_df['bm25_tokens'] = articles_df.apply(
        lambda row: get_bm25_tokens(row['title']) + get_bm25_tokens(row['body']),
        axis=1
    )
    return BM25Okapi(articles_df['bm25_tokens'].tolist(), k1=Config.BM25_K1, b=Config.BM25_B)


def build_vector_index(articles_df: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    """Генерация и нормализация эмбеддингов для базы статей."""
    articles_df['embed_text'] = (
        "passage: " + articles_df['title'].astype(str) + " " + articles_df['body'].astype(str)
    ).apply(get_embedding_text)
    
    embeddings = model.encode(
        articles_df['embed_text'].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def hybrid_search(
    query_text: str,
    bm25: BM25Okapi,
    article_embeddings: np.ndarray,
    model: SentenceTransformer,
    idx_to_id: Dict[int, int],
    top_k: int = 10
) -> List[str]:
    """Гибридный поиск с объединением скоров BM25 и векторной модели."""
    # Расчет скоров BM25
    query_tokens = get_bm25_tokens(query_text)
    bm25_scores = bm25.get_scores(query_tokens)
    
    # Расчет векторных скоров
    clean_query = get_embedding_text(query_text)
    q_emb = model.encode("query: " + clean_query, convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb)
    vector_scores = np.dot(article_embeddings, q_emb)
    
    # Нормализация и взвешенное суммирование
    norm_bm25 = normalize_scores(bm25_scores)
    norm_vector = normalize_scores(vector_scores)
    hybrid_scores = (Config.ALPHA * norm_vector) + ((1 - Config.ALPHA) * norm_bm25)
    
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    return [str(idx_to_id[idx]) for idx in top_indices]


def generate_predictions(
    test_df: pd.DataFrame,
    bm25: BM25Okapi,
    article_embeddings: np.ndarray,
    model: SentenceTransformer,
    idx_to_id: Dict[int, int]
) -> List[str]:
    """Генерация списка релевантных статей для каждого запроса."""
    predictions = []
    for _, row in test_df.iterrows():
        top_ids = hybrid_search(row['query_text'], bm25, article_embeddings, model, idx_to_id)
        predictions.append(" ".join(top_ids))
    return predictions


def main():
    print("Init models")
    model = SentenceTransformer(Config.MODEL_NAME)
    
    print("Load data")
    articles_df = pd.read_feather("articles.f")
    test_df = pd.read_feather("test.f")
    idx_to_id = {idx: aid for idx, aid in enumerate(articles_df['article_id'])}
    
    print("Build indexes")
    bm25 = build_bm25_index(articles_df)
    article_embeddings = build_vector_index(articles_df, model)
    
    print("Generate predictions")
    predictions = generate_predictions(test_df, bm25, article_embeddings, model, idx_to_id)
    test_df['answer'] = predictions
    
    output_df = test_df[["query_id", "answer"]].copy()
    output_df.to_csv("answer.csv", index=False)
    print("Done")


if __name__ == "__main__":
    main()