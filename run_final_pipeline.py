"""
Run Final Pipeline — Script-Bridged Hinglish Intent Classifier
Data Sprint 2.0 (Anveshan, NIE Mysuru) — NLP Track
"""
import os
import sys
import time
import json
import argparse
import hashlib
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix, hstack

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate Hinglish Intent Classifier")
    parser.add_argument("--from-scratch", action="store_true", help="Recompute all folds from scratch ignoring any cache")
    return parser.parse_args()

def main():
    args = parse_args()
    print("="*75)
    print("RUN FINAL PIPELINE — SCRIPT-BRIDGED HINGLISH INTENT CLASSIFIER")
    print(f"Mode: {'FROM-SCRATCH (No Cache)' if args.from_scratch else 'STANDARD'}")
    print("="*75)
    
    SEED = 42
    np.random.seed(SEED)
    
    # Transliteration mapper
    VOWELS = {'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo', 'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'ऑ': 'o', 'ऍ': 'e'}
    MATRAS = {'ा': 'aa', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo', 'ृ': 'ri', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ॉ': 'o', 'ॅ': 'e'}
    CONSONANTS = {'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng', 'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny', 'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n', 'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n', 'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm', 'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h', 'क़': 'q', 'ख़': 'kh', 'ग़': 'gh', 'ज़': 'z', 'ड़': 'd', 'ढ़': 'dh', 'फ़': 'f'}
    HALANT = '्'; ANUSVARA = 'ं'; CHANDRABINDU = 'ँ'; VISARGA = 'ः'; NUKTA = '़'

    def devanagari_to_roman(text):
        text = str(text)
        res = []
        i = 0
        n = len(text)
        while i < n:
            c = text[i]
            if i + 1 < n and text[i+1] == NUKTA:
                c = c + NUKTA; i += 1
            if c in VOWELS: res.append(VOWELS[c])
            elif c in CONSONANTS:
                base = CONSONANTS[c]
                if i + 1 < n:
                    nxt = text[i+1]
                    if nxt == HALANT: res.append(base); i += 1
                    elif nxt in MATRAS: res.append(base + MATRAS[nxt]); i += 1
                    elif nxt in [ANUSVARA, CHANDRABINDU]: res.append(base + 'a')
                    elif nxt in CONSONANTS or nxt in VOWELS: res.append(base + 'a')
                    elif nxt.isspace() or nxt in '.,!?\"\'()[]-;:': res.append(base)
                    else: res.append(base + 'a')
                else: res.append(base)
            elif c == ANUSVARA or c == CHANDRABINDU: res.append('n')
            elif c == VISARGA: res.append('h')
            elif c in MATRAS: res.append(MATRAS[c])
            else: res.append(c)
            i += 1
        return ''.join(res)

    def classify_script(text):
        has_dev = any('\u0900' <= ch <= '\u097F' for ch in text)
        has_lat = any('a' <= ch.lower() <= 'z' for ch in text)
        if has_dev and has_lat: return 'mixed'
        elif has_dev: return 'devanagari'
        elif has_lat: return 'roman'
        return 'other'

    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    sample_sub = pd.read_csv('sample_submission.csv')

    train_df['script'] = train_df['query_text'].apply(classify_script)
    test_df['script'] = test_df['query_text'].apply(classify_script)

    train_df['trans'] = train_df['query_text'].apply(lambda x: devanagari_to_roman(x) if any('\u0900' <= ch <= '\u097F' for ch in x) else str(x))
    test_df['trans'] = test_df['query_text'].apply(lambda x: devanagari_to_roman(x) if any('\u0900' <= ch <= '\u097F' for ch in x) else str(x))

    # Build Twin Grouping
    exact_map = defaultdict(list)
    for idx, s in enumerate(train_df['trans']):
        exact_map[s.strip().lower()].append(idx)

    rows, cols = [], []
    for indices in exact_map.values():
        if len(indices) > 1:
            first = indices[0]
            for other in indices[1:]:
                rows.extend([first, other])
                cols.extend([other, first])

    vec_grp = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=1)
    X_grp = vec_grp.fit_transform(train_df['trans'])
    nn = NearestNeighbors(n_neighbors=15, metric='cosine')
    nn.fit(X_grp)
    dists, indices = nn.kneighbors(X_grp)

    for i in range(len(train_df)):
        for j, d in zip(indices[i][1:], dists[i][1:]):
            if d < 0.25:
                rows.append(i); cols.append(j)

    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(train_df), len(train_df)))
    n_components, labels = connected_components(adj, directed=False)
    train_df['group_id'] = labels

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_df['fold'] = -1
    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(train_df, train_df['target'], train_df['group_id'])):
        train_df.loc[val_idx, 'fold'] = fold

    # Strict Assertions
    for gid, grp in train_df.groupby('group_id'):
        assert grp['fold'].nunique() == 1, f"Group {gid} spans multiple folds!"
    for (s, lbl), grp in train_df.groupby([train_df['trans'].str.strip().str.lower(), 'target']):
        assert grp['fold'].nunique() == 1, f"Duplicate transliteration spans folds!"

    print(f"Twin groups: {n_components} | Assertions verified.")

    # Train Production Model
    aug_rows_full = train_df[train_df['script'].isin(['devanagari', 'mixed'])].copy()
    aug_rows_full['query_text'] = aug_rows_full['trans']
    train_full = pd.concat([train_df, aug_rows_full], ignore_index=True)

    vec_c_prod = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    vec_w_prod = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, sublinear_tf=True)

    X_c_tr = vec_c_prod.fit_transform(train_full['query_text'])
    X_w_tr = vec_w_prod.fit_transform(train_full['query_text'])
    X_full = hstack([X_c_tr, X_w_tr])

    clf_prod = LogisticRegression(C=6.0, max_iter=1000, random_state=SEED)
    clf_prod.fit(X_full, train_full['target'])

    class_counts = train_df['target'].value_counts().sort_index().values
    prior = class_counts / len(train_df)
    weights_prod = 1.0 / (prior ** 0.200)

    def predict(text: str) -> dict:
        t0 = time.perf_counter()
        if len(text) > 250:
            return {
                'label': 55,
                'confidence': 1.0,
                'latency_ms': (time.perf_counter() - t0) * 1000,
                'is_fallback': True,
                'reason': 'open_set_long_prose'
            }
        
        has_dev = any('\u0900' <= ch <= '\u097F' for ch in text)
        trans = devanagari_to_roman(text) if has_dev else text
        
        if trans != text:
            texts_to_vec = [text, trans]
            fc = vec_c_prod.transform(texts_to_vec)
            fw = vec_w_prod.transform(texts_to_vec)
            f_comb = hstack([fc, fw])
            probs = clf_prod.predict_proba(f_comb)
            p_avg = 0.5 * probs[0] + 0.5 * probs[1]
        else:
            fc = vec_c_prod.transform([text])
            fw = vec_w_prod.transform([text])
            f_comb = hstack([fc, fw])
            p_avg = clf_prod.predict_proba(f_comb)[0]
            
        p_adj = p_avg * weights_prod
        p_norm = p_adj / p_adj.sum()
        pred_label = int(np.argmax(p_norm))
        conf = float(np.max(p_norm))
        latency_ms = (time.perf_counter() - t0) * 1000
        
        return {
            'label': pred_label,
            'confidence': conf,
            'latency_ms': latency_ms,
            'is_fallback': False,
            'reason': 'model_prediction'
        }

    # Generate Test Predictions
    print("Generating final test predictions...")
    test_preds = []
    for q in test_df['query_text']:
        test_preds.append(predict(q)['label'])

    sub = pd.DataFrame({
        'example_id': test_df['example_id'],
        'target': test_preds
    })

    assert len(sub) == 4912
    assert (sub['example_id'].values == sample_sub['example_id'].values).all()
    assert sub['target'].dtype in [np.int64, np.int32, int]
    assert not sub.isnull().any().any()
    assert sub['target'].min() >= 0 and sub['target'].max() <= 55

    sub.to_csv('submission.csv', index=False)
    sub_sha256 = hashlib.sha256(open('submission.csv', 'rb').read()).hexdigest()
    print(f"submission.csv successfully generated and verified! SHA-256: {sub_sha256}")

if __name__ == '__main__':
    main()
