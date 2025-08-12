# -*- coding: utf-8 -*-
"""
Continuation of Medical Concwpt Annotation Tool with Relation Extraction
complete_medical_pipeline.py

Refactored pipeline: n-gram entity recognition -> SapBERT + FAISS normalization -> rule-based relation extraction.

Author: assistant (refactor for user)
Date: 2025-08-12

Notes:
- Update CONFIG paths to point to your FAISS index, MCN CSV, and relation CSV.
- Requires: transformers, torch, faiss, pandas, numpy, nltk, fuzzywuzzy
  (Install python-Levenshtein to speed up fuzzywuzzy if desired)
- Ensure NLTK punkt + stopwords are downloaded:
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
"""

from typing import List, Dict, Tuple, Optional
import logging
import re
import math
import itertools

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from fuzzywuzzy import fuzz
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# -----------------------------
# CONFIGURATION (edit these)
# -----------------------------
CONFIG = {
    # SapBERT model used for vectorisation (tokenizer + model)
    "SAPBERT_MODEL": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",

    # CSV containing concepts - must include columns: conceptId, term, semantic_type (or adapt in code)
    "DESCRIPTION_CSV": "/mnt/Data/CS4H-Lab/akhila/data/conceptmapping/data/en/MCN_data_new.csv",

    # FAISS index path (pre-built embedding index compatible with SapBERT embeddings)
    "FAISS_INDEX_PATH": "/mnt/Data/CS4H-Lab/akhila/data/conceptmapping/data/embeddingspace/FAISS_IP_new_prepro_SCT.idx",

    # Relation guideline CSV (must include 'Domain concept tag', 'Range concept tag', 'Relation')
    "RELATION_CSV": "/mnt/Data/CS4H-Lab/akhila/data/knowledgegraphs/data/relation_guidelines.csv",

    # ngram hyper-parameters (min & max tokens per n-gram)
    "NGRAM_MIN": 1,
    "NGRAM_MAX": 5,

    # FAISS / matching hyper-parameters
    "TOP_K": 10,
    # Minimum similarity (FAISS returned score) to keep candidate. Keep this consistent with your index (0-1).
    "SIM_THRESHOLD": 0.85,
    "SYN_THRESHOLD": 70,    # syntactic similarity threshold (0–100)
    # Whether to filter by semantic types list (set to False to disable)
    "USE_SEMANTIC_FILTER": True,

    # Semantic types allowed when semantic filter is active
    "SEMANTIC_TYPE_FILTER": [
        'procedure', 'finding', 'disorder', 'body structure', 'qualifier value',
        'substance', 'Morphologic abnormality', 'Observable entity', 'Physical object', 'Regime therapy'
    ],
    # Max characters allowed for a candidate span (avoid overly long spans)
    "MAX_SPAN_CHARS": 120,
}

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MedicalPipeline")

# -----------------------------
# NLTK stopwords setup
# -----------------------------
try:
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    logger.warning("NLTK stopwords not available; proceeding with empty stopword set.")
    STOPWORDS = set()

NEGATIONS = {"n't", 'neither', 'never', 'no', 'nobody', 'none', 'nor', 'not', 'nothing', 'nowhere'}
ATTRS = {'so', 'am', 'about', 'above', 'after', 'against', 'all', 'and', 'before'}
STOPWORDS = STOPWORDS.difference(NEGATIONS).difference(ATTRS)

# -----------------------------
# Utility functions
# -----------------------------
def simple_preprocess(text: str) -> str:
    """Light preprocessing: tokenize, remove stopwords, punct normalization, collapse whitespace."""
    if not text:
        return ""
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.lower() not in STOPWORDS]
    cleaned = " ".join(tokens)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def generate_token_spans(text: str) -> List[Tuple[str, int, int]]:
    """
    Return list of (token, char_start, char_end) for word tokens.
    Uses regex to find word tokens so we preserve character offsets.
    """
    tokens = []
    for m in re.finditer(r"\w+", text):
        tokens.append((m.group(0), m.start(), m.end()))
    return tokens

def generate_ngrams_with_offsets(text: str, min_n: int, max_n: int) -> List[Dict]:
    """
    Generate n-grams (min_n..max_n tokens) with character offsets.
    Returns list of dicts: {'text': span_text, 'start': start_char, 'end': end_char, 'token_count': n}
    """
    token_spans = generate_token_spans(text)
    n_tokens = len(token_spans)
    ngrams = []
    for n in range(min_n, max_n + 1):
        if n > n_tokens:
            break
        for i in range(0, n_tokens - n + 1):
            start_char = token_spans[i][1]
            end_char = token_spans[i + n - 1][2]
            span_text = text[start_char:end_char]
            ngrams.append({
                'text': span_text,
                'start': start_char,
                'end': end_char,
                'token_count': n
            })
    return ngrams

# -----------------------------
# Pipeline class
# -----------------------------
class MedicalPipeline:
    def __init__(self, config: Dict = CONFIG, device: Optional[int] = None):
        self.config = config

        # Device selection (-1 for cpu, >=0 for cuda device)
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device

        # Load SapBERT tokenizer + model
        logger.info("Loading SapBERT model: %s", config["SAPBERT_MODEL"])
        self.tokenizer = AutoTokenizer.from_pretrained(config["SAPBERT_MODEL"], use_fast=True)
        self.model = AutoModel.from_pretrained(config["SAPBERT_MODEL"])
        if torch.cuda.is_available() and self.device >= 0:
            self.model = self.model.to("cuda")

        # Load concept descriptions CSV
        logger.info("Loading description CSV: %s", config["DESCRIPTION_CSV"])
        # Ensure columns exist: conceptId, term, semantic_type (if 'semantic_type' is missing adjust below)
        self.descriptions = pd.read_csv(config["DESCRIPTION_CSV"], dtype=str, na_filter=False)
        # normalize column names if needed
        for col in ['conceptId', 'term', 'semantic_type']:
            if col not in self.descriptions.columns:
                logger.warning("Column '%s' not found in DESCRIPTION_CSV. Available cols: %s", col, self.descriptions.columns.tolist())

        # Load FAISS index
        logger.info("Loading FAISS index: %s", config["FAISS_INDEX_PATH"])
        self.index = faiss.read_index(config["FAISS_INDEX_PATH"])

        # Load relation guidelines
        logger.info("Loading relation guidelines: %s", config["RELATION_CSV"])
        self.relation_df = pd.read_csv(config["RELATION_CSV"], dtype=str, na_filter=False)

        # parameters
        self.min_n = int(config.get("NGRAM_MIN", 1))
        self.max_n = int(config.get("NGRAM_MAX", 4))
        self.top_k = int(config.get("TOP_K", 10))
        self.threshold = float(config.get("SIM_THRESHOLD", 0.85))
        self.use_sem_filter = bool(config.get("USE_SEMANTIC_FILTER", True))
        self.sem_filter_list = config.get("SEMANTIC_TYPE_FILTER", [])
        self.max_span_chars = int(config.get("MAX_SPAN_CHARS", 120))

    # -----------------------------
    # Embedding helper
    # -----------------------------
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Batch-embed a list of texts (returns float32 numpy array, shape=(len(texts), dim))
        Pools using CLS token (last_hidden_state[:,0,:]).
        """
        if not texts:
            return np.zeros((0, self.model.config.hidden_size), dtype='float32')

        # Tokenize batch
        inputs = self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        if torch.cuda.is_available() and self.device >= 0:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            self.model = self.model.to("cuda")

        with torch.no_grad():
            out = self.model(**inputs)
        # CLS pooling
        vectors = out.last_hidden_state[:, 0, :].cpu().numpy().astype('float32')
        return vectors

    # -----------------------------
    # N-gram entity recognition
    # -----------------------------
    def ngram_candidates(self, text: str) -> pd.DataFrame:
        """
        Generate candidate n-gram spans from text (with offsets). Returns a DataFrame
        with columns: span_text, start, end, token_count, cleaned_text.
        Filters out empty / stopword-only spans and overly long spans.
        """
        ngrams = generate_ngrams_with_offsets(text, self.min_n, self.max_n)
        records = []
        seen = set()
        for ng in ngrams:
            span = ng['text'].strip()
            if not span:
                continue
            if len(span) > self.max_span_chars:
                continue
            cleaned = simple_preprocess(span)
            if cleaned == "":
                continue
            # deduplicate using cleaned + start position
            key = (cleaned.lower(), ng['start'])
            if key in seen:
                continue
            seen.add(key)
            records.append({
                'span_text': span,
                'cleaned': cleaned,
                'start': ng['start'],
                'end': ng['end'],
                'token_count': ng['token_count']
            })
        df = pd.DataFrame.from_records(records)
        # Prefer longer token_count for identical positions in later overlap resolution
        return df

    # -----------------------------
    # Normalization using SapBERT + FAISS
    # -----------------------------
    def normalize_candidates(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        For each candidate span (row in candidates_df with column 'cleaned'), embed and query FAISS.
        Returns dataframe with candidate matches and scoring columns.
        """
        if candidates_df is None or candidates_df.empty:
            return pd.DataFrame()

        texts = candidates_df['cleaned'].tolist()
        vectors = self.embed_texts(texts)  # shape (N, dim)
        # Ensure float32
        vectors = np.asarray(vectors, dtype='float32')
        # normalize for cosine / inner-product indices (common pattern)
        faiss.normalize_L2(vectors)

        distances, indices = self.index.search(vectors, self.top_k)  # distances shape (N, K)
        records = []
        for i, cand_row in candidates_df.reset_index(drop=True).iterrows():
            cand_text = cand_row['span_text']
            cleaned = cand_row['cleaned']
            start = int(cand_row['start'])
            end = int(cand_row['end'])
            token_count = int(cand_row['token_count'])
            for rank, idx in enumerate(indices[i]):
                if idx < 0 or idx >= len(self.descriptions):
                    continue
                dist = float(distances[i][rank])
                # depending on how index was built, 'dist' might represent inner product / similarity.
                # We keep the same convention as your earlier code: higher is better and threshold compared to self.threshold
                score = float(dist)
                if score < self.threshold:
                    continue
                desc_row = self.descriptions.iloc[idx]
                candidate_term = str(desc_row.get('term', ''))
                sem_type = str(desc_row.get('semantic_type', ''))
                records.append({
                    'entity_text': cand_text,
                    'cleaned': cleaned,
                    'start': start,
                    'end': end,
                    'token_count': token_count,
                    'candidate_conceptId': str(desc_row.get('conceptId', '')),
                    'candidate_term': candidate_term,
                    'candidate_semantic_type': sem_type,
                    'faiss_score': score,
                    'faiss_rank': rank
                })
        df = pd.DataFrame.from_records(records)
        if df.empty:
            return df

        # syntactic similarity (fuzzy token_set_ratio)
        df['syntactic_sim'] = df.apply(
            lambda r: fuzz.token_set_ratio(
                r['cleaned'],
                ' '.join(simple_preprocess(r['candidate_term']))
            ), axis=1
        )

        if hasattr(self, 'syntactic_threshold') and self.syntactic_threshold is not None:
            df = df[df['syntactic_sim'] >= self.syntactic_threshold]
        # optional semantic filter
        if self.use_sem_filter and 'candidate_semantic_type' in df.columns:
            df = df[df['candidate_semantic_type'].isin(self.sem_filter_list)]

        # rank and drop duplicates per (entity_text, candidate_conceptId) keeping best faiss_score -> syntactic_sim
        df = df.sort_values(['start', 'faiss_score', 'syntactic_sim'], ascending=[True, False, False])
        df = df.drop_duplicates(subset=['entity_text', 'candidate_conceptId'], keep='first').reset_index(drop=True)
        return df

    # -----------------------------
    # Resolve overlaps (choose longer token_count)
    # -----------------------------
    @staticmethod
    def resolve_overlapping_entities(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove overlapping entity matches. Preference given to longer spans (token_count), then higher faiss_score.
        Expects columns: start, end, token_count
        """
        if df is None or df.empty:
            return pd.DataFrame()
        # sort by start asc, token_count desc, faiss_score desc
        df_sorted = df.sort_values(['start', 'token_count', 'faiss_score'], ascending=[True, False, False]).reset_index(drop=True)
        kept = []
        occupied_intervals = []
        for _, row in df_sorted.iterrows():
            s = int(row['start']); e = int(row['end'])
            overlap = False
            for (a, b) in occupied_intervals:
                if not (e <= a or s >= b):  # overlap exists
                    overlap = True
                    break
            if not overlap:
                kept.append(row.to_dict())
                occupied_intervals.append((s, e))
        if not kept:
            return pd.DataFrame()
        out = pd.DataFrame(kept).reset_index(drop=True)
        return out

    # -----------------------------
    # Relation extraction: pair entities within same sentence & match guidelines
    # -----------------------------
    def extract_relations(self, resolved_df: pd.DataFrame, text: str) -> pd.DataFrame:
        """
        Produce relation predictions for entity pairs that fall within same sentence window.
        Matches are found in self.relation_df using 'Domain concept tag' and 'Range concept tag'.
        Returns DataFrame of matched relations with subject/object info and context.
        """
        if resolved_df is None or resolved_df.empty:
            return pd.DataFrame()

        # map each entity to a sentence index
        sentences = list(sent_tokenize(text))
        sentence_offsets = []
        # compute sentence offsets (start char in original text)
        # naive approach: find each sentence in order using str.find starting from cursor
        cursor = 0
        for sent in sentences:
            idx = text.find(sent, cursor)
            if idx == -1:
                # fallback: approximate
                idx = cursor
            sentence_offsets.append((idx, idx + len(sent)))
            cursor = idx + len(sent)

        def find_sentence_index(ent_start, ent_end):
            for si, (a, b) in enumerate(sentence_offsets):
                # consider entity inside sentence if overlap with sentence interval
                if not (ent_end <= a or ent_start >= b):
                    return si
            return None

        # add sentence_index to resolved_df
        df = resolved_df.copy()
        df['sentence_index'] = df.apply(lambda r: find_sentence_index(int(r['start']), int(r['end'])), axis=1)
        df = df.dropna(subset=['sentence_index'])

        # group by sentence and create all unordered pairs inside sentence
        relations = []
        for si, group in df.groupby('sentence_index'):
            rows = group.to_dict('records')
            # all unordered pairs
            for a_idx in range(len(rows)):
                for b_idx in range(a_idx + 1, len(rows)):
                    a = rows[a_idx]; b = rows[b_idx]
                    type_a = a.get('candidate_semantic_type', '')
                    type_b = b.get('candidate_semantic_type', '')
                    # match guidelines bidirectionally
                    cond = (
                            ((self.relation_df['Domain concept tag'] == type_a) & (self.relation_df['Range concept tag'] == type_b)) |
                            ((self.relation_df['Domain concept tag'] == type_b) & (self.relation_df['Range concept tag'] == type_a))
                    )
                    matches = self.relation_df[cond]
                    if not matches.empty:
                        relations.append({
                            'subject_text': a['entity_text'],
                            'subject_cid': a['candidate_conceptId'],
                            'subject_type': type_a,
                            'object_text': b['entity_text'],
                            'object_cid': b['candidate_conceptId'],
                            'object_type': type_b,
                            'matched_relations': '; '.join(matches['Relation'].astype(str).tolist()),
                            'sentence_index': int(si),
                            'context': text[sentence_offsets[si][0]:sentence_offsets[si][1]]
                        })
        rel_df = pd.DataFrame(relations)
        return rel_df

    # -----------------------------
    # Full pipeline runner
    # -----------------------------
    def run_pipeline(self, text: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run full pipeline on a single text string.
        Returns (normalized_entities_df, relations_df)
        normalized_entities_df has columns:
            entity_text, cleaned, start, end, token_count, candidate_conceptId, candidate_term,
            candidate_semantic_type, faiss_score, syntactic_sim
        """
        # 1. generate n-gram candidates
        cand_df = self.ngram_candidates(text)
        logger.info("Generated %d raw n-gram candidates.", 0 if cand_df is None else len(cand_df))

        # 2. normalize with SapBERT + FAISS
        norm_df = self.normalize_candidates(cand_df)
        if norm_df is None or norm_df.empty:
            logger.info("No normalization candidates above threshold.")
            return pd.DataFrame(), pd.DataFrame()
        logger.info("Normalization found %d candidate matches (pre-overlap).", len(norm_df))

        # 3. resolve overlaps
        resolved = self.resolve_overlapping_entities(norm_df)
        logger.info("After overlap resolution: %d entities.", 0 if resolved is None else len(resolved))

        # 4. relation extraction (within sentence windows)
        rel_df = self.extract_relations(resolved, text)
        logger.info("Extracted %d relation(s).", 0 if rel_df is None else len(rel_df))

        # final DF formatting
        # ensure important columns exist (fill empty strings if not)
        if not resolved.empty:
            for c in ['candidate_term', 'candidate_semantic_type', 'candidate_conceptId']:
                if c not in resolved.columns:
                    resolved[c] = ""
            resolved = resolved.reset_index(drop=True)
        return resolved, rel_df

# -----------------------------
# Example usage (update paths in CONFIG before running)
# -----------------------------
if __name__ == "__main__":
    sample_text = ("Alongside the diagnosis of a right heart blockage, the patient presented with a distinct "
                   "issue: a left leg fracture, necessitating a comprehensive treatment plan that addresses "
                   "both cardiac and orthopedic conditions.")
    mp = MedicalPipeline(CONFIG)
    normalized_entities, relations = mp.run_pipeline(sample_text)

    print("\n=== Normalized Entities ===")
    if normalized_entities is not None and not normalized_entities.empty:
        print(normalized_entities.to_string(index=False))
    else:
        print("No entities found / normalized.")

    print("\n=== Relations ===")
    if relations is not None and not relations.empty:
        print(relations.to_string(index=False))
    else:
        print("No relations predicted.")
