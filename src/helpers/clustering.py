# service/src/helpers/clustering.py

import itertools
import logging
import re
from typing import Dict, List, Set

import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

# ---- Configuration -------------------------------------------------------------
CLUSTERING_SIM_ENTITY = 0.40
CLUSTERING_SIM_GLOBAL = 0.60
CLUSTERING_BIG_COMM = 30
CLUSTERING_AVG_SIM_MIN = 0.50
CLUSTERING_TOPIC_LABELS = ["economy", "politics", "sports", "conflict", "misc"]
CLUSTERING_ALIAS_THR = 0.20
CLUSTERING_TOK_REMOVE = {"the", "a", "an", "of"}
CLUSTERING_PUN_RE = re.compile(r"[^\w\s]")

logger = logging.getLogger(__name__)


def scrub_text(s: str) -> str:
    """Clean text for alias discovery"""
    s = CLUSTERING_PUN_RE.sub("", s.lower()).strip()
    return " ".join(t for t in s.split() if t not in CLUSTERING_TOK_REMOVE)


def grouper(seq, n):
    """Group sequence into chunks of size n"""
    for i in range(0, len(seq), n):
        yield seq[i: i + n]


def discover_aliases(
        ents_per_doc: List[Set[str]], embedder: SentenceTransformer
) -> Dict[str, str]:
    """Discover entity aliases using cosine similarity clustering"""
    if not embedder:
        return {}

    raw = sorted({e for ents in ents_per_doc for e in ents if e != "__NOENT__"})
    if not raw:
        return {}

    clean = [scrub_text(r) for r in raw]
    embs = (
        embedder.encode(clean, convert_to_tensor=True, normalize_embeddings=True)
        .cpu()
        .numpy()
    )

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=CLUSTERING_ALIAS_THR,
    )
    labels = model.fit_predict(embs)

    canons, by_label = {}, {}
    for r, c, l in zip(raw, clean, labels):
        by_label.setdefault(l, []).append((len(c), r))
    for items in by_label.values():
        items.sort()
        canon = items[0][1]
        for _, raw_txt in items:
            canons[raw_txt] = canon
    return canons


def get_entities(docs, embedder) -> List[Set[str]]:
    """Extract and normalize entities from spaCy documents"""
    sets = []
    for d in docs:
        ents = {
            e.text.strip()
            for e in d.ents
            if e.label_ in {"GPE", "LOC", "ORG", "FAC", "PERSON"}
        }
        sets.append(ents if ents else {"__NOENT__"})

    alias = discover_aliases(sets, embedder)
    return [{alias.get(e, e) for e in s} for s in sets]


def get_primary_topics(
        texts: List[str], classifier, show_bar: bool = False
) -> List[str]:
    """Get primary topic classification for texts"""
    if not classifier:
        return ["misc"] * len(texts)

    prim = []
    for chunk in tqdm(list(grouper(texts, 16)), desc="classify", disable=not show_bar):
        try:
            res = classifier(chunk, CLUSTERING_TOPIC_LABELS, multi_label=True)
            prim.extend(r["labels"][0] for r in res)
        except Exception as e:
            logger.warning("Topic classification failed for chunk: %s", e)
            prim.extend(["misc"] * len(chunk))
    return prim


def build_clustering_graph(
        texts: List[str], nlp, embedder, classifier, show_bar: bool = False, config: Dict = None
):
    """Build similarity graph for clustering"""
    if not nlp or not embedder:
        raise Exception("Clustering models not available")

    sim_entity = (
        config.get("similarity_entity", CLUSTERING_SIM_ENTITY)
        if config
        else CLUSTERING_SIM_ENTITY
    )
    sim_global = (
        config.get("similarity_global", CLUSTERING_SIM_GLOBAL)
        if config
        else CLUSTERING_SIM_GLOBAL
    )

    docs = list(
        tqdm(nlp.pipe(texts, batch_size=64), total=len(texts), desc="spaCy", disable=not show_bar)
    )
    entities = get_entities(docs, embedder)
    topics = get_primary_topics(texts, classifier, show_bar)

    emb = embedder.encode(texts, batch_size=64, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(emb, emb).cpu().numpy()

    G, why = nx.Graph(), {}
    for i, j in itertools.combinations(range(len(texts)), 2):
        if topics[i] != topics[j] or topics[i] == "misc":
            continue
        shared = bool(entities[i] & entities[j])
        sim = sims[i, j]
        ok = False
        if shared and sim >= sim_entity:
            ok = True
            why[(i, j)] = f"ENT & {sim:.2f}"
        elif sim >= sim_global:
            ok = True
            why[(i, j)] = f"{sim:.2f}"
        if ok:
            G.add_edge(i, j, weight=float(sim))
    return G, sims, why, topics, entities


def split_large_communities(comm, sims, config: Dict = None, depth: int = 0):
    """Recursively split large communities"""
    big_comm = (
        config.get("big_community_size", CLUSTERING_BIG_COMM)
        if config
        else CLUSTERING_BIG_COMM
    )
    avg_sim_min = (
        config.get("avg_similarity_min", CLUSTERING_AVG_SIM_MIN)
        if config
        else CLUSTERING_AVG_SIM_MIN
    )
    sim_global = (
        config.get("similarity_global", CLUSTERING_SIM_GLOBAL)
        if config
        else CLUSTERING_SIM_GLOBAL
    )

    if len(comm) <= big_comm:
        return [comm]
    
    # Safety check: avoid division by zero for small communities
    if len(comm) <= 1:
        return [comm]
    
    subM = sims[np.ix_(comm, comm)]
    # Calculate average similarity, avoiding division by zero
    denominator = len(comm) * (len(comm) - 1)
    if denominator == 0:
        return [comm]
    
    avg = (subM.sum() - len(comm)) / denominator
    if avg >= avg_sim_min or depth > 2:
        return [comm]
    tight = nx.Graph()
    tight.add_nodes_from(range(len(comm)))
    for a, b in itertools.combinations(range(len(comm)), 2):
        if subM[a, b] >= sim_global + 0.05:
            tight.add_edge(a, b)
    sub = louvain_communities(tight, weight=None, resolution=1.1)
    out = []
    for c in sub:
        mapped = [comm[i] for i in c]
        out.extend(split_large_communities(mapped, sims, config, depth + 1))
    return out
