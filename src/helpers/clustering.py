# service/src/helpers/clustering.py

import itertools
import logging
import re
from typing import Dict, List, Set, Optional

import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

# ---- Configuration -------------------------------------------------------------
CLUSTERING_SIM_ENTITY = 0.40  # Restored for event-specific entities
CLUSTERING_SIM_GLOBAL = 0.80  # Very high threshold for same-event clustering
CLUSTERING_BIG_COMM = 15      # Smaller communities for tighter event clustering
CLUSTERING_AVG_SIM_MIN = 0.75 # Very high minimum for same-event stories
CLUSTERING_TOPIC_LABELS = ["economy", "politics", "sports", "conflict", "misc"]
CLUSTERING_ALIAS_THR = 0.20
CLUSTERING_TOK_REMOVE = {"the", "a", "an", "of"}
CLUSTERING_PUN_RE = re.compile(r"[^\w\s]")

# New parameters for event-specific clustering
CLUSTERING_ENTITY_CONTEXT_WEIGHT = 0.25
CLUSTERING_MIN_SHARED_ENTITIES = 1
CLUSTERING_TOPIC_STRICT_MODE = False  # Allow cross-topic for same events
CLUSTERING_EVENT_SPECIFIC_MODE = True

# Entity type weights for context-aware similarity
ENTITY_TYPE_WEIGHTS = {
    "GPE": 0.3,     # Geopolitical entities (countries, cities) - lower weight
    "LOC": 0.3,     # Locations - lower weight
    "ORG": 0.8,     # Organizations - higher weight for specificity
    "FAC": 0.6,     # Facilities - medium weight
    "PERSON": 0.9,  # Person names - highest weight for specificity
}

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


def get_entities_with_types(docs, embedder) -> List[Dict[str, Set[str]]]:
    """Extract and normalize entities with their types from spaCy documents"""
    entity_sets = []
    for d in docs:
        entity_dict = {
            "GPE": set(),  # Geopolitical entities
            "LOC": set(),  # Locations
            "ORG": set(),  # Organizations
            "FAC": set(),  # Facilities
            "PERSON": set()  # Person names
        }

        for e in d.ents:
            if e.label_ in entity_dict:
                entity_dict[e.label_].add(e.text.strip())

        # If no entities found, add marker
        if not any(entity_dict.values()):
            entity_dict["__NOENT__"] = {"__NOENT__"}

        entity_sets.append(entity_dict)

    # Apply alias discovery per entity type
    all_entities_by_type = {}
    for entity_type in ["GPE", "LOC", "ORG", "FAC", "PERSON"]:
        type_entities = [doc_entities.get(entity_type, set()) for doc_entities in entity_sets]
        if any(type_entities):
            alias = discover_aliases(type_entities, embedder)
            all_entities_by_type[entity_type] = alias

    # Apply aliases to entity sets
    normalized_sets = []
    for entity_dict in entity_sets:
        normalized_dict = {}
        for entity_type, entities in entity_dict.items():
            if entity_type in all_entities_by_type:
                alias = all_entities_by_type[entity_type]
                normalized_dict[entity_type] = {alias.get(e, e) for e in entities}
            else:
                normalized_dict[entity_type] = entities
        normalized_sets.append(normalized_dict)

    return normalized_sets


def calculate_entity_context_similarity(entities_i: Dict[str, Set[str]], entities_j: Dict[str, Set[str]], config: Dict = None) -> float:
    """Calculate weighted entity similarity based on entity types and context"""
    if not entities_i or not entities_j:
        return 0.0

    min_shared_entities = config.get("min_shared_entities", CLUSTERING_MIN_SHARED_ENTITIES) if config else CLUSTERING_MIN_SHARED_ENTITIES

    total_score = 0.0
    total_weight = 0.0
    shared_entity_count = 0

    for entity_type in ["GPE", "LOC", "ORG", "FAC", "PERSON"]:
        entities_set_i = entities_i.get(entity_type, set())
        entities_set_j = entities_j.get(entity_type, set())

        # Skip empty sets or marker entities
        if not entities_set_i or not entities_set_j or "__NOENT__" in entities_set_i or "__NOENT__" in entities_set_j:
            continue

        shared_entities = entities_set_i & entities_set_j
        if shared_entities:
            shared_entity_count += len(shared_entities)

            # Calculate Jaccard similarity for this entity type
            union_entities = entities_set_i | entities_set_j
            jaccard_sim = len(shared_entities) / len(union_entities) if union_entities else 0.0

            # Weight by entity type importance
            type_weight = ENTITY_TYPE_WEIGHTS.get(entity_type, 0.5)
            total_score += jaccard_sim * type_weight
            total_weight += type_weight

    # Require minimum number of shared entities
    if shared_entity_count < min_shared_entities:
        return 0.0

    # Return weighted average similarity
    return total_score / total_weight if total_weight > 0 else 0.0


def should_cluster_by_topic(topic_i: str, topic_j: str, config: Dict = None) -> bool:
    """Determine if two texts should be clustered based on their topics"""
    topic_strict_mode = config.get("topic_strict_mode", CLUSTERING_TOPIC_STRICT_MODE) if config else CLUSTERING_TOPIC_STRICT_MODE

    # Never cluster if either topic is misc (too generic)
    if topic_i == "misc" or topic_j == "misc":
        return False

    # In strict mode, topics must match exactly
    if topic_strict_mode:
        return topic_i == topic_j

    # In non-strict mode, allow some compatible topic pairs
    compatible_pairs = {
        ("economy", "politics"),
        ("politics", "economy"),
        # Add more compatible pairs as needed
    }

    return topic_i == topic_j or (topic_i, topic_j) in compatible_pairs


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
    """Build similarity graph for event-specific clustering that identifies posts about the same event.

    Optionally reuses precomputed embeddings if provided via config['precomputed_embeddings'].
    Optionally uses precomputed entities if provided via config['precomputed_entities'].
    """
    if not nlp or not embedder:
        raise Exception("Clustering models not available")

    # Get configuration parameters
    sim_entity = config.get("similarity_entity", CLUSTERING_SIM_ENTITY) if config else CLUSTERING_SIM_ENTITY
    sim_global = config.get("similarity_global", CLUSTERING_SIM_GLOBAL) if config else CLUSTERING_SIM_GLOBAL
    entity_context_weight = config.get("entity_context_weight", CLUSTERING_ENTITY_CONTEXT_WEIGHT) if config else CLUSTERING_ENTITY_CONTEXT_WEIGHT
    event_specific_mode = config.get("event_specific_mode", CLUSTERING_EVENT_SPECIFIC_MODE) if config else CLUSTERING_EVENT_SPECIFIC_MODE

    logger.info(f"🔧 Building graph for event-specific clustering: sim_entity={sim_entity}, sim_global={sim_global}, event_mode={event_specific_mode}")

    docs = list(
        tqdm(nlp.pipe(texts, batch_size=64), total=len(texts), desc="spaCy", disable=not show_bar)
    )

    # Check for precomputed entities
    precomputed_entities = config.get("precomputed_entities") if config else None

    if precomputed_entities and len(precomputed_entities) == len(texts):
        logger.info(f"🔧 Using precomputed entities for {len(texts)} texts")
        entities = precomputed_entities
        entities_typed = get_entities_with_types(docs, embedder)
    else:
        entities = get_entities(docs, embedder)
        entities_typed = get_entities_with_types(docs, embedder)

    topics = get_primary_topics(texts, classifier, show_bar)

    # Handle precomputed embeddings
    pre_embs: Optional[List[List[float]]] = None
    if config and isinstance(config, dict):
        pre_embs = config.get("precomputed_embeddings")

    if pre_embs is not None:
        try:
            emb_tensor = torch.tensor(pre_embs, dtype=torch.float32)
            if emb_tensor.shape[0] != len(texts):
                logger.warning("Precomputed embeddings count (%d) doesn't match texts (%d); recomputing.", emb_tensor.shape[0], len(texts))
                emb = embedder.encode(texts, batch_size=64, convert_to_tensor=True, normalize_embeddings=True)
            else:
                norms = torch.norm(emb_tensor, dim=1)
                avg_norm = float(torch.mean(norms)) if norms.numel() else 1.0
                if abs(avg_norm - 1.0) > 0.05:
                    logger.info("Normalizing provided embeddings (avg norm %.3f)", avg_norm)
                    emb_tensor = torch.nn.functional.normalize(emb_tensor, p=2, dim=1)
                emb = emb_tensor
                logger.info("Using %d precomputed embeddings (dim=%d)", emb.shape[0], emb.shape[1])
        except Exception as e:
            logger.warning("Failed to use precomputed embeddings (%s); falling back to encode", e)
            emb = embedder.encode(texts, batch_size=64, convert_to_tensor=True, normalize_embeddings=True)
    else:
        emb = embedder.encode(texts, batch_size=64, convert_to_tensor=True, normalize_embeddings=True)

    sims = util.cos_sim(emb, emb).cpu().numpy()

    G, why = nx.Graph(), {}
    edges_added = 0
    edges_rejected_topic = 0
    edges_rejected_entity = 0
    edges_rejected_similarity = 0

    for i, j in itertools.combinations(range(len(texts)), 2):
        # Relaxed topic gating for event-specific clustering (allow cross-topic for same events)
        if not should_cluster_by_topic(topics[i], topics[j], config):
            edges_rejected_topic += 1
            continue

        # Calculate semantic similarity
        semantic_sim = float(sims[i, j])

        # Calculate entity-based similarity
        has_shared_entities = bool(entities[i] & entities[j])
        entity_context_sim = calculate_entity_context_similarity(entities_typed[i], entities_typed[j], config)

        # Determine if we should add an edge for event-specific clustering
        ok = False
        reason = ""

        if event_specific_mode:
            # Event-specific clustering methods prioritize entity overlap + high semantic similarity

            # Method 1: Strong entity overlap + good semantic similarity (main entities of the event)
            if entity_context_sim > 0.3 and semantic_sim >= sim_entity:
                combined_sim = (semantic_sim * (1 - entity_context_weight)) + (entity_context_sim * entity_context_weight)
                if combined_sim >= sim_entity:
                    ok = True
                    reason = f"EVENT_ENT & {semantic_sim:.2f} (ctx: {entity_context_sim:.2f})"

            # Method 2: Very high semantic similarity (same event, different angle)
            elif semantic_sim >= sim_global:
                ok = True
                reason = f"EVENT_SEM {semantic_sim:.2f}"

            # Method 3: Shared key entities + decent semantic similarity (related event stories)
            elif has_shared_entities and semantic_sim >= sim_entity * 0.85:
                ok = True
                reason = f"EVENT_SHARED & {semantic_sim:.2f}"

        else:
            # Fallback to original clustering logic
            if entity_context_sim > 0 and semantic_sim >= sim_entity:
                combined_sim = (semantic_sim * (1 - entity_context_weight)) + (entity_context_sim * entity_context_weight)
                if combined_sim >= sim_entity:
                    ok = True
                    reason = f"ENT_CTX & {semantic_sim:.2f} (ctx: {entity_context_sim:.2f})"
            elif semantic_sim >= sim_global:
                ok = True
                reason = f"SEM {semantic_sim:.2f}"
            elif has_shared_entities and semantic_sim >= sim_entity * 0.8:
                ok = True
                reason = f"ENT_FALL & {semantic_sim:.2f}"

        if ok:
            G.add_edge(i, j, weight=semantic_sim)
            why[(i, j)] = reason
            edges_added += 1
        else:
            if entity_context_sim == 0 and not has_shared_entities:
                edges_rejected_entity += 1
            else:
                edges_rejected_similarity += 1

    logger.info(f"🔧 Event clustering graph complete: {edges_added} edges added, "
                f"{edges_rejected_topic} rejected by topic, "
                f"{edges_rejected_entity} rejected by entities, "
                f"{edges_rejected_similarity} rejected by similarity")

    return G, sims, why, topics, entities


def split_large_communities(comm, sims, config: Dict = None, depth: int = 0):
    """Recursively split large communities with stricter similarity enforcement"""
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

    # Always check average similarity first, even for small communities
    if len(comm) <= 1:
        return [comm]
    
    subM = sims[np.ix_(comm, comm)]
    # Calculate average similarity, avoiding division by zero
    denominator = len(comm) * (len(comm) - 1)
    if denominator == 0:
        return [comm]
    
    avg = (subM.sum() - len(comm)) / denominator

    # STRICT ENFORCEMENT: If average similarity is too low, split into individual items
    if avg < avg_sim_min:
        logger.info(f"🔧 Splitting community of size {len(comm)} due to low avg similarity ({avg:.3f} < {avg_sim_min})")
        if depth > 3:  # Prevent infinite recursion
            return [[item] for item in comm]

        # Try to form tighter subcommunities
        tight = nx.Graph()
        tight.add_nodes_from(range(len(comm)))
        for a, b in itertools.combinations(range(len(comm)), 2):
            if subM[a, b] >= sim_global:  # Use strict threshold
                tight.add_edge(a, b)

        if tight.number_of_edges() == 0:
            # No edges meet the threshold - split into individual items
            return [[item] for item in comm]

        sub = louvain_communities(tight, weight=None, resolution=1.2)  # Higher resolution for tighter communities
        out = []
        for c in sub:
            mapped = [comm[i] for i in c]
            out.extend(split_large_communities(mapped, sims, config, depth + 1))
        return out

    # If community is small enough and similarity is acceptable, keep it
    if len(comm) <= big_comm:
        return [comm]

    # For large communities with good similarity, still try to split them
    tight = nx.Graph()
    tight.add_nodes_from(range(len(comm)))
    for a, b in itertools.combinations(range(len(comm)), 2):
        if subM[a, b] >= sim_global + 0.05:  # Slightly higher threshold for large community splitting
            tight.add_edge(a, b)
    sub = louvain_communities(tight, weight=None, resolution=1.1)
    out = []
    for c in sub:
        mapped = [comm[i] for i in c]
        out.extend(split_large_communities(mapped, sims, config, depth + 1))
    return out


def calculate_category_similarity(categories_i: List[str], categories_j: List[str]) -> float:
    """Calculate similarity based on shared categories"""
    if not categories_i or not categories_j:
        return 0.0

    # Convert to sets for intersection/union operations
    set_i = set(categories_i)
    set_j = set(categories_j)

    # Calculate Jaccard similarity
    intersection = set_i & set_j
    union = set_i | set_j

    if not union:
        return 0.0

    return len(intersection) / len(union)


def should_cluster_by_domain(categories_i: List[str], categories_j: List[str], config: Dict = None) -> bool:
    """Determine if two texts should be clustered based on their domain categories"""
    domain_filtering = config.get("domain_filtering", CLUSTERING_DOMAIN_FILTERING) if config else CLUSTERING_DOMAIN_FILTERING

    if not domain_filtering or not categories_i or not categories_j:
        return True  # Allow clustering if no domain filtering or no category data

    # Define domain category groups that are incompatible
    entertainment_domains = {"Cine y Series", "Actualidad Cultural", "seleccion-tendencias"}
    sports_domains = {"sports", "Copa Sudamericana", "Alianza Lima", "Racing"}
    politics_domains = {"politics", "conflict", "Ministerio de las Culturas"}

    # Get domain types for each post
    domains_i = set()
    domains_j = set()

    for cat in categories_i:
        if any(domain in cat for domain in entertainment_domains):
            domains_i.add("entertainment")
        elif any(domain in cat for domain in sports_domains):
            domains_i.add("sports")
        elif any(domain in cat for domain in politics_domains):
            domains_i.add("politics")

    for cat in categories_j:
        if any(domain in cat for domain in entertainment_domains):
            domains_j.add("entertainment")
        elif any(domain in cat for domain in sports_domains):
            domains_j.add("sports")
        elif any(domain in cat for domain in politics_domains):
            domains_j.add("politics")

    # If both posts have clear domain assignments and they don't overlap, don't cluster
    if domains_i and domains_j and not (domains_i & domains_j):
        return False

    return True
