"""Language-aware lexical retrieval: tokenization, BM25 and evidence ranking.

Extracted from ``rag.py`` so the tokenizer, stemmer, stop-word lists and the
client-side BM25 index live in one place, independent from the retrieval
orchestration.  The same ``lexical_terms`` pipeline is applied to index and
query sides, so a Russian stemmer never touches English tokens and vice versa.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from itertools import pairwise
from typing import Any

from llama_index.core.vector_stores.utils import metadata_dict_to_node

# Unicode letters and digits (no underscore): accented Latin, Greek symbols
# and other scripts stay whole instead of splitting into noise fragments.
LEXICAL_TOKEN_RE = re.compile(r"[^\W_]+")
LEXICAL_STOP_WORDS = {
    "а",
    "без",
    "бы",
    "в",
    "во",
    "для",
    "и",
    "из",
    "или",
    "как",
    "какая",
    "какие",
    "какой",
    "кратко",
    "на",
    "не",
    "мне",
    "можешь",
    "но",
    "о",
    "по",
    "пожалуйста",
    "подробнее",
    "подробно",
    "про",
    "при",
    "с",
    "со",
    "что",
    "ты",
    "знаешь",
    "такое",
    "такую",
    "таком",
    "это",
    "about",
    "are",
    "describe",
    "explain",
    "is",
    "me",
    "please",
    "tell",
    "what",
    "gts",
    "midas",
    "nx",
}
LEXICAL_IGNORED_PREFIXES = (
    "использ",
    "объясн",
    "опиш",
    "поясн",
    "расскаж",
)
OVERVIEW_PREFIXES = (
    "дай обзор ",
    "дай краткий обзор ",
    "кратко расскажи ",
    "можешь рассказать ",
    "обзор ",
    "объясни ",
    "опиши ",
    "подробно расскажи ",
    "расскажи ",
    "расскажи подробнее ",
    "что известно о ",
    "что такое ",
    "что ты знаешь о ",
    "describe ",
    "explain ",
    "overview ",
    "tell me about ",
    "what are ",
    "what is ",
)
# Wh-words that turn a survey opener into a concrete question:
# «расскажи, как настроить …» asks for one procedure, not a topic survey.
FOCUSED_QUESTION_WORDS = {
    "как",
    "какая",
    "какие",
    "каким",
    "какими",
    "какого",
    "какой",
    "каком",
    "какую",
    "где",
    "зачем",
    "когда",
    "куда",
    "ли",
    "откуда",
    "почему",
    "сколько",
    "чем",
    "how",
    "why",
    "when",
    "where",
    "which",
    "who",
}
# Imperative verbs and procedure nouns mark a focused request even without a
# wh-word («расскажи про установку …», «опиши шаги создания …»).  Erring
# towards ``focused`` is the safe direction: it keeps the full context, while
# a false ``overview`` used to starve the answer of evidence.
FOCUSED_QUESTION_STEMS = (
    "включ",
    "выключ",
    "выполн",
    "вычисл",
    "вылет",
    "генер",
    "добав",
    "зависа",
    "зада",
    "запус",
    "измен",
    "импорт",
    "исправ",
    "настро",
    "назнач",
    "найд",
    "напиш",
    "не выходит",
    "не запуска",
    "не могу",
    "не получа",
    "не работа",
    "не удает",
    "обнов",
    "открыв",
    "откро",
    "ошибк",
    "падает",
    "показ",
    "посчит",
    "постро",
    "прилож",
    "проблем",
    "провер",
    "рассчит",
    "сдела",
    "смоделир",
    "созда",
    "удал",
    "установ",
    "фиксир",
    "экспорт",
    "apply",
    "assign",
    "build",
    "can't",
    "cannot",
    "cant",
    "config",
    "connect",
    "crash",
    "creat",
    "delet",
    "doesn't",
    "doesnt",
    "error",
    "fail",
    "fix",
    "import",
    "install",
    "issue",
    "open",
    "problem",
    "remov",
    "run",
    "set",
    "step",
    "unable",
    "won't",
    "wont",
)
RUSSIAN_SUFFIXES = tuple(
    sorted(
        {
            "иями",
            "ыми",
            "ими",
            "ами",
            "ями",
            "ого",
            "его",
            "ому",
            "ему",
            "иях",
            "ах",
            "ях",
            "ов",
            "ев",
            "ей",
            "ий",
            "ый",
            "ой",
            "ая",
            "яя",
            "ое",
            "ее",
            "ие",
            "ые",
            "ую",
            "юю",
            "ам",
            "ям",
            "ом",
            "ем",
            "а",
            "я",
            "ы",
            "и",
            "е",
            "о",
            "у",
            "ю",
        },
        key=len,
        reverse=True,
    )
)


class LexicalBM25Index:
    """In-memory BM25 index over the collection's clean chunk texts.

    Replaces the former ``scroll`` + ``MatchText`` retrieval: instead of an
    unordered filter over the ``_node_content`` JSON payload (where field
    names and file metadata could match the query), documents are ranked by
    BM25 over ``node.text`` terms normalized by ``lexical_terms``.  The same
    language-aware tokenizer is applied to both index and query sides, so a
    Russian stemmer never touches English tokens and vice versa.  Corpus size
    (~4k chunks) keeps the whole index comfortably in memory.
    """

    def __init__(self) -> None:
        self.nodes: list[Any] = []
        self.postings: dict[str, list[tuple[int, float]]] = {}
        self.doc_lengths: list[int] = []
        self.avg_doc_length = 1.0

    @property
    def doc_count(self) -> int:
        return len(self.nodes)

    @classmethod
    def build(cls, payloads: Sequence[Mapping[str, Any]], tokenize: Callable[[str], list[str]]) -> "LexicalBM25Index":
        nodes = []
        for payload in payloads:
            try:
                nodes.append(metadata_dict_to_node(dict(payload)))
            except Exception:
                continue
        return cls.build_nodes(nodes, tokenize)

    @classmethod
    def build_nodes(cls, nodes: Sequence[Any], tokenize: Callable[[str], list[str]]) -> LexicalBM25Index:
        """Build the same immutable view from already decoded local nodes."""
        index = cls()
        total = 0
        for node in nodes:
            terms = tokenize(node.text or "")
            doc_index = len(index.nodes)
            index.nodes.append(node)
            index.doc_lengths.append(len(terms))
            total += len(terms)
            for term, frequency in Counter(terms).items():
                index.postings.setdefault(term, []).append((doc_index, float(frequency)))
        if index.nodes:
            index.avg_doc_length = max(total / len(index.nodes), 1.0)
        return index

    def query(self, terms: Sequence[str], limit: int, *, allowed_ids: set[str] | None = None) -> list[tuple[Any, float]]:
        """Return BM25-ranked ``(node, score)`` pairs, best first."""

        if not terms or not self.nodes:
            return []
        k1, b = 1.2, 0.75
        doc_count = len(self.nodes)
        scores: dict[int, float] = defaultdict(float)
        for term in set(terms):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = math.log(1 + (doc_count - len(postings) + 0.5) / (len(postings) + 0.5))
            for doc_index, frequency in postings:
                if allowed_ids is not None and self.nodes[doc_index].node_id not in allowed_ids:
                    continue
                length = self.doc_lengths[doc_index]
                denominator = frequency + k1 * (1 - b + b * length / self.avg_doc_length)
                scores[doc_index] += idf * (frequency * (k1 + 1)) / denominator
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [(self.nodes[doc_index], score) for doc_index, score in ranked]


def is_lexical_term(token: str) -> bool:
    """Accept normal terms and short digit-bearing identifiers.

    Two-character codes (``3D``, ``2D``, ``R2``, ``E5``) carry product
    meaning and must not be dropped with one-letter noise; bare letters
    below three characters stay out.
    """

    return len(token) >= 3 or (
        len(token) == 2 and any(char.isdigit() for char in token)
    )


def sufficient_lexical_terms(terms: Sequence[str]) -> bool:
    """Strict all-token recall needs enough specificity to avoid noise.

    Three terms always qualify; a pair qualifies when it is long enough
    or carries a digit-bearing identifier («3d модель»); a lone
    identifier («4021») is already a precise match on its own.
    """

    if len(terms) >= 3:
        return True
    has_identifier = any(
        len(term) >= 2 and any(char.isdigit() for char in term)
        for term in terms
    )
    if len(terms) == 2:
        return has_identifier or sum(map(len, terms)) >= 9
    return len(terms) == 1 and has_identifier


def fulltext_terms(text: str, *, excluded: set[str] | frozenset[str] = frozenset()) -> list[str]:
    """Raw query tokens for the strict fulltext recall pass."""

    return [
        token
        for token in LEXICAL_TOKEN_RE.findall(text.casefold().replace("ё", "е"))
        if is_lexical_term(token)
        and token not in LEXICAL_STOP_WORDS
        and token not in excluded
        and not any(token.startswith(prefix) for prefix in LEXICAL_IGNORED_PREFIXES)
    ]


def minimum_cover_span(terms: Sequence[str], required: set[str]) -> int | None:
    """Return the shortest token window containing every required term."""

    if not required:
        return None
    counts: dict[str, int] = {}
    missing = len(required)
    left = 0
    best: int | None = None
    for right, term in enumerate(terms):
        if term not in required:
            continue
        counts[term] = counts.get(term, 0) + 1
        if counts[term] == 1:
            missing -= 1
        while missing == 0:
            best = min(best or len(terms), right - left + 1)
            left_term = terms[left]
            left += 1
            if left_term not in required:
                continue
            counts[left_term] -= 1
            if counts[left_term] == 0:
                missing += 1
    return best


def minimum_cover_char_span(text: str, required: set[str]) -> tuple[int, int] | None:
    """Locate the shortest character passage containing all required stems."""

    if not required:
        return None
    occurrences = [
        (stem, match.start(), match.end())
        for match in LEXICAL_TOKEN_RE.finditer(text.casefold().replace("ё", "е"))
        if (stem := light_stem(match.group())) in required
    ]
    counts: dict[str, int] = {}
    missing = len(required)
    left = 0
    best: tuple[int, int] | None = None
    for right, (term, _start, end) in enumerate(occurrences):
        counts[term] = counts.get(term, 0) + 1
        if counts[term] == 1:
            missing -= 1
        while missing == 0:
            start = occurrences[left][1]
            if best is None or end - start < best[1] - best[0]:
                best = (start, end)
            left_term = occurrences[left][0]
            left += 1
            counts[left_term] -= 1
            if counts[left_term] == 0:
                missing += 1
    return best


def mixed_query_groups(terms: Sequence[str]) -> list[set[str]]:
    latin = {term for term in terms if re.fullmatch(r"[a-z]+", term)}
    cyrillic = {term for term in terms if re.fullmatch(r"[а-я]+", term)}
    if len(latin) < 5 or len(cyrillic) < 3:
        return []
    identifiers = set(terms) - latin - cyrillic
    return [latin | identifiers, cyrillic | identifiers]


def light_stem(token: str) -> str:
    if re.search(r"[а-я]", token):
        if len(token) < 5:
            return token
        for suffix in RUSSIAN_SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                return token[: -len(suffix)]
        return token
    if not token.isascii() or not token.isalpha():
        return token
    # English terms get conservative plural/gerund reductions so
    # «boundaries» meets «boundary».  The stems are approximate, but the
    # same rules run on query and document tokens, so an imperfect stem
    # still matches itself.
    if len(token) >= 6 and token.endswith("sses"):
        token = token[:-2]
    elif len(token) >= 5 and token.endswith("ies") and token[-4] not in "aeiou":
        token = token[:-3] + "y"
    elif len(token) >= 5 and token.endswith(("ches", "shes", "xes", "zes")):
        token = token[:-2]
    elif len(token) >= 4 and token.endswith("s") and token[-2] not in "siu":
        token = token[:-1]
    if len(token) >= 7 and token.endswith("ing"):
        token = token[:-3]
    return token


def lexical_terms(text: str) -> list[str]:
    terms: list[str] = []
    for raw_token in LEXICAL_TOKEN_RE.findall(text.casefold().replace("ё", "е")):
        if raw_token in LEXICAL_STOP_WORDS or any(
            raw_token.startswith(prefix) for prefix in LEXICAL_IGNORED_PREFIXES
        ):
            continue
        token = light_stem(raw_token)
        if is_lexical_term(token):
            terms.append(token)
    return terms


def is_overview_question(question: str) -> bool:
    # Rejoining lexical tokens makes the prefix check punctuation-blind:
    # «расскажи, как …» and «расскажи как …» normalize identically.
    normalized = " ".join(
        LEXICAL_TOKEN_RE.findall(question.casefold().replace("ё", "е"))
    )
    if not normalized.startswith(OVERVIEW_PREFIXES):
        return False
    # A survey opener alone does not make the question broad: «расскажи,
    # как настроить фильтрацию сетки» asks for one concrete procedure and
    # needs the complete focused context, not an overview-shaped answer.
    tokens = set(normalized.split())
    if tokens & FOCUSED_QUESTION_WORDS:
        return False
    return not any(stem in normalized for stem in FOCUSED_QUESTION_STEMS)


def is_inventory_question(question: str) -> bool:
    """Recognize an explicit list operation, independent of documented topics.

    Conservative by design: unknown/ambiguous requests keep ordinary retrieval.
    A mixed procedure/comparison/definition does not reserve catalog context.
    """
    if not isinstance(question, str) or not question.strip() or len(question) > 1_000:
        return False
    text = " ".join(LEXICAL_TOKEN_RE.findall(question.casefold().replace("ё", "е")))
    if re.search(r"\b(?:как|почему|сравни\w*|отлич\w*|разниц\w*|инструкци\w*|настро\w*|"
                 r"порядок|процедур\w*|шаг\w*|определени\w*|установ\w*|задавать|задать|вычисл\w*|"
                 r"how|why|compare\w*|comparison|difference\w*|versus|vs|configur\w*|install\w*|"
                 r"calculate|define|definitions?|procedures?|steps?)\b", text):
        return False
    if re.search(r"\b(?:что такое|что означает|what is|what does)\b", text):
        return False
    text = re.sub(r"^(?:(?:пожалуйста|please)\s+)?(?:расскажи\w*|подскажи\w*|скажи\w*|"
                  r"(?:could you\s+)?tell me)\s+(?:пожалуйста\s+)?", "", text)
    return bool(re.search(
        r"^(?:(?:пожалуйста|please)\s+)?(?:перечисли\w*|список|перечень|каталог|list|enumerate)\b"
        r"|\b(?:дай\w*|покажи\w*|приведи\w*|show|give)\s+(?:(?:мне|me|a|the|all)\s+)*(?:список|перечень|list|catalog)\b"
        r"|^(?:какие|каковы)\b.*\b(?:есть|доступ\w*|существ\w*|поддерж\w*|предусмотр\w*|бывают|можно|виды|типы|варианты)\b"
        r"|^(?:what|which)\b.*\b(?:available|supported|exist|there|types|kinds|categories|options|can be used)\b", text))


def inventory_subject_terms(question: str) -> list[str]:
    """Remove operation words only; never expand a subject or model registry."""
    operation = {light_stem(word) for word in (  # noqa: SIM905 - compact operation vocabulary
        "какие каковы есть доступны доступные существуют поддерживаются предусмотрены бывают виды типы варианты "
        "перечисли перечислите список перечень каталог дай покажи приведи мне пожалуйста "
        "расскажи подскажи скажи можно tell could you can be used "
        "what which are the a all list enumerate available supported exist there types kinds categories options "
        "show give me please brief briefly describe").split()}
    return [term for term in lexical_terms(question) if term not in operation]


def rank_fulltext_evidence(
    question: str,
    source_nodes: Sequence[Any],
    *,
    retrieval_query: str | None = None,
    fulltext_excluded: set[str] | frozenset[str] = frozenset(),
    node_text: Callable[[Any], str],
) -> list[Any]:
    """Prefer compact direct evidence over incidental all-token matches."""

    query_terms = lexical_terms(retrieval_query or question)
    if not query_terms:
        return list(source_nodes)
    query_unique = set(query_terms)
    query_pairs = set(zip(query_terms, query_terms[1:]))
    current_terms = set(
        lexical_terms(" ".join(fulltext_terms(question, excluded=fulltext_excluded)))
    )

    def rank_key(indexed_node: tuple[int, Any]) -> tuple[float, float, float, float, float, int]:
        index, source_node = indexed_node
        full_terms = lexical_terms(node_text(source_node))
        lead_terms = lexical_terms(node_text(source_node)[:400])
        full_coverage = len(query_unique.intersection(full_terms)) / len(query_unique)
        current_lead_coverage = (
            len(current_terms.intersection(lead_terms)) / len(current_terms)
            if current_terms
            else 0.0
        )
        pair_coverage = 0.0
        if query_pairs:
            full_pairs = set(zip(full_terms, full_terms[1:]))
            pair_coverage = len(query_pairs.intersection(full_pairs)) / len(query_pairs)
        span = minimum_cover_span(full_terms, query_unique)
        proximity = 1.0 / span if span is not None else 0.0
        dense_score = float(source_node.score or 0.0)
        return (
            full_coverage,
            current_lead_coverage,
            proximity,
            pair_coverage,
            dense_score,
            -index,
        )

    return [
        source_node
        for _, source_node in sorted(
            enumerate(source_nodes),
            key=rank_key,
            reverse=True,
        )
    ]


def rank_by_lexical_evidence(
    question: str,
    source_nodes: Sequence[Any],
    *,
    extra_questions: Sequence[str] = (),
    term_weight: float,
    pair_weight: float,
    node_text: Callable[[Any], str],
) -> list[Any]:
    """Boost literal query evidence while retaining dense similarity.

    Only informative query terms contribute.  Matching both the terms and
    their adjacent pairs rewards passages that contain the user's actual
    concept (for example ``конструкционный расчёт``), rather than generic
    manual boilerplate that happens to repeat one common word.  Term and
    pair coverage are weighted by ``term_weight`` and ``pair_weight``;
    setting both to ``0`` disables the lexical boost entirely and restores
    a pure dense ordering.  ``extra_questions`` (typically the English
    translation) contribute their own per-language coverage, so an English
    passage matching the translated terms is not disadvantaged against a
    Russian question.
    """

    node_terms = [lexical_terms(node_text(item)) for item in source_nodes]
    variants: list[tuple[set[str], set[tuple[str, str]], list[set[str]], list[set[tuple[str, str]]], dict[str, float]]] = []
    for variant_question in (question, *extra_questions):
        query_terms = lexical_terms(variant_question)
        if not query_terms:
            continue
        query_unique = set(query_terms)
        query_pairs = set(zip(query_terms, query_terms[1:]))
        # Compare substantial translations independently so bilingual
        # boilerplate cannot displace a direct monolingual passage.
        script_groups = mixed_query_groups(query_terms)
        group_pairs = []
        for group in script_groups:
            terms = [term for term in query_terms if term in group]
            group_pairs.append(set(pairwise(terms)))
        weights = {
            term: 1 + math.log((1 + len(node_terms)) / (1 + sum(term in terms for terms in node_terms)))
            for term in query_unique
        } if script_groups else {}
        variants.append((query_unique, query_pairs, script_groups, group_pairs, weights))
    if not variants:
        return list(source_nodes)

    def variant_relevance(indexed_node: tuple[int, Any], variant: tuple) -> float:
        query_unique, query_pairs, script_groups, group_pairs, weights = variant
        text_terms = node_terms[indexed_node[0]]
        text_unique = set(text_terms)
        term_coverage = len(query_unique & text_unique) / len(query_unique)
        pair_coverage = 0.0
        if query_pairs:
            text_pairs = set(zip(text_terms, text_terms[1:]))
            pair_coverage = len(query_pairs & text_pairs) / len(query_pairs)
        if script_groups:
            term_coverage = max(
                sum(weights[term] for term in group & text_unique) / sum(weights[term] for term in group)
                for group in script_groups
            )
            # Pairs never cross the translation boundary.
            pair_coverage = max(
                len(pairs & text_pairs) / len(pairs) if pairs else 0.0
                for pairs in group_pairs
            )
        return term_weight * term_coverage + pair_weight * pair_coverage

    def rank_key(indexed_node: tuple[int, Any]) -> tuple[float, float, int]:
        index, source_node = indexed_node
        lexical_relevance = max(variant_relevance(indexed_node, variant) for variant in variants)
        dense_score = float(source_node.score or 0.0)
        # Dense and lexical evidence have equal bounded influence.  The
        # original cosine score remains untouched for confidence checks.
        return dense_score + lexical_relevance, dense_score, -index

    return [
        source_node
        for _, source_node in sorted(
            enumerate(source_nodes),
            key=rank_key,
            reverse=True,
        )
    ]
