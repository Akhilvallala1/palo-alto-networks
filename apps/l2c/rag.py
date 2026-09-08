"""Local embeddings and a sqlite-vec index over the policy corpus.

No managed vector database and no hosted embedding endpoint. Both would add
cost, add a key to the zero-key path, and — the reason that actually matters —
hide the retrieval step behind an API call, when the retrieval step is the part
a reviewer wants to read.

**The embedder is lexical, and says so.** `HashingEmbedder` is a hashed
bag-of-features: word unigrams, word bigrams and character 4-grams, each hashed
into a fixed number of dimensions with a signed weight, then L2-normalised. It
is the classic hashing vectoriser, not a learned model, so it captures term
overlap and nothing else — "discount above twenty percent" retrieves the RVP
clause because the words are there, not because it understands escalation. That
is an honest fit for a corpus of eighteen short clauses with distinctive
vocabulary, it needs no model download, and it is deterministic across machines
and processes because it hashes with BLAKE2b rather than `hash()`, which is
randomised per interpreter by PYTHONHASHSEED.

Character n-grams are included so that a query saying "22% off" still reaches a
clause written "twenty percent (20%)" through the shared surface forms around
it. Where they are not enough, `search` is not doing semantic matching and no
amount of tuning here will make it; the fix would be a real embedding model,
which is a dependency this app deliberately does not take.

The index is a `vec0` virtual table. `sqlite-vec` is loaded as a SQLite
extension, so a Python whose `sqlite3` was built without extension support
raises `RagUnavailableError` with the reason rather than silently degrading to
a linear scan and reporting different neighbours.
"""

import hashlib
import itertools
import math
import re
import sqlite3
import struct
from collections.abc import Iterable, Sequence
from pathlib import Path

import sqlite_vec

from .models import PolicyEvidence
from .policies import POLICY_SECTIONS, PolicySection

__all__ = [
    "DEFAULT_DIMENSIONS",
    "MAX_DIMENSIONS",
    "HashingEmbedder",
    "PolicyIndex",
    "RagUnavailableError",
    "build_index",
]

#: Character 4-grams mean a clause contributes roughly as many features as it
#: has characters, so a narrow vector collides them into each other and signed
#: hashing then cancels genuine matches. 2048 is the width at which top-1
#: retrieval over the shipped corpus becomes exact; 512 does not manage it.
DEFAULT_DIMENSIONS = 2048

#: sqlite-vec's `vec0` refuses a vector column wider than this.
MAX_DIMENSIONS = 8192

_WORD_RE = re.compile(r"[a-z0-9]+")

#: Words that carry no retrieval signal in a corpus that is entirely policy
#: prose. Kept deliberately short: an aggressive stop list on an eighteen-chunk
#: corpus removes more signal than noise.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "before",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
    }
)


class RagUnavailableError(RuntimeError):
    """The sqlite-vec extension could not be loaded into this SQLite build."""


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


#: Per-family feature weights. Character grams are numerous — a clause yields
#: roughly as many of them as it has characters, against a few dozen words — so
#: at equal weight they dominate the vector and push every similarity toward
#: zero. They earn their place as a bridge across spelling variants, not as the
#: main signal, and are weighted accordingly.
_W_WORD = 1.0
_W_BIGRAM = 0.7
_W_CHARGRAM = 0.25


def _features(text: str) -> list[tuple[str, float]]:
    """Weighted word unigrams, word bigrams and character 4-grams.

    Bigrams are what separate "written approval" from two unrelated mentions of
    each word, and the character grams give partial credit across the
    "20%" / "twenty percent" spelling split the corpus uses on purpose.
    """
    words = _tokens(text)
    kept = [word for word in words if word not in _STOPWORDS]
    features: list[tuple[str, float]] = [(f"w:{word}", _W_WORD) for word in kept]
    features += [(f"b:{first}_{second}", _W_BIGRAM) for first, second in itertools.pairwise(words)]
    squashed = "".join(words)
    features += [
        (f"c:{squashed[i : i + 4]}", _W_CHARGRAM) for i in range(max(0, len(squashed) - 3))
    ]
    return features


class HashingEmbedder:
    """Deterministic lexical embeddings. No model, no network, no state."""

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS, *, seed: int = 0) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if dimensions > MAX_DIMENSIONS:
            raise ValueError(
                f"dimensions must be <= {MAX_DIMENSIONS}; sqlite-vec's vec0 rejects "
                f"a wider vector column, and {dimensions} would fail at index build time"
            )
        self.dimensions = dimensions
        self.seed = seed

    def _bucket(self, feature: str) -> tuple[int, float]:
        """Hash a feature to a dimension and a sign.

        BLAKE2b keyed on the seed, so the mapping is stable across processes and
        machines. The sign bit is the standard hashing-trick trick: it makes
        collisions cancel on average instead of always reinforcing.
        """
        digest = hashlib.blake2b(
            feature.encode("utf-8"), digest_size=8, key=str(self.seed).encode("utf-8")
        ).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dimensions, 1.0 if (value >> 63) & 1 else -1.0

    def embed(self, text: str) -> list[float]:
        """A unit-length vector. Empty text yields an all-zero vector."""
        vector = [0.0] * self.dimensions
        counts: dict[str, int] = {}
        weights: dict[str, float] = {}
        for feature, weight in _features(text):
            counts[feature] = counts.get(feature, 0) + 1
            weights[feature] = weight
        for feature, count in counts.items():
            index, sign = self._bucket(feature)
            # Sublinear term frequency: a clause that says "discount" nine times
            # should not out-rank one that says it twice by a factor of four.
            vector[index] += sign * weights[feature] * (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


def _pack(vector: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


class PolicyIndex:
    """A `vec0` table over the policy corpus, plus citation resolution.

    Holds the chunk text itself alongside the vectors so that a citation can be
    resolved — and quoted — without a second store to keep in sync.
    """

    def __init__(
        self,
        sections: Sequence[PolicySection],
        *,
        embedder: HashingEmbedder | None = None,
        path: Path | None = None,
    ) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.sections: dict[str, PolicySection] = {s.section_id: s for s in sections}
        self._conn = self._connect(path)
        self._build(sections)

    @staticmethod
    def _connect(path: Path | None) -> sqlite3.Connection:
        conn = sqlite3.connect(str(path) if path is not None else ":memory:")
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except (AttributeError, sqlite3.OperationalError) as exc:
            conn.close()
            raise RagUnavailableError(
                "could not load the sqlite-vec extension into this SQLite build "
                f"({type(exc).__name__}: {exc}). The policy index needs it; a Python "
                "built with --disable-loadable-sqlite-extensions cannot run the L2C RAG."
            ) from exc
        return conn

    def _build(self, sections: Iterable[PolicySection]) -> None:
        dims = self.embedder.dimensions
        with self._conn:
            self._conn.execute("DROP TABLE IF EXISTS policy_chunks")
            self._conn.execute(
                # `distance_metric=cosine` is not the default — vec0 uses L2
                # unless told otherwise. On unit vectors the two rank identically,
                # so the wrong metric would still retrieve the right clause while
                # reporting `1 - distance` as a number that is not a similarity.
                f"CREATE VIRTUAL TABLE policy_chunks USING vec0("
                f"section_id TEXT PRIMARY KEY, embedding float[{dims}] distance_metric=cosine)"
            )
            for section in sections:
                self._conn.execute(
                    "INSERT INTO policy_chunks(section_id, embedding) VALUES (?, ?)",
                    (section.section_id, _pack(self.embedder.embed(section.embedding_text))),
                )

    def __len__(self) -> int:
        return len(self.sections)

    @property
    def section_ids(self) -> tuple[str, ...]:
        return tuple(self.sections)

    def search(self, query: str, k: int = 4) -> list[PolicyEvidence]:
        """Nearest `k` clauses by cosine distance, best first.

        `score` is `1 - distance`, which for sqlite-vec's cosine metric is the
        cosine similarity itself: it runs -1..1, not 0..1, and signed hashing
        makes small negative values ordinary for an unrelated chunk. Treat it as
        a ranking, not as a calibrated confidence.
        """
        if k <= 0:
            return []
        vector = self.embedder.embed(query)
        if not any(vector):
            return []
        rows = self._conn.execute(
            """
            SELECT section_id, distance FROM policy_chunks
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (_pack(vector), k),
        ).fetchall()
        evidence: list[PolicyEvidence] = []
        for section_id, distance in rows:
            section = self.sections.get(str(section_id))
            if section is None:  # pragma: no cover - index and dict are built together
                continue
            evidence.append(
                PolicyEvidence(
                    section_id=section.section_id,
                    doc_id=section.doc_id,
                    title=section.title,
                    text=section.text,
                    score=round(1.0 - float(distance), 6),
                )
            )
        return evidence

    def resolve(self, section_id: str) -> PolicySection | None:
        """The clause behind a citation, or None if it does not exist.

        This is the function that makes "the decision cites a real chunk" a
        checkable claim: a citation the index cannot resolve is a bug, and the
        tests assert on this rather than on the presence of a plausible string.
        """
        return self.sections.get(section_id)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PolicyIndex":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def build_index(
    sections: Sequence[PolicySection] | None = None,
    *,
    dimensions: int = DEFAULT_DIMENSIONS,
    path: Path | None = None,
) -> PolicyIndex:
    """Index the shipped corpus, or a supplied one in tests."""
    return PolicyIndex(
        POLICY_SECTIONS if sections is None else sections,
        embedder=HashingEmbedder(dimensions),
        path=path,
    )
