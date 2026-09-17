"""Sensory front-end: fetch raw text. This is NOT the memory store.

Wikipedia/local files are the analogue of eyes and ears. Learned structure
still exists only in synaptic weights.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


USER_AGENT = (
    "BrainMemory/0.1 (experimental educational project; "
    "https://github.com/TheNickCoolst/brainmemory)"
)


@dataclass
class Document:
    title: str
    text: str
    source: str
    related: list[str] = field(default_factory=list)


class LocalFolderSense:
    """Read .txt/.md files from a folder. Offline, no API key."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def search(self, query: str, limit: int = 5) -> list[Document]:
        if not self.root.exists():
            return []
        q = query.lower().strip()
        hits: list[Document] = []
        for path in sorted(self.root.rglob("*")):
            if path.suffix.lower() not in {".txt", ".md"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            hay = f"{path.stem} {text}".lower()
            if q in hay or all(part in hay for part in q.split() if len(part) > 2):
                hits.append(
                    Document(
                        title=path.stem.replace("_", " "),
                        text=text.strip(),
                        source=str(path),
                        related=_related_headings(text),
                    )
                )
            if len(hits) >= limit:
                break
        return hits

    def fetch(self, ref: str) -> Document | None:
        path = Path(ref)
        if not path.exists():
            cand = list(self.root.glob(f"{ref}*"))
            path = cand[0] if cand else path
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8", errors="ignore")
        return Document(title=path.stem.replace("_", " "), text=text.strip(), source=str(path))


class WikipediaSense:
    """MediaWiki REST/opensearch. Requires network; no API key."""

    def __init__(self, language: str = "de", timeout: float = 12.0):
        self.language = language
        self.timeout = timeout
        self.base = f"https://{language}.wikipedia.org"

    def search(self, query: str, limit: int = 5) -> list[Document]:
        q = urllib.parse.quote(query)
        url = f"{self.base}/w/api.php?action=opensearch&search={q}&limit={int(limit)}&namespace=0&format=json"
        raw = _get_json(url, self.timeout)
        if not isinstance(raw, list) or len(raw) < 4:
            return []
        titles, descs, urls = raw[1], raw[2], raw[3]
        docs: list[Document] = []
        for title, desc, page_url in zip(titles, descs, urls):
            full = self.fetch(title)
            if full and full.text:
                docs.append(full)
            elif desc:
                docs.append(Document(title=str(title), text=str(desc), source=str(page_url)))
        return docs

    def fetch(self, title: str) -> Document | None:
        slug = urllib.parse.quote(title.replace(" ", "_"))
        url = f"{self.base}/api/rest_v1/page/summary/{slug}"
        data = _get_json(url, self.timeout)
        if not data or data.get("type") == "disambiguation":
            extract = str(data.get("extract", "") if data else "")
            if not extract:
                return None
        extract = str(data.get("extract") or "")
        if not extract:
            return None
        related = self._links(title)
        return Document(
            title=str(data.get("title") or title),
            text=extract.strip(),
            source=str(data.get("content_urls", {}).get("desktop", {}).get("page") or url),
            related=related,
        )

    def _links(self, title: str, limit: int = 8) -> list[str]:
        q = urllib.parse.quote(title)
        url = (
            f"{self.base}/w/api.php?action=query&prop=links&pllimit={int(limit)}"
            f"&plnamespace=0&titles={q}&format=json"
        )
        data = _get_json(url, self.timeout) or {}
        pages = (data.get("query") or {}).get("pages") or {}
        names: list[str] = []
        for page in pages.values():
            for link in page.get("links") or []:
                t = str(link.get("title") or "").strip()
                if t and ":" not in t:
                    names.append(t)
        return names[:limit]


class CompositeSense:
    def __init__(self, senses: list):
        self.senses = senses

    def search(self, query: str, limit: int = 5) -> list[Document]:
        out: list[Document] = []
        for sense in self.senses:
            try:
                out.extend(sense.search(query, limit=limit))
            except Exception:
                continue
            if len(out) >= limit:
                break
        return out[:limit]


def chunk_text(text: str, max_chars: int = 420) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    buf = ""
    for p in parts:
        if not p:
            continue
        if buf and len(buf) + 1 + len(p) > max_chars:
            chunks.append(buf.strip())
            buf = p
        else:
            buf = f"{buf} {p}".strip()
    if buf:
        chunks.append(buf.strip())
    return chunks


def _related_headings(text: str) -> list[str]:
    heads = re.findall(r"^#+\s+(.+)$", text, flags=re.MULTILINE)
    return [h.strip() for h in heads if h.strip()][:8]


def _get_json(url: str, timeout: float) -> dict | list | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
