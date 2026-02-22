"""Academic paper search: Semantic Scholar and arXiv."""
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

# Semantic Scholar: no API key required for basic use (rate-limited)
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
ARXIV_API = "http://export.arxiv.org/api/query"


def _get(url: str) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ResearchAgent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8")
    except Exception:
        return None


def search_semantic_scholar(query: str, limit: int = 10) -> list[dict]:
    """Search Semantic Scholar for papers. Returns list of dicts with title, authors, abstract, url, year."""
    q = urllib.parse.quote(query.strip().replace("-", " "))
    fields = "title,url,abstract,authors,year,citationCount"
    url = f"{SEMANTIC_SCHOLAR_BASE}/paper/search?query={q}&limit={min(limit, 100)}&fields={fields}"
    raw = _get(url)
    if not raw:
        return []
    try:
        data = json.loads(raw)
        papers = data.get("data") or []
        out = []
        for p in papers:
            authors = p.get("authors") or []
            author_names = [a.get("name", "") for a in authors[:5]]
            out.append({
                "title": p.get("title") or "",
                "authors": ", ".join(author_names) + (" et al." if len(authors) > 5 else ""),
                "abstract": (p.get("abstract") or "")[:500],
                "url": p.get("url") or f"https://www.semanticscholar.org/paper/{p.get('paperId', '')}",
                "year": p.get("year"),
                "citationCount": p.get("citationCount"),
            })
        return out
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def search_arxiv(query: str, max_results: int = 10) -> list[dict]:
    """Search arXiv for preprints. Returns list of dicts with title, authors, summary, link, published."""
    q = urllib.parse.quote(query.strip())
    url = f"{ARXIV_API}?search_query=all:{q}&start=0&max_results={max_results}"
    raw = _get(url)
    if not raw:
        return []
    try:
        root = ET.fromstring(raw)
        # arXiv uses default namespace; ElementTree stores tags as {uri}localname
        ATOM = "http://www.w3.org/2005/Atom"
        entries = root.findall(f".//{{{ATOM}}}entry")
        out = []
        for e in entries:
            title_el = e.find(f"{{{ATOM}}}title")
            title = (title_el.text or "").strip() if title_el is not None else ""
            if title and title != "Error":
                summary_el = e.find(f"{{{ATOM}}}summary")
                summary = (summary_el.text or "").strip().replace("\n", " ")[:500] if summary_el is not None else ""
                link_el = e.find(f"{{{ATOM}}}id")
                link = (link_el.text or "").strip() if link_el is not None else ""
                published_el = e.find(f"{{{ATOM}}}published")
                published = (published_el.text or "") if published_el is not None else ""
                authors = e.findall(f"{{{ATOM}}}author")
                author_names = []
                for a in authors[:5]:
                    name_el = a.find(f"{{{ATOM}}}name")
                    if name_el is not None and name_el.text:
                        author_names.append(name_el.text)
                out.append({
                    "title": title,
                    "authors": ", ".join(author_names) + (" et al." if len(authors) > 5 else ""),
                    "abstract": summary,
                    "url": link,
                    "year": published[:4] if len(published) >= 4 else None,
                    "citationCount": None,
                })
        return out
    except (ET.ParseError, AttributeError, TypeError):
        return []


def search_academic_papers(
    query: str,
    max_results: int = 10,
    source: str = "both",
) -> str:
    """
    Search for academic papers by topic. Use when the user wants to find relevant papers,
    do a literature search, or discover research on a topic.
    source: "semantic_scholar" (published papers, many fields), "arxiv" (preprints, CS/math/physics),
    or "both" (default) to search both and merge results.
    """
    if not query or not query.strip():
        return "Please provide a search query (e.g. a topic or research question)."
    limit_each = max(5, (max_results + 1) // 2) if source == "both" else max_results
    papers = []
    if source in ("both", "semantic_scholar"):
        papers.extend(search_semantic_scholar(query, limit=limit_each))
    if source in ("both", "arxiv"):
        arxiv_list = search_arxiv(query, max_results=limit_each)
        # Prefer Semantic Scholar when both; add arXiv results that don't duplicate by title
        seen_titles = {p["title"].lower()[:60] for p in papers}
        for p in arxiv_list:
            if p["title"].lower()[:60] not in seen_titles:
                papers.append(p)
                seen_titles.add(p["title"].lower()[:60])
    if not papers:
        return "No academic papers found for that query. Try a different or broader search."
    lines = []
    for i, p in enumerate(papers[:max_results], 1):
        title = p.get("title") or "No title"
        authors = p.get("authors") or "Unknown"
        abstract = (p.get("abstract") or "")[:400]
        url = p.get("url") or ""
        year = p.get("year")
        cites = p.get("citationCount")
        year_str = f" ({year})" if year else ""
        cite_str = f" [cited {cites}×]" if cites is not None else ""
        lines.append(f"{i}. {title}{year_str}{cite_str}\n   Authors: {authors}\n   {abstract}\n   {url}")
    return "\n\n".join(lines)
