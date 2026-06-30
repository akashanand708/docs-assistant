"""Web scraper for documentation sites."""

from __future__ import annotations

import re
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from rich.console import Console
from tqdm import tqdm

from .. import config

console = Console()


@dataclass
class ScrapedPage:
    """Represents a scraped documentation page."""

    url: str
    title: str
    content: str
    breadcrumbs: list[str]

    def to_document_text(self) -> str:
        """Convert to text suitable for embedding."""
        breadcrumb_str = " > ".join(self.breadcrumbs) if self.breadcrumbs else ""
        return f"# {self.title}\n\nPath: {breadcrumb_str}\nURL: {self.url}\n\n{self.content}"


class DocsScraper:
    """Scraper for a documentation site."""

    def __init__(
        self,
        base_url: str = config.DOCS_BASE_URL,
        max_pages: int = config.MAX_PAGES,
        request_delay: float = config.REQUEST_DELAY,
    ):
        self.base_url = base_url.rstrip("/")
        self.max_pages = max_pages
        self.request_delay = request_delay
        self.visited: set[str] = set()
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (compatible; DocsAssistant/1.0; Documentation Scraper)"}
        )

    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments and trailing slashes."""
        parsed = urlparse(url)
        # Remove fragment and normalize
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        return normalized

    def _is_valid_doc_url(self, url: str) -> bool:
        """Check if URL is a valid documentation page to scrape."""
        if not url:
            return False

        parsed = urlparse(url)

        # Must be same domain
        base_parsed = urlparse(self.base_url)
        if parsed.netloc != base_parsed.netloc:
            return False

        # Skip non-HTML resources
        skip_extensions = (
            ".pdf",
            ".zip",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
            ".css",
            ".js",
            ".json",
            ".xml",
        )
        if any(parsed.path.lower().endswith(ext) for ext in skip_extensions):
            return False

        # Skip common non-doc paths
        skip_paths = ("/search", "/api/", "/_", "/static/", "/assets/")
        if any(skip in parsed.path for skip in skip_paths):
            return False

        return True

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from the page, removing navigation and boilerplate."""
        # Remove unwanted elements
        for selector in [
            "nav",
            "header",
            "footer",
            "script",
            "style",
            "aside",
            ".sidebar",
            ".navigation",
            ".nav",
            ".menu",
            ".toc",
            ".breadcrumb",
            "#cookie-banner",
            ".cookie-consent",
        ]:
            for elem in soup.select(selector):
                elem.decompose()

        # Try to find main content area
        content_selectors = [
            "main",
            "article",
            ".content",
            ".main-content",
            ".doc-content",
            ".documentation",
            "#content",
            "#main",
        ]

        content_elem = None
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                break

        if not content_elem:
            content_elem = soup.body or soup

        # Get text and clean up whitespace
        text = content_elem.get_text(separator="\n", strip=True)

        # Clean up excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        # Try various title selectors
        for selector in ["h1", "title", ".page-title", ".doc-title"]:
            elem = soup.select_one(selector)
            if elem:
                title = elem.get_text(strip=True)
                if title:
                    return title
        return "Untitled"

    def _extract_breadcrumbs(self, soup: BeautifulSoup) -> list[str]:
        """Extract breadcrumb navigation if available."""
        breadcrumb_selectors = [
            ".breadcrumb",
            ".breadcrumbs",
            "nav[aria-label='breadcrumb']",
            ".doc-breadcrumb",
        ]

        for selector in breadcrumb_selectors:
            elem = soup.select_one(selector)
            if elem:
                links = elem.find_all(["a", "span", "li"])
                crumbs = [link.get_text(strip=True) for link in links]
                crumbs = [c for c in crumbs if c and c not in ("›", ">", "/", "»")]
                if crumbs:
                    return crumbs

        return []

    def _extract_links(self, soup: BeautifulSoup, current_url: str) -> list[str]:
        """Extract all documentation links from the page."""
        links = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]

            # Convert relative URLs to absolute
            absolute_url = urljoin(current_url, href)
            normalized = self._normalize_url(absolute_url)

            if self._is_valid_doc_url(normalized) and normalized not in self.visited:
                links.append(normalized)

        return list(set(links))

    def _fetch_page(self, url: str) -> BeautifulSoup | None:
        """Fetch and parse a single page."""
        try:
            resp = self.session.get(url, timeout=config.REQUEST_TIMEOUT)
            resp.raise_for_status()

            # Check content type
            content_type = resp.headers.get("content-type", "").lower()
            if "text/html" not in content_type:
                return None

            return BeautifulSoup(resp.text, "lxml")
        except requests.RequestException as e:
            console.print(f"[yellow]Warning: Failed to fetch {url}: {e}[/yellow]")
            return None

    def crawl(self) -> Iterator[ScrapedPage]:
        """Crawl the documentation site and yield scraped pages."""
        queue: deque[str] = deque([self.base_url])
        pages_scraped = 0

        console.print(f"[bold blue]Starting crawl from {self.base_url}[/bold blue]")
        console.print(f"[dim]Max pages: {self.max_pages}, Delay: {self.request_delay}s[/dim]")

        with tqdm(total=self.max_pages, desc="Scraping pages", unit="page") as pbar:
            while queue and pages_scraped < self.max_pages:
                url = queue.popleft()

                if url in self.visited:
                    continue

                self.visited.add(url)

                # Fetch and parse page
                soup = self._fetch_page(url)
                if not soup:
                    continue

                # Extract content
                content = self._extract_content(soup)
                if content and len(content) >= 50:
                    page = ScrapedPage(
                        url=url,
                        title=self._extract_title(soup),
                        content=content,
                        breadcrumbs=self._extract_breadcrumbs(soup),
                    )
                    pages_scraped += 1
                    pbar.update(1)
                    pbar.set_postfix({"queued": len(queue)})
                    yield page

                # Add new links to queue
                new_links = self._extract_links(soup, url)
                queue.extend(new_links)

                # Rate limiting
                if self.request_delay > 0:
                    time.sleep(self.request_delay)

        console.print(
            f"[bold green]Crawl complete! Scraped {pages_scraped} pages, visited {len(self.visited)} URLs[/bold green]"
        )
