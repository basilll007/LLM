#!/usr/bin/env python3
"""
Robust async web fetcher and extractor.

Usage:
  - Set SEARCH_API_KEY in your environment if using search-api integration.
  - Either:
      * Edit fetch_search_api_results() to call your search API and return URL list,
        OR
      * Create a seed_urls.txt file (one URL per line).
  - Run: python scraper.py

Outputs:
  output/
    raw_html/<hostname>/<safe_filename>.html
    text/<hostname>/<safe_filename>.txt
    meta/<hostname>/<safe_filename>.json
  state.json   (resume state)
"""

import asyncio
import aiohttp
import aiofiles
import os
import json
import time
import hashlib
import math
import logging
import re
from urllib.parse import urlparse, urljoin
from urllib import robotparser
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime
from dateutil import parser as dateparser
from tqdm.asyncio import tqdm
import trafilatura
from bs4 import BeautifulSoup
from lxml_html_clean import Cleaner
import PyPDF2
import pdfplumber
import io

# Config
OUTPUT_DIR = Path("output")
RAW_DIR = OUTPUT_DIR / "raw_html"
TEXT_DIR = OUTPUT_DIR / "text"
META_DIR = OUTPUT_DIR / "meta"
STATE_FILE = OUTPUT_DIR / "state.json"
SEED_FILE = Path("seed_urls.txt")  # fallback
TARGET_BYTES = 1 * 1024**3  # 1 GiB, change to 1_000_000_000 if you want decimal GB
CONCURRENT_HOSTS = 10           # concurrent hosts (semaphores)
CONCURRENT_FETCH_PER_HOST = 3   # concurrency per host
REQUEST_TIMEOUT = 30            # seconds
MAX_RETRIES = 5
BACKOFF_FACTOR = 1.5
USER_AGENT = "ResearchScraper/1.0 (+https://example.org)"

# LLM Training Quality Standards
MIN_TEXT_LENGTH = 500           # Minimum text length for LLM training
MIN_WORD_COUNT = 100           # Minimum word count
MAX_TEXT_LENGTH = 50000        # Maximum text length to avoid extremely long documents
MIN_SENTENCE_COUNT = 5         # Minimum number of sentences
MIN_PARAGRAPH_COUNT = 2        # Minimum number of paragraphs
QUALITY_SCORE_THRESHOLD = 0.6  # Minimum quality score (0-1)

# Nested Link Discovery Settings
ENABLE_LINK_DISCOVERY = True    # Enable discovery of nested links
MAX_DEPTH = 4                   # Maximum crawl depth from seed URLs (increased for deeper crawling)
MAX_LINKS_PER_PAGE = 100        # Maximum links to extract per page (increased for more discovery)
SAME_DOMAIN_ONLY = False        # Allow cross-domain crawling for better content discovery
RELEVANT_LINK_PATTERNS = [      # Patterns for relevant links (quantum science focus)
    r'quantum', r'physics', r'research', r'science', r'theory', r'computing',
    r'mechanics', r'field', r'particle', r'academic', r'paper', r'article',
    r'lecture', r'course', r'tutorial', r'documentation', r'guide', r'pdf'
]
EXCLUDE_LINK_PATTERNS = [       # Patterns to exclude (removed PDF exclusion)
    r'login', r'register', r'cart', r'checkout', r'admin', r'api',
    r'\.doc$', r'\.ppt$', r'\.zip$', r'\.exe$',
    r'facebook\.com', r'twitter\.com', r'linkedin\.com', r'instagram\.com'
]

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("scraper")

# Ensure directories
for d in (RAW_DIR, TEXT_DIR, META_DIR):
    d.mkdir(parents=True, exist_ok=True)

# State for resume
state = {
    "processed_hashes": {},   # sha256 -> {"url":..., "filename":...}
    "queued_urls": [],       # list of pending URLs
    "discovered_urls": set(), # set of discovered URLs to avoid re-discovery
    "url_depths": {},        # url -> depth mapping for crawl depth tracking
    "bytes_saved": 0,
    "last_run": None
}

# Load state if present
if STATE_FILE.exists():
    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            st = json.load(f)
            # Convert discovered_urls back to set if it was saved as list
            if "discovered_urls" in st and isinstance(st["discovered_urls"], list):
                st["discovered_urls"] = set(st["discovered_urls"])
            state.update(st)
            # Ensure new fields exist
            state.setdefault("discovered_urls", set())
            state.setdefault("url_depths", {})
            logger.info(f"Resumed state: {len(state['processed_hashes'])} items processed, {len(state['queued_urls'])} queued, {state['bytes_saved']} bytes saved")
    except Exception as e:
        logger.warning("Could not load state file, starting fresh: %s", e)


def save_state():
    state["last_run"] = datetime.utcnow().isoformat() + "Z"
    # Convert set to list for JSON serialization
    state_copy = state.copy()
    if isinstance(state_copy.get("discovered_urls"), set):
        state_copy["discovered_urls"] = list(state_copy["discovered_urls"])
    with STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(state_copy, f, indent=2)


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def extract_links_from_html(html_bytes: bytes, base_url: str) -> List[str]:
    """
    Extract relevant links from HTML content for further crawling.
    
    Args:
        html_bytes: Raw HTML content
        base_url: Base URL for resolving relative links
        
    Returns:
        List of discovered URLs that match relevance criteria
    """
    if not ENABLE_LINK_DISCOVERY:
        return []
    
    try:
        soup = BeautifulSoup(html_bytes, 'html.parser')
        base_domain = urlparse(base_url).netloc
        discovered_links = []
        
        # Find all links
        for link in soup.find_all('a', href=True):
            href = link.get('href', '').strip()
            if not href or href.startswith('#') or href.startswith('mailto:') or href.startswith('tel:'):
                continue
                
            # Resolve relative URLs
            full_url = urljoin(base_url, href)
            parsed_url = urlparse(full_url)
            
            # Skip if not HTTP/HTTPS
            if parsed_url.scheme not in ('http', 'https'):
                continue
                
            # Domain filtering
            if SAME_DOMAIN_ONLY and parsed_url.netloc != base_domain:
                continue
                
            # Check exclude patterns
            if any(re.search(pattern, full_url, re.IGNORECASE) for pattern in EXCLUDE_LINK_PATTERNS):
                continue
                
            # Check relevance patterns (URL + link text)
            link_text = link.get_text(strip=True).lower()
            url_text = full_url.lower()
            combined_text = f"{url_text} {link_text}"
            
            if any(re.search(pattern, combined_text, re.IGNORECASE) for pattern in RELEVANT_LINK_PATTERNS):
                discovered_links.append(full_url)
                
            # Limit links per page
            if len(discovered_links) >= MAX_LINKS_PER_PAGE:
                break
                
        logger.debug(f"Discovered {len(discovered_links)} relevant links from {base_url}")
        return discovered_links
        
    except Exception as e:
        logger.warning(f"Failed to extract links from {base_url}: {e}")
        return []


def should_crawl_url(url: str, current_depth: int) -> bool:
    """
    Determine if a URL should be crawled based on depth and discovery rules.
    
    Args:
        url: URL to check
        current_depth: Current crawl depth
        
    Returns:
        True if URL should be crawled
    """
    # Check depth limit
    if current_depth >= MAX_DEPTH:
        return False
        
    # Check if already discovered
    if url in state["discovered_urls"]:
        return False
        
    # Check if already processed
    url_hash = sha256_bytes(url.encode('utf-8'))
    if url_hash in state["processed_hashes"]:
        return False
        
    return True


def safe_filename_from_url(url: str) -> str:
    """Generate a safe filename from a URL."""
    parsed = urlparse(url)
    base = parsed.path.strip("/").split("/")[-1] or "index"
    base = "".join(ch for ch in base if ch.isalnum() or ch in "-_.")[:120] or "index"
    return f"{base}_{sha256_bytes(url.encode())[:12]}"


async def fetch_search_api_results() -> List[str]:
    """
    Modify this function to call your search-api according to its docs.
    It should return a list of URLs (strings).

    Example approach (pseudo):
      - Use aiohttp to POST/GET the search endpoint with your query ("quantum") and the API key in headers.
      - Parse JSON and return URLs.

    For safety, this function will first try to read SEED_FILE if present.
    """
    # If a seed file exists, use it as the source of URLs:
    if SEED_FILE.exists():
        urls = [line.strip() for line in SEED_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
        logger.info("Loaded %d seed URLs from %s", len(urls), SEED_FILE)
        return urls

    # Placeholder: raise so user knows to implement or add seed file
    logger.warning("No seed_urls.txt found. You MUST implement fetch_search_api_results() to call your search API (see code comments)")
    return []


class RobotsCache:
    def __init__(self):
        self._cache: Dict[str, robotparser.RobotFileParser] = {}

    async def allowed(self, session: aiohttp.ClientSession, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._cache:
            robots_url = f"{base}/robots.txt"
            parser = robotparser.RobotFileParser()
            try:
                async with session.get(robots_url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        parser.parse(text.splitlines())
                    else:
                        # If robots.txt missing or inaccessible, assume allowed (conservative choice could be disallow)
                        parser = robotparser.RobotFileParser()
                        parser.parse([])
                self._cache[base] = parser
            except Exception as e:
                logger.debug("Robots fetch error for %s: %s", base, e)
                parser = robotparser.RobotFileParser()
                parser.parse([])
                self._cache[base] = parser
        return self._cache[base].can_fetch(USER_AGENT, url)


async def fetch_with_retries(session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
    backoff = BACKOFF_FACTOR
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    return data
                elif resp.status in (429, 503):
                    # Rate limited or service unavailable -> backoff
                    logger.warning("Status %d for %s (attempt %d). Backing off %s sec", resp.status, url, attempt, backoff)
                    await asyncio.sleep(backoff)
                    backoff *= BACKOFF_FACTOR
                    continue
                else:
                    logger.info("Skipping %s (HTTP %d)", url, resp.status)
                    return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning("Fetch error %s for %s (attempt %d)", e, url, attempt)
            await asyncio.sleep(backoff)
            backoff *= BACKOFF_FACTOR
    logger.error("Max retries reached for %s", url)
    return None


def extract_text_from_pdf(pdf_bytes: bytes, url: str) -> str:
    """
    Extract text from PDF bytes using multiple methods for best results.
    """
    text = ""
    
    try:
        # Method 1: Try pdfplumber first (better for complex layouts)
        with io.BytesIO(pdf_bytes) as pdf_file:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
        
        if text.strip():
            logger.info(f"Successfully extracted text from PDF using pdfplumber: {url}")
            return clean_and_structure_text(text)
    
    except Exception as e:
        logger.warning(f"pdfplumber failed for {url}: {e}")
    
    try:
        # Method 2: Fallback to PyPDF2
        with io.BytesIO(pdf_bytes) as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        
        if text.strip():
            logger.info(f"Successfully extracted text from PDF using PyPDF2: {url}")
            return clean_and_structure_text(text)
    
    except Exception as e:
        logger.warning(f"PyPDF2 failed for {url}: {e}")
    
    logger.warning(f"Failed to extract text from PDF: {url}")
    return ""


def extract_text_from_html(html_bytes: bytes, url: str) -> str:
    """
    Enhanced text extraction optimized for LLM training data.
    Preserves structure while removing noise and improving quality.
    """
    # Try trafilatura first with enhanced settings
    try:
        text = trafilatura.extract(
            html_bytes.decode('utf-8', errors='ignore'), 
            url=url,
            include_comments=False,
            include_tables=True,
            include_formatting=True,
            favor_precision=True,
            favor_recall=False
        )
        if text and len(text.strip()) > MIN_TEXT_LENGTH:
            return clean_and_structure_text(text.strip())
    except Exception as e:
        logger.debug(f"Trafilatura extraction failed for {url}: {e}")
    
    # Enhanced BeautifulSoup fallback
    try:
        soup = BeautifulSoup(html_bytes, "html.parser")
        
        # Remove unwanted elements more comprehensively
        unwanted_tags = [
            "script", "style", "noscript", "header", "footer", "nav", "aside",
            "advertisement", "ads", "sidebar", "menu", "breadcrumb", "cookie",
            "popup", "modal", "overlay", "banner", "social", "share", "comment"
        ]
        
        for tag in unwanted_tags:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Remove elements by class/id patterns (common noise)
        noise_patterns = [
            r'.*ad.*', r'.*banner.*', r'.*popup.*', r'.*modal.*', r'.*cookie.*',
            r'.*social.*', r'.*share.*', r'.*comment.*', r'.*sidebar.*', r'.*menu.*'
        ]
        
        for pattern in noise_patterns:
            for element in soup.find_all(attrs={"class": re.compile(pattern, re.I)}):
                element.decompose()
            for element in soup.find_all(attrs={"id": re.compile(pattern, re.I)}):
                element.decompose()
        
        # Extract text with better structure preservation
        text_parts = []
        
        # Process main content areas first
        main_content = soup.find(['main', 'article', 'div'], class_=re.compile(r'.*content.*|.*article.*|.*main.*', re.I))
        if main_content:
            text_parts.append(extract_structured_text(main_content))
        else:
            # Fallback to body content
            body = soup.find('body')
            if body:
                text_parts.append(extract_structured_text(body))
            else:
                text_parts.append(extract_structured_text(soup))
        
        raw_text = '\n\n'.join(filter(None, text_parts))
        return clean_and_structure_text(raw_text)
        
    except Exception as e:
        logger.debug(f"BeautifulSoup extraction failed for {url}: {e}")
        return ""


def extract_structured_text(element) -> str:
    """Extract text while preserving document structure."""
    text_parts = []
    
    # Handle different content types
    for child in element.descendants:
        if child.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            text = child.get_text().strip()
            if text:
                text_parts.append(f"\n\n## {text}\n")
        elif child.name in ['p', 'div']:
            text = child.get_text().strip()
            if text and len(text) > 20:  # Skip very short paragraphs
                text_parts.append(f"{text}\n")
        elif child.name in ['li']:
            text = child.get_text().strip()
            if text:
                text_parts.append(f"• {text}\n")
        elif child.name in ['blockquote']:
            text = child.get_text().strip()
            if text:
                text_parts.append(f"\n> {text}\n")
    
    return ''.join(text_parts)


def clean_and_structure_text(text: str) -> str:
    """Clean and structure text for optimal LLM training."""
    if not text:
        return ""
    
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Remove common noise patterns
    noise_patterns = [
        r'Cookie Policy.*?(?=\n|$)',
        r'Privacy Policy.*?(?=\n|$)',
        r'Terms of Service.*?(?=\n|$)',
        r'Subscribe to.*?(?=\n|$)',
        r'Follow us on.*?(?=\n|$)',
        r'Share this.*?(?=\n|$)',
        r'Click here.*?(?=\n|$)',
        r'Read more.*?(?=\n|$)',
        r'Advertisement.*?(?=\n|$)',
        r'\b\d{1,2}\/\d{1,2}\/\d{2,4}\b',  # Remove standalone dates
        r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\b',  # Remove standalone times
    ]
    
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Normalize punctuation and spacing
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    text = re.sub(r'\s+([.!?,:;])', r'\1', text)
    
    # Ensure proper paragraph breaks
    sentences = re.split(r'(?<=[.!?])\s+', text)
    paragraphs = []
    current_paragraph = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            current_paragraph.append(sentence)
            # Start new paragraph after 3-5 sentences or at natural breaks
            if (len(current_paragraph) >= 3 and 
                (sentence.endswith('.') or sentence.endswith('!') or sentence.endswith('?'))):
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
    
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    return '\n\n'.join(paragraphs).strip()


def calculate_text_quality_score(text: str) -> float:
    """Calculate a quality score for text content (0-1 scale)."""
    if not text:
        return 0.0
    
    score = 0.0
    factors = 0
    
    # Length factor (optimal range: 1000-10000 chars)
    length = len(text)
    if 1000 <= length <= 10000:
        score += 1.0
    elif 500 <= length < 1000 or 10000 < length <= 20000:
        score += 0.7
    elif 200 <= length < 500 or 20000 < length <= 50000:
        score += 0.4
    factors += 1
    
    # Word count factor
    words = text.split()
    word_count = len(words)
    if word_count >= MIN_WORD_COUNT:
        score += min(1.0, word_count / 500)  # Normalize to 500 words = 1.0
    factors += 1
    
    # Sentence structure factor
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    if sentence_count >= MIN_SENTENCE_COUNT:
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        if 10 <= avg_sentence_length <= 30:  # Optimal sentence length
            score += 1.0
        elif 5 <= avg_sentence_length < 10 or 30 < avg_sentence_length <= 50:
            score += 0.7
        else:
            score += 0.3
    factors += 1
    
    # Paragraph structure factor
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    paragraph_count = len(paragraphs)
    if paragraph_count >= MIN_PARAGRAPH_COUNT:
        score += min(1.0, paragraph_count / 10)  # Normalize to 10 paragraphs = 1.0
    factors += 1
    
    # Content diversity factor (vocabulary richness)
    unique_words = set(word.lower() for word in words if word.isalpha())
    if word_count > 0:
        diversity_ratio = len(unique_words) / word_count
        score += min(1.0, diversity_ratio * 2)  # Normalize
    factors += 1
    
    # Coherence factor (check for proper capitalization and punctuation)
    coherence_score = 0.0
    if re.search(r'[A-Z]', text):  # Has uppercase letters
        coherence_score += 0.3
    if re.search(r'[.!?]', text):  # Has sentence endings
        coherence_score += 0.3
    if not re.search(r'[^\w\s.!?,:;()-]', text):  # No weird characters
        coherence_score += 0.4
    score += coherence_score
    factors += 1
    
    return score / factors if factors > 0 else 0.0


def is_content_suitable_for_llm(text: str, url: str) -> Tuple[bool, str, Dict]:
    """
    Determine if content is suitable for LLM training with detailed reasoning.
    Returns (is_suitable, reason, metrics)
    """
    if not text or not text.strip():
        return False, "Empty content", {}
    
    text = text.strip()
    
    # Calculate metrics
    char_count = len(text)
    word_count = len(text.split())
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    paragraph_count = len(paragraphs)
    quality_score = calculate_text_quality_score(text)
    
    metrics = {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "quality_score": quality_score,
        "avg_sentence_length": word_count / sentence_count if sentence_count > 0 else 0
    }
    
    # Check minimum requirements
    if char_count < MIN_TEXT_LENGTH:
        return False, f"Too short: {char_count} < {MIN_TEXT_LENGTH} chars", metrics
    
    if char_count > MAX_TEXT_LENGTH:
        return False, f"Too long: {char_count} > {MAX_TEXT_LENGTH} chars", metrics
    
    if word_count < MIN_WORD_COUNT:
        return False, f"Too few words: {word_count} < {MIN_WORD_COUNT}", metrics
    
    if sentence_count < MIN_SENTENCE_COUNT:
        return False, f"Too few sentences: {sentence_count} < {MIN_SENTENCE_COUNT}", metrics
    
    if paragraph_count < MIN_PARAGRAPH_COUNT:
        return False, f"Too few paragraphs: {paragraph_count} < {MIN_PARAGRAPH_COUNT}", metrics
    
    if quality_score < QUALITY_SCORE_THRESHOLD:
        return False, f"Low quality score: {quality_score:.2f} < {QUALITY_SCORE_THRESHOLD}", metrics
    
    # Check for common low-quality indicators
    text_lower = text.lower()
    
    # Skip if mostly navigation/menu content
    nav_indicators = ['home', 'about', 'contact', 'login', 'register', 'menu', 'navigation']
    nav_count = sum(1 for indicator in nav_indicators if indicator in text_lower)
    if nav_count > word_count * 0.1:  # More than 10% navigation words
        return False, "Appears to be navigation/menu content", metrics
    
    # Skip if mostly error pages
    error_indicators = ['404', 'not found', 'error', 'page not found', 'access denied']
    if any(indicator in text_lower for indicator in error_indicators) and word_count < 200:
        return False, "Appears to be error page", metrics
    
    # Skip if mostly form content
    form_indicators = ['submit', 'form', 'input', 'button', 'field', 'required']
    form_count = sum(1 for indicator in form_indicators if indicator in text_lower)
    if form_count > word_count * 0.15:  # More than 15% form words
        return False, "Appears to be form content", metrics
    
    return True, "Suitable for LLM training", metrics


async def save_record(url: str, html_bytes: bytes, extracted_text: str, quality_metrics: Dict):
    """
    Enhanced save function with comprehensive metadata for LLM training data.
    """
    parsed = urlparse(url)
    host = parsed.netloc.replace(":", "_")
    raw_host_dir = RAW_DIR / host
    text_host_dir = TEXT_DIR / host
    meta_host_dir = META_DIR / host
    for d in (raw_host_dir, text_host_dir, meta_host_dir):
        d.mkdir(parents=True, exist_ok=True)

    fname = safe_filename_from_url(url)

    raw_path = raw_host_dir / (fname + ".html")
    text_path = text_host_dir / (fname + ".txt")
    meta_path = meta_host_dir / (fname + ".json")

    # Write raw HTML
    async with aiofiles.open(raw_path, "wb") as f:
        await f.write(html_bytes)
    # Write text
    async with aiofiles.open(text_path, "w", encoding="utf-8") as f:
        await f.write(extracted_text)

    # Extract additional metadata from HTML
    try:
        soup = BeautifulSoup(html_bytes, 'html.parser')
        
        # Extract title
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else ""
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc.get('content', '').strip() if meta_desc else ""
        
        # Extract meta keywords
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        keywords = meta_keywords.get('content', '').strip() if meta_keywords else ""
        
        # Extract author information
        author_meta = soup.find('meta', attrs={'name': 'author'})
        author = author_meta.get('content', '').strip() if author_meta else ""
        
        # Extract publication date
        pub_date = ""
        date_selectors = [
            ('meta', {'name': 'article:published_time'}),
            ('meta', {'property': 'article:published_time'}),
            ('meta', {'name': 'pubdate'}),
            ('meta', {'name': 'date'}),
            ('time', {'datetime': True}),
        ]
        
        for selector, attrs in date_selectors:
            date_elem = soup.find(selector, attrs)
            if date_elem:
                if selector == 'meta':
                    pub_date = date_elem.get('content', '')
                else:
                    pub_date = date_elem.get('datetime', '') or date_elem.get_text()
                if pub_date:
                    break
        
        # Extract language
        lang = soup.get('lang', '') or soup.find('html', {'lang': True})
        if hasattr(lang, 'get'):
            lang = lang.get('lang', '')
        
        # Extract canonical URL
        canonical = soup.find('link', {'rel': 'canonical'})
        canonical_url = canonical.get('href', '') if canonical else ""
        
        # Count structural elements
        headings = {f'h{i}': len(soup.find_all(f'h{i}')) for i in range(1, 7)}
        lists = len(soup.find_all(['ul', 'ol']))
        tables = len(soup.find_all('table'))
        images = len(soup.find_all('img'))
        links = len(soup.find_all('a', href=True))
        
        # Extract main content indicators
        main_content_selectors = ['main', 'article', '[role="main"]', '.content', '#content']
        main_content_found = any(soup.select(selector) for selector in main_content_selectors)
        
    except Exception as e:
        logger.warning("Failed to extract additional metadata from %s: %s", url, e)
        title = description = keywords = author = pub_date = lang = canonical_url = ""
        headings = {f'h{i}': 0 for i in range(1, 7)}
        lists = tables = images = links = 0
        main_content_found = False

    # Enhanced metadata with comprehensive information
    meta = {
        # Basic information
        "url": url,
        "canonical_url": canonical_url,
        "domain": parsed.netloc,
        "path": parsed.path,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        
        # Content metadata
        "title": title,
        "description": description,
        "keywords": keywords,
        "author": author,
        "publication_date": pub_date,
        "language": lang,
        
        # File information
        "raw_file": str(raw_path),
        "text_file": str(text_path),
        "meta_file": str(meta_path),
        "raw_bytes": len(html_bytes),
        "text_chars": len(extracted_text),
        
        # Quality and training readiness
        "quality_metrics": quality_metrics,
        "llm_training_ready": True,  # Only saved if suitable for LLM training
        "extraction_method": "enhanced_trafilatura_with_structure",
        
        # Structural analysis
        "structure": {
            "headings": headings,
            "total_headings": sum(headings.values()),
            "lists": lists,
            "tables": tables,
            "images": images,
            "links": links,
            "has_main_content": main_content_found
        },
        
        # Content categorization hints
        "content_indicators": {
            "academic": any(term in (title + description + extracted_text).lower() 
                          for term in ['research', 'paper', 'study', 'journal', 'academic', 'university']),
            "technical": any(term in (title + description + extracted_text).lower() 
                           for term in ['documentation', 'api', 'tutorial', 'guide', 'manual']),
            "educational": any(term in (title + description + extracted_text).lower() 
                             for term in ['course', 'lecture', 'lesson', 'learn', 'education']),
            "scientific": any(term in (title + description + extracted_text).lower() 
                            for term in ['quantum', 'physics', 'science', 'theory', 'experiment'])
        },
        
        # Crawl information
        "crawl_depth": state["url_depths"].get(url, 0),
        "discovered_from": "seed" if url in state.get("queued_urls", [])[:len(state.get("queued_urls", []))] else "link_discovery"
    }
    
    async with aiofiles.open(meta_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(meta, indent=2, ensure_ascii=False))

    # update state with enhanced tracking
    h = sha256_bytes(extracted_text.encode("utf-8"))
    state["processed_hashes"][h] = {
        "url": url, 
        "raw_path": str(raw_path), 
        "text_path": str(text_path), 
        "meta_path": str(meta_path),
        "quality_score": quality_metrics.get("quality_score", 0.0),
        "timestamp": meta["timestamp"],
        "title": title,
        "domain": parsed.netloc
    }
    state["bytes_saved"] += os.path.getsize(raw_path) + os.path.getsize(text_path) + os.path.getsize(meta_path)
    save_state()


async def worker_for_host(session: aiohttp.ClientSession, host_queue: asyncio.Queue, robots: RobotsCache, host_sem: asyncio.Semaphore):
    """
    Enhanced worker with intelligent content filtering and quality validation for LLM training data.
    Each host has a queue of URLs. This worker drains it with limited concurrency.
    """
    processed_count = 0
    skipped_count = 0
    
    while not host_queue.empty() and state["bytes_saved"] < TARGET_BYTES:
        url = await host_queue.get()
        if state["bytes_saved"] >= TARGET_BYTES:
            host_queue.task_done()
            break

        # Enhanced URL filtering - now handle PDFs separately
        url_lower = url.lower()
        
        # Check if it's a PDF
        is_pdf = url_lower.endswith('.pdf')
        
        # Skip non-PDF binary files
        skip_extensions = ['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', 
                          '.zip', '.rar', '.tar', '.gz', '.jpg', '.jpeg', '.png', 
                          '.gif', '.svg', '.mp4', '.avi', '.mov', '.mp3', '.wav']
        
        if any(url_lower.endswith(ext) for ext in skip_extensions):
            logger.info("Skipping non-text content: %s", url)
            skipped_count += 1
            host_queue.task_done()
            continue

        # Skip URLs that are likely to be low-quality
        skip_patterns = ['/tag/', '/category/', '/archive/', '/search/', '/login/', 
                        '/register/', '/cart/', '/checkout/', '/admin/', '/wp-admin/']
        if any(pattern in url_lower for pattern in skip_patterns):
            logger.info("Skipping likely low-quality URL: %s", url)
            skipped_count += 1
            host_queue.task_done()
            continue

        # Check robots.txt
        allowed = await robots.allowed(session, url)
        if not allowed:
            logger.info("Blocked by robots.txt: %s", url)
            skipped_count += 1
            host_queue.task_done()
            continue

        # Fetch content
        html = await fetch_with_retries(session, url)
        if not html:
            skipped_count += 1
            host_queue.task_done()
            continue

        # Extract text based on content type
        if is_pdf:
            extracted = extract_text_from_pdf(html, url)
            if not extracted:
                logger.info("Failed to extract text from PDF: %s", url)
                skipped_count += 1
                host_queue.task_done()
                continue
        else:
            extracted = extract_text_from_html(html, url)
        
        # Enhanced content validation for LLM training
        is_suitable, reason, metrics = is_content_suitable_for_llm(extracted, url)
        
        if not is_suitable:
            logger.info("Skipping %s: %s (metrics: %s)", url, reason, 
                       {k: v for k, v in metrics.items() if k in ['char_count', 'word_count', 'quality_score']})
            skipped_count += 1
            host_queue.task_done()
            continue

        # Enhanced deduplication - check content similarity
        h = sha256_bytes(extracted.encode("utf-8"))
        if h in state["processed_hashes"]:
            existing_url = state["processed_hashes"][h]["url"]
            logger.info("Duplicate content detected: %s (original: %s)", url, existing_url)
            skipped_count += 1
            host_queue.task_done()
            continue

        # Additional similarity check for near-duplicates
        # Check if we have very similar content (first 500 chars)
        content_preview = extracted[:500].lower()
        is_near_duplicate = False
        
        for existing_hash, existing_data in state["processed_hashes"].items():
            if "content_preview" in existing_data:
                existing_preview = existing_data["content_preview"]
                # Simple similarity check - if 80% of words match, consider duplicate
                preview_words = set(content_preview.split())
                existing_words = set(existing_preview.split())
                if len(preview_words) > 0 and len(existing_words) > 0:
                    intersection = len(preview_words.intersection(existing_words))
                    similarity = intersection / max(len(preview_words), len(existing_words))
                    if similarity > 0.8:
                        logger.info("Near-duplicate content detected: %s (similar to: %s, similarity: %.2f)", 
                                  url, existing_data["url"], similarity)
                        is_near_duplicate = True
                        break
        
        if is_near_duplicate:
            skipped_count += 1
            host_queue.task_done()
            continue

        try:
            # Save the high-quality content
            await save_record(url, html, extracted, metrics)
            
            # Store content preview for similarity checking
            state["processed_hashes"][h]["content_preview"] = content_preview
            
            # Link discovery - extract and queue new URLs if enabled
            if ENABLE_LINK_DISCOVERY:
                current_depth = state["url_depths"].get(url, 0)
                if current_depth < MAX_DEPTH:
                    discovered_links = extract_links_from_html(html, url)
                    new_links_count = 0
                    
                    for link in discovered_links[:MAX_LINKS_PER_PAGE]:
                        if should_crawl_url(link, current_depth + 1):
                            # Add to discovered URLs to prevent re-discovery
                            if link not in state["discovered_urls"]:
                                state["discovered_urls"].add(link)
                                state["url_depths"][link] = current_depth + 1
                                state["queued_urls"].append(link)
                                new_links_count += 1
                    
                    if new_links_count > 0:
                        logger.info("Discovered %d new links from %s (depth %d)", 
                                   new_links_count, url, current_depth)
            
            processed_count += 1
            logger.info("✓ Saved high-quality content from %s (score: %.2f, %d chars, %d words) - Total: %d processed, %d skipped", 
                       url, metrics["quality_score"], metrics["char_count"], 
                       metrics["word_count"], processed_count, skipped_count)
            
            # Log progress every 10 successful saves
            if processed_count % 10 == 0:
                logger.info("Progress update - Processed: %d, Skipped: %d, Total bytes: %d", 
                           processed_count, skipped_count, state["bytes_saved"])
                
        except Exception as e:
            logger.exception("Error saving record for %s: %s", url, e)
            skipped_count += 1

        host_queue.task_done()
    
    logger.info("Worker completed - Final stats: %d processed, %d skipped", processed_count, skipped_count)


async def main():
    # get initial urls
    urls = await fetch_search_api_results()
    if not urls:
        logger.error("No URLs to process. Provide seed_urls.txt or implement fetch_search_api_results(). Exiting.")
        return

    # Initialize seed URLs with depth 0
    for url in urls:
        if url not in state["url_depths"]:
            state["url_depths"][url] = 0
        if url not in state["queued_urls"]:
            state["queued_urls"].append(url)

    logger.info("Starting with %d seed URLs, %d queued URLs total", len(urls), len(state["queued_urls"]))

    # Process URLs dynamically as they are discovered
    robots = RobotsCache()
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=REQUEST_TIMEOUT, sock_read=REQUEST_TIMEOUT)
    connector = aiohttp.TCPConnector(limit_per_host=CONCURRENT_FETCH_PER_HOST, ttl_dns_cache=300)
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=True) as session:
        # Create host-specific queues and semaphores
        host_queues: Dict[str, asyncio.Queue] = {}
        host_semaphores: Dict[str, asyncio.Semaphore] = {}
        active_tasks: Dict[str, asyncio.Task] = {}
        
        # Global semaphore to limit concurrent hosts
        global_sem = asyncio.Semaphore(CONCURRENT_HOSTS)
        
        async def process_host_queue(host: str, queue: asyncio.Queue):
            """Process a single host's queue with proper concurrency control"""
            async with global_sem:
                await worker_for_host(session, queue, robots, global_sem)
        
        # Main processing loop
        processed_urls = set()
        
        while state["bytes_saved"] < TARGET_BYTES and state["queued_urls"]:
            # Distribute queued URLs to host-specific queues
            current_batch = state["queued_urls"].copy()
            state["queued_urls"].clear()
            
            for url in current_batch:
                if url in processed_urls:
                    continue
                    
                processed_urls.add(url)
                parsed = urlparse(url)
                host = parsed.netloc
                
                # Create host queue and semaphore if not exists
                if host not in host_queues:
                    host_queues[host] = asyncio.Queue()
                    host_semaphores[host] = asyncio.Semaphore(CONCURRENT_FETCH_PER_HOST)
                
                # Add URL to host queue
                await host_queues[host].put(url)
                
                # Start host worker if not already running
                if host not in active_tasks or active_tasks[host].done():
                    active_tasks[host] = asyncio.create_task(
                        process_host_queue(host, host_queues[host])
                    )
            
            # Wait for some tasks to complete before checking for new URLs
            if active_tasks:
                # Wait for at least one task to complete or timeout
                try:
                    done, pending = await asyncio.wait(
                        active_tasks.values(), 
                        timeout=10.0,  # Check every 10 seconds
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Clean up completed tasks
                    for task in done:
                        for host, host_task in list(active_tasks.items()):
                            if host_task == task:
                                del active_tasks[host]
                                break
                                
                except asyncio.TimeoutError:
                    pass  # Continue to check for new URLs
            
            # Save state periodically
            save_state()
            
            # Log progress
            if len(processed_urls) % 50 == 0:
                logger.info("Progress: %d URLs processed, %d queued, %d bytes saved, %d active hosts", 
                           len(processed_urls), len(state["queued_urls"]), 
                           state["bytes_saved"], len(active_tasks))
        
        # Wait for remaining tasks to complete
        if active_tasks:
            logger.info("Waiting for %d remaining host tasks to complete...", len(active_tasks))
            await asyncio.gather(*active_tasks.values(), return_exceptions=True)

    logger.info("Scraping complete. Processed %d URLs, Saved bytes: %d", 
               len(processed_urls), state["bytes_saved"])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Saving state and exiting.")
        save_state()
