#!/usr/bin/env python3
"""
LLM Training Data Consolidator

This script consolidates scraped text data into formats optimized for LLM training:
1. Single consolidated text file with document separators
2. JSONL format for structured training
3. Metadata summary and quality reports
4. Deduplication and quality filtering
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime
import hashlib
import logging

# Configuration
OUTPUT_DIR = Path("output")
TEXT_DIR = OUTPUT_DIR / "text"
META_DIR = OUTPUT_DIR / "meta"
CONSOLIDATED_DIR = OUTPUT_DIR / "consolidated"
STATE_FILE = OUTPUT_DIR / "state.json"

# Consolidation settings
MIN_QUALITY_SCORE = 0.7  # Higher threshold for training data
MIN_TEXT_LENGTH = 1000   # Longer minimum for training
MAX_TEXT_LENGTH = 100000 # Maximum length per document
DOCUMENT_SEPARATOR = "\n\n" + "="*80 + "\n\n"
CHUNK_SIZE = 50 * 1024 * 1024  # 50MB chunks for large datasets

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("consolidator")

# Create output directories
CONSOLIDATED_DIR.mkdir(parents=True, exist_ok=True)


def load_metadata_files() -> List[Dict]:
    """Load all metadata files and return sorted by quality score."""
    metadata_files = []
    
    if not META_DIR.exists():
        logger.warning("Metadata directory not found: %s", META_DIR)
        return []
    
    for meta_file in META_DIR.rglob("*.json"):
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                metadata['meta_file_path'] = str(meta_file)
                metadata_files.append(metadata)
        except Exception as e:
            logger.warning("Failed to load metadata file %s: %s", meta_file, e)
    
    # Sort by quality score (highest first)
    metadata_files.sort(key=lambda x: x.get('quality_metrics', {}).get('quality_score', 0), reverse=True)
    
    logger.info("Loaded %d metadata files", len(metadata_files))
    return metadata_files


def load_text_content(text_file_path: str) -> Optional[str]:
    """Load text content from file."""
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.warning("Failed to load text file %s: %s", text_file_path, e)
        return None


def calculate_content_hash(content: str) -> str:
    """Calculate SHA-256 hash of content for deduplication."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def is_high_quality_content(metadata: Dict) -> Tuple[bool, str]:
    """Determine if content meets high-quality standards for LLM training."""
    quality_metrics = metadata.get('quality_metrics', {})
    
    # Check quality score
    quality_score = quality_metrics.get('quality_score', 0)
    if quality_score < MIN_QUALITY_SCORE:
        return False, f"Quality score {quality_score:.2f} below threshold {MIN_QUALITY_SCORE}"
    
    # Check text length
    char_count = quality_metrics.get('char_count', 0)
    if char_count < MIN_TEXT_LENGTH:
        return False, f"Text too short: {char_count} chars"
    
    if char_count > MAX_TEXT_LENGTH:
        return False, f"Text too long: {char_count} chars"
    
    # Check if marked as LLM training ready
    if not metadata.get('llm_training_ready', False):
        return False, "Not marked as LLM training ready"
    
    # Check for minimum structural requirements
    word_count = quality_metrics.get('word_count', 0)
    sentence_count = quality_metrics.get('sentence_count', 0)
    paragraph_count = quality_metrics.get('paragraph_count', 0)
    
    if word_count < 200:
        return False, f"Too few words: {word_count}"
    
    if sentence_count < 10:
        return False, f"Too few sentences: {sentence_count}"
    
    if paragraph_count < 3:
        return False, f"Too few paragraphs: {paragraph_count}"
    
    return True, "High quality content"


def clean_text_for_training(text: str) -> str:
    """Clean and normalize text for LLM training."""
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces and tabs
    
    # Remove common web artifacts
    text = re.sub(r'Cookie Policy|Privacy Policy|Terms of Service', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Subscribe to newsletter|Sign up for updates', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Share on social media|Follow us on', '', text, flags=re.IGNORECASE)
    
    # Remove navigation artifacts
    text = re.sub(r'Home\s*>\s*|Breadcrumb|Navigation', '', text, flags=re.IGNORECASE)
    
    # Clean up common formatting issues
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)  # Ensure space after sentence endings
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase words
    
    return text.strip()


def create_document_header(metadata: Dict) -> str:
    """Create a header for each document in the consolidated format."""
    url = metadata.get('url', 'Unknown')
    title = metadata.get('title', 'Untitled')
    timestamp = metadata.get('timestamp', 'Unknown')
    quality_score = metadata.get('quality_metrics', {}).get('quality_score', 0)
    
    header = f"""Document: {title}
Source: {url}
Extracted: {timestamp}
Quality Score: {quality_score:.3f}
Content Type: Academic/Scientific Text
"""
    return header


def consolidate_to_text_format(metadata_files: List[Dict]) -> Dict:
    """Consolidate all high-quality content into text format."""
    logger.info("Creating consolidated text format...")
    
    consolidated_stats = {
        'total_documents': 0,
        'high_quality_documents': 0,
        'total_characters': 0,
        'total_words': 0,
        'unique_domains': set(),
        'quality_scores': [],
        'rejected_reasons': {}
    }
    
    seen_hashes = set()
    current_chunk = 0
    current_chunk_size = 0
    current_chunk_content = []
    
    for metadata in metadata_files:
        consolidated_stats['total_documents'] += 1
        
        # Check quality
        is_quality, reason = is_high_quality_content(metadata)
        if not is_quality:
            consolidated_stats['rejected_reasons'][reason] = consolidated_stats['rejected_reasons'].get(reason, 0) + 1
            continue
        
        # Load text content
        text_file_path = metadata.get('text_file')
        if not text_file_path or not Path(text_file_path).exists():
            consolidated_stats['rejected_reasons']['Text file not found'] = consolidated_stats['rejected_reasons'].get('Text file not found', 0) + 1
            continue
        
        content = load_text_content(text_file_path)
        if not content:
            consolidated_stats['rejected_reasons']['Failed to load content'] = consolidated_stats['rejected_reasons'].get('Failed to load content', 0) + 1
            continue
        
        # Check for duplicates
        content_hash = calculate_content_hash(content)
        if content_hash in seen_hashes:
            consolidated_stats['rejected_reasons']['Duplicate content'] = consolidated_stats['rejected_reasons'].get('Duplicate content', 0) + 1
            continue
        
        seen_hashes.add(content_hash)
        
        # Clean content
        cleaned_content = clean_text_for_training(content)
        
        # Create document entry
        header = create_document_header(metadata)
        document_entry = f"{header}\n{'-'*60}\n{cleaned_content}"
        
        # Check if we need to start a new chunk
        entry_size = len(document_entry.encode('utf-8'))
        if current_chunk_size + entry_size > CHUNK_SIZE and current_chunk_content:
            # Save current chunk
            chunk_file = CONSOLIDATED_DIR / f"training_data_chunk_{current_chunk:03d}.txt"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(DOCUMENT_SEPARATOR.join(current_chunk_content))
            
            logger.info("Saved chunk %d with %d documents (%d MB)", 
                       current_chunk, len(current_chunk_content), current_chunk_size // (1024*1024))
            
            current_chunk += 1
            current_chunk_size = 0
            current_chunk_content = []
        
        # Add to current chunk
        current_chunk_content.append(document_entry)
        current_chunk_size += entry_size
        
        # Update stats
        consolidated_stats['high_quality_documents'] += 1
        consolidated_stats['total_characters'] += len(cleaned_content)
        consolidated_stats['total_words'] += len(cleaned_content.split())
        consolidated_stats['unique_domains'].add(metadata.get('domain', 'unknown'))
        consolidated_stats['quality_scores'].append(metadata.get('quality_metrics', {}).get('quality_score', 0))
    
    # Save final chunk
    if current_chunk_content:
        chunk_file = CONSOLIDATED_DIR / f"training_data_chunk_{current_chunk:03d}.txt"
        with open(chunk_file, 'w', encoding='utf-8') as f:
            f.write(DOCUMENT_SEPARATOR.join(current_chunk_content))
        
        logger.info("Saved final chunk %d with %d documents (%d MB)", 
                   current_chunk, len(current_chunk_content), current_chunk_size // (1024*1024))
    
    # Convert set to list for JSON serialization
    consolidated_stats['unique_domains'] = list(consolidated_stats['unique_domains'])
    
    return consolidated_stats


def consolidate_to_jsonl_format(metadata_files: List[Dict]) -> Dict:
    """Consolidate content into JSONL format for structured training."""
    logger.info("Creating JSONL format...")
    
    jsonl_file = CONSOLIDATED_DIR / "training_data.jsonl"
    jsonl_stats = {
        'total_records': 0,
        'total_characters': 0,
        'average_quality_score': 0,
        'domains': {}
    }
    
    seen_hashes = set()
    quality_scores = []
    
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for metadata in metadata_files:
            # Check quality
            is_quality, reason = is_high_quality_content(metadata)
            if not is_quality:
                continue
            
            # Load text content
            text_file_path = metadata.get('text_file')
            if not text_file_path or not Path(text_file_path).exists():
                continue
            
            content = load_text_content(text_file_path)
            if not content:
                continue
            
            # Check for duplicates
            content_hash = calculate_content_hash(content)
            if content_hash in seen_hashes:
                continue
            
            seen_hashes.add(content_hash)
            
            # Clean content
            cleaned_content = clean_text_for_training(content)
            
            # Create JSONL record
            record = {
                'text': cleaned_content,
                'url': metadata.get('url'),
                'title': metadata.get('title'),
                'domain': metadata.get('domain'),
                'timestamp': metadata.get('timestamp'),
                'quality_score': metadata.get('quality_metrics', {}).get('quality_score'),
                'word_count': len(cleaned_content.split()),
                'char_count': len(cleaned_content),
                'content_hash': content_hash
            }
            
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            # Update stats
            jsonl_stats['total_records'] += 1
            jsonl_stats['total_characters'] += len(cleaned_content)
            quality_scores.append(record['quality_score'])
            
            domain = record['domain']
            jsonl_stats['domains'][domain] = jsonl_stats['domains'].get(domain, 0) + 1
    
    if quality_scores:
        jsonl_stats['average_quality_score'] = sum(quality_scores) / len(quality_scores)
    
    return jsonl_stats


def generate_consolidation_report(text_stats: Dict, jsonl_stats: Dict) -> None:
    """Generate a comprehensive report of the consolidation process."""
    report = {
        'consolidation_timestamp': datetime.now().isoformat(),
        'text_format': text_stats,
        'jsonl_format': jsonl_stats,
        'summary': {
            'total_training_documents': text_stats['high_quality_documents'],
            'total_training_characters': text_stats['total_characters'],
            'total_training_words': text_stats['total_words'],
            'average_quality_score': sum(text_stats['quality_scores']) / len(text_stats['quality_scores']) if text_stats['quality_scores'] else 0,
            'unique_domains': len(text_stats['unique_domains']),
            'data_size_mb': text_stats['total_characters'] / (1024 * 1024),
            'rejection_summary': text_stats['rejected_reasons']
        }
    }
    
    # Save detailed report
    report_file = CONSOLIDATED_DIR / "consolidation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Create human-readable summary
    summary_file = CONSOLIDATED_DIR / "README.md"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"""# LLM Training Data Consolidation Report

Generated: {report['consolidation_timestamp']}

## Summary Statistics

- **Total Training Documents**: {report['summary']['total_training_documents']:,}
- **Total Characters**: {report['summary']['total_training_characters']:,}
- **Total Words**: {report['summary']['total_training_words']:,}
- **Data Size**: {report['summary']['data_size_mb']:.1f} MB
- **Average Quality Score**: {report['summary']['average_quality_score']:.3f}
- **Unique Domains**: {report['summary']['unique_domains']}

## Quality Filtering Results

""")
        
        for reason, count in report['summary']['rejection_summary'].items():
            f.write(f"- **{reason}**: {count:,} documents\n")
        
        f.write(f"""
## Output Files

### Text Format (for general training)
- `training_data_chunk_*.txt`: Consolidated text files with document separators
- Each chunk is approximately 50MB for manageable processing

### JSONL Format (for structured training)
- `training_data.jsonl`: One JSON record per line with metadata
- Suitable for frameworks that require structured input

### Domain Distribution

""")
        
        for domain in sorted(text_stats['unique_domains']):
            domain_count = jsonl_stats['domains'].get(domain, 0)
            f.write(f"- **{domain}**: {domain_count:,} documents\n")
        
        f.write("""
## Usage

### For Text-based Training
```python
# Load consolidated text chunks
for chunk_file in Path('consolidated').glob('training_data_chunk_*.txt'):
    with open(chunk_file, 'r', encoding='utf-8') as f:
        content = f.read()
        documents = content.split('=' * 80)
        # Process documents...
```

### For Structured Training
```python
import json

# Load JSONL format
with open('consolidated/training_data.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        record = json.loads(line)
        text = record['text']
        metadata = {k: v for k, v in record.items() if k != 'text'}
        # Process record...
```
""")
    
    logger.info("Consolidation report saved to %s", report_file)
    logger.info("Human-readable summary saved to %s", summary_file)


def main():
    """Main consolidation process."""
    logger.info("Starting LLM training data consolidation...")
    
    # Load all metadata
    metadata_files = load_metadata_files()
    if not metadata_files:
        logger.error("No metadata files found. Run the scraper first.")
        return
    
    # Create consolidated formats
    text_stats = consolidate_to_text_format(metadata_files)
    jsonl_stats = consolidate_to_jsonl_format(metadata_files)
    
    # Generate report
    generate_consolidation_report(text_stats, jsonl_stats)
    
    logger.info("Consolidation complete!")
    logger.info("High-quality documents: %d", text_stats['high_quality_documents'])
    logger.info("Total training data: %.1f MB", text_stats['total_characters'] / (1024 * 1024))
    logger.info("Average quality score: %.3f", sum(text_stats['quality_scores']) / len(text_stats['quality_scores']) if text_stats['quality_scores'] else 0)


if __name__ == "__main__":
    main()