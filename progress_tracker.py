#!/usr/bin/env python3
"""
Progress Tracker for LLM Training Data Collection

This script monitors the progress of data collection toward the 1GB target,
provides quality metrics, and generates progress reports.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import logging

# Configuration
OUTPUT_DIR = Path("output")
STATE_FILE = OUTPUT_DIR / "state.json"
META_DIR = OUTPUT_DIR / "meta"
PROGRESS_DIR = OUTPUT_DIR / "progress"
TARGET_BYTES = 1 * 1024**3  # 1 GiB target

# Create progress directory
PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("progress_tracker")


def load_current_state() -> Dict:
    """Load the current scraping state."""
    if not STATE_FILE.exists():
        return {}
    
    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load state file: %s", e)
        return {}


def analyze_metadata_files() -> Dict:
    """Analyze all metadata files to get comprehensive statistics."""
    if not META_DIR.exists():
        return {
            'total_documents': 0,
            'total_text_bytes': 0,
            'total_raw_bytes': 0,
            'quality_distribution': {},
            'domain_distribution': {},
            'content_type_distribution': {},
            'language_distribution': {},
            'crawl_depth_distribution': {},
            'average_quality_score': 0,
            'high_quality_documents': 0,
            'documents_by_hour': {},
            'processing_rate': 0
        }
    
    stats = {
        'total_documents': 0,
        'total_text_bytes': 0,
        'total_raw_bytes': 0,
        'quality_scores': [],
        'domains': {},
        'content_types': {
            'academic': 0,
            'technical': 0,
            'educational': 0,
            'scientific': 0,
            'other': 0
        },
        'languages': {},
        'crawl_depths': {},
        'timestamps': [],
        'high_quality_count': 0
    }
    
    for meta_file in META_DIR.rglob("*.json"):
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            stats['total_documents'] += 1
            stats['total_text_bytes'] += metadata.get('text_chars', 0)
            stats['total_raw_bytes'] += metadata.get('raw_bytes', 0)
            
            # Quality metrics
            quality_score = metadata.get('quality_metrics', {}).get('quality_score', 0)
            stats['quality_scores'].append(quality_score)
            if quality_score >= 0.7:
                stats['high_quality_count'] += 1
            
            # Domain distribution
            domain = metadata.get('domain', 'unknown')
            stats['domains'][domain] = stats['domains'].get(domain, 0) + 1
            
            # Content type analysis
            content_indicators = metadata.get('content_indicators', {})
            content_type = 'other'
            for ctype in ['academic', 'technical', 'educational', 'scientific']:
                if content_indicators.get(ctype, False):
                    content_type = ctype
                    break
            stats['content_types'][content_type] += 1
            
            # Language distribution
            lang = metadata.get('language', 'unknown')
            if lang:
                stats['languages'][lang] = stats['languages'].get(lang, 0) + 1
            
            # Crawl depth
            depth = metadata.get('crawl_depth', 0)
            stats['crawl_depths'][depth] = stats['crawl_depths'].get(depth, 0) + 1
            
            # Timestamp for rate analysis
            timestamp = metadata.get('timestamp', '')
            if timestamp:
                stats['timestamps'].append(timestamp)
                
        except Exception as e:
            logger.warning("Failed to process metadata file %s: %s", meta_file, e)
    
    # Calculate derived statistics
    result = {
        'total_documents': stats['total_documents'],
        'total_text_bytes': stats['total_text_bytes'],
        'total_raw_bytes': stats['total_raw_bytes'],
        'quality_distribution': _calculate_quality_distribution(stats['quality_scores']),
        'domain_distribution': dict(sorted(stats['domains'].items(), key=lambda x: x[1], reverse=True)),
        'content_type_distribution': stats['content_types'],
        'language_distribution': dict(sorted(stats['languages'].items(), key=lambda x: x[1], reverse=True)),
        'crawl_depth_distribution': stats['crawl_depths'],
        'average_quality_score': sum(stats['quality_scores']) / len(stats['quality_scores']) if stats['quality_scores'] else 0,
        'high_quality_documents': stats['high_quality_count'],
        'documents_by_hour': _calculate_hourly_distribution(stats['timestamps']),
        'processing_rate': _calculate_processing_rate(stats['timestamps'])
    }
    
    return result


def _calculate_quality_distribution(quality_scores: List[float]) -> Dict:
    """Calculate distribution of quality scores."""
    if not quality_scores:
        return {}
    
    ranges = {
        '0.9-1.0': 0,
        '0.8-0.9': 0,
        '0.7-0.8': 0,
        '0.6-0.7': 0,
        '0.5-0.6': 0,
        '0.0-0.5': 0
    }
    
    for score in quality_scores:
        if score >= 0.9:
            ranges['0.9-1.0'] += 1
        elif score >= 0.8:
            ranges['0.8-0.9'] += 1
        elif score >= 0.7:
            ranges['0.7-0.8'] += 1
        elif score >= 0.6:
            ranges['0.6-0.7'] += 1
        elif score >= 0.5:
            ranges['0.5-0.6'] += 1
        else:
            ranges['0.0-0.5'] += 1
    
    return ranges


def _calculate_hourly_distribution(timestamps: List[str]) -> Dict:
    """Calculate documents processed per hour."""
    hourly_counts = {}
    
    for timestamp_str in timestamps:
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            hour_key = timestamp.strftime('%Y-%m-%d %H:00')
            hourly_counts[hour_key] = hourly_counts.get(hour_key, 0) + 1
        except Exception:
            continue
    
    return dict(sorted(hourly_counts.items()))


def _calculate_processing_rate(timestamps: List[str]) -> float:
    """Calculate average processing rate (documents per hour)."""
    if len(timestamps) < 2:
        return 0
    
    try:
        sorted_timestamps = sorted([
            datetime.fromisoformat(ts.replace('Z', '+00:00')) 
            for ts in timestamps
        ])
        
        time_span = sorted_timestamps[-1] - sorted_timestamps[0]
        hours = time_span.total_seconds() / 3600
        
        if hours > 0:
            return len(timestamps) / hours
        else:
            return 0
    except Exception:
        return 0


def calculate_progress_metrics(stats: Dict, state: Dict) -> Dict:
    """Calculate progress toward targets and efficiency metrics."""
    total_bytes = stats['total_text_bytes']
    progress_percentage = (total_bytes / TARGET_BYTES) * 100
    
    # Estimate time to completion
    processing_rate = stats['processing_rate']
    if processing_rate > 0 and total_bytes < TARGET_BYTES:
        remaining_bytes = TARGET_BYTES - total_bytes
        avg_bytes_per_doc = total_bytes / stats['total_documents'] if stats['total_documents'] > 0 else 1000
        remaining_docs = remaining_bytes / avg_bytes_per_doc
        hours_to_completion = remaining_docs / processing_rate
    else:
        hours_to_completion = 0
    
    # Quality efficiency
    high_quality_ratio = stats['high_quality_documents'] / stats['total_documents'] if stats['total_documents'] > 0 else 0
    
    # Storage efficiency
    compression_ratio = stats['total_text_bytes'] / stats['total_raw_bytes'] if stats['total_raw_bytes'] > 0 else 0
    
    return {
        'target_bytes': TARGET_BYTES,
        'current_bytes': total_bytes,
        'progress_percentage': progress_percentage,
        'bytes_remaining': max(0, TARGET_BYTES - total_bytes),
        'estimated_hours_to_completion': hours_to_completion,
        'high_quality_ratio': high_quality_ratio,
        'compression_ratio': compression_ratio,
        'average_document_size': total_bytes / stats['total_documents'] if stats['total_documents'] > 0 else 0,
        'target_achieved': total_bytes >= TARGET_BYTES
    }


def generate_progress_report() -> Dict:
    """Generate a comprehensive progress report."""
    logger.info("Generating progress report...")
    
    state = load_current_state()
    stats = analyze_metadata_files()
    progress = calculate_progress_metrics(stats, state)
    
    report = {
        'report_timestamp': datetime.now().isoformat(),
        'scraping_state': {
            'processed_hashes': len(state.get('processed_hashes', {})),
            'queued_urls': len(state.get('queued_urls', [])),
            'discovered_urls': len(state.get('discovered_urls', [])),
            'bytes_saved_total': state.get('bytes_saved', 0),
            'last_run': state.get('last_run')
        },
        'content_statistics': stats,
        'progress_metrics': progress,
        'recommendations': _generate_recommendations(stats, progress)
    }
    
    return report


def _generate_recommendations(stats: Dict, progress: Dict) -> List[str]:
    """Generate recommendations based on current progress and statistics."""
    recommendations = []
    
    # Progress recommendations
    if progress['progress_percentage'] < 10:
        recommendations.append("Consider adding more seed URLs or enabling deeper crawling to increase data collection rate")
    elif progress['progress_percentage'] < 50:
        recommendations.append("Good progress! Consider optimizing quality filters if too many documents are being rejected")
    
    # Quality recommendations
    if progress['high_quality_ratio'] < 0.5:
        recommendations.append("Low high-quality document ratio. Review quality thresholds or target more academic/scientific sources")
    elif progress['high_quality_ratio'] > 0.9:
        recommendations.append("Excellent quality ratio! Consider slightly relaxing filters to increase volume if needed")
    
    # Domain diversity recommendations
    domain_count = len(stats['domain_distribution'])
    if domain_count < 5:
        recommendations.append("Limited domain diversity. Consider adding seed URLs from more varied sources")
    elif domain_count > 50:
        recommendations.append("High domain diversity achieved - good for training data variety")
    
    # Content type recommendations
    content_types = stats['content_type_distribution']
    scientific_ratio = (content_types.get('scientific', 0) + content_types.get('academic', 0)) / stats['total_documents'] if stats['total_documents'] > 0 else 0
    if scientific_ratio < 0.3:
        recommendations.append("Consider targeting more scientific and academic sources for specialized LLM training")
    
    # Processing rate recommendations
    if stats['processing_rate'] < 10:
        recommendations.append("Low processing rate. Consider increasing concurrency or optimizing network settings")
    elif stats['processing_rate'] > 100:
        recommendations.append("High processing rate achieved - monitor for quality to ensure not overwhelming target sites")
    
    return recommendations


def save_progress_report(report: Dict) -> None:
    """Save the progress report to files."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed JSON report
    json_file = PROGRESS_DIR / f"progress_report_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Save latest report (overwrite)
    latest_file = PROGRESS_DIR / "latest_progress.json"
    with open(latest_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Create human-readable summary
    summary_file = PROGRESS_DIR / f"progress_summary_{timestamp}.md"
    with open(summary_file, 'w', encoding='utf-8') as f:
        _write_progress_summary(f, report)
    
    logger.info("Progress report saved to %s", json_file)
    logger.info("Progress summary saved to %s", summary_file)


def _write_progress_summary(f, report: Dict) -> None:
    """Write a human-readable progress summary."""
    stats = report['content_statistics']
    progress = report['progress_metrics']
    state = report['scraping_state']
    
    f.write(f"""# LLM Training Data Collection Progress Report

Generated: {report['report_timestamp']}

## 🎯 Progress Overview

- **Target**: {progress['target_bytes'] / (1024**3):.1f} GB
- **Current**: {progress['current_bytes'] / (1024**3):.3f} GB
- **Progress**: {progress['progress_percentage']:.1f}%
- **Remaining**: {progress['bytes_remaining'] / (1024**3):.3f} GB

{'✅ **TARGET ACHIEVED!**' if progress['target_achieved'] else f"⏱️ **Estimated completion**: {progress['estimated_hours_to_completion']:.1f} hours"}

## 📊 Content Statistics

- **Total Documents**: {stats['total_documents']:,}
- **High Quality Documents**: {stats['high_quality_documents']:,} ({progress['high_quality_ratio']:.1%})
- **Average Quality Score**: {stats['average_quality_score']:.3f}
- **Average Document Size**: {progress['average_document_size']:,.0f} characters
- **Processing Rate**: {stats['processing_rate']:.1f} documents/hour

## 🌐 Domain Distribution

""")
    
    for domain, count in list(stats['domain_distribution'].items())[:10]:
        f.write(f"- **{domain}**: {count:,} documents\n")
    
    f.write(f"""
## 📚 Content Types

- **Academic**: {stats['content_type_distribution']['academic']:,}
- **Scientific**: {stats['content_type_distribution']['scientific']:,}
- **Technical**: {stats['content_type_distribution']['technical']:,}
- **Educational**: {stats['content_type_distribution']['educational']:,}
- **Other**: {stats['content_type_distribution']['other']:,}

## 🔍 Quality Distribution

""")
    
    for quality_range, count in stats['quality_distribution'].items():
        f.write(f"- **{quality_range}**: {count:,} documents\n")
    
    f.write(f"""
## 🚀 System Status

- **Processed Hashes**: {state['processed_hashes']:,}
- **Queued URLs**: {state['queued_urls']:,}
- **Discovered URLs**: {state['discovered_urls']:,}
- **Total Storage Used**: {state['bytes_saved_total'] / (1024**2):.1f} MB

## 💡 Recommendations

""")
    
    for recommendation in report['recommendations']:
        f.write(f"- {recommendation}\n")


def main():
    """Generate and save progress report."""
    try:
        report = generate_progress_report()
        save_progress_report(report)
        
        # Print summary to console
        progress = report['progress_metrics']
        stats = report['content_statistics']
        
        print(f"\n🎯 Progress: {progress['progress_percentage']:.1f}% ({progress['current_bytes'] / (1024**3):.3f} GB / {progress['target_bytes'] / (1024**3):.1f} GB)")
        print(f"📊 Documents: {stats['total_documents']:,} total, {stats['high_quality_documents']:,} high quality")
        print(f"⚡ Rate: {stats['processing_rate']:.1f} docs/hour")
        print(f"🏆 Quality: {stats['average_quality_score']:.3f} average score")
        
        if progress['target_achieved']:
            print("✅ Target achieved!")
        else:
            print(f"⏱️ ETA: {progress['estimated_hours_to_completion']:.1f} hours")
        
    except Exception as e:
        logger.error("Failed to generate progress report: %s", e)
        raise


if __name__ == "__main__":
    main()