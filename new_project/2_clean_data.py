"""
PHASE 1: DATA CLEANING & PREPARATION
=====================================
This script cleans the 36 scraped pages, removing:
- Navigation menus
- Footers
- Duplicate content
- Advertisements
- Scripts and styles
- Empty sections
"""

import json
import re
import os
from datetime import datetime
from collections import Counter
import hashlib

class DataCleaner:
    def __init__(self, input_file):
        """Initialize the cleaner with input JSON file"""
        self.input_file = input_file
        self.raw_data = []
        self.cleaned_data = []
        self.stats = {
            'total_pages': 0,
            'cleaned_pages': 0,
            'total_chars_before': 0,
            'total_chars_after': 0,
            'duplicates_removed': 0,
            'garbage_patterns_found': Counter()
        }
        
    def load_data(self):
        """Load the scraped data"""
        print(f"📂 Loading data from: {self.input_file}")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        self.stats['total_pages'] = len(self.raw_data)
        print(f"✅ Loaded {self.stats['total_pages']} pages\n")
        
    def identify_garbage_patterns(self, text):
        """Identify common garbage patterns in text"""
        patterns = {
            'navigation': r'(Home|About|Services|Contact|Blog|Menu|Navigation)\s*\|',
            'footer': r'(Copyright|©|All Rights Reserved|Privacy Policy|Terms)',
            'social_media': r'(Facebook|Twitter|LinkedIn|Instagram|Follow us)',
            'cookie_notice': r'(cookie|Cookie Policy|We use cookies)',
            'script_tags': r'<script.*?</script>',
            'style_tags': r'<style.*?</style>',
            'html_comments': r'<!--.*?-->',
            'excessive_whitespace': r'\s{3,}',
            'email_pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone_pattern': r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}',
        }
        
        found_patterns = []
        for pattern_name, pattern_regex in patterns.items():
            if re.search(pattern_regex, text, re.IGNORECASE | re.DOTALL):
                found_patterns.append(pattern_name)
                self.stats['garbage_patterns_found'][pattern_name] += 1
        
        return found_patterns
    
    def clean_text(self, text):
        """Clean individual text content"""
        if not text:
            return ""
        
        original_length = len(text)
        
        # Remove HTML tags
        text = re.sub(r'<script.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<.*?>', '', text)
        
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        text = re.sub(r'&#\d+;', ' ', text)
        
        # Remove navigation-like patterns
        text = re.sub(r'(Home|About|Services|Contact|Blog|Menu)\s*\|\s*', '', text, flags=re.IGNORECASE)
        
        # Remove social media buttons/text
        text = re.sub(r'(Share on|Follow us on)\s+(Facebook|Twitter|LinkedIn|Instagram|YouTube)', '', text, flags=re.IGNORECASE)
        
        # Remove cookie notices
        text = re.sub(r'(This website uses cookies|We use cookies|Cookie Policy|Accept cookies).*?(\.|$)', '', text, flags=re.IGNORECASE)
        
        # Remove copyright notices
        text = re.sub(r'Copyright\s*©?\s*\d{4}.*?(\.|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'©\s*\d{4}.*?(\.|$)', '', text)
        text = re.sub(r'All Rights Reserved.*?(\.|$)', '', text, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\t+', ' ', text)
        
        # Remove lines with only special characters
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if len(re.sub(r'[^a-zA-Z0-9]', '', line)) > 3]
        text = '\n'.join(cleaned_lines)
        
        # Remove repeated characters (e.g., "====", "----")
        text = re.sub(r'([=\-_*#])\1{4,}', '', text)
        
        # Clean up spacing
        text = text.strip()
        
        return text
    
    def is_duplicate_content(self, text, seen_hashes):
        """Check if content is duplicate using hash"""
        # Create hash of normalized text (ignore whitespace differences)
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        content_hash = hashlib.md5(normalized.encode()).hexdigest()
        
        if content_hash in seen_hashes:
            return True
        
        seen_hashes.add(content_hash)
        return False
    
    def is_meaningful_content(self, text):
        """Check if content is meaningful (not just garbage)"""
        # Must have minimum length
        if len(text) < 100:
            return False
        
        # Must have reasonable word count
        words = text.split()
        if len(words) < 20:
            return False
        
        # Must have reasonable sentence structure
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) < 3:
            return False
        
        # Check for minimum alphabetic character ratio
        alpha_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(text)
        if total_chars > 0 and alpha_chars / total_chars < 0.6:
            return False
        
        return True
    
    def clean_all_pages(self):
        """Clean all scraped pages"""
        print("🧹 Starting data cleaning process...")
        print("=" * 60)
        
        seen_hashes = set()
        
        for i, page in enumerate(self.raw_data):
            print(f"\nProcessing page {i+1}/{len(self.raw_data)}...")
            
            # Extract data
            markdown = page.get('markdown', '')
            metadata = page.get('metadata', {})
            url = metadata.get('sourceURL', 'unknown')
            title = metadata.get('title', 'No Title')
            
            print(f"  URL: {url}")
            print(f"  Title: {title}")
            
            # Track original stats
            original_length = len(markdown)
            self.stats['total_chars_before'] += original_length
            
            # Identify garbage patterns (for reporting)
            garbage_found = self.identify_garbage_patterns(markdown)
            if garbage_found:
                print(f"  ⚠️  Found: {', '.join(garbage_found)}")
            
            # Clean the text
            cleaned_text = self.clean_text(markdown)
            
            # Check if duplicate
            if self.is_duplicate_content(cleaned_text, seen_hashes):
                print(f"  ❌ DUPLICATE - Skipping")
                self.stats['duplicates_removed'] += 1
                continue
            
            # Check if meaningful
            if not self.is_meaningful_content(cleaned_text):
                print(f"  ❌ NOT MEANINGFUL - Skipping (too short or low quality)")
                continue
            
            # Calculate cleaning stats
            cleaned_length = len(cleaned_text)
            reduction = ((original_length - cleaned_length) / original_length * 100) if original_length > 0 else 0
            
            print(f"  ✅ Cleaned: {original_length:,} → {cleaned_length:,} chars ({reduction:.1f}% reduction)")
            
            self.stats['total_chars_after'] += cleaned_length
            
            # Save cleaned data
            self.cleaned_data.append({
                'page_id': i + 1,
                'url': url,
                'title': title,
                'content': cleaned_text,
                'word_count': len(cleaned_text.split()),
                'char_count': cleaned_length,
                'metadata': metadata
            })
            
            self.stats['cleaned_pages'] += 1
        
        print("\n" + "=" * 60)
        print("✅ Cleaning complete!")
    
    def generate_report(self):
        """Generate cleaning report"""
        print("\n" + "=" * 60)
        print("📊 CLEANING REPORT")
        print("=" * 60)
        
        print(f"\n📄 Pages:")
        print(f"  Total scraped: {self.stats['total_pages']}")
        print(f"  Duplicates removed: {self.stats['duplicates_removed']}")
        print(f"  Low quality removed: {self.stats['total_pages'] - self.stats['cleaned_pages'] - self.stats['duplicates_removed']}")
        print(f"  Final cleaned pages: {self.stats['cleaned_pages']}")
        
        print(f"\n📝 Content:")
        print(f"  Before cleaning: {self.stats['total_chars_before']:,} characters")
        print(f"  After cleaning: {self.stats['total_chars_after']:,} characters")
        reduction = ((self.stats['total_chars_before'] - self.stats['total_chars_after']) / 
                    self.stats['total_chars_before'] * 100) if self.stats['total_chars_before'] > 0 else 0
        print(f"  Reduction: {reduction:.1f}%")
        
        if self.stats['garbage_patterns_found']:
            print(f"\n🗑️  Garbage patterns found:")
            for pattern, count in self.stats['garbage_patterns_found'].most_common():
                print(f"  - {pattern}: {count} occurrences")
        
        # Word count statistics
        if self.cleaned_data:
            word_counts = [page['word_count'] for page in self.cleaned_data]
            avg_words = sum(word_counts) / len(word_counts)
            min_words = min(word_counts)
            max_words = max(word_counts)
            
            print(f"\n📊 Word Count Statistics:")
            print(f"  Average: {avg_words:.0f} words/page")
            print(f"  Min: {min_words} words")
            print(f"  Max: {max_words} words")
    
    def save_cleaned_data(self, output_dir='processed_data'):
        """Save cleaned data to files"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON
        json_file = os.path.join(output_dir, f'clean_data_{timestamp}.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.cleaned_data, f, indent=2, ensure_ascii=False)
        
        # Save report
        report_file = os.path.join(output_dir, f'cleaning_report_{timestamp}.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("DATA CLEANING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {self.input_file}\n\n")
            
            f.write(f"Pages:\n")
            f.write(f"  Total scraped: {self.stats['total_pages']}\n")
            f.write(f"  Duplicates removed: {self.stats['duplicates_removed']}\n")
            f.write(f"  Final cleaned pages: {self.stats['cleaned_pages']}\n\n")
            
            f.write(f"Content:\n")
            f.write(f"  Before: {self.stats['total_chars_before']:,} chars\n")
            f.write(f"  After: {self.stats['total_chars_after']:,} chars\n")
            
            f.write(f"\nCleaned Pages:\n")
            f.write("-" * 60 + "\n")
            for page in self.cleaned_data:
                f.write(f"\n{page['page_id']}. {page['title']}\n")
                f.write(f"   URL: {page['url']}\n")
                f.write(f"   Words: {page['word_count']}\n")
        
        # Save individual markdown files
        pages_dir = os.path.join(output_dir, 'individual_pages')
        if not os.path.exists(pages_dir):
            os.makedirs(pages_dir)
        
        for page in self.cleaned_data:
            page_file = os.path.join(pages_dir, f"page_{page['page_id']:03d}.md")
            with open(page_file, 'w', encoding='utf-8') as f:
                f.write(f"# {page['title']}\n\n")
                f.write(f"**URL:** {page['url']}\n\n")
                f.write(f"**Words:** {page['word_count']}\n\n")
                f.write("-" * 60 + "\n\n")
                f.write(page['content'])
        
        print(f"\n💾 Saved files:")
        print(f"  ✅ JSON: {json_file}")
        print(f"  ✅ Report: {report_file}")
        print(f"  ✅ Individual pages: {pages_dir}/ ({len(self.cleaned_data)} files)")
        
        return json_file


def main():
    """Main execution"""
    print("\n" + "=" * 60)
    print("🚀 PHASE 1: DATA CLEANING & PREPARATION")
    print("=" * 60 + "\n")
    
    # Find the most recent scraped data
    input_dir = 'data'
    
    if not os.path.exists(input_dir):
        print(f"❌ Error: '{input_dir}' directory not found!")
        print(f"💡 Please run 'firecrawl_save_data.py' first to scrape the website.")
        return
    
    # Find latest JSON file
    json_files = [f for f in os.listdir(input_dir) 
                  if f.startswith('complete_data') and f.endswith('.json')]
    
    if not json_files:
        print(f"❌ Error: No data files found in '{input_dir}'!")
        print(f"💡 Please run 'firecrawl_save_data.py' first.")
        return
    
    json_files.sort(reverse=True)
    input_file = os.path.join(input_dir, json_files[0])
    
    print(f"📂 Using input file: {input_file}\n")
    
    # Initialize cleaner
    cleaner = DataCleaner(input_file)
    
    # Execute cleaning pipeline
    cleaner.load_data()
    cleaner.clean_all_pages()
    cleaner.generate_report()
    output_file = cleaner.save_cleaned_data()
    
    print("\n" + "=" * 60)
    print("✅ PHASE 1 COMPLETE!")
    print("=" * 60)
    print(f"\n📁 Cleaned data saved to: processed_data/")
    print(f"\n🎯 Next: Run Phase 2 to create vector database")
    print("=" * 60 + "\n")
    
    return output_file


if __name__ == "__main__":
    main()