import json
import os
from datetime import datetime

"""
This script shows you how to LOAD and USE the saved crawl data
Run this AFTER you've scraped and saved the data
"""

# ============================================================
# LOAD DATA FROM JSON FILE
# ============================================================

def load_saved_data(json_file):
    """Load the saved JSON data"""
    print(f"📂 Loading data from: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} pages\n")
    return data

# ============================================================
# ANALYZE THE DATA
# ============================================================

def analyze_data(data):
    """Analyze the scraped data"""
    print(f"{'='*60}")
    print(f"📊 DATA ANALYSIS")
    print(f"{'='*60}\n")
    
    total_pages = len(data)
    total_words = sum(len(page.get('markdown', '').split()) for page in data)
    
    print(f"📄 Total Pages: {total_pages}")
    print(f"📝 Total Words: {total_words:,}")
    print(f"📊 Average Words per Page: {total_words // total_pages if total_pages > 0 else 0:,}")
    
    # Show all page titles and URLs
    print(f"\n{'='*60}")
    print(f"📋 ALL PAGES:")
    print(f"{'='*60}\n")
    
    for i, page in enumerate(data):
        metadata = page.get('metadata', {})
        title = metadata.get('title', 'No Title')
        url = metadata.get('sourceURL', 'No URL')
        word_count = len(page.get('markdown', '').split())
        
        print(f"{i+1:3}. {title[:50]:50} | {word_count:5} words")
        print(f"     URL: {url}")
        print()

# ============================================================
# SEARCH WITHIN THE DATA
# ============================================================

def search_in_pages(data, search_term):
    """Search for a term in all pages"""
    print(f"\n{'='*60}")
    print(f"🔍 SEARCHING FOR: '{search_term}'")
    print(f"{'='*60}\n")
    
    results = []
    
    for i, page in enumerate(data):
        markdown = page.get('markdown', '').lower()
        if search_term.lower() in markdown:
            results.append((i, page))
    
    if results:
        print(f"✅ Found in {len(results)} pages:\n")
        for i, page in results:
            title = page.get('metadata', {}).get('title', 'No Title')
            url = page.get('metadata', {}).get('sourceURL', 'No URL')
            print(f"📄 Page {i+1}: {title}")
            print(f"   URL: {url}\n")
    else:
        print(f"❌ No results found for '{search_term}'")

# ============================================================
# EXPORT SPECIFIC PAGES
# ============================================================

def export_page(data, page_number, output_file='exported_page.md'):
    """Export a specific page to a file"""
    if page_number < 1 or page_number > len(data):
        print(f"❌ Invalid page number. Must be between 1 and {len(data)}")
        return
    
    page = data[page_number - 1]
    metadata = page.get('metadata', {})
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# {metadata.get('title', 'No Title')}\n\n")
        f.write(f"**URL:** {metadata.get('sourceURL', 'No URL')}\n")
        f.write(f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("-"*80 + "\n\n")
        f.write(page.get('markdown', ''))
    
    print(f"✅ Page {page_number} exported to: {output_file}")

# ============================================================
# COMBINE SPECIFIC PAGES
# ============================================================

def combine_pages(data, page_numbers, output_file='combined_pages.md'):
    """Combine specific pages into one file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Combined Pages\n")
        f.write(f"**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("="*80 + "\n\n")
        
        for page_num in page_numbers:
            if page_num < 1 or page_num > len(data):
                continue
            
            page = data[page_num - 1]
            metadata = page.get('metadata', {})
            
            f.write(f"\n\n## Page {page_num}: {metadata.get('title', 'No Title')}\n")
            f.write(f"**URL:** {metadata.get('sourceURL', 'No URL')}\n\n")
            f.write("-"*80 + "\n\n")
            f.write(page.get('markdown', ''))
            f.write("\n\n" + "="*80 + "\n")
    
    print(f"✅ Combined {len(page_numbers)} pages into: {output_file}")

# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔄 LOADING SAVED CRAWL DATA")
    print("="*60 + "\n")
    
    # STEP 1: Find your JSON file
    # Look in the 'crawl_output' folder for a file like 'complete_data_YYYYMMDD_HHMMSS.json'
    
    # Option A: Specify the exact file
    json_file = 'crawl_output/complete_data_20250109_123456.json'  # UPDATE THIS
    
    # Option B: Find the most recent JSON file automatically
    output_dir = 'crawl_output'
    if os.path.exists(output_dir):
        json_files = [f for f in os.listdir(output_dir) if f.startswith('complete_data') and f.endswith('.json')]
        if json_files:
            json_files.sort(reverse=True)  # Get most recent
            json_file = os.path.join(output_dir, json_files[0])
            print(f"📁 Found latest file: {json_file}\n")
        else:
            print("❌ No JSON files found in 'crawl_output' folder")
            print("💡 Run 'firecrawl_save_data.py' first to scrape and save data")
            exit()
    else:
        print("❌ 'crawl_output' folder not found")
        print("💡 Run 'firecrawl_save_data.py' first to scrape and save data")
        exit()
    
    # STEP 2: Load the data
    try:
        data = load_saved_data(json_file)
        
        # STEP 3: Analyze the data
        analyze_data(data)
        
        # STEP 4: Search for something (example)
        print("\n" + "="*60)
        search_term = input("🔍 Enter search term (or press Enter to skip): ").strip()
        if search_term:
            search_in_pages(data, search_term)
        
        # STEP 5: Export specific page (example)
        print("\n" + "="*60)
        page_num = input(f"📄 Enter page number to export (1-{len(data)}) or press Enter to skip: ").strip()
        if page_num.isdigit():
            export_page(data, int(page_num), f'page_{page_num}.md')
        
        # STEP 6: Access data programmatically (examples)
        print("\n" + "="*60)
        print("💻 PROGRAMMATIC ACCESS EXAMPLES:")
        print("="*60 + "\n")
        
        print("# Get first page title:")
        print(f">>> data[0]['metadata']['title']")
        print(f"'{data[0]['metadata']['title']}'")
        
        print("\n# Get first page URL:")
        print(f">>> data[0]['metadata']['sourceURL']")
        print(f"'{data[0]['metadata']['sourceURL']}'")
        
        print("\n# Get first page content (first 200 chars):")
        print(f">>> data[0]['markdown'][:200]")
        print(f"'{data[0]['markdown'][:200]}...'")
        
        print("\n# Get all URLs:")
        print(f">>> urls = [page['metadata']['sourceURL'] for page in data]")
        print(f">>> print(urls[:3])")
        print(f"{[page['metadata']['sourceURL'] for page in data[:3]]}")
        
        print("\n# Count total words across all pages:")
        print(f">>> total_words = sum(len(page['markdown'].split()) for page in data)")
        total_words = sum(len(page['markdown'].split()) for page in data)
        print(f">>> print(total_words)")
        print(f"{total_words:,} words")
        
        print("\n" + "="*60)
        print("✅ Data loaded and ready to use!")
        print("="*60)
        
    except FileNotFoundError:
        print(f"\n❌ File not found: {json_file}")
        print("💡 Make sure you've run 'firecrawl_save_data.py' first")
    except Exception as e:
        print(f"\n❌ Error: {e}")