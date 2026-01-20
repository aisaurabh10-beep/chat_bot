import requests
import time
import urllib3
import json
import os
from datetime import datetime

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

print('Starting Firecrawl Data Collection & Saving...')

# ============================================================
# CONFIGURATION
# ============================================================
API_KEY = "fc-7de9c85379534de6969dbaa8d8b0bf0a"  # Replace with your key
BASE_URL = "https://api.firecrawl.dev/v2"
TARGET_URL = "https://bharathatechno.com/"

headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

# ============================================================
# FUNCTIONS
# ============================================================

def start_crawl(url, limit=100):
    """Start a crawl job"""
    endpoint = f"{BASE_URL}/crawl"
    payload = {
        'url': url,
        'limit': limit,
        'scrapeOptions': {
            'formats': ['markdown', 'html']  # Get both markdown and HTML
        }
    }
    
    response = requests.post(
        endpoint, 
        json=payload, 
        headers=headers, 
        verify=False
    )
    
    response.raise_for_status()
    return response.json()

def check_crawl_status(job_id):
    """Check the status of a crawl job"""
    endpoint = f"{BASE_URL}/crawl/{job_id}"
    
    response = requests.get(
        endpoint, 
        headers=headers, 
        verify=False
    )
    
    response.raise_for_status()
    return response.json()

def save_data_to_files(data, output_dir='crawl_output'):
    """Save scraped data in multiple formats"""
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"\n💾 Saving data to '{output_dir}' folder...")
    print(f"{'='*60}")
    
    # 1. Save complete JSON file (all data)
    json_file = os.path.join(output_dir, f'complete_data_{timestamp}.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Complete JSON: {json_file}")
    
    # 2. Save all markdown in one combined file
    combined_md_file = os.path.join(output_dir, f'all_pages_{timestamp}.md')
    with open(combined_md_file, 'w', encoding='utf-8') as f:
        f.write(f"# Complete Website Crawl\n")
        f.write(f"**Source:** {TARGET_URL}\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Pages:** {len(data)}\n\n")
        f.write("="*80 + "\n\n")
        
        for i, page in enumerate(data):
            url = page.get('metadata', {}).get('sourceURL', 'Unknown URL')
            title = page.get('metadata', {}).get('title', 'No Title')
            
            f.write(f"\n\n# Page {i+1}: {title}\n")
            f.write(f"**URL:** {url}\n\n")
            f.write("-"*80 + "\n\n")
            f.write(page.get('markdown', ''))
            f.write("\n\n" + "="*80 + "\n")
    print(f"✅ Combined Markdown: {combined_md_file}")
    
    # 3. Save individual markdown files for each page
    pages_dir = os.path.join(output_dir, 'individual_pages')
    if not os.path.exists(pages_dir):
        os.makedirs(pages_dir)
    
    for i, page in enumerate(data):
        url = page.get('metadata', {}).get('sourceURL', 'unknown')
        # Create a safe filename from URL
        safe_filename = url.replace('https://', '').replace('http://', '')
        safe_filename = safe_filename.replace('/', '_').replace('?', '_').replace('&', '_')
        safe_filename = safe_filename[:100]  # Limit filename length
        
        page_file = os.path.join(pages_dir, f'page_{i+1:03d}_{safe_filename}.md')
        
        with open(page_file, 'w', encoding='utf-8') as f:
            f.write(f"# {page.get('metadata', {}).get('title', 'No Title')}\n\n")
            f.write(f"**URL:** {url}\n\n")
            f.write("-"*80 + "\n\n")
            f.write(page.get('markdown', ''))
    print(f"✅ Individual Pages: {pages_dir}/ ({len(data)} files)")
    
    # 4. Save a summary CSV file
    csv_file = os.path.join(output_dir, f'pages_summary_{timestamp}.csv')
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("Page_Number,URL,Title,Word_Count\n")
        for i, page in enumerate(data):
            url = page.get('metadata', {}).get('sourceURL', '')
            title = page.get('metadata', {}).get('title', 'No Title').replace(',', ';')
            word_count = len(page.get('markdown', '').split())
            f.write(f"{i+1},\"{url}\",\"{title}\",{word_count}\n")
    print(f"✅ Summary CSV: {csv_file}")
    
    # 5. Save URLs list
    urls_file = os.path.join(output_dir, f'urls_list_{timestamp}.txt')
    with open(urls_file, 'w', encoding='utf-8') as f:
        for i, page in enumerate(data):
            url = page.get('metadata', {}).get('sourceURL', '')
            f.write(f"{i+1}. {url}\n")
    print(f"✅ URLs List: {urls_file}")
    
    print(f"{'='*60}")
    print(f"✅ All data saved successfully!")
    
    return {
        'json_file': json_file,
        'combined_md': combined_md_file,
        'pages_dir': pages_dir,
        'csv_file': csv_file,
        'urls_file': urls_file
    }

def preview_data(data):
    """Display a preview of the scraped data"""
    print(f"\n{'='*60}")
    print(f"📊 DATA PREVIEW")
    print(f"{'='*60}")
    print(f"Total Pages Crawled: {len(data)}")
    print(f"\n{'First 3 Pages:'}")
    print(f"{'-'*60}")
    
    for i, page in enumerate(data[:3]):
        metadata = page.get('metadata', {})
        title = metadata.get('title', 'No Title')
        url = metadata.get('sourceURL', 'No URL')
        word_count = len(page.get('markdown', '').split())
        
        print(f"\n📄 Page {i+1}:")
        print(f"   Title: {title}")
        print(f"   URL: {url}")
        print(f"   Words: {word_count}")
        print(f"   Preview: {page.get('markdown', '')[:150]}...")
    
    print(f"\n{'-'*60}")
    print(f"... and {len(data) - 3} more pages")
    print(f"{'='*60}")

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    try:
        print(f"\n🚀 Starting crawl of: {TARGET_URL}")
        print(f"{'='*60}\n")
        
        # Start the crawl
        crawl_result = start_crawl(TARGET_URL, limit=100)
        job_id = crawl_result.get('id')
        print(f"✅ Crawl Job Started!")
        print(f"Job ID: {job_id}\n")
        
        # Monitor progress
        print(f"📊 Monitoring Progress...")
        print(f"{'-'*60}")
        
        while True:
            status = check_crawl_status(job_id)
            
            current_status = status.get('status', 'unknown')
            completed = status.get('completed', 0)
            total = status.get('total', 0)
            credits_used = status.get('creditsUsed', 0)
            
            print(f"Status: {current_status:12} | Pages: {completed:2}/{total:2} | Credits: {credits_used}")
            
            if current_status == 'completed':
                print(f"{'-'*60}")
                print(f"\n✅ Crawl Completed Successfully!\n")
                
                # Get all the data
                all_data = status.get('data', [])
                
                if len(all_data) > 0:
                    # Preview the data
                    preview_data(all_data)
                    
                    # Save the data
                    saved_files = save_data_to_files(all_data)
                    
                    # Show what was saved
                    print(f"\n📁 FILES CREATED:")
                    print(f"{'='*60}")
                    for file_type, file_path in saved_files.items():
                        print(f"   {file_type}: {file_path}")
                    
                    print(f"\n💡 HOW TO USE THE DATA:")
                    print(f"{'='*60}")
                    print(f"1. View all pages together: Open '{saved_files['combined_md']}'")
                    print(f"2. Browse individual pages: Check folder '{saved_files['pages_dir']}'")
                    print(f"3. Get page list: Open '{saved_files['csv_file']}'")
                    print(f"4. Load in Python:")
                    print(f"   >>> import json")
                    print(f"   >>> with open('{saved_files['json_file']}', 'r') as f:")
                    print(f"   >>>     data = json.load(f)")
                    print(f"   >>> print(f'Loaded {{len(data)}} pages')")
                    
                else:
                    print("⚠️  No data returned from crawl")
                
                break
                
            elif current_status == 'failed':
                print(f"\n❌ Crawl Failed!")
                print(f"Error details: {status}")
                break
                
            time.sleep(5)
            
    except requests.exceptions.SSLError as e:
        print(f"\n❌ SSL Certificate Error: {e}")
        print(f"\n💡 Make sure verify=False is set in the requests")
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request Error: {e}")
        
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()