import requests
import time
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

print('Starting Firecrawl v2 with Direct API calls (SSL bypass)...')

# Your API configuration
API_KEY = "fc-7de9c85379534de6969dbaa8d8b0bf0a"  # Replace with your key
BASE_URL = "https://api.firecrawl.dev/v2"

# Headers for authentication
headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

# ==============================================================
# SOLUTION 3: Use requests library directly with verify=False
# ==============================================================

def start_crawl(url, limit=100):
    """Start a crawl job"""
    endpoint = f"{BASE_URL}/crawl"
    payload = {
        'url': url,
        'limit': limit,
        'scrapeOptions': {
            'formats': ['markdown']
        }
    }
    
    # verify=False bypasses SSL certificate verification
    response = requests.post(
        endpoint, 
        json=payload, 
        headers=headers, 
        verify=False  # THIS IS THE KEY: Bypass SSL verification
    )
    
    response.raise_for_status()
    return response.json()

def check_crawl_status(job_id):
    """Check the status of a crawl job"""
    endpoint = f"{BASE_URL}/crawl/{job_id}"
    
    response = requests.get(
        endpoint, 
        headers=headers, 
        verify=False  # Bypass SSL verification
    )
    
    response.raise_for_status()
    return response.json()

# ==============================================================
# Now use the functions
# ==============================================================

try:
    print("\n=== Starting Crawl ===")
    crawl_result = start_crawl('https://bharathatechno.com/', limit=100)
    
    job_id = crawl_result.get('id')
    print(f"Crawl Job ID: {job_id}")
    
    # Poll for status
    print("\n=== Monitoring Progress ===")
    while True:
        status = check_crawl_status(job_id)
        
        current_status = status.get('status', 'unknown')
        completed = status.get('completed', 0)
        total = status.get('total', 0)
        
        print(f"Status: {current_status} - Pages: {completed}/{total}")
        
        if current_status == 'completed':
            print("\n✅ Crawl finished!")
            if 'data' in status and len(status['data']) > 0:
                print(f"Total pages crawled: {len(status['data'])}")
                print(f"\nSample from first page:")
                print(f"{status['data'][0].get('markdown', '')[:300]}...")
            break
        elif current_status == 'failed':
            print(f"\n❌ Crawl failed: {status}")
            break
            
        time.sleep(5)
        
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")