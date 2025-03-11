from google.cloud import storage
import os
import requests
import xml.etree.ElementTree as ET
import json

# Explore bucket: https://console.cloud.google.com/storage/browser/arxiv-dataset

OUTPUT_DIR = "./pdfs"
SEARCH_QUERY = "cat:cs.AI"  # category
START_DATE = "20250101"
END_DATE = "20251231"
MAX_RESULTS = 300        # Number of papers per request (max 300)
TOTAL_PAPERS = 9999999999999999999        # Total number of papers to fetch

def download_arxiv_paper(id, output_dir=OUTPUT_DIR):
  try:
    # Set the GCS bucket and prefix (folder) for the arXiv dataset
    bucket_name = "arxiv-dataset"
    year = id[:2]
    month = id[2:4]
    path = f"arxiv/arxiv/pdf/{year}{month}/{id}.pdf"

    # Initialize the GCS client without authentication (public bucket)
    client = storage.Client.create_anonymous_client()

    # Get the bucket
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(path)
    
    # Download the file to the local directory
    local_path = os.path.join(output_dir, f"{id}.pdf")
    blob.download_to_filename(local_path)
    print(f"Downloaded {id} to {output_dir} successfully")
  except Exception as e:
    print(f"Error downloading {id}: {e}")

def parse_arxiv_response(xml_data):
    root = ET.fromstring(xml_data)
    ns = {"arxiv": "http://www.w3.org/2005/Atom"}
    results = []
    
    for entry in root.findall("arxiv:entry", ns):
        # Extract paper ID - use URL as fallback
        paper_id = None
        if entry.find("arxiv:id", ns) is not None:
            paper_id = entry.find("arxiv:id", ns).text
            # Extract the arXiv ID part from the full URL
            if paper_id.startswith("http://arxiv.org/abs/"):
                paper_id = paper_id.split("/")[-1]
        
        # Find PDF URL
        pdf_url = None
        for link in entry.findall("arxiv:link", ns):
            if 'title' in link.attrib and link.attrib['title'] == 'pdf':
                pdf_url = link.attrib['href']
                break
            elif 'rel' in link.attrib and link.attrib['rel'] == 'alternate' and 'type' in link.attrib and link.attrib['type'] == 'application/pdf':
                pdf_url = link.attrib['href']
                break
        
        paper = {
            "id": paper_id,
            "title": entry.find("arxiv:title", ns).text.strip(),
            "summary": entry.find("arxiv:summary", ns).text.strip(),
            "published": entry.find("arxiv:published", ns).text,
            "updated": entry.find("arxiv:updated", ns).text,
            "authors": [author.find("arxiv:name", ns).text for author in entry.findall("arxiv:author", ns)],
            "categories": entry.find("arxiv:category", ns).attrib["term"] if entry.find("arxiv:category", ns) else None,
            "pdf_url": pdf_url
        }
        results.append(paper)
    
    return results

def fetch_arxiv_papers(search_query, start_date, end_date, start=0):
    BASE_URL = "http://export.arxiv.org/api/query?"
    params = {
        "search_query": f"{search_query} AND submittedDate:[{start_date} TO {end_date}]",
        "start": start,
        "max_results": min(MAX_RESULTS, TOTAL_PAPERS - start),
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.text
    return None

# Create a local directory to store the downloaded files
if not os.path.exists(OUTPUT_DIR):
  os.makedirs(OUTPUT_DIR)

# Loop to fetch multiple batches
papers = []
for start in range(0, TOTAL_PAPERS, MAX_RESULTS):
    to_download = min(MAX_RESULTS, TOTAL_PAPERS - start)
    print(f"Fetching papers {start} to {start + to_download}...")
    xml_data = fetch_arxiv_papers(SEARCH_QUERY, START_DATE, END_DATE, start)
    if xml_data:
        batch_papers = parse_arxiv_response(xml_data)
        papers.extend(batch_papers)
        
        # Download PDFs for this batch
        for paper in batch_papers:
            if paper['pdf_url']:
                print(f"Downloading PDF for {paper['id']}: {paper['title']}")
                download_arxiv_paper(paper['id'])

# Save metadata as JSON file
with open(f"{OUTPUT_DIR}/_arxiv_papers.json", "w", encoding="utf-8") as f:
    json.dump(papers, f, indent=2)

print(f"Saved {len(papers)} papers metadata to {OUTPUT_DIR}/_arxiv_papers.json")
print(f"PDFs downloaded to {OUTPUT_DIR} directory")