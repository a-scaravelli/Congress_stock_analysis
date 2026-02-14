

import requests
import os
import time

# --- CONFIGURATION ---
# ⚠️ CRITICAL: You will likely hit the rate limit (60/hr) without a token.
# Generate one here: https://github.com/settings/tokens (Select 'public_repo' scope)
GITHUB_TOKEN = "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" 

OWNER = "unitedstates"
REPO = "congress-legislators"
FILE_PATH = "committee-membership-current.yaml"
OUTPUT_DIR = "committee_history"
# ---------------------

def download_history():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if "YOUR_TOKEN" not in GITHUB_TOKEN else {}
    
    # 1. Fetch all commits that touched this file
    print(f"Fetching commit history for {FILE_PATH}...")
    commits = []
    page = 1
    
    while True:
        url = f"https://api.github.com/repos/{OWNER}/{REPO}/commits"
        params = {"path": FILE_PATH, "per_page": 100, "page": page}
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching commits: {response.status_code} {response.text}")
            break
            
        data = response.json()
        if not data:
            break  # No more commits
            
        commits.extend(data)
        print(f"Found {len(data)} commits on page {page}...")
        page += 1
        
    print(f"Total revisions found: {len(commits)}")

    # 2. Download each version
    for i, commit in enumerate(commits):
        sha = commit['sha']
        short_sha = sha[:7]
        date = commit['commit']['author']['date'].replace(':', '-')
        
        # Naming format: YYYY-MM-DD_SHA_filename
        filename = f"{date}_{short_sha}_committee-membership.yaml"
        local_path = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(local_path):
            print(f"[{i+1}/{len(commits)}] Skipping {short_sha} (already exists)")
            continue

        raw_url = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{sha}/{FILE_PATH}"
        
        print(f"[{i+1}/{len(commits)}] Downloading {date} ({short_sha})...")
        try:
            r = requests.get(raw_url, headers=headers)
            if r.status_code == 200:
                with open(local_path, 'wb') as f:
                    f.write(r.content)
            else:
                print(f"Failed to download {short_sha}: {r.status_code}")
        except Exception as e:
            print(f"Error downloading {short_sha}: {e}")
            
        # Be nice to the API
        time.sleep(0.5)

if __name__ == "__main__":
    download_history()
