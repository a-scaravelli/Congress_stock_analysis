import requests
import json

API_KEY = "XQfuCF7MJSZWwgFxZF8pbzx6NfVfEdmY45Oebjqp"  # Replace this

def safe_get(url, params):
    params = dict(params or {})
    params["api_key"] = API_KEY
    
    try:
        r = requests.get(url, params=params, timeout=30)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            return r.json()
        else:
            print(f"Error: {r.text[:300]}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

print("="*60)
print("TESTING CORRECT COMMITTEE ENDPOINT")
print("="*60)

# Test 1: Just chamber (current committees)
print("\n1. Testing: /committee/house")
url1 = "https://api.congress.gov/v3/committee/house"
data1 = safe_get(url1, {"limit": 3})
if data1 and 'committees' in data1:
    print(f"✓ Found {len(data1['committees'])} committees")
    if data1['committees']:
        print(f"  Example: {data1['committees'][0].get('name')}")

# Test 2: Just chamber for senate
print("\n2. Testing: /committee/senate")
url2 = "https://api.congress.gov/v3/committee/senate"
data2 = safe_get(url2, {"limit": 3})
if data2 and 'committees' in data2:
    print(f"✓ Found {len(data2['committees'])} committees")

# Test 3: Specific committee details
print("\n3. Testing: /committee/{chamber}/{code}")
if data1 and 'committees' in data1 and data1['committees']:
    first_cmte = data1['committees'][0]
    code = first_cmte.get('systemCode')
    chamber = 'house'
    
    print(f"Getting details for: {first_cmte.get('name')}")
    print(f"Code: {code}")
    
    url3 = f"https://api.congress.gov/v3/committee/{chamber}/{code}"
    data3 = safe_get(url3, {})
    
    if data3:
        print("\n✓ Full committee response:")
        print(json.dumps(data3, indent=2)[:1000])  # First 1000 chars
        
        if 'committee' in data3:
            cmte_keys = list(data3['committee'].keys())
            print(f"\nCommittee object keys: {cmte_keys}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("The correct endpoints are:")
print("  List: /committee/{chamber}")
print("  Detail: /committee/{chamber}/{systemCode}")
print("\nNOTE: These return CURRENT committees, not historical ones!")