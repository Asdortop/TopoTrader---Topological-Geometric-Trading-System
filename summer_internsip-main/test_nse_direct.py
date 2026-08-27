"""Quick test of NSE direct API for TCS 2022 Q1."""
import sys, os, time
sys.stdout.reconfigure(line_buffering=True)
import requests
from datetime import date

HEADERS = {
    "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept"     : "application/json, text/plain, */*",
    "Referer"    : "https://www.nseindia.com/",
}
NSE_API = (
    "https://www.nseindia.com/api/historical/cm/equity"
    "?symbol={symbol}&series=[%22EQ%22]&from={from_date}&to={to_date}"
)

print("Step 1: Creating session...", flush=True)
session = requests.Session()
session.headers.update(HEADERS)

print("Step 2: Warming up NSE homepage...", flush=True)
r = session.get("https://www.nseindia.com", timeout=15)
print(f"  homepage status: {r.status_code}", flush=True)
time.sleep(1)

print("Step 3: Fetching TCS data 01-01-2022 to 31-03-2022...", flush=True)
url = NSE_API.format(symbol="TCS", from_date="01-01-2022", to_date="31-03-2022")
print(f"  URL: {url[:80]}...", flush=True)

resp = session.get(url, timeout=20)
print(f"  status: {resp.status_code}", flush=True)

if resp.status_code == 200:
    data = resp.json()
    records = data.get("data", [])
    print(f"  records returned: {len(records)}", flush=True)
    if records:
        print(f"  first record keys: {list(records[0].keys())}", flush=True)
        print(f"  first record: {records[0]}", flush=True)
        print("SUCCESS", flush=True)
    else:
        print(f"  full response: {str(data)[:500]}", flush=True)
else:
    print(f"  response text: {resp.text[:500]}", flush=True)
