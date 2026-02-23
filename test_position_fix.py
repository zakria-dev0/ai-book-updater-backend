"""
Quick test to verify position tracking is working
"""
import requests

BASE_URL = "http://127.0.0.1:8000/api/v1"
TEST_EMAIL = "position_test@example.com"
TEST_PASSWORD = "testpass123"

# 1. Register/Login
print("1. Logging in...")
requests.post(f"{BASE_URL}/auth/register", json={
    "email": TEST_EMAIL,
    "password": TEST_PASSWORD
})

response = requests.post(f"{BASE_URL}/auth/login", json={
    "email": TEST_EMAIL,
    "password": TEST_PASSWORD
})
token = response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. Upload existing test file
print("2. Uploading test document...")
with open('test_equations.docx', 'rb') as f:
    files = {'file': ('test_equations.docx', f, 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')}
    response = requests.post(f"{BASE_URL}/upload/", files=files, headers=headers)
    document_id = response.json()['document_id']
    print(f"   Document ID: {document_id}")

# 3. Process document
print("3. Processing document...")
response = requests.post(f"{BASE_URL}/documents/{document_id}/process", headers=headers)
print(f"   Status: {response.json()['status']}")

# 4. Get details and check positions
print("\n4. Checking if positions are populated...")
response = requests.get(f"{BASE_URL}/documents/{document_id}", headers=headers)
data = response.json()

print(f"\n{'='*60}")
print("POSITION VERIFICATION")
print(f"{'='*60}")

# Check equations
equations = data.get('equations', [])
print(f"\nEquations ({len(equations)}):")
if equations:
    eq = equations[0]
    pos = eq.get('position', {})
    print(f"  Sample Equation:")
    print(f"    LaTeX: {eq.get('latex')[:50]}...")
    print(f"    Position:")
    print(f"      - Page: {pos.get('page')}")
    print(f"      - Paragraph: {pos.get('paragraph')}")
    print(f"      - Line: {pos.get('line')}")

    if pos.get('page') is not None or pos.get('paragraph') is not None or pos.get('line') is not None:
        print(f"\n  [OK] Position is populated!")
    else:
        print(f"\n  [ERROR] Position is still NULL!")
else:
    print("  No equations found")

# Check tables
tables = data.get('tables', [])
print(f"\nTables ({len(tables)}):")
if tables:
    tbl = tables[0]
    pos = tbl.get('position', {})
    print(f"  Sample Table:")
    print(f"    Position:")
    print(f"      - Page: {pos.get('page')}")
    print(f"      - Paragraph: {pos.get('paragraph')}")
    print(f"      - Line: {pos.get('line')}")

    if pos.get('page') is not None or pos.get('paragraph') is not None or pos.get('line') is not None:
        print(f"\n  [OK] Position is populated!")
    else:
        print(f"\n  [ERROR] Position is still NULL!")
else:
    print("  No tables found")

print(f"\n{'='*60}")
print("\nTest completed!")
