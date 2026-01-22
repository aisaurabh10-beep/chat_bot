import google.generativeai as genai

# REPLACE WITH YOUR ACTUAL KEY
API_KEY = "AIzaSyBBDrDHerfWSqTb4eIFe6B5thr08P6FYTE"

print(f"1. Configuring with key: {API_KEY[:5]}...")
genai.configure(api_key=API_KEY)

print("2. Attempting to list models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"   - Found: {m.name}")
    print("\n✅ SUCCESS! Your key works.")
except Exception as e:
    print(f"\n❌ FAILURE. Detailed Error:\n{e}")