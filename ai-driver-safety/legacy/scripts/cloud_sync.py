import requests
import json

SUPABASE_URL = "https://ybcfbbclcamcccqfkgbi.supabase.co"
SUPABASE_KEY = "sb_publishable_gjLy3MxPn4NgPWPsmhk5kQ_E45BAsrU"

def send_alert(event_type, risk_score):
    """
    Sends driver violations straight to the Supabase Cloud Table
    """
    url = f"{SUPABASE_URL}/rest/v1/incidents"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    payload = {
        "driver_name": "Ahmed", 
        "event_type": str(event_type),
        "risk_index": float(risk_score)
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 201:
            print(f" [CLOUD LOG] Successfully uploaded alert: {event_type} ({risk_score})")
        else:
            print(f" [CLOUD ERROR] Status code: {response.status_code}")
    except Exception as e:
        print(f" [CLOUD FAILED] Connection Error: {e}")