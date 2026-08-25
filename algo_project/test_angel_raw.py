import requests

from angel_one.login import AngelOneLogin


print()
print("=" * 70)
print("ANGEL ONE RAW API TEST")
print("=" * 70)


# ============================================================
# LOGIN
# ============================================================

client = AngelOneLogin.connect_from_env()

api = client.get_api()


print()
print("LOGIN : OK")


# ============================================================
# TOKEN
# ============================================================

access_token = client.access_token
api_key = client.api_key


print()
print("ACCESS TOKEN : OK")
print("API KEY      : OK")


# ============================================================
# API URL
# ============================================================

url = (
    "https://apiconnect.angelone.in/"
    "rest/secure/angelbroking/historical/v1/"
    "getCandleData"
)


# ============================================================
# HEADERS (reuse the exact headers SmartConnect itself builds)
# ============================================================

headers = api.requestHeaders()
headers["Authorization"] = f"Bearer {access_token}"


# ============================================================
# REQUEST BODY
# (fromdate/todate MUST be "YYYY-MM-DD HH:MM" - this was the bug:
# the previous "DD-MM-YYYY HH:MM" order caused Angel One's date
# parser to silently reject the request with an empty HTTP 400 body)
# ============================================================

from_date, to_date = AngelOneLogin.market_session_range(days=1)

payload = {
    "exchange": "NSE",
    "symboltoken": "26000",
    "interval": "ONE_MINUTE",
    "fromdate": from_date,
    "todate": to_date,
}


print()
print("=" * 70)
print("REQUEST")
print("=" * 70)

print()
print("URL :", url)

print()
print("PAYLOAD :")
print(payload)


# ============================================================
# REQUEST
# ============================================================

try:

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=20,
    )

    print()
    print("=" * 70)
    print("SERVER RESPONSE")
    print("=" * 70)

    print()
    print("HTTP STATUS :", response.status_code)

    print()
    print("RAW RESPONSE :")
    print(response.text)

    print()
    print("RESPONSE LENGTH :")
    print(len(response.text))


except Exception as error:

    print()
    print("=" * 70)
    print("HTTP ERROR")
    print("=" * 70)

    print()
    print(type(error).__name__)
    print(error)


print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)