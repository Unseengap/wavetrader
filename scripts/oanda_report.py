import requests
from datetime import datetime, timezone, timedelta

api_key = '684418077ea627c0070b8dd46b8124e8-449a590a81cfd30e79951b5babb09bed'
acct = '101-001-30902818-001'
base = 'https://api-fxpractice.oanda.com'
h = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

r = requests.get(f'{base}/v3/accounts/{acct}/summary', headers=h)
a = r.json()['account']
print('=== ACCOUNT SUMMARY ===')
for k in ['balance','NAV','unrealizedPL','pl','openTradeCount','financing','marginUsed']:
    print(f'  {k}: {a[k]}')

r2 = requests.get(f'{base}/v3/accounts/{acct}/openTrades', headers=h)
trades = r2.json().get('trades', [])
print(f'\n=== OPEN TRADES ({len(trades)}) ===')
for t in trades:
    sl = t.get('stopLossOrder', {}).get('price', 'none')
    tp = t.get('takeProfitOrder', {}).get('price', 'none')
    print(f'  ID={t["id"]} {t["instrument"]} units={t["currentUnits"]} entry={t["price"]} PnL={t["unrealizedPL"]} SL={sl} TP={tp}')

since = (datetime.now(timezone.utc) - timedelta(hours=12)).strftime('%Y-%m-%dT%H:%M:%SZ')
r3 = requests.get(f'{base}/v3/accounts/{acct}/transactions', headers=h, params={'from': since, 'pageSize': 1000})
pages = r3.json().get('pages', [])
print(f'\n=== ALL TRANSACTIONS (since {since}) ===')
all_txns = []
for page_url in pages:
    r4 = requests.get(page_url, headers=h)
    all_txns.extend(r4.json().get('transactions', []))
print(f'Total: {len(all_txns)}')
for tx in all_txns:
    typ = tx.get('type', '')
    t = tx.get('time', '')[:19]
    inst = tx.get('instrument', '')
    units = tx.get('units', '')
    price = tx.get('price', '')
    reason = tx.get('reason', '')
    pl = tx.get('pl', '')
    reject = tx.get('rejectReason', '')
    to_id = tx.get('tradeOpened', {}).get('tradeID', '')
    tc_list = tx.get('tradesClosed', [])
    tc_id = tc_list[0].get('tradeID', '') if tc_list else ''
    tid = to_id or tc_id
    print(f'  [{tx["id"]:>4}] {t}  {typ:35s} {inst:10s} u={str(units):8s} p={str(price):10s} pl={str(pl):8s} {reason} T={tid} {reject}')
