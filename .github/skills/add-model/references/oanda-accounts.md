# OANDA Account & API Key Reference

## Account Structure

```
OANDA User (single login)
├── Primary Account  (101-001-XXXXXXXX-001)  ← MTF model
├── Sub-Account #2   (101-001-XXXXXXXX-002)  ← WaveFollower
├── Sub-Account #3   (101-001-XXXXXXXX-003)  ← New model
└── ...
```

**One API key covers ALL accounts under the same user.**

## API Key Lifecycle When Adding Accounts

```
1. Create sub-account → new account ID
2. Revoke OLD API key → ALL models get 403 ⚠️
3. Generate NEW API key → covers all accounts (old + new)
4. Update .env on VPS with new key for EVERY model
5. docker compose restart → all models recover
6. Verify: curl each account → 200 OK
```

**Expected downtime**: 2-5 minutes (key propagation + container restart)

## Env Var Naming Convention

| Model | API Key Env Var | Account ID Env Var |
|-------|----------------|--------------------|
| MTF (default) | `OANDA_DEMO_API_KEY` | `OANDA_DEMO_ACCOUNT_ID` |
| WaveFollower | `WF_OANDA_DEMO_API_KEY` | `WF_OANDA_DEMO_ACCOUNT_ID` |
| New model | `{PREFIX}_OANDA_DEMO_API_KEY` | `{PREFIX}_OANDA_DEMO_ACCOUNT_ID` |

## Verification Commands

```bash
# Test account access
curl -s -H "Authorization: Bearer $API_KEY" \
  "https://api-fxpractice.oanda.com/v3/accounts/$ACCOUNT_ID/summary" | jq '.account.balance'

# List all accounts for this key
curl -s -H "Authorization: Bearer $API_KEY" \
  "https://api-fxpractice.oanda.com/v3/accounts" | jq '.accounts[].id'
```

## Common 403 Causes

1. **Key not regenerated** after adding sub-account
2. **Old key still in .env** — must update ALL entries
3. **OANDA propagation delay** — wait 1-2 min after regeneration
4. **Wrong environment** — practice key on live URL (or vice versa)
5. **Account not linked** to user — verify in OANDA Hub
