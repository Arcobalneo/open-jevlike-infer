#!/usr/bin/env bash
# A choice, a score and a noul question in one request.
BASE=${BASE:-http://127.0.0.1:8000}
curl -s "$BASE/v1/systemone" -H 'content-type: application/json' -d '{
  "model": "clef-flash",
  "state": "Our checkout started returning errors and orders are blocked.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle the message?",
      "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}
    },
    "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
    "outage": {"type": "noul", "instructions": "Is a service down?"}
  }
}'
echo
