"""Generate data/golden_triage.jsonl (synthetic, fictional tickets). Run from the repo root:

    python tools/generate_golden.py

The file is committed; this script shows how it was made and keeps it reproducible.
"""
import json
from pathlib import Path

TICKETS = {
    "billing": [
        "I was charged twice for my last order.",
        "My invoice shows a total I do not recognise.",
        "Please send me a refund for the damaged lamp.",
        "The payment page took my card but the order never appeared.",
        "I was billed for a plan I never chose.",
        "Can I get a receipt for my purchase in March?",
        "The discount code was ignored on my invoice.",
        "I would like a refund because the item was not as described.",
        "There is an extra charge on my statement from your store.",
        "My payment failed but the money left my account.",
        "The invoice has the wrong company name on it.",
        "I need a corrected receipt for my accountant.",
    ],
    "login": [
        "I forgot my password and the reset email never arrives.",
        "I am locked out of my account after too many tries.",
        "The verification code never reaches my phone.",
        "I cannot sign in with my new email address.",
        "My 2fa app shows a code that is always rejected.",
        "Every time I try to log in the page just reloads.",
        "I changed my password and now it says it is invalid.",
        "Please help me sign in, the form says my account does not exist.",
        "I lost my phone and need to move my 2fa to a new one.",
        "The login link in the welcome email has expired.",
        "I keep getting logged out and have to sign in again every minute.",
        "My password reset link says it was already used.",
    ],
    "shipping": [
        "My package has not arrived and it was due last Monday.",
        "The tracking page has shown the same status for five days.",
        "The courier left my delivery at the wrong address.",
        "My order shipped but the tracking number does not work.",
        "The delivery arrived with a crushed box.",
        "I only received two of the three items that shipped.",
        "The courier says nobody was home but I was there all day.",
        "Can the package be redirected to my office address?",
        "My delivery date moved three times this week.",
        "The parcel arrived but the contents are for another customer.",
        "Tracking says delivered but I have nothing.",
        "The courier asked me to pay a fee on delivery.",
    ],
    "bug": [
        "The app crash happens every time I open the cart.",
        "I get an error when I upload a profile photo.",
        "The checkout button is broken on my phone.",
        "The product page keeps not loading on mobile data.",
        "The search screen freezes after I type two letters.",
        "There is a bug where my wishlist shows empty items.",
        "The site shows a blank page after I apply a filter.",
        "The order history page throws an error with a long list.",
        "The app crash started after the latest update.",
        "Saved addresses are broken, they all show the same street.",
        "The language switch is broken and resets to English.",
        "The notification badge freezes at 9 and never clears.",
    ],
    "cancellation": [
        "Please cancel my subscription before the next renewal.",
        "I want to unsubscribe from the monthly box.",
        "Please close my account and delete my saved data.",
        "I need to stop my subscription starting next month.",
        "Cancel my order, I ordered the wrong size.",
        "How do I cancel the annual plan?",
        "I would like to unsubscribe from all marketing emails.",
        "Cancel the standing order for coffee beans please.",
        "I want to close my account, I no longer shop with you.",
        "Please cancel the gift subscription I bought by mistake.",
        "Can you cancel my membership and confirm by email?",
        "I want to unsubscribe from the weekly deals newsletter.",
    ],
    "feature_request": [
        "It would be great to have a dark mode in the app.",
        "My wish is a way to save several delivery addresses.",
        "A suggestion: let us reorder a past basket in one tap.",
        "Please add support for paying with bank transfer.",
        "Could you add a feature to compare two products side by side?",
        "It would be great if the app could show prices in my currency.",
        "I wish the wishlist could be shared with friends.",
        "A suggestion for the team: add size guides to every page.",
        "Please add a button to export my order history.",
        "A feature to schedule a delivery time would help a lot.",
        "It would be great to filter products by rating.",
        "Please add a way to leave notes for the courier.",
    ],
}
# Hand-written cases that a keyword-driven prompt can get wrong or that need no category.
HARD = [
    ("billing", ["hard"], "My money vanished after the update, what is going on?", "normal"),
    ("cancellation", ["hard"], "Hello, I would like to leave your service next month.", "normal"),
    ("shipping", ["hard"], "Where is my stuff? It has been two weeks.", "normal"),
    ("bug", ["hard"], "The app does not do what I expect when I tap save.", "normal"),
    ("feature_request", ["hard"], "Can you also support dark mode? Not a bug, just a wish.", "normal"),
    ("other", ["easy"], "Do you have a shop in Dhaka?", "normal"),
    ("other", ["easy"], "Thanks for the quick help yesterday!", "normal"),
    ("other", ["easy"], "Can I get a copy of your partnership brochure? No rush.", "low"),
    ("billing", ["hard"], "I was charged twice and now I cannot log in to check.", "normal"),
]
VARIANTS = [("URGENT: {}", "high"), ("{} Please fix this asap.", "high"),
            ("{} No rush on this.", "low"), ("{}", "normal")]
REGEX = {"billing": "(?i)invoice|refund|billing", "login": "(?i)password|sign.?in|access",
         "shipping": "(?i)delivery|package|tracking", "bug": "(?i)bug|fix|engineer",
         "cancellation": "(?i)cancel", "feature_request": "(?i)feature|suggest|roadmap",
         "other": "(?i)team|help"}
SCHEMA = {"type": "object", "required": ["category", "priority", "sla_hours", "reply"],
          "properties": {"category": {"type": "string", "enum": list(REGEX)},
                         "priority": {"type": "string", "enum": ["high", "normal", "low"]},
                         "sla_hours": {"type": "number", "minimum": 1},
                         "reply": {"type": "string", "minLength": 1}}}
SLA = {"high": 4, "normal": 24, "low": 72}


def case(cid, text, cat, prio, tags):
    crit = [{"type": "json_schema", "schema": SCHEMA},
            {"type": "exact", "field": "category"},
            {"type": "exact", "field": "priority"},
            {"type": "regex", "field": "reply", "pattern": REGEX[cat]},
            {"type": "custom", "callable": "regression_guard.examples.triage:reply_is_concise"}]
    if "sla" in tags:
        crit.append({"type": "numeric", "field": "sla_hours", "tolerance": 2})
    return {"id": cid, "input": text, "expected": {"category": cat, "priority": prio, "sla_hours": SLA[prio]},
            "criteria": crit, "tags": tags, "weight": 2 if prio == "high" else 1}


def main():
    rows, n = [], 0
    for cat, texts in TICKETS.items():
        for i, text in enumerate(texts):
            fmt, prio = VARIANTS[i % 4]
            n += 1
            tags = [cat, "easy"] + (["sla"] if i % 4 == 0 else [])
            rows.append(case(f"t{n:03d}", fmt.format(text), cat, prio, tags))
    for cat, tags, text, prio in HARD:
        n += 1
        rows.append(case(f"t{n:03d}", text, cat, prio, [cat] + tags))
    out = Path(__file__).resolve().parent.parent / "data" / "golden_triage.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} cases to {out}")


if __name__ == "__main__":
    main()
