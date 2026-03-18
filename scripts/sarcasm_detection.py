"""
=============================================================================
Step 4: Lightweight Sarcasm Detection for Indian Political Tweets
=============================================================================
Project : Twitter Sentiment Analysis with Lightweight Sarcasm Detection
Script  : scripts/sarcasm_detection.py
Input   : data/political_tweets_sentiment.csv
Output  : data/political_tweets_final.csv
=============================================================================

APPROACH
--------
Rule-based sarcasm detection tuned for Indian political Twitter discourse.
When sarcasm is detected, the original model sentiment is corrected.

This is the novel contribution of this project — existing sentiment models
do not account for sarcasm in Indian political tweets, leading to systematic
misclassification of ironic praise and coded negative language.

SIX DETECTION RULES
--------------------
Rule 1 — Explicit sarcasm hashtags      (#sarcasm, #irony, #justsaying)
Rule 2 — Punctuation overload           (!!!, ???, ...... patterns)
Rule 3 — Indian political sarcasm words (andhbhakt, jumla, CONgress, etc.)
Rule 4 — Sarcastic emoji signals        (🤡, 🙄, 😏, 😂 in political context)
Rule 5 — Ironic praise pattern          (great/brilliant/genius + negative ctx)
Rule 6 — Contradictory structure        (positive opener + "but" + complaint)

CORRECTION LOGIC
----------------
If sarcasm detected:
  - Positive sentiment → corrected to Negative  (ironic praise caught)
  - Neutral  sentiment → corrected to Negative  (disguised negativity caught)
  - Negative sentiment → kept as Negative       (already correct)

=============================================================================
"""

import os
import re
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

INPUT_CSV  = "data/political_tweets_sentiment.csv"
OUTPUT_CSV = "data/political_tweets_final.csv"

# ─────────────────────────────────────────────────────────────────────────────
# RULE 1 — Explicit sarcasm/irony hashtags
# ─────────────────────────────────────────────────────────────────────────────

SARCASM_HASHTAGS = [
    "sarcasm", "irony", "ironic", "justsaying", "just_saying",
    "notreally", "suretotally", "obviously", "clearly",
]

def rule_hashtag(text: str) -> bool:
    """Detects explicit sarcasm/irony hashtag markers."""
    lowered = text.lower()
    # Match hashtags directly or as words
    for tag in SARCASM_HASHTAGS:
        if f"#{tag}" in lowered or f"# {tag}" in lowered:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# RULE 2 — Punctuation overload (!!!, ???, ......)
# ─────────────────────────────────────────────────────────────────────────────

def rule_punctuation(text: str) -> bool:
    """
    Detects excessive punctuation — a strong signal of sarcastic/ironic tone.
    Triggers on 3+ consecutive ! or ?, or 4+ consecutive dots.
    """
    if re.search(r"[!]{3,}", text):   # !!! or more
        return True
    if re.search(r"[?]{3,}", text):   # ??? or more
        return True
    if re.search(r"[.]{4,}", text):   # .... or more
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# RULE 3 — Indian political sarcasm vocabulary
# ─────────────────────────────────────────────────────────────────────────────

# Words/phrases commonly used sarcastically in Indian political Twitter
INDIAN_SARCASM_WORDS = [
    # BJP/Modi criticism markers
    "andhbhakt", "bhakt",        # blind follower — always derogatory
    "jumla", "jumlebaazi",        # empty promise — always negative
    "feku",                       # liar (Modi nickname) — always negative
    "godi media",                 # lapdog media — always negative
    "pappu",                      # Rahul Gandhi insult — always negative
    "congress",                   # intentional misspelling used sarcastically
    "sickulars",                  # sarcastic term for secularists
    "libtard",                    # sarcastic liberal insult
    "andh bhakt",                 # space variant
    "namo app",                   # NaMo app tweets — often ironic praise
    "vishwaguru",                 # Vishwaguru — often used sarcastically
    "double engine",              # BJP slogan used sarcastically
    "achhe din",                  # "good days" — BJP slogan, now ironic
    "sabka saath",                # BJP slogan used ironically
    "modiji",                     # used sarcastically in critical tweets
    "modi ji",                    # same with space
    "great leader",               # ironic praise in political context
    "our great",                  # ironic opener
    "master stroke",              # used sarcastically for BJP decisions
    "masterstroke",               # variant
    "clapping",                   # reference to applauding Modi
    "thali bajao",                # COVID plate banging — used sarcastically
    "diya jalao",                 # COVID candle — used sarcastically
]

def rule_indian_sarcasm_vocab(text: str) -> bool:
    """Detects Indian political sarcasm vocabulary."""
    lowered = text.lower()
    return any(word in lowered for word in INDIAN_SARCASM_WORDS)


# ─────────────────────────────────────────────────────────────────────────────
# RULE 4 — Sarcastic emoji signals
# ─────────────────────────────────────────────────────────────────────────────

# These emojis in a political tweet context strongly indicate sarcasm/irony
SARCASM_EMOJIS = [
    "🤡",   # clown — very common sarcasm signal in Indian political Twitter
    "🙄",   # eye roll
    "😏",   # smirk
    "🤣",   # laughing — used sarcastically at political claims
    "😂",   # crying laugh — used to mock
    "👏",   # clapping — often sarcastic applause
    "🐄",   # cow — used to mock BJP/Hindutva
    "🐍",   # snake — used to call someone a traitor
]

def rule_sarcasm_emoji(text: str) -> bool:
    """Detects sarcastic emoji usage in political tweet context."""
    return any(emoji in text for emoji in SARCASM_EMOJIS)


# ─────────────────────────────────────────────────────────────────────────────
# RULE 5 — Ironic praise pattern
# Positive/praise word + negative context word in same tweet
# ─────────────────────────────────────────────────────────────────────────────

PRAISE_WORDS = [
    "genius", "brilliant", "great job", "well done", "excellent",
    "bravo", "amazing", "incredible", "fantastic", "wonderful",
    "superb", "outstanding", "perfect", "best pm", "best leader",
    "legend", "god", "demi-god", "superhero", "marvel",
]

NEGATIVE_CONTEXT = [
    "corrupt", "failed", "failure", "useless", "incompetent",
    "destroyed", "ruined", "disaster", "shame", "pathetic",
    "embarrassing", "joke", "fraud", "liar", "lies", "scam",
    "rape", "murder", "crime", "criminal", "thief", "chor",
    "poor", "poverty", "unemployment", "inflation",
]

def rule_ironic_praise(text: str) -> bool:
    """
    Detects ironic praise — a positive/complimentary word appearing
    in a tweet that also contains negative context words.
    """
    lowered = text.lower()
    has_praise   = any(p in lowered for p in PRAISE_WORDS)
    has_negative = any(n in lowered for n in NEGATIVE_CONTEXT)
    return has_praise and has_negative


# ─────────────────────────────────────────────────────────────────────────────
# RULE 6 — Contradictory structure (positive opener + "but" + complaint)
# ─────────────────────────────────────────────────────────────────────────────

POSITIVE_OPENERS = [
    "great", "good", "nice", "wonderful", "excellent", "love",
    "appreciate", "respect", "well done", "congratulations",
    "finally", "at last", "thankfully",
]

CONTRADICTION_MARKERS = [
    " but ", " however ", " yet ", " still ", " though ",
    " unfortunately ", " sadly ", " except ", " despite ",
    " while ", " whereas ",
]

def rule_contradictory_structure(text: str) -> bool:
    """
    Detects tweets that open positively but pivot to a complaint —
    a common sarcasm/backhanded-compliment pattern in political tweets.
    """
    lowered = text.lower()

    has_opener       = any(lowered.startswith(op) or f" {op} " in lowered[:50]
                           for op in POSITIVE_OPENERS)
    has_contradiction = any(marker in lowered for marker in CONTRADICTION_MARKERS)

    return has_opener and has_contradiction


# ─────────────────────────────────────────────────────────────────────────────
# MASTER SARCASM DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

RULES = {
    "hashtag"          : rule_hashtag,
    "punctuation"      : rule_punctuation,
    "indian_vocab"     : rule_indian_sarcasm_vocab,
    "sarcasm_emoji"    : rule_sarcasm_emoji,
    "ironic_praise"    : rule_ironic_praise,
    "contradictory"    : rule_contradictory_structure,
}

def detect_sarcasm(text: str) -> tuple[bool, str]:
    """
    Runs all six rules against a tweet.

    Returns:
        (sarcasm_detected: bool, sarcasm_type: str)
        sarcasm_type is "none" if no rule fired,
        or comma-separated rule names if one or more fired.
    """
    triggered = []
    for rule_name, rule_fn in RULES.items():
        if rule_fn(str(text)):
            triggered.append(rule_name)

    if triggered:
        return True, ",".join(triggered)
    return False, "none"


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT CORRECTION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def correct_sentiment(original_sentiment: str, sarcasm_detected: bool) -> str:
    """
    Applies sarcasm correction to model's original sentiment prediction.

    Rules:
      - No sarcasm detected → keep original sentiment unchanged
      - Sarcasm + Positive  → correct to Negative (ironic praise)
      - Sarcasm + Neutral   → correct to Negative (disguised negativity)
      - Sarcasm + Negative  → keep Negative (already correct)
    """
    if not sarcasm_detected:
        return original_sentiment

    if original_sentiment == "Positive":
        return "Negative"   # Ironic praise flipped
    elif original_sentiment == "Neutral":
        return "Negative"   # Disguised negativity corrected
    else:
        return "Negative"   # Already negative, confirmed


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Step 4: Lightweight Sarcasm Detection")
    print("=" * 65)

    # ── Load sentiment CSV ────────────────────────────────────────────────────
    print(f"\n[Load]  Reading '{INPUT_CSV}'...")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    print(f"[Load]  ✅ {len(df)} tweets loaded.")

    # ── Run sarcasm detection ─────────────────────────────────────────────────
    print(f"\n[Detect] Running sarcasm detection across 6 rules...")

    results = df["clean_text"].apply(lambda t: detect_sarcasm(t))
    df["sarcasm_detected"] = results.apply(lambda x: x[0])
    df["sarcasm_type"]     = results.apply(lambda x: x[1])

    # ── Apply sentiment correction ────────────────────────────────────────────
    df["corrected_sentiment"] = df.apply(
        lambda row: correct_sentiment(row["sentiment"], row["sarcasm_detected"]),
        axis=1
    )

    # ── Save output ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[Save]  ✅ Final dataset saved to: {OUTPUT_CSV}")
    print(f"[Save]     Columns: {list(df.columns)}")

    # ── Summary report ────────────────────────────────────────────────────────
    total          = len(df)
    sarcasm_count  = df["sarcasm_detected"].sum()
    sarcasm_pct    = sarcasm_count / total * 100

    print("\n" + "=" * 65)
    print("  SARCASM DETECTION REPORT")
    print("=" * 65)
    print(f"\n  Total tweets        : {total}")
    print(f"  Sarcasm detected    : {sarcasm_count} ({sarcasm_pct:.1f}%)")
    print(f"  Non-sarcastic       : {total - sarcasm_count} ({100-sarcasm_pct:.1f}%)")

    # Rule breakdown
    print(f"\n  Rule Breakdown:")
    print("  " + "-" * 45)
    for rule_name in RULES:
        count = df["sarcasm_type"].str.contains(rule_name).sum()
        pct   = count / total * 100
        print(f"  {rule_name:<22} : {count:>5} tweets ({pct:.1f}%)")

    # Sentiment before vs after correction
    print(f"\n  Sentiment Distribution — BEFORE correction:")
    before = df["sentiment"].value_counts()
    for label in ["Positive", "Neutral", "Negative"]:
        n   = before.get(label, 0)
        pct = n / total * 100
        print(f"    {label:<10} : {n:>5} ({pct:.1f}%)")

    print(f"\n  Sentiment Distribution — AFTER sarcasm correction:")
    after = df["corrected_sentiment"].value_counts()
    for label in ["Positive", "Neutral", "Negative"]:
        n   = after.get(label, 0)
        pct = n / total * 100
        print(f"    {label:<10} : {n:>5} ({pct:.1f}%)")

    # How many tweets had sentiment changed
    changed = (df["sentiment"] != df["corrected_sentiment"]).sum()
    print(f"\n  Tweets with corrected sentiment : {changed} ({changed/total*100:.1f}%)")

    # Sample sarcastic tweets
    print(f"\n  Sample detected sarcastic tweets:")
    print("  " + "-" * 55)
    sarcastic = df[df["sarcasm_detected"] == True]
    for _, row in sarcastic.head(6).iterrows():
        print(f"  Rule     : {row['sarcasm_type']}")
        print(f"  Original : {row['sentiment']} → Corrected: {row['corrected_sentiment']}")
        print(f"  Tweet    : {row['clean_text'][:100]}")
        print()

    print("=" * 65)
    print("  ✅ Step 4 complete! Proceed to Step 5 (Aggregation & Visualisation).")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
