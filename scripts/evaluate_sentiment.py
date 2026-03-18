"""
=============================================================================
Step 3b: Model Evaluation — Manual Labels vs Model Predictions
=============================================================================
Project : Twitter Sentiment Analysis with Lightweight Sarcasm Detection
Script  : scripts/evaluate_sentiment.py
Input   : data/political_tweets_sentiment.csv
Output  : data/evaluation_report.txt
=============================================================================
"""

import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ─────────────────────────────────────────────────────────────────────────────
# MANUAL GROUND TRUTH LABELS
# Sampled from political_tweets_sentiment.csv (2313 rows), random_state=42
# Skipped = US/foreign noise tweets excluded from evaluation
# ─────────────────────────────────────────────────────────────────────────────

MANUAL_LABELS = {
    # Tweets 1-50
    1 : 'Neutral',   # Haryana Rajya Sabha — analytical
    2 : 'Negative',  # Criticising Modi on history curriculum
    3 : 'Neutral',   # Factual debunking of fake video
    4 : 'Negative',  # Criticising BJP on census
    5 : 'Negative',  # Sarcastic list mocking Modi's biography
    6 : 'Neutral',   # Ambiguous Telugu — asking if BJP supported
    7 : 'Negative',  # BJP ruling = worst roads, rapes, corruption
    8 : 'Neutral',   # Congress promise — news report
    9 : 'Negative',  # BJP won't give justice to general caste victim
    10: 'Skip',      # US noise
    11: 'Neutral',   # Poetic justice — ambiguous
    12: 'Negative',  # TMC walks out — protest against EC/BJP
    13: 'Negative',  # Mocking BJP supporters
    14: 'Negative',  # AAP reduced to clowns
    15: 'Negative',  # Modi cutout torn — blaming anti-BJP
    16: 'Neutral',   # Factual — BJP cells in Mizoram
    17: 'Negative',  # BJP keeps fooling people
    18: 'Negative',  # BJP fixed election + Bengali race conspiracy
    19: 'Neutral',   # Iran ship — ambiguous credit claim
    20: 'Skip',      # US noise
    21: 'Neutral',   # MIGA joke — ambiguous
    22: 'Neutral',   # BJP not in TN game — humorous analytical
    23: 'Negative',  # BJP destroying Kshatriya community
    24: 'Negative',  # Criticising war allies killing innocents
    25: 'Negative',  # Gas 45 days — BJP IT cell misleading
    26: 'Neutral',   # BJP-PK-CBN political analysis
    27: 'Negative',  # Modi obsessed with optics, Manmohan better
    28: 'Skip',      # US noise
    29: 'Neutral',   # North Indian voting pattern — analytical
    30: 'Neutral',   # Ambiguous
    31: 'Negative',  # Congress lying — sarcastic
    32: 'Negative',  # Sarcastic — BJP provides luxury to be alive
    33: 'Negative',  # Modi idiot on international relations
    34: 'Negative',  # Times of India fake news criticism
    35: 'Negative',  # BJP acting as both opposition and ruling party
    36: 'Negative',  # Opposition Iran appeasement narrative
    37: 'Negative',  # Things will go worse — pessimistic
    38: 'Negative',  # Modi compromised — EVM manipulation
    39: 'Negative',  # EVM manipulation to inflate BJP votes
    40: 'Skip',      # US noise
    41: 'Negative',  # Nehru giant, Modi what? — anti-Modi
    42: 'Negative',  # SHAMEonBJP FM penalising taxpayers
    43: 'Negative',  # BJP will lose WB, TN, Kerala
    44: 'Neutral',   # Ambiguous
    45: 'Negative',  # BJP worker misconduct
    46: 'Positive',  # India leads BRICS — pro-Modi
    47: 'Neutral',   # Sunil Grover never mimicked Modi — observational
    48: 'Neutral',   # Sanatan trishul — religious, no party sentiment
    49: 'Positive',  # Defending Modi — fake video propaganda
    50: 'Skip',      # US noise

    # Tweets 51-100
    51: 'Neutral',   # Mixed — criticism but wants Modi to stay
    52: 'Neutral',   # Prosenjit/BJP/TMC — humorous observation
    53: 'Negative',  # Modi divider in chief
    54: 'Skip',      # US noise
    55: 'Negative',  # BJP RSS over active, home wars
    56: 'Negative',  # Teacher suspended — dangerous signs
    57: 'Neutral',   # Neither Modi nor Congress can solve — balanced
    58: 'Negative',  # Sarcastic — Modi mother of democracy 🤡
    59: 'Negative',  # Corruption rapes rampant, remove BJP
    60: 'Negative',  # BJP game over, govt will fall
    61: 'Negative',  # Mocking BJP andhbhakts
    62: 'Neutral',   # Factual Ladakh/BJP Sixth Schedule analysis
    63: 'Neutral',   # Breaking news — Haryana voting
    64: 'Neutral',   # News headline — mimicry suspension
    65: 'Neutral',   # BJP 272 seats — Telugu analysis
    66: 'Negative',  # Anti-DMK, mixed negative
    67: 'Positive',  # India car sales growth via NaMo App
    68: 'Negative',  # Mocking BJP/RSS — cow piss
    69: 'Neutral',   # Defending Modi on jet — analytical
    70: 'Positive',  # Pro-Modi anti-Congress on terrorism
    71: 'Skip',      # Uganda noise
    72: 'Neutral',   # IAS spying — sarcastic ambiguous
    73: 'Negative',  # Harsh attack on Modi — colonized mindset
    74: 'Skip',      # Potato Congress noise
    75: 'Negative',  # Congress cowardly lions
    76: 'Negative',  # Pro-BJP aligns with Israel killing Palestinians
    77: 'Neutral',   # Political analysis — no party won alliance
    78: 'Negative',  # BJP Epstein Sanskaar — mocking
    79: 'Neutral',   # This is india bro — ambiguous
    80: 'Neutral',   # BJP can pull 2+ seats — analytical
    81: 'Negative',  # BJP IT fake news makers
    82: 'Skip',      # US noise
    83: 'Positive',  # People of India voted BJP — defending
    84: 'Negative',  # BJP RSS biggest chor
    85: 'Neutral',   # Sarcastic list about Modi's knowledge
    86: 'Positive',  # Defending BJP/Modi on oil stability
    87: 'Skip',      # US noise
    88: 'Negative',  # Coward Hindutva fascist BJP — aggressive
    89: 'Skip',      # US noise
    90: 'Negative',  # AIADMK shameless on CAA
    91: 'Neutral',   # Sarcastic Modi/Pushpak Vimana — humorous
    92: 'Neutral',   # Congress Vegetarian Anda — ambiguous
    93: 'Negative',  # BJP workers targeting opposition
    94: 'Positive',  # Defending BJP/Modi on oil — same as 86
    95: 'Negative',  # Shameless fellows — hostile
    96: 'Negative',  # Criticising communal rhetoric
    97: 'Negative',  # BJP fixed election
    98: 'Neutral',   # Vijay/Modi alliance analysis
    99: 'Skip',      # US noise
    100:'Skip',      # US noise

    # Tweets 101-150
    101:'Positive',  # My vote was for BJP not Congress
    102:'Negative',  # EVM sarcastic criticism
    103:'Negative',  # Upper caste show aukat of BJP — hostile
    104:'Negative',  # Self called Gods chosen — mocking
    105:'Neutral',   # Saw this driving — observational
    106:'Negative',  # Living in fear under BJP regime
    107:'Negative',  # Mocking Modi cultists
    108:'Negative',  # BJP alcohol, rape crimes
    109:'Positive',  # Bodoland peace — pro-Modi
    110:'Negative',  # Mocking Congress and AAP
    111:'Neutral',   # Price rollback 2015 — factual
    112:'Neutral',   # Iran ships — ambiguous
    113:'Skip',      # US noise
    114:'Neutral',   # WB Hindus BJP voting — analytical
    115:'Negative',  # Congress incompetent and corrupt
    116:'Negative',  # Congress wouldn't win fair elections
    117:'Negative',  # BJP IT cell boot lickers
    118:'Negative',  # bjpigs — hostile
    119:'Positive',  # TMC hooliganism ending — pro-Modi
    120:'Negative',  # BJP-TN optics bad
    121:'Positive',  # Strong pro-Modi anti-Congress foreign policy
    122:'Skip',      # US noise
    123:'Skip',      # US noise
    124:'Negative',  # Failed Congress journalist
    125:'Negative',  # Modi Yogi Shah go away
    126:'Neutral',   # BJP war ambassador Zionists — observational
    127:'Neutral',   # Lost ur mind TVK — ambiguous
    128:'Negative',  # BJP extremist mobs threatening Muslims
    129:'Negative',  # Anti-Congress on partition blame
    130:'Skip',      # US noise
    131:'Negative',  # Congress stooges false narratives
    132:'Negative',  # Modi liar, optics over accountability
    133:'Negative',  # Congress legalised infiltrators
    134:'Positive',  # Pro-Modi, TMC jungle raj ending
    135:'Negative',  # BJP jumlebaazi, Modi destroyed temples
    136:'Positive',  # Pro-BJP save Bengali Hindus
    137:'Neutral',   # Assam BJP alliance — factual
    138:'Skip',      # US noise
    139:'Skip',      # US noise
    140:'Negative',  # Teacher suspended for mimicking Modi
    141:'Negative',  # Modi Shah won't be around — hostile
    142:'Negative',  # BJP fixed election
    143:'Neutral',   # Kerala Congress seats — factual news
    144:'Negative',  # BJP double engine = only corruption
    145:'Positive',  # Pro-BJP Bengal election appeal
    146:'Negative',  # Criticising Modi/Stalin/Vijayan ads
    147:'Skip',      # US noise
    148:'Skip',      # US noise
    149:'Negative',  # Sarcastic — Modi govt is useless
    150:'Neutral',   # Sarcastic Rahul Gandhi joke — ambiguous
}


def main():
    print("=" * 65)
    print("  Step 3b: Sentiment Model Evaluation")
    print("=" * 65)

    # Load FINAL csv — has both sentiment and corrected_sentiment columns
    df = pd.read_csv("data/political_tweets_final.csv", encoding="utf-8-sig")
    print(f"\n[Load]  {len(df)} tweets loaded from political_tweets_final.csv.")

    # Recreate exact same 150-tweet sample used for manual labelling
    sample = df.sample(150, random_state=42).reset_index(drop=True)
    sample.index += 1  # 1-based to match manual label keys

    # Align manual labels with BOTH model and corrected predictions
    records = []
    skipped = 0
    for idx, manual_label in MANUAL_LABELS.items():
        if manual_label == 'Skip':
            skipped += 1
            continue
        if idx in sample.index:
            records.append({
                "tweet_num"          : idx,
                "manual"             : manual_label,
                "model_raw"          : sample.loc[idx, "sentiment"],
                "model_corrected"    : sample.loc[idx, "corrected_sentiment"],
                "sarcasm_detected"   : sample.loc[idx, "sarcasm_detected"],
                "sarcasm_type"       : sample.loc[idx, "sarcasm_type"],
                "text"               : sample.loc[idx, "clean_text"][:80],
            })

    eval_df = pd.DataFrame(records)
    print(f"[Eval]  {len(eval_df)} tweets used for evaluation.")
    print(f"[Eval]  {skipped} tweets skipped (US/foreign noise).")
    print(f"[Eval]  {eval_df['sarcasm_detected'].sum()} tweets flagged as sarcastic in this sample.\n")

    y_true         = eval_df["manual"].tolist()
    y_pred_before  = eval_df["model_raw"].tolist()
    y_pred_after   = eval_df["model_corrected"].tolist()

    # Metrics — BEFORE
    acc_before    = accuracy_score(y_true, y_pred_before)
    report_before = classification_report(
        y_true, y_pred_before,
        labels=["Positive", "Neutral", "Negative"],
        digits=3
    )
    cm_before = confusion_matrix(
        y_true, y_pred_before,
        labels=["Positive", "Neutral", "Negative"]
    )

    # Metrics — AFTER
    acc_after    = accuracy_score(y_true, y_pred_after)
    report_after = classification_report(
        y_true, y_pred_after,
        labels=["Positive", "Neutral", "Negative"],
        digits=3
    )
    cm_after = confusion_matrix(
        y_true, y_pred_after,
        labels=["Positive", "Neutral", "Negative"]
    )

    # Use after-correction as primary for confusion matrix display
    acc    = acc_after
    report = report_after
    cm     = cm_after

    # Print report
    print("=" * 65)
    print("  EVALUATION REPORT")
    print("=" * 65)
    improvement = (acc_after - acc_before) * 100
    print(f"\n  Sample Size : {len(eval_df)} manually annotated tweets")
    print(f"    - Positive  : {y_true.count('Positive')}")
    print(f"    - Neutral   : {y_true.count('Neutral')}")
    print(f"    - Negative  : {y_true.count('Negative')}")
    print(f"\n  ── Accuracy Comparison ──────────────────────────────")
    print(f"  Before sarcasm correction : {acc_before:.3f} ({acc_before*100:.1f}%)")
    print(f"  After  sarcasm correction : {acc_after:.3f}  ({acc_after*100:.1f}%)")
    print(f"  Improvement               : {improvement:+.1f} percentage points")

    print(f"\n  ── Per-Class Metrics BEFORE ─────────────────────────")
    print(report_before)
    print(f"  ── Per-Class Metrics AFTER ──────────────────────────")
    print(report_after)

    print("  ── Confusion Matrix BEFORE (rows=actual, cols=predicted) ──")
    print(f"  {'':12} {'Positive':>10} {'Neutral':>10} {'Negative':>10}")
    for i, label in enumerate(["Positive", "Neutral", "Negative"]):
        print(f"  {label:12} {cm_before[i][0]:>10} {cm_before[i][1]:>10} {cm_before[i][2]:>10}")

    print(f"\n  ── Confusion Matrix AFTER (rows=actual, cols=predicted) ───")
    print(f"  {'':12} {'Positive':>10} {'Neutral':>10} {'Negative':>10}")
    for i, label in enumerate(["Positive", "Neutral", "Negative"]):
        print(f"  {label:12} {cm_after[i][0]:>10} {cm_after[i][1]:>10} {cm_after[i][2]:>10}")

    # Show misclassified after correction
    wrong_after = eval_df[eval_df["manual"] != eval_df["model_corrected"]]
    print(f"\n  Misclassified after correction: {len(wrong_after)} / {len(eval_df)} tweets")
    print(f"\n  Sample remaining misclassifications:")
    print("  " + "-" * 55)
    for _, row in wrong_after.head(6).iterrows():
        print(f"  [Human: {row['manual']:<8} | Corrected: {row['model_corrected']:<8} | Sarcasm: {row['sarcasm_type']}]")
        print(f"  {row['text']}")
        print()

    # Save to file
    os.makedirs("data", exist_ok=True)
    with open("data/evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write("SENTIMENT MODEL EVALUATION REPORT\n")
        f.write("=" * 65 + "\n")
        f.write(f"Model        : cardiffnlp/twitter-roberta-base-sentiment-latest\n")
        f.write(f"Dataset      : Indian Political Tweets (2026)\n")
        f.write(f"Sample Size  : {len(eval_df)} manually annotated tweets\n")
        f.write(f"Skipped      : {skipped} (US/foreign noise)\n\n")
        f.write(f"Accuracy BEFORE sarcasm correction : {acc_before:.3f} ({acc_before*100:.1f}%)\n")
        f.write(f"Accuracy AFTER  sarcasm correction : {acc_after:.3f} ({acc_after*100:.1f}%)\n")
        f.write(f"Improvement                        : {improvement:+.1f} percentage points\n\n")
        f.write("Per-Class Metrics BEFORE:\n")
        f.write(report_before)
        f.write("\nPer-Class Metrics AFTER:\n")
        f.write(report_after)
        f.write("\nConfusion Matrix BEFORE (rows=actual, cols=predicted):\n")
        f.write(f"{'':12} {'Positive':>10} {'Neutral':>10} {'Negative':>10}\n")
        for i, label in enumerate(["Positive", "Neutral", "Negative"]):
            f.write(f"{label:12} {cm_before[i][0]:>10} {cm_before[i][1]:>10} {cm_before[i][2]:>10}\n")
        f.write("\nConfusion Matrix AFTER (rows=actual, cols=predicted):\n")
        f.write(f"{'':12} {'Positive':>10} {'Neutral':>10} {'Negative':>10}\n")
        for i, label in enumerate(["Positive", "Neutral", "Negative"]):
            f.write(f"{label:12} {cm_after[i][0]:>10} {cm_after[i][1]:>10} {cm_after[i][2]:>10}\n")

    print(f"\n[Save]  ✅ Report saved to: data/evaluation_report.txt")
    print("\n" + "=" * 65)
    print("  ✅ Evaluation complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
