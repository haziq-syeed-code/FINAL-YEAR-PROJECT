"""
=============================================================================
Step 5: Party-wise Aggregation & Visualization
=============================================================================
Project : Twitter Sentiment Analysis with Lightweight Sarcasm Detection
Script  : scripts/visualize.py
Input   : data/political_tweets_final.csv
Output  : dashboard/sentiment_dashboard.png
=============================================================================

INSTALL
-------
    pip install matplotlib seaborn

RUN
---
    python scripts/visualize.py
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

INPUT_CSV  = "data/political_tweets_final.csv"
OUTPUT_DIR = "dashboard"

COLORS = {
    "Positive"   : "#2ecc71",
    "Neutral"    : "#f39c12",
    "Negative"   : "#e74c3c",
    "BJP"        : "#FF6B35",
    "Congress"   : "#4A90D9",
    "AAP"        : "#8B5CF6",
    "TMC"        : "#10B981",
    "DMK/AIADMK" : "#F59E0B",
    "bg"         : "#0F1117",
    "card"       : "#1A1D27",
    "text"       : "#E2E8F0",
    "subtext"    : "#94A3B8",
    "grid"       : "#2D3148",
}

plt.rcParams.update({
    "figure.facecolor" : COLORS["bg"],
    "axes.facecolor"   : COLORS["card"],
    "axes.edgecolor"   : COLORS["grid"],
    "axes.labelcolor"  : COLORS["text"],
    "xtick.color"      : COLORS["subtext"],
    "ytick.color"      : COLORS["subtext"],
    "text.color"       : COLORS["text"],
    "grid.color"       : COLORS["grid"],
    "grid.linewidth"   : 0.5,
    "font.family"      : "DejaVu Sans",
})

MAIN_PARTIES = ["BJP", "Congress", "TMC", "DMK/AIADMK", "AAP"]


# ─────────────────────────────────────────────────────────────────────────────
# PARTY TAGGING
# ─────────────────────────────────────────────────────────────────────────────

def tag_party(text):
    t = str(text).lower()
    parties = []
    if any(w in t for w in ['bjp','modi','amit shah','yogi','nda','hindutva','rss']):
        parties.append('BJP')
    if any(w in t for w in ['congress','rahul','kharge','upa','sonia','priyanka']):
        parties.append('Congress')
    if any(w in t for w in ['aap','kejriwal','aam aadmi']):
        parties.append('AAP')
    if any(w in t for w in ['tmc','trinamool','mamata','didi']):
        parties.append('TMC')
    if any(w in t for w in ['dmk','stalin','aiadmk','eps','annamalai']):
        parties.append('DMK/AIADMK')
    return parties if parties else ['Other']


# ─────────────────────────────────────────────────────────────────────────────
# LOAD & PREPARE DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["parties"] = df["clean_text"].apply(tag_party)
    df_exp = df.explode("parties")
    df_parties = df_exp[df_exp["parties"].isin(MAIN_PARTIES)].copy()
    return df, df_parties


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — Overall Sentiment Donut
# ─────────────────────────────────────────────────────────────────────────────

def chart_overall_sentiment(df, ax):
    counts = df["corrected_sentiment"].value_counts()
    labels = ["Negative", "Neutral", "Positive"]
    sizes  = [counts.get(l, 0) for l in labels]
    colors = [COLORS[l] for l in labels]
    total  = sum(sizes)

    wedges, _, autotexts = ax.pie(
        sizes, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops={"linewidth": 3, "edgecolor": COLORS["bg"], "width": 0.55},
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
        at.set_color(COLORS["text"])

    ax.text(0, 0.1, str(total), ha="center", va="center",
            fontsize=22, fontweight="bold", color=COLORS["text"])
    ax.text(0, -0.2, "tweets", ha="center", va="center",
            fontsize=10, color=COLORS["subtext"])

    patches = [mpatches.Patch(color=COLORS[l], label=f"{l}  {counts.get(l,0)}")
               for l in labels]
    ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.15),
              ncol=3, frameon=False, fontsize=9)
    ax.set_title("Overall Sentiment\n(After Sarcasm Correction)",
                 fontsize=12, fontweight="bold", pad=15, color=COLORS["text"])


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — Party-wise Sentiment Stacked Bar
# ─────────────────────────────────────────────────────────────────────────────

def chart_party_sentiment(df_parties, ax):
    sentiments = ["Positive", "Neutral", "Negative"]
    bar_width  = 0.55

    data = {}
    for party in MAIN_PARTIES:
        subset = df_parties[df_parties["parties"] == party]
        total  = len(subset)
        if total == 0:
            data[party] = [0, 0, 0]
            continue
        data[party] = [
            subset[subset["corrected_sentiment"] == s].shape[0] / total * 100
            for s in sentiments
        ]

    bottoms = [0] * len(MAIN_PARTIES)
    for i, sentiment in enumerate(sentiments):
        values = [data[p][i] for p in MAIN_PARTIES]
        bars = ax.bar(MAIN_PARTIES, values, bar_width,
                      bottom=bottoms, color=COLORS[sentiment],
                      label=sentiment, alpha=0.92)
        for j, (bar, val) in enumerate(zip(bars, values)):
            if val > 6:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bottoms[j] + val / 2,
                        f"{val:.0f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
        bottoms = [b + v for b, v in zip(bottoms, values)]

    for i, party in enumerate(MAIN_PARTIES):
        count = len(df_parties[df_parties["parties"] == party])
        ax.text(i, 102, f"n={count}", ha="center", va="bottom",
                fontsize=8, color=COLORS["subtext"])

    ax.set_ylim(0, 115)
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.set_title("Sentiment by Political Party",
                 fontsize=12, fontweight="bold", pad=15)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    party_colors = [COLORS.get(p, COLORS["text"]) for p in MAIN_PARTIES]
    for tick, color in zip(ax.get_xticklabels(), party_colors):
        tick.set_color(color)
        tick.set_fontweight("bold")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — Sarcasm Detection Rule Breakdown
# ─────────────────────────────────────────────────────────────────────────────

def chart_sarcasm_rules(df, ax):
    rules = {
        "Indian Vocab"   : "indian_vocab",
        "Punctuation"    : "punctuation",
        "Election Fraud" : "election_fraud",
        "Sarcasm Emoji"  : "sarcasm_emoji",
        "So Called"      : "so_called",
        "Rhetorical"     : "rhetorical",
        "Ironic Praise"  : "ironic_compliment",
        "Hashtag"        : "hashtag",
    }

    counts = {}
    for label, key in rules.items():
        counts[label] = df["sarcasm_type"].str.contains(key, na=False).sum()

    counts = {k: v for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True) if v > 0}
    labels = list(counts.keys())
    values = list(counts.values())

    bar_colors = ["#e74c3c","#f39c12","#3498db","#9b59b6",
                  "#1abc9c","#e67e22","#2ecc71","#e91e63"][:len(labels)]

    bars = ax.barh(labels[::-1], values[::-1],
                   color=bar_colors[::-1], alpha=0.85, height=0.6)

    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", ha="left",
                fontsize=10, fontweight="bold", color=COLORS["text"])

    ax.set_xlabel("Tweets Detected", fontsize=10)
    ax.set_title("Sarcasm Detection by Rule",
                 fontsize=12, fontweight="bold", pad=15)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — Accuracy Before vs After
# ─────────────────────────────────────────────────────────────────────────────

def chart_accuracy_comparison(ax):
    metrics = ["Overall\nAccuracy", "Negative\nF1", "Neutral\nF1", "Positive\nF1"]
    before  = [66.7, 76.1, 60.6, 23.5]
    after   = [75.2, 86.1, 66.7, 25.0]

    x     = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, before, width,
                   label="Before", color="#4A5568", alpha=0.85)
    bars2 = ax.bar(x + width/2, after, width,
                   label="After", color="#4A90D9", alpha=0.92)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontsize=8, color=COLORS["subtext"])

    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=COLORS["text"])

    for i, (b, a) in enumerate(zip(before, after)):
        ax.annotate(f"+{a-b:.1f}",
                    xy=(x[i] + width/2, a + 2.5),
                    fontsize=8, color="#2ecc71",
                    fontweight="bold", ha="center")

    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.set_title("Model Performance\nBefore vs After Sarcasm Correction",
                 fontsize=12, fontweight="bold", pad=15)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 5 — Sarcasm % by Party (REPLACES trend chart)
# Shows what percentage of each party's tweets were detected as sarcastic
# and how many had their sentiment corrected as a result
# ─────────────────────────────────────────────────────────────────────────────

def chart_sarcasm_by_party(df_parties, ax):
    parties = ["BJP", "Congress", "TMC", "DMK/AIADMK", "AAP"]

    sarc_pct      = []
    corrected_pct = []
    counts        = []

    for party in parties:
        subset = df_parties[df_parties["parties"] == party]
        total  = len(subset)
        sarc   = subset["sarcasm_detected"].sum()
        corr   = (subset["sentiment"] != subset["corrected_sentiment"]).sum()
        sarc_pct.append(sarc / total * 100 if total > 0 else 0)
        corrected_pct.append(corr / total * 100 if total > 0 else 0)
        counts.append(total)

    x     = np.arange(len(parties))
    width = 0.38

    bars1 = ax.bar(x - width/2, sarc_pct, width,
                   label="Sarcasm Detected %",
                   color=[COLORS.get(p, "#888") for p in parties],
                   alpha=0.85)

    bars2 = ax.bar(x + width/2, corrected_pct, width,
                   label="Sentiment Corrected %",
                   color=[COLORS.get(p, "#888") for p in parties],
                   alpha=0.45, hatch="//")

    # Value labels
    for bar in bars1:
        if bar.get_height() > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}%",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=COLORS["text"])

    for bar in bars2:
        if bar.get_height() > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}%",
                    ha="center", va="bottom",
                    fontsize=8, color=COLORS["subtext"])

    # Tweet count below x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{p}\n(n={c})" for p, c in zip(parties, counts)],
        fontsize=9
    )
    for tick, party in zip(ax.get_xticklabels(), parties):
        tick.set_color(COLORS.get(party, COLORS["text"]))
        tick.set_fontweight("bold")

    ax.set_ylim(0, max(sarc_pct) * 1.35)
    ax.set_ylabel("Percentage of Party Tweets (%)", fontsize=10)
    ax.set_title("Sarcasm Detection Rate by Political Party\n"
                 "Solid = tweets flagged as sarcastic  ·  Hatched = sentiment corrected",
                 fontsize=12, fontweight="bold", pad=15)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Insight annotation
    max_party = parties[sarc_pct.index(max(sarc_pct))]
    ax.annotate(
        f"BJP & AAP have highest\nsarcasm rate (~11.6%)",
        xy=(0, max(sarc_pct)), xytext=(1.5, max(sarc_pct) * 1.15),
        fontsize=8, color=COLORS["subtext"],
        arrowprops=dict(arrowstyle="->", color=COLORS["subtext"], lw=0.8),
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard(df, df_parties):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig = plt.figure(figsize=(20, 14), facecolor=COLORS["bg"])
    fig.suptitle(
        "Indian Political Tweet Sentiment Analysis  ·  2026\n"
        "Lightweight Sarcasm Detection Pipeline  ·  n=2,313 tweets",
        fontsize=16, fontweight="bold", color=COLORS["text"], y=0.98
    )

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.50, wspace=0.35,
                           top=0.92, bottom=0.06,
                           left=0.06, right=0.97)

    # Row 1: Donut | Party stacked bar (spans 2 cols)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1:])

    # Row 2: Sarcasm rules | Accuracy comparison
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[1, 2])

    # Row 3: Sarcasm by party (full width — replaces trend chart)
    ax5 = fig.add_subplot(gs[2, :])

    chart_overall_sentiment(df, ax1)
    chart_party_sentiment(df_parties, ax2)
    chart_sarcasm_rules(df, ax3)
    chart_accuracy_comparison(ax4)
    chart_sarcasm_by_party(df_parties, ax5)

    fig.text(
        0.5, 0.01,
        "Model: cardiffnlp/twitter-roberta-base-sentiment-latest  ·  "
        "Accuracy: 66.7% → 75.2% (+8.5pp after sarcasm correction)  ·  "
        "Sarcasm detected in 8.6% of tweets",
        ha="center", fontsize=8, color=COLORS["subtext"]
    )

    out_path = os.path.join(OUTPUT_DIR, "sentiment_dashboard.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"[Save]  ✅ Dashboard saved to: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Step 5: Aggregation & Visualization")
    print("=" * 65)

    print(f"\n[Load]  Reading '{INPUT_CSV}'...")
    df, df_parties = load_data()
    print(f"[Load]  ✅ {len(df)} tweets loaded.")

    print(f"\n[Party] Distribution:")
    for party in MAIN_PARTIES:
        n = len(df_parties[df_parties["parties"] == party])
        sarc = df_parties[df_parties["parties"] == party]["sarcasm_detected"].sum()
        print(f"         {party:<12}: {n} tweets, {sarc} sarcastic ({sarc/n*100:.1f}%)")

    print(f"\n[Chart] Building dashboard...")
    build_dashboard(df, df_parties)

    print("\n" + "=" * 65)
    print("  ✅ Step 5 complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()