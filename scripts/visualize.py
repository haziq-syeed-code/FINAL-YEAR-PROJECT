"""
=============================================================================
Step 5: Party-wise Aggregation & Visualization
=============================================================================
Project : Twitter Sentiment Analysis with Lightweight Sarcasm Detection
Script  : scripts/visualize.py
Input   : data/political_tweets_final.csv
Output  : dashboard/sentiment_dashboard.png
          dashboard/party_sentiment.png
          dashboard/sarcasm_analysis.png
          dashboard/accuracy_comparison.png
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
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

INPUT_CSV    = "data/political_tweets_final.csv"
OUTPUT_DIR   = "dashboard"

# Color palette — consistent across all charts
COLORS = {
    "Positive"  : "#2ecc71",   # green
    "Neutral"   : "#f39c12",   # amber
    "Negative"  : "#e74c3c",   # red
    "BJP"       : "#FF6B35",   # saffron-orange
    "Congress"  : "#4A90D9",   # congress blue
    "AAP"       : "#8B5CF6",   # purple
    "TMC"       : "#10B981",   # green
    "DMK/AIADMK": "#F59E0B",   # amber
    "bg"        : "#0F1117",   # dark background
    "card"      : "#1A1D27",   # card background
    "text"      : "#E2E8F0",   # light text
    "subtext"   : "#94A3B8",   # subtle text
    "grid"      : "#2D3148",   # grid lines
}

plt.rcParams.update({
    "figure.facecolor"  : COLORS["bg"],
    "axes.facecolor"    : COLORS["card"],
    "axes.edgecolor"    : COLORS["grid"],
    "axes.labelcolor"   : COLORS["text"],
    "xtick.color"       : COLORS["subtext"],
    "ytick.color"       : COLORS["subtext"],
    "text.color"        : COLORS["text"],
    "grid.color"        : COLORS["grid"],
    "grid.linewidth"    : 0.5,
    "font.family"       : "DejaVu Sans",
})


# ─────────────────────────────────────────────────────────────────────────────
# PARTY TAGGING
# ─────────────────────────────────────────────────────────────────────────────

def tag_party(text):
    text = str(text).lower()
    parties = []
    if any(w in text for w in ['bjp', 'modi', 'amit shah', 'yogi', 'nda', 'hindutva', 'rss']):
        parties.append('BJP')
    if any(w in text for w in ['congress', 'rahul', 'kharge', 'upa', 'sonia', 'priyanka']):
        parties.append('Congress')
    if any(w in text for w in ['aap', 'kejriwal', 'aam aadmi']):
        parties.append('AAP')
    if any(w in text for w in ['tmc', 'trinamool', 'mamata', 'didi']):
        parties.append('TMC')
    if any(w in text for w in ['dmk', 'stalin', 'aiadmk', 'eps', 'annamalai']):
        parties.append('DMK/AIADMK')
    return parties if parties else ['Other']


# ─────────────────────────────────────────────────────────────────────────────
# LOAD & PREPARE DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Party tagging — one tweet can have multiple tags
    df["parties"] = df["clean_text"].apply(tag_party)
    df_exploded = df.explode("parties")

    # Only keep main parties for party-level analysis
    main_parties = ["BJP", "Congress", "AAP", "TMC", "DMK/AIADMK"]
    df_parties = df_exploded[df_exploded["parties"].isin(main_parties)].copy()

    return df, df_parties


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — Overall Sentiment Distribution (Donut)
# ─────────────────────────────────────────────────────────────────────────────

def chart_overall_sentiment(df, ax):
    counts = df["corrected_sentiment"].value_counts()
    labels = ["Negative", "Neutral", "Positive"]
    sizes  = [counts.get(l, 0) for l in labels]
    colors = [COLORS[l] for l in labels]
    total  = sum(sizes)

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        wedgeprops={"linewidth": 3, "edgecolor": COLORS["bg"], "width": 0.55},
    )

    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
        at.set_color(COLORS["text"])

    # Center text
    ax.text(0, 0.1, str(total), ha="center", va="center",
            fontsize=22, fontweight="bold", color=COLORS["text"])
    ax.text(0, -0.2, "tweets", ha="center", va="center",
            fontsize=10, color=COLORS["subtext"])

    # Legend
    patches = [mpatches.Patch(color=COLORS[l], label=f"{l}  {counts.get(l,0)}") for l in labels]
    ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.15),
              ncol=3, frameon=False, fontsize=9)

    ax.set_title("Overall Sentiment\n(After Sarcasm Correction)",
                 fontsize=12, fontweight="bold", pad=15, color=COLORS["text"])


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — Party-wise Sentiment Stacked Bar
# ─────────────────────────────────────────────────────────────────────────────

def chart_party_sentiment(df_parties, ax):
    parties     = ["BJP", "Congress", "TMC", "DMK/AIADMK", "AAP"]
    sentiments  = ["Positive", "Neutral", "Negative"]
    bar_width   = 0.55

    # Build percentage matrix
    data = {}
    for party in parties:
        subset = df_parties[df_parties["parties"] == party]
        total  = len(subset)
        if total == 0:
            data[party] = [0, 0, 0]
            continue
        data[party] = [
            subset[subset["corrected_sentiment"] == s].shape[0] / total * 100
            for s in sentiments
        ]

    bottoms = [0] * len(parties)
    for i, sentiment in enumerate(sentiments):
        values = [data[p][i] for p in parties]
        bars = ax.bar(parties, values, bar_width,
                      bottom=bottoms,
                      color=COLORS[sentiment],
                      label=sentiment,
                      alpha=0.92)

        # Add percentage labels inside bars
        for j, (bar, val) in enumerate(zip(bars, values)):
            if val > 6:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bottoms[j] + val / 2,
                        f"{val:.0f}%",
                        ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white")
        bottoms = [b + v for b, v in zip(bottoms, values)]

    # Tweet count labels on top
    for i, party in enumerate(parties):
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

    # Color party labels
    party_colors = [COLORS.get(p, COLORS["text"]) for p in parties]
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

    # Sort descending
    counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    counts = {k: v for k, v in counts.items() if v > 0}

    labels = list(counts.keys())
    values = list(counts.values())

    # Horizontal bar chart
    bar_colors = [
        "#e74c3c", "#f39c12", "#3498db", "#9b59b6",
        "#1abc9c", "#e67e22", "#2ecc71", "#e91e63"
    ][:len(labels)]

    bars = ax.barh(labels[::-1], values[::-1], color=bar_colors[::-1],
                   alpha=0.85, height=0.6)

    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
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
# CHART 4 — Accuracy Before vs After (grouped bar)
# ─────────────────────────────────────────────────────────────────────────────

def chart_accuracy_comparison(ax):
    metrics = ["Overall\nAccuracy", "Negative\nF1", "Neutral\nF1", "Positive\nF1"]
    before  = [66.7, 76.1, 60.6, 23.5]
    after   = [76.0, 86.1, 66.7, 25.0]

    x      = np.arange(len(metrics))
    width  = 0.35

    bars1 = ax.bar(x - width/2, before, width, label="Before Sarcasm Correction",
                   color="#4A5568", alpha=0.85)
    bars2 = ax.bar(x + width/2, after,  width, label="After Sarcasm Correction",
                   color="#4A90D9", alpha=0.92)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontsize=9, color=COLORS["subtext"])

    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=COLORS["text"])

    # Improvement arrows
    for i, (b, a) in enumerate(zip(before, after)):
        diff = a - b
        ax.annotate(f"+{diff:.1f}",
                    xy=(x[i] + width/2, a + 2),
                    fontsize=8, color="#2ecc71",
                    fontweight="bold", ha="center")

    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.set_title("Model Performance: Before vs After Sarcasm Correction\n(Evaluated on 129 manually annotated tweets)",
                 fontsize=12, fontweight="bold", pad=15)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 5 — Sentiment Trend Over Time
# ─────────────────────────────────────────────────────────────────────────────

def chart_sentiment_trend(df, ax):
    df_time = df.dropna(subset=["date"]).copy()
    df_time["hour"] = df_time["date"].dt.floor("h")

    # Group by hour and sentiment
    grouped = df_time.groupby(["hour", "corrected_sentiment"]).size().unstack(fill_value=0)

    for sent in ["Positive", "Neutral", "Negative"]:
        if sent in grouped.columns:
            ax.plot(grouped.index, grouped[sent],
                    color=COLORS[sent], linewidth=1.5,
                    label=sent, alpha=0.85)
            ax.fill_between(grouped.index, grouped[sent],
                            color=COLORS[sent], alpha=0.08)

    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Tweet Count", fontsize=10)
    ax.set_title("Sentiment Trend Over Time",
                 fontsize=12, fontweight="bold", pad=15)
    ax.legend(frameon=False, fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Format x-axis
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=8)


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

    # Layout: 3 rows
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           top=0.92, bottom=0.06,
                           left=0.06, right=0.97)

    # Row 1: Overall donut | Party stacked bar (spans 2 cols)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1:])

    # Row 2: Sarcasm rules | Accuracy comparison
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[1, 2])

    # Row 3: Trend (full width)
    ax5 = fig.add_subplot(gs[2, :])

    chart_overall_sentiment(df, ax1)
    chart_party_sentiment(df_parties, ax2)
    chart_sarcasm_rules(df, ax3)
    chart_accuracy_comparison(ax4)
    chart_sentiment_trend(df, ax5)

    # Footer
    fig.text(0.5, 0.01,
             "Model: cardiffnlp/twitter-roberta-base-sentiment-latest  ·  "
             "Accuracy: 66.7% → 76.0% (+9.3pp after sarcasm correction)  ·  "
             "Sarcasm detected: 8.6% of tweets",
             ha="center", fontsize=8, color=COLORS["subtext"])

    out_path = os.path.join(OUTPUT_DIR, "sentiment_dashboard.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
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
    for party in ["BJP", "Congress", "TMC", "DMK/AIADMK", "AAP"]:
        n = len(df_parties[df_parties["parties"] == party])
        print(f"         {party:<12}: {n} tweets")

    print(f"\n[Chart] Building dashboard...")
    build_dashboard(df, df_parties)

    print("\n" + "=" * 65)
    print("  ✅ Step 5 complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
