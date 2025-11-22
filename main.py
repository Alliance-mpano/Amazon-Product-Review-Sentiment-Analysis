# requirements: pandas, numpy, matplotlib, seaborn, nltk, vaderSentiment, wordcloud, adjustText

#%%
import os
import numpy as np
import pandas as pd
from adjustText import adjust_text
import seaborn as sns
import matplotlib.pyplot as plt
import nltk, string
from nltk.corpus import stopwords
nltk.download('stopwords')
from wordcloud import WordCloud
from textwrap import shorten
from matplotlib.lines import Line2D

#%%
sns.set_theme(
    style="whitegrid",    # clean grid background
    context="talk",       # larger fonts (great for presentation)
    palette="Blues_r"     # consistent color scheme
)

# setting default figure size & font scaling
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.titlepad'] = 15
#%%

df = pd.read_csv('data/amazon_reviews.csv')
df = df[['reviews.rating', 'reviews.text', 'reviews.title', 'categories', 'brand']]
df.head()
# %%



stop = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = [w for w in text.split() if w not in stop]
    return " ".join(words)

df['cleaned_reviews'] = df['reviews.text'].fillna('').apply(clean_text)
df.head()
# %%

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
df['sentiment_scores'] = df['cleaned_reviews'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

def label_sentiment(score):
    if score >= 0.05: return 'positive'
    elif score <= -0.05: return 'negative'
    else: return 'neutral'

df['sentiment_label'] = df['sentiment_scores'].apply(label_sentiment)
df['sentiment_label'].value_counts(normalize=True)
# %%


sns.countplot(data=df, x = 'sentiment_label', order = ['positive', 'neutral', 'negative'])
plt.title('Sentiment Distribution of Amazon Product Reviews')
plt.savefig("outputs/sentiment_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

sns.barplot(data = df, x = 'reviews.rating', y = 'sentiment_scores')
plt.title('Average Sentiment Scores by Review Ratings')
plt.savefig("outputs/avg_sentiment_by_rating.png", dpi=300, bbox_inches="tight")
plt.show()
df.head()
# %%


for label in ['positive','negative']:
    texts = df.loc[df['sentiment_label'] == label, 'cleaned_reviews'].dropna().astype(str)
    text = " ".join(texts)
    if not text.strip():
        # skip generating a word cloud for empty text
        continue
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Top Words in {label} Reviews")
    plt.savefig(f"outputs/top_words_{label}.png", dpi=300, bbox_inches="tight")
    plt.show()

# %%
# Summary statistics by category and brand

category_summary = df.groupby('categories')['sentiment_scores'].mean().sort_values(ascending=False).head(10)
print(category_summary)

brand_summary = df.groupby('brand')['sentiment_scores'].mean().sort_values(ascending=False).head(16)
print(brand_summary)
# %%
print("Sentiment Analysis Summary:")
print("Total reviews analyzed:", len(df))
print("Positive reviews:", (df['sentiment_label']=='positive').sum())
print("Negative reviews:", (df['sentiment_label']=='negative').sum())
print("Neutral reviews:", (df['sentiment_label']=='neutral').sum())
print("Average sentiment score:", df['sentiment_scores'].mean().round(3))
# %%
print("Top 5 Positive Reviews:")
top_positive = df.sort_values(by='sentiment_scores', ascending=False).head(5)
for index, row in top_positive.iterrows():
    print(f"Rating: {row['reviews.rating']}, Score: {row['sentiment_scores']:.3f}")
    print(f"Title: {row['reviews.title']}")
    print(f"Review: {row['reviews.text']}\n")
print("Top 5 Negative Reviews:")
top_negative = df.sort_values(by='sentiment_scores').head(5)
for index, row in top_negative.iterrows():
    print(f"Rating: {row['reviews.rating']}, Score: {row['sentiment_scores']:.3f}")
    print(f"Title: {row['reviews.title']}")
    print(f"Review: {row['reviews.text']}\n")

# %%
# print("Correlation between sentiment and review ratings")
# x = df['reviews.rating']
# y = df['sentiment_scores']
# corr = x.corr(y)
# print(f"Correlation coefficient: {corr:.3f}")
# plt.title('Sentiment Scores vs Review Ratings')
# plt.scatter(x,y,alpha=0.5)
# plt.plot(np.unique(x), 
#          np.poly1d(np.polyfit(x, y, 1))
#          (np.unique(x)), color='red')
# plt.xlabel('Review Ratings')
# plt.ylabel('Sentiment Scores')

print("Correlation between sentiment and review ratings")
# pairwise drop NaNs for a fair correlation
corr = df['reviews.rating'].corr(df['sentiment_scores'])
print(f"Correlation coefficient: {corr:.3f}")

plt.figure(figsize=(9,6))
sns.regplot(
    data=df,
    x='reviews.rating', y='sentiment_scores',
    scatter_kws={'alpha':0.5, 's':40},
    line_kws={'color':'red', 'lw':2}
)
plt.title('Sentiment Scores vs Review Ratings')
plt.xlabel('Review Ratings')
plt.ylabel('Sentiment Scores')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/sentiment_by_rating.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
#correlation by category
#one might find, for example, that:

# Electronics have lower correlation (delivery/technical complaints).

# Books or Beauty categories have higher correlation (emotional reviews match ratings better).

# That makes the insight even richer
df = df.dropna(subset=['categories', 'reviews.rating', 'sentiment_scores'])
# Calculate correlation between sentiment scores and review ratings for each category
category_corr = (
    df.groupby('categories')[['reviews.rating', 'sentiment_scores']]
    .corr().unstack().iloc[:, 1]
    .sort_values(ascending=False)
    .rename('correlation_sentiment_rating')
)
category_corr.head(10)
#%%
def plot_clean_bar(series, title, color_palette="Blues_r", truncate=40):
    # ensure output directory exists
    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(10,6))
    sns.barplot(
        x=series.values,
        y=[cat[:truncate] + ('...' if len(cat) > truncate else '') for cat in series.index],
        palette=color_palette
    )
    plt.title(title, fontsize=13, weight='bold', pad=15)
    plt.xlabel("Value")
    plt.ylabel("")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # sanitize title to create a safe filename
    safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).rstrip()
    safe_filename = safe_title.replace(" ", "_")
    plt.savefig(f"outputs/{safe_filename}.png", dpi=300, bbox_inches="tight")
    plt.show()
# %%
#Average sentiment and rating by category
category_summary = (
    df.groupby('categories')
    .agg(avg_sentiment=('sentiment_scores', 'mean'),
         avg_rating=('reviews.rating', 'mean'),
         count=('sentiment_scores', 'size'))
    .sort_values('avg_sentiment', ascending=False)
)
print(category_summary
)

category_summary.head(10)
# %%
# Visualize top and bottom correlations
top_corr = category_corr.dropna().head(10)
low_corr = category_corr.dropna().tail(10)

#Tells where ratings don't reflect emotion - a user might rate a product highly but express frustration in the review text, or vice versa.
#Great discussion point for customer experience teams to identify such gaps.
#
plot_clean_bar(top_corr, "Top 10 Categories: Highest Sentiment–Rating Correlation") #What people say align with ratings
# what people say does not align with ratings
plot_clean_bar(low_corr, "Bottom 10 Categories: Lowest Sentiment–Rating Correlation")

# %%
brand_corr = (
    df.groupby('brand')[['reviews.rating', 'sentiment_scores']]
    .corr().unstack().iloc[:, 1]
    .sort_values(ascending=False)
    .rename("corr_sentiment_rating")
)

brand_corr.head(10)
# %%
brand_summary = (
    df.groupby('brand')
    .agg(avg_sentiment=('sentiment_scores', 'mean'),
         avg_rating=('reviews.rating', 'mean'),
         count=('sentiment_scores', 'size'))
    .sort_values('avg_sentiment', ascending=False)
)
brand_summary.head(10)
# %%
top_brands = brand_corr.dropna().head(10)
plot_clean_bar(top_brands, "Top 10 Brands: Highest Sentiment–Rating Correlation")
print(f"Sample size: {df[['reviews.rating','sentiment_scores']].dropna().shape[0]} pairs used for correlation")


#%%
# 1) Prepare data safely
category_scatter = category_summary.reset_index()[["categories","avg_rating","avg_sentiment","count"]].dropna()

x = category_scatter["avg_rating"].values
y = category_scatter["avg_sentiment"].values

# robust linear fit
if len(category_scatter) >= 2:
    m, b = np.polyfit(x, y, 1)
else:
    m, b = 0.0, float(np.mean(y)) if len(y) else 0.0

category_scatter["predicted_sentiment"] = m * category_scatter["avg_rating"] + b
category_scatter["residual"] = category_scatter["avg_sentiment"] - category_scatter["predicted_sentiment"]

# filter tiny categories to reduce clutter
dfp = category_scatter[category_scatter["count"] > 30].copy()

# label top K positive/negative residuals
K = 4
top_pos = dfp.nlargest(K, "residual").copy()
top_neg = dfp.nsmallest(K, "residual").copy()

# 2) Plot
plt.figure(figsize=(10.8, 6.6))

sc = sns.scatterplot(data=dfp, x="avg_rating", y="avg_sentiment",
                     hue="residual", palette="coolwarm", s=80, alpha=0.9, edgecolor="none")

# regression line (expected sentiment baseline)
x_min, x_max = x.min(), x.max()
plt.plot([x_min, x_max], [m*x_min + b, m*x_max + b], color="0.25", lw=1.6)

# titles / labels
plt.title("Amazon Categories: Sentiment vs Rating (Residual Analysis)", fontsize=14, weight="bold")
plt.xlabel("Average Star Rating")
plt.ylabel("Average Sentiment (VADER Compound)")
plt.grid(True, linestyle="--", alpha=0.35)

# legend outside
plt.legend(title="Residual (Δ Sentiment)", bbox_to_anchor=(1.02, 0.5), loc="center left", frameon=True)

# small caption explaining colors
plt.figtext(0.5, -0.04,
            "Red = Happier than expected   |   Blue = Less happy than expected",
            ha="center", fontsize=10, color="gray")

# 3) Annotate extremes
def _annotate_block(df_block, color):
    for _, r in df_block.iterrows():
        label = shorten(str(r["categories"]), width=28, placeholder="…")
        plt.annotate(label,
                     (r["avg_rating"], r["avg_sentiment"]),
                     xytext=(8, 8), textcoords="offset points",
                     fontsize=9, weight="bold",
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.95),
                     arrowprops=dict(arrowstyle="-", lw=0.7, color=color, alpha=0.85))

_annotate_block(top_pos, color="#b22222")  # positive residuals (overperformers)
_annotate_block(top_neg, color="#1f77b4")  # negative residuals (underperformers)

plt.tight_layout()
plt.savefig("outputs/residual_scatter_clean.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()
#%%
# --- Residual distribution plot ---
plt.figure(figsize=(8,5))
sns.histplot(dfp["residual"], bins=20, kde=True, color="steelblue")
plt.title("Distribution of Category Residuals (Δ Sentiment)", fontsize=13, weight="bold")
plt.xlabel("Residual (Actual – Predicted Sentiment)")
plt.ylabel("Number of Categories")
plt.axvline(0, color="red", linestyle="--", label="Expected sentiment baseline")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/residual_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# quick summary statistics
print("Residual Summary:")
print(dfp['residual'].describe())
print("\nCount of categories:")
print("Negative residuals:", (dfp['residual'] < 0).sum())
print("Positive residuals:", (dfp['residual'] > 0).sum())

# %%
#a small table summarizing the top overperforming and underperforming categories
summary_tbl = (
    pd.concat([top_pos.assign(flag="Overperforming (+)"),
               top_neg.assign(flag="Underperforming (−)")])
    [["flag","categories","avg_rating","avg_sentiment","predicted_sentiment","residual"]]
    .round(3)
    .sort_values(["flag","residual"], ascending=[False, False])
)
summary_tbl
summary_tbl.to_csv("outputs/category_performance_summary.csv", index=False)
# %%
if __name__ == "__main__":
    print("Run all cells in order to reproduce figures in outputs/")

# %%
df.head()
# %%
x = df["reviews.rating"]
y = df["sentiment_scores"]
m1, b1 = np.polyfit(x, y, 1)
df["predicted_sentiment"] = m1 * x + b1
df["residual"] = y - df["predicted_sentiment"]

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='reviews.rating', y='residual',  palette=sns.color_palette("coolwarm", 5))
plt.title("Distribution of Residuals by Review Rating", fontsize=13, weight='bold')
plt.xlabel("Review Rating (Stars)")
plt.ylabel("Residual (Actual – Predicted Sentiment)")
plt.axhline(0, color='black', linestyle='--', lw=1)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/residuals_by_rating.png", dpi=300, bbox_inches="tight")
plt.show()

#quick summary
print(df.groupby('reviews.rating')['residual'].describe().round(3))
# %%
