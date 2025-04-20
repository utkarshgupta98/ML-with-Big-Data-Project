import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import plotly.graph_objects as go
import networkx as nx

# Setup
st.set_page_config(page_title="Palate Patterns: Amazon Reviews", layout="wide")
st.title("📊 Palate Patterns: Big Data Insights from Amazon Fine Food Reviews")

# --- SECTION 1: Ratings & Reviews ---
st.header("⭐ Ratings & Review Analysis")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Score Distribution")
    score_df = pd.read_csv("score_distribution.csv")
    st.bar_chart(score_df.set_index("Score"))

with col2:
    st.subheader("Average Review Length by Score")
    review_len_df = pd.read_csv("review_length_stats.csv")
    st.line_chart(review_len_df.set_index("Score"))

st.divider()

# --- SECTION 2: Sentiment Analysis ---
st.header("😊 Sentiment Distribution")
sent_df = pd.read_csv("sentiment_distribution.csv")
fig, ax = plt.subplots()
sns.barplot(data=sent_df, x="Sentiment", y="count", palette="coolwarm", ax=ax)
st.pyplot(fig)

st.divider()

# --- SECTION 3: Model Evaluation Metrics ---
st.header("📊 Sentiment Classifier Performance")
with open("model_metrics.json") as f:
    metrics = json.load(f)

metric_cols = st.columns(len(metrics))
for i, (metric, value) in enumerate(metrics.items()):
    metric_cols[i].metric(label=metric.capitalize(), value=f"{value:.2f}")

st.divider()

# --- SECTION 4: Frequent Patterns ---
st.header("🔁 Frequent Pattern Mining")

st.subheader("Top Frequent Itemsets")
freq_df = pd.read_csv("freq_itemsets_final.csv")
st.dataframe(freq_df.head(10), use_container_width=True)

st.subheader("Association Rules Graph")
assoc_df = pd.read_csv("assoc_rules_csv_final.csv")
assoc_df = assoc_df[assoc_df["antecedent"].notna() & assoc_df["consequent"].notna()]

G = nx.DiGraph()
for _, row in assoc_df.iterrows():
    src = row["antecedent"]
    tgt = row["consequent"]
    conf = round(row["confidence"], 2)
    if src and tgt:
        G.add_edge(src, tgt, weight=conf)

pos = nx.spring_layout(G, k=0.6)
edge_x, edge_y = [], []
for u, v in G.edges():
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

node_x, node_y, text = [], [], []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    text.append(node)

fig = go.Figure()
fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                         line=dict(width=1, color='gray'), hoverinfo='none'))
fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=text,
                         marker=dict(size=10, color='skyblue'), textposition='top center'))
fig.update_layout(title='Association Rules Network', showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# ---------- SECTION 5: Jaccard Similarity of Words ----------
st.divider()
st.header("🔗 Jaccard Similarity Network (Word Co-occurrence)")

try:
    jaccard_df = pd.read_csv("jaccard_similarity_top.csv")  # Ensure this file exists

    G_jaccard = nx.Graph()
    for _, row in jaccard_df.iterrows():
        word1 = row["word1"]
        word2 = row["word2"]
        score = row["jaccard"]
        G_jaccard.add_edge(word1, word2, weight=score)

    pos = nx.spring_layout(G_jaccard, k=0.5, seed=42)
    edge_x, edge_y = [], []
    for u, v in G_jaccard.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x, node_y, labels = [], [], []
    for node in G_jaccard.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        labels.append(node)

    fig_jaccard = go.Figure()
    fig_jaccard.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                     line=dict(width=1, color='gray'), hoverinfo='none'))
    fig_jaccard.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
                                     marker=dict(size=10, color='lightgreen'),
                                     text=labels, textposition="top center"))
    fig_jaccard.update_layout(title="Top Jaccard Similarity Network", showlegend=False)
    st.plotly_chart(fig_jaccard, use_container_width=True)

except Exception as e:
    st.warning("⚠️ Jaccard Similarity file not found. Please upload 'jaccard_similarity_top.csv' to enable this section.")
