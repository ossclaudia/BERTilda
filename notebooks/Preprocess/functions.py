# ---------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------Importing Packages----------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

import re
import string
import html
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
import spacy
from nltk.corpus import stopwords
import nltk
import emoji
from rapidfuzz import process, fuzz
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from sklearn.metrics.pairwise import cosine_similarity
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
nlp_ner = spacy.load("en_core_web_sm")
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import timedelta
from itertools import product
from matplotlib.gridspec import GridSpec
import statsmodels.api as sm

# ---------------------------------------------------------------------------------------------------------------------------------
# --------------------------------Text Cleaning Utilities--------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def replace_html(text):
    return html.unescape(text) if isinstance(text, str) else ""

def clean_hashtags(hashtags):
    if pd.isna(hashtags):
        return ""
    hashtags = re.sub(r"#", "", hashtags)
    hashtags = re.sub(r",", " ", hashtags)
    return hashtags.strip()

def clean_urls(urls):
    if pd.isna(urls):
        return ""
    urls = re.sub(r"https?://", "", urls)
    urls = re.sub(r"www\.", "", urls)
    urls = re.sub(r"[^a-zA-Z/]", " ", urls)
    urls = re.sub(r"/", " ", urls)
    urls = re.sub(r"\s+", " ", urls).strip()
    irrelevant = {"com", "org", "gov", "net", "html"}
    words = [w for w in urls.split() if w not in irrelevant]
    return " ".join(words)

def clean_tweet_row(row):
    text, hashtags, urls = row['text'], row['hashtags'], row['urls']
    text = replace_html(text)
    hashtags_clean = clean_hashtags(hashtags)
    url_words = clean_urls(urls)
    combined = f"{text} {hashtags_clean} {url_words}"
    combined = re.sub(r'\bRT\b', '', combined)
    combined = emoji.replace_emoji(combined, replace='')
    combined = re.sub(r'http\S+|www\S+', '', combined)
    combined = re.sub(r'#', '', combined)
    combined = re.sub(r'@', '', combined)
    combined = re.sub(r'[^a-zA-Z\s]', ' ', combined)
    combined = combined.lower()
    combined = re.sub(r'\s+', ' ', combined).strip()
    tokens = [word for word in combined.split() if word not in stop_words]
    doc = nlp(" ".join(tokens))
    lemmas = [token.lemma_ for token in doc]
    return " ".join(lemmas)

# ---------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------Preprocess for Top2Vec-------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------


def preprocess_for_top2vec(text):
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove mentions
    text = re.sub(r'@\w+', '', text)

    # Remove hashtags (mas mantém a palavra)
    text = re.sub(r'#(\w+)', r'\1', text)

    # Remove emojis
    text = emoji.replace_emoji(text, replace='')

    # Remove non-alphanumeric junk but keep punctuation helpful for meaning
    text = re.sub(r'[^A-Za-z0-9.,!?;:\'\"()\-\s]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ---------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------Parallel Processing----------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def parallel_apply(df, func):
    with Pool(cpu_count()) as p:
        result = p.map(func, [row for _, row in df.iterrows()])
    return result

# ---------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------Entity Normalization---------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def normalize_entities(text, canonical_entities, canonical_keys, threshold=90):
    if not text:
        return ""
    doc = nlp_ner(text)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG"] and ent.text in canonical_entities:
            text = re.sub(r'\b{}\b'.format(re.escape(ent.text)), canonical_entities[ent.text], text)
    normalized_words = []
    for word in text.split():
        match, score, _ = process.extractOne(word, canonical_keys, scorer=fuzz.ratio)
        normalized_words.append(canonical_entities[match] if score >= threshold else word)
    return " ".join(normalized_words)

# ---------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------Tokenization-----------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def spacy_tokenize(text):
    doc = nlp(text)
    tokens = [ent.text.replace(" ", "_") for ent in doc.ents if ent.label_ in ["PERSON", "ORG"]]
    tokens += [token.text for token in doc if token.is_alpha]
    return tokens

# ---------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------Static Metrics--------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def topic_diversity(topics_words):
    K = len(topics_words)
    if K <= 1:
        return np.nan
    diversities = []
    for k, topic_k in enumerate(topics_words):
        redundancy_sum = 0
        for w in topic_k:
            redundancy_sum += sum(w in topic_q for i, topic_q in enumerate(topics_words) if i != k)
        r_kC = redundancy_sum / (len(topic_k) * (K - 1))
        d_kC = 1 - r_kC
        diversities.append(d_kC)
    return np.mean(diversities)

def topic_coherence_metrics(topics_words, texts):
    dictionary = corpora.Dictionary(texts)
    cv_model = CoherenceModel(topics=topics_words, texts=texts, dictionary=dictionary, coherence='c_v')
    npmi_model = CoherenceModel(topics=topics_words, texts=texts, dictionary=dictionary, coherence='c_npmi')
    umass_model = CoherenceModel(topics=topics_words, texts=texts, dictionary=dictionary, coherence='u_mass')
    
    return (
        float(cv_model.get_coherence()),
        float(np.mean(cv_model.get_coherence_per_topic())),
        float(np.mean(npmi_model.get_coherence_per_topic())),
        float(np.mean(umass_model.get_coherence_per_topic()))
    )

# ---------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------LDA------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def train_lda(subset, n_topics=10, dictionary=None, top_n=10):
    if dictionary is None:
        raise ValueError("A global dictionary must be provided for LDA")
    
    corpus = [dictionary.doc2bow(tokens) for tokens in subset['tokens']]
    
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        random_state=42,
        passes=10,
        alpha='auto',
        chunksize=2000
    )
    
    topics_words = []
    topic_vecs = []
    for topic in lda_model.show_topics(num_topics=n_topics, num_words=top_n, formatted=False):
        words = [w for w, _ in topic[1]]
        topics_words.append(words)
        vec = np.zeros(len(dictionary))
        for word, weight in topic[1]:
            if word in dictionary.token2id:
                vec[dictionary.token2id[word]] = weight
        topic_vecs.append(vec)
    
    cv, cv_per_topic, npmi, umass = topic_coherence_metrics(topics_words, subset['tokens'])
    diversity = topic_diversity(topics_words)
    tq = cv * diversity

    return topics_words, topic_vecs, cv, npmi, umass, diversity, tq

# ---------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------NMF------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def train_nmf(subset, global_vocab, n_topics=10):
    docs = [" ".join(tokens) for tokens in subset['tokens']]
    if len(docs) < 5:
        return None
    
    vectorizer = TfidfVectorizer(vocabulary=global_vocab)
    tfidf = vectorizer.fit_transform(docs)
    vocab = vectorizer.get_feature_names_out()
    
    nmf_model = NMF(n_components=n_topics, random_state=42, max_iter=1000)
    W = nmf_model.fit_transform(tfidf)
    H = nmf_model.components_
    
    topics_words = []
    topic_vecs = []
    for topic_vec in H:
        top_indices = topic_vec.argsort()[-10:][::-1]
        top_words = [vocab[i] for i in top_indices]
        topics_words.append(top_words)
        topic_vecs.append(topic_vec)
    
    cv, cv_per_topic, npmi, umass = topic_coherence_metrics(topics_words, subset['tokens'])
    diversity = topic_diversity(topics_words)
    tq = cv * diversity
    
    return topics_words, topic_vecs, cv, npmi, umass, diversity, tq

# ---------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------Sliding Window-------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def process_window(model_type, start_date, tweets, window_size, global_vocab=None,
                   prev_topics=None, prev_vecs=None, n_topics=10, dictionary=None):

    current_end = start_date + window_size
    subset = tweets[(tweets['created_at'] >= start_date) & (tweets['created_at'] < current_end)]
    if len(subset) < 10:
        return None

    if model_type == 'lda':
        topics, vecs, cv, npmi, umass, diversity, tq = train_lda(subset, n_topics, dictionary=dictionary)
    elif model_type == 'nmf':
        res = train_nmf(subset, global_vocab, n_topics)
        if res is None:
            return None
        topics, vecs, cv, npmi, umass, diversity, tq = res
    else:
        raise ValueError("model_type must be 'lda' or 'nmf'")

    return {
        'start_date': start_date.date(),
        'end_date': current_end.date(),
        'num_docs': len(subset),
        'cv': cv,
        'npmi': npmi,
        'umass': umass,
        'diversity': diversity,
        'tq': tq,
        'topics': [" | ".join(t) for t in topics],
        'topic_vecs': vecs,
        'topic_words': topics
    }


# ---------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------Execution------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def run_dynamic_pipeline(tweets, model_type='lda', n_topics=10, window_days=7, step_days=3):

    start_date = tweets['created_at'].min()
    end_date = tweets['created_at'].max()
    window_size = timedelta(days=window_days)
    step_size = timedelta(days=step_days)

    global_vocab = None
    global_dictionary = None

    if model_type == 'nmf':
        vectorizer = TfidfVectorizer(max_df=0.4, min_df=10)
        _ = vectorizer.fit([" ".join(tokens) for tokens in tweets['tokens']])
        global_vocab = vectorizer.get_feature_names_out()
    elif model_type == 'lda':
        global_dictionary = corpora.Dictionary(tweets['tokens'])
        global_dictionary.filter_extremes(no_below=10, no_above=0.4)

    current_start = start_date
    results, prev_topics, prev_vecs = [], None, None

    while current_start + window_size <= end_date:
        res = process_window(
            model_type, current_start, tweets, window_size,
            global_vocab, prev_topics, prev_vecs, n_topics, dictionary=global_dictionary
        )
        if res:
            prev_topics = res['topic_words']
            prev_vecs = res['topic_vecs']
            results.append(res)
        current_start += step_size

    df = pd.DataFrame(results)
    df.drop(columns=['topic_vecs', 'topic_words'], inplace=True, errors='ignore')
    return df

# ---------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------Show Static Metrics--------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

def describe_metric(series):
    return {
        "mean": series.mean(),
        "std": series.std(),
        "p25": series.quantile(0.25),
        "p50": series.quantile(0.50),
        "p75": series.quantile(0.75),
    }

def plot_static_metrics(df, model_name):
    x = range(1, len(df) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"{model_name} Metrics: CV, NPMI, Diversity & UMass", fontsize=14, fontweight='bold')

    axes[0, 0].plot(x, df['cv'], marker='o', color='tab:blue')
    axes[0, 0].set_title("Coherence (CV)")
    axes[0, 0].set_xlabel("Window")
    axes[0, 0].set_ylabel("CV")
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)

    axes[0, 1].plot(x, df['npmi'], marker='o', color='tab:orange')
    axes[0, 1].set_title("Coherence (NPMI)")
    axes[0, 1].set_xlabel("Window")
    axes[0, 1].set_ylabel("NPMI")
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)

    axes[1, 0].plot(x, df['diversity'], marker='o', color='tab:green')
    axes[1, 0].set_title("Diversity")
    axes[1, 0].set_xlabel("Window")
    axes[1, 0].set_ylabel("Diversity")
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)

    axes[1, 1].plot(x, df['umass'], marker='o', color='tab:purple')
    axes[1, 1].set_title("Coherence (UMass)")
    axes[1, 1].set_xlabel("Window")
    axes[1, 1].set_ylabel("UMass")
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, df['tq'], marker='o', color='tab:red')
    ax.set_title(f"{model_name} Metric: Topic Quality (TQ)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Window")
    ax.set_ylabel("TQ")
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def summarize_static_metrics(df, model_name):
    print(f"=== {model_name} – Overall Metrics Summary ===")

    metrics = {
        "CV": df["cv"],
        "NPMI": df["npmi"],
        "UMass": df["umass"],
        "Diversity": df["diversity"],
        "Topic Quality (TQ)": df["tq"],
    }

    for name, series in metrics.items():
        stats = describe_metric(series)
        print(
            f"{name}: "
            f"mean={stats['mean']:.4f}, "
            f"std={stats['std']:.4f}, "
            f"p25={stats['p25']:.4f}, "
            f"median={stats['p50']:.4f}, "
            f"p75={stats['p75']:.4f}"
        )

def plot_temporal_metrics_tomotopy(df, model_name):
    x = df["year"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f"{model_name} – Topic Quality Metrics Over Time",
        fontsize=14,
        fontweight="bold"
    )

    # Coherence CV
    axes[0, 0].plot(x, df["cv"], marker="o", color="tab:blue")
    axes[0, 0].set_title("Coherence (CV)")
    axes[0, 0].set_xlabel("Year")
    axes[0, 0].set_ylabel("CV")
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)

    # Coherence NPMI
    axes[0, 1].plot(x, df["npmi"], marker="o", color="tab:orange")
    axes[0, 1].set_title("Coherence (NPMI)")
    axes[0, 1].set_xlabel("Year")
    axes[0, 1].set_ylabel("NPMI")
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)

    # Diversity
    axes[1, 0].plot(x, df["diversity"], marker="o", color="tab:green")
    axes[1, 0].set_title("Topic Diversity")
    axes[1, 0].set_xlabel("Year")
    axes[1, 0].set_ylabel("Diversity")
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)

    # Coherence UMass
    axes[1, 1].plot(x, df["umass"], marker="o", color="tab:purple")
    axes[1, 1].set_title("Coherence (UMass)")
    axes[1, 1].set_xlabel("Year")
    axes[1, 1].set_ylabel("UMass")
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

def plot_temporal_tq_tomotopy(df, model_name):
    x = df["year"]

    plt.figure(figsize=(10, 4))
    plt.plot(x, df["tq"], marker="o", color="tab:red")
    plt.title(
        f"{model_name} – Topic Quality (TQ) Over Time",
        fontsize=14,
        fontweight="bold"
    )
    plt.xlabel("Year")
    plt.ylabel("TQ")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

def summarize_temporal_metrics_tomotopy(df, model_name):
    print(f"=== {model_name} – Temporal Metrics Summary ===")

    metrics = {
        "CV": df["cv"],
        "NPMI": df["npmi"],
        "UMass": df["umass"],
        "Diversity": df["diversity"],
        "Topic Quality (TQ)": df["tq"],
    }

    for name, series in metrics.items():
        stats = describe_metric(series)
        print(
            f"{name}: "
            f"mean={stats['mean']:.4f}, "
            f"std={stats['std']:.4f}, "
            f"p25={stats['p25']:.4f}, "
            f"median={stats['p50']:.4f}, "
            f"p75={stats['p75']:.4f}"
        )

def plot_temporal_metrics_detm(df, model_name):
    x = df["year"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        f"{model_name} – Topic Quality Metrics Over Time",
        fontsize=14,
        fontweight="bold"
    )

    # Coherence CV
    axes[0].plot(x, df["cv"], marker="o", color="tab:blue")
    axes[0].set_title("Coherence (CV)")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("CV")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Diversity
    axes[1].plot(x, df["diversity"], marker="o", color="tab:green")
    axes[1].set_title("Topic Diversity")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Diversity")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

def plot_temporal_tq_detm(df, model_name):
    x = df["year"]

    plt.figure(figsize=(10, 4))
    plt.plot(x, df["topic_quality"], marker="o", color="tab:red")
    plt.title(
        f"{model_name} – Topic Quality (TQ) Over Time",
        fontsize=14,
        fontweight="bold"
    )
    plt.xlabel("Year")
    plt.ylabel("TQ")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

def summarize_temporal_metrics_detm(df, model_name):
    print(f"=== {model_name} – Temporal Metrics Summary ===")

    metrics = {
        "CV": df["cv"],
        "Diversity": df["diversity"],
        "Topic Quality (TQ)": df["topic_quality"],
    }

    for name, series in metrics.items():
        stats = describe_metric(series)
        print(
            f"{name}: "
            f"mean={stats['mean']:.4f}, "
            f"std={stats['std']:.4f}, "
            f"p25={stats['p25']:.4f}, "
            f"median={stats['p50']:.4f}, "
            f"p75={stats['p75']:.4f}"
        )


def _get_time_variable(df, time_col="year",
                       start_col="start_year",
                       end_col="end_year"):

    if time_col in df.columns:
        return df[time_col]

    if start_col in df.columns and end_col in df.columns:
        return (df[start_col] + df[end_col]) / 2

    raise ValueError(
        "No valid time information found: "
        "expected `year` or (`start_year`, `end_year`)."
    )


def compute_drift(df, metric,
                  time_col="year",
                  start_col="start_year",
                  end_col="end_year"):

    df = df.copy()
    df["__time"] = _get_time_variable(
        df, time_col=time_col,
        start_col=start_col,
        end_col=end_col
    )

    df = df[["__time", metric]].dropna().sort_values("__time")

    X = sm.add_constant(df["__time"])
    y = df[metric]

    model = sm.OLS(y, X).fit()

    return {
        "metric": metric,
        "drift": model.params["__time"],
        "p_value": model.pvalues["__time"],
        "r2": model.rsquared
    }


def drift_table(df, metrics, model_name=None,
                time_col="year",
                start_col="start_year",
                end_col="end_year"):

    if model_name is not None:
        print(f"Drift metrics for model {model_name}")

    results = []
    for m in metrics:
        res = compute_drift(
            df, m,
            time_col=time_col,
            start_col=start_col,
            end_col=end_col
        )
        results.append(res)

    return pd.DataFrame(results)

def compute_volatility(
    df,
    metric,
    window=3,
    time_col="year",
    start_col="start_year",
    end_col="end_year"
):

    df = df.copy()
    df["__time"] = _get_time_variable(
        df,
        time_col=time_col,
        start_col=start_col,
        end_col=end_col
    )

    df = (
        df[["__time", metric]]
        .dropna()
        .sort_values("__time")
        .reset_index(drop=True)
    )

    rolling_mean = df[metric].rolling(window=window).mean()
    rolling_std = df[metric].rolling(window=window).std()

    return {
        "metric": metric,
        "window": window,
        "rolling_mean_avg": rolling_mean.mean(),
        "rolling_std_avg": rolling_std.mean(),
        "rolling_std_max": rolling_std.max()
    }


def volatility_table(
    df,
    metrics,
    window=3,
    model_name=None,
    dataset_name=None,
    time_col="year",
    start_col="start_year",
    end_col="end_year",
    plot=False
):
    if model_name is not None:
        print(f"Volatility metrics for model {model_name} (window={window})")

    results = []

    for m in metrics:
        res = compute_volatility(
            df,
            m,
            window=window,
            time_col=time_col,
            start_col=start_col,
            end_col=end_col
        )
        results.append(res)

    results_df = pd.DataFrame(results)

    if plot and len(metrics) == 3:
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2)

        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, :])
        ]

        for ax, metric in zip(axes, metrics):
            dft = df.copy()
            dft["__time"] = _get_time_variable(
                dft,
                time_col=time_col,
                start_col=start_col,
                end_col=end_col
            )

            dft = (
                dft[["__time", metric]]
                .dropna()
                .sort_values("__time")
                .reset_index(drop=True)
            )

            rolling_mean = dft[metric].rolling(window=window).mean()
            rolling_std = dft[metric].rolling(window=window).std()

            ax.plot(dft["__time"], dft[metric], alpha=0.4, label=metric)
            ax.plot(dft["__time"], rolling_mean, label="Rolling mean")
            ax.fill_between(
                dft["__time"],
                rolling_mean - rolling_std,
                rolling_mean + rolling_std,
                alpha=0.2,
                label="± Rolling std"
            )

            ax.set_title(metric)
            ax.set_xlabel("Time")
            ax.set_ylabel(metric)
            ax.legend()

        title_parts = []
        if model_name:
            title_parts.append(f"Model: {model_name}")
        if dataset_name:
            title_parts.append(f"Dataset: {dataset_name}")

        title = " | ".join(title_parts)
        title = (
            f"{title} — Volatility (window={window})"
            if title
            else f"Volatility (window={window})"
        )

        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        plt.show()

    return results_df
