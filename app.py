import ast
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import pandas as pd
import networkx as nx
import numpy as np
from datetime import datetime
from dash import Dash, dcc, html, Input, Output, State, ctx, dash_table
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import math
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
DATA_DIR = Path("./datasets")

DATASETS = {
    "Congress Tweets": {
        "topics": DATA_DIR / "119congresstweets/bertilda.csv",
        "relations": DATA_DIR / "119congresstweets/bertilda_mergeandsplits.csv",
    },
    "UN Speeches": {
        "topics": DATA_DIR / "undebates/un_bertilda_topics_per_window.csv",
        "relations": DATA_DIR / "undebates/un_bertilda_merge_split_relations.csv",
    },
    "State of the Union": {
        "topics": DATA_DIR / "stateofunion/stateofunion_bertilda_topics_per_window.csv",
        "relations": DATA_DIR / "stateofunion/stateofunion_bertilda_merge_split_relations.csv",
    },
}

NODE_STATE_COLOR = {
    "split": "blue",
    "continued": "green",
    "disappeared": "red",
    "merge": "yellow",
    "multiple": "orange",
    "unclear/unknown": "grey",
}
PLASMA_COLORS = px.colors.sequential.Greens

# ---------- UTILITIES ----------


def safe_eval(cell):
    """Safely parse list-like strings (e.g. '[1,2]') or return []"""
    import numpy as _np

    if cell is None:
        return []
    if isinstance(cell, (list, tuple, set, _np.ndarray, pd.Series)):
        return list(cell)
    try:
        if pd.isna(cell):
            return []
    except Exception:
        pass
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return []
        # try literal eval
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple, set)):
                return list(val)
            return [val]
        except Exception:
            # fallback: return the raw string
            return [s]
    # fallback: single value
    return [cell]


def map_similarity_to_hex(sim: float) -> str:
    if sim is None:
        return "#888888"
    try:
        idx = min(int(sim * (len(PLASMA_COLORS) - 1)), len(PLASMA_COLORS) - 1)
        return PLASMA_COLORS[idx]
    except Exception:
        return "#888888"


def rgba_from_hex(hexc: str, alpha: float) -> str:
    h = hexc.lstrip("#")
    if len(h) == 3:
        h = "".join([c * 2 for c in h])
    r, g, b = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ---------- LOAD DATA ----------

def load_dataset(dataset_name):
    paths = DATASETS[dataset_name]
    df_b = pd.read_csv(paths["topics"])
    df_m = pd.read_csv(paths["relations"])

    for c in ["start_date", "end_date"]:
        if c in df_b.columns:
            try:
                df_b[c] = pd.to_datetime(df_b[c])
            except Exception:
                pass

    # parse representative tweets
    tweets_per_window: Dict[int, List[List[str]]] = {}
    if "representative_tweets" in df_b.columns:
        for idx, row in df_b.iterrows():
            w = int(idx)
            raw_tweets = row.get("representative_tweets", None)
            parsed = safe_eval(raw_tweets) 
            tweets_list: List[List[str]] = []
            for p in parsed:
                if isinstance(p, str):
                    tweets_list.append([t.strip() for t in p.split("||") if t.strip()])
                else:
                    tweets_list.append([str(p)])
            tweets_per_window[w] = tweets_list

    # Parse topics and counts into dictionaries keyed by window index
    topics_per_window: Dict[int, List[str]] = {}
    topic_words_per_window: Dict[int, List[List[str]]] = {}
    doccounts_per_window: Dict[int, List[int]] = {}
    n_windows = len(df_b)

    for idx, row in df_b.iterrows():
        w = int(idx)
        raw_topics = row.get("topics", None)
        parsed = safe_eval(raw_topics)  # returns list
        # parsed may be ['a|b|c', 'd|e'] or ["topic string"] etc.
        final_topics: List[str] = []
        words_lists: List[List[str]] = []
        for p in parsed:
            if isinstance(p, str) and "|" in p:
                final_topics.append(p)
                words_lists.append([x.strip() for x in p.split("|") if x.strip()])
            elif isinstance(p, str):
                final_topics.append(p)
                # split by space if no '|' found
                words_lists.append([x.strip() for x in p.split() if x.strip()])
            else:
                final_topics.append(str(p))
                words_lists.append([str(p)])
        topics_per_window[w] = final_topics
        topic_words_per_window[w] = words_lists

        # doc counts
        raw_counts = row.get("topic_doc_counts", None)
        pc = safe_eval(raw_counts)
        # try to coerce to ints and align length
        counts: List[int] = []
        if len(pc) == len(final_topics):
            for v in pc:
                try:
                    counts.append(int(v))
                except Exception:
                    counts.append(1)
        elif len(pc) == 1 and len(final_topics) > 0:
            try:
                val = int(pc[0])
            except Exception:
                val = 1
            counts = [val] * len(final_topics)
        else:
            counts = [1] * len(final_topics)
        doccounts_per_window[w] = counts

    # normalize column names
    df_m.columns = [c.strip() for c in df_m.columns]

    list_cols = [
        "successors",
        "successors_words",
        "successors_similarities",
        "topics_from",
        "topics_from_words",
        "antecedents_similarities",
        "coverage_forward",
        "coverage_backward",
    ]
    for c in list_cols:
        if c in df_m.columns:
            df_m[c] = df_m[c].apply(safe_eval)

    # ---------- BUILD GRAPH ----------
    def node_key(window: int, topic_idx: int) -> str:
        return f"{int(window)}_{int(topic_idx)}"


    edges: List[Dict[str, Any]] = []
    nodes_set: Set[str] = set()

    # ensure all topics from df_b are present as nodes (even if isolated)
    for w, topics in topics_per_window.items():
        for tid in range(len(topics)):
            nodes_set.add(node_key(w, tid))

    for _, row in df_m.iterrows():
        try:
            w1 = int(row.get("window1"))
            t1 = row.get("topic1")
            w2 = int(row.get("window2"))
        except Exception:
            # some rows may have NaNs; skip those
            continue
        relation = str(row.get("relation", "")).lower().strip()
        # successors: used for continued/split/unclear
        successors = safe_eval(row.get("successors"))
        succ_sims = safe_eval(row.get("successors_similarities"))

        cov_f_list = safe_eval(row.get("coverage_forward"))
        cov_b_list = safe_eval(row.get("coverage_backward"))

        # topics_from / topic_to: used for merges
        topics_from = safe_eval(row.get("topics_from"))
        topic_to = row.get("topic_to")
        ante_sims = safe_eval(row.get("antecedents_similarities"))

        # handle successors
        if isinstance(successors, list) and t1 is not None:
            try:
                t1i = int(t1)
            except Exception:
                t1i = None
            if t1i is not None:
                for i, s in enumerate(successors):
                    if s is None:
                        continue
                    try:
                        s_i = int(s)
                    except Exception:
                        continue
                    src = node_key(w1, t1i)
                    tgt = node_key(w2, s_i)
                    sim = None
                    cov_f = None
                    cov_b = None        
                    if isinstance(succ_sims, list) and i < len(succ_sims):
                        try:
                            sim = float(succ_sims[i])
                        except Exception:
                            sim = None
                    if isinstance(cov_f_list, list) and i < len(cov_f_list):
                        try:
                            cov_f = float(cov_f_list[i])
                        except:
                            cov_f = None

                    if isinstance(cov_b_list, list) and i < len(cov_b_list):
                        try:
                            cov_b = float(cov_b_list[i])
                        except:
                            cov_b = None

                    edges.append({
                        "src": src,
                        "tgt": tgt,
                        "relation": relation,
                        "similarity": sim,
                        "coverage_forward": cov_f,
                        "coverage_backward": cov_b,
                        "w_src": w1,
                        "w_tgt": w2,
                    })
                    nodes_set.add(src)
                    nodes_set.add(tgt)

        # handle merges (topics_from -> topic_to)
        if isinstance(topics_from, list) and topic_to is not None:
            try:
                topic_to_i = int(topic_to)
            except Exception:
                topic_to_i = None
            if topic_to_i is not None:
                for i, a in enumerate(topics_from):
                    if a is None:
                        continue
                    try:
                        a_i = int(a)
                    except Exception:
                        continue
                    src = node_key(w1, a_i)
                    tgt = node_key(w2, topic_to_i)
                    sim = None
                    cov_f = None
                    cov_b = None
                    if isinstance(ante_sims, list) and i < len(ante_sims):
                        try:
                            sim = float(ante_sims[i])
                        except Exception:
                            sim = None
                    if isinstance(cov_f_list, list) and i < len(cov_f_list):
                        cov_f = float(cov_f_list[i])

                    if isinstance(cov_b_list, list) and i < len(cov_b_list):
                        cov_b = float(cov_b_list[i])
                    edges.append(
                        {
                            "src": src,
                            "tgt": tgt,
                            "relation": relation,
                            "similarity": sim,
                            "coverage_forward": cov_f,
                            "coverage_backward": cov_b,
                            "w_src": w1,
                            "w_tgt": w2,
                        }
                    )
                    nodes_set.add(src)
                    nodes_set.add(tgt)

    # Build NetworkX DiGraph (for BFS/traversals)
    G = nx.DiGraph()
    for n in nodes_set:
        G.add_node(n)

    for e in edges:
        relation = str(e.get("relation", "")).lower()
        if relation == "disappeared":
            continue

        G.add_edge(
            e["src"],
            e["tgt"],
            **{k: v for k, v in e.items() if k not in ("src", "tgt")}
        )


    return df_b, G, topics_per_window, topic_words_per_window, doccounts_per_window, tweets_per_window

df_b, G, topics_per_window, topic_words_per_window, doccounts_per_window, tweets_per_window = load_dataset("Congress Tweets")
n_windows = len(df_b)

def node_key(window: int, topic_idx: int) -> str:
    return f"{int(window)}_{int(topic_idx)}"

 # helper function to extract info from node key
def topic_info_from_nodekey(nkey: str) -> Tuple[int, int, str, List[str], int, List[str]]:
    w_str, t_str = nkey.split("_")
    w = int(w_str)
    t = int(t_str)
    label = f"{w}:{t}"
    words: List[str] = []
    if w in topic_words_per_window and t < len(topic_words_per_window[w]):
        words = topic_words_per_window[w][t]
    elif w in topics_per_window and t < len(topics_per_window[w]):
        raw = topics_per_window[w][t]
        if isinstance(raw, str):
            words = raw.split("|") if "|" in raw else raw.split()
        else:
            words = [str(raw)]
    doccount = 0
    if w in doccounts_per_window and t < len(doccounts_per_window[w]):
        doccount = int(doccounts_per_window[w][t])
    tweets: List[str] = []
    if w in tweets_per_window and t < len(tweets_per_window[w]):
        tweets = tweets_per_window[w][t]
    return w, t, label, words, doccount, tweets


def node_future_state(nkey: str) -> str:
        """Determina o estado futuro de um nó (olhando para janelas futuras)."""
        if nkey not in G.nodes():
            return "disappeared"

        outs = list(G.out_edges(nkey, data=True))
        if not outs:
            return "disappeared"

        rels = [str(d.get("relation", "")).lower().strip() for (_, _, d) in outs if d.get("relation")]
        if not rels:
            return "disappeared"

        rel_types = set([r for r in rels if r not in ("disappeared", "")])

        if len(rel_types) > 1:
            return "multiple"

        rel = next(iter(rel_types))
        if "continue" in rel:
            return "continued"
        if "split" in rel:
            return "split"
        if "merge" in rel:
            return "merge"
        if "unclear" in rel or "unknown" in rel:
            return "unclear"
        return "disappeared"


    # ---------- PLOTTING HELPERS ----------

plasma_cmap = plt.get_cmap("Greens")


def similarity_to_color(sim):
        norm_sim = (sim - 0.7) / 0.4 
        norm_sim = max(0, min(1, norm_sim))  
        rgba = plasma_cmap(norm_sim) 
        return f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]:.2f})"


def build_network_plot(nodes: list, edges_list: list):
        if not nodes:
            return go.Figure()

        by_window = {}
        for n in nodes:
            w, t, _, _, _, _ = topic_info_from_nodekey(n)
            by_window.setdefault(w, []).append(n)

        positions = {}
        for w, nlist in by_window.items():
            nlist_sorted = sorted(nlist, key=lambda x: int(x.split("_")[1]))
            for i, n in enumerate(nlist_sorted):
                x = float(w)
                y = i - (len(nlist_sorted) - 1) / 2.0
                positions[n] = (x, y)

        fig = go.Figure()
        sims_for_color = []

        # ---- Edges ----
        for e in edges_list:
            src, tgt = e["src"], e["tgt"]
            if src not in positions or tgt not in positions:
                continue

            x0, y0 = positions[src]
            x1, y1 = positions[tgt]

            sim = e.get("similarity", None)
            cov_f = e.get("coverage_forward", None)
            cov_b = e.get("coverage_backward", None)
            cov = cov_f if cov_f is not None else 0
            rel = str(e.get("relation", "")).lower()

            norm_sim = max(0.7, sim if sim is not None else 0.7)
            sims_for_color.append(norm_sim)
            color = similarity_to_color(norm_sim)
            width = max(0.8, 4 * cov)
            dash_style = "solid"

            if rel in ("unclear", "unknown"):
                dash_style = "dash"


            fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=color, width=width, dash=dash_style),
                hoverinfo="none",
                showlegend=False,
            ))


            dx, dy = x1 - x0, y1 - y0
            length = math.sqrt(dx**2 + dy**2)
            if length > 0:
                offset = 0.05 
                perp_x = -dy / length * offset
                perp_y = dx / length * offset
            else:
                perp_x = perp_y = 0

            x_mid = (x0 + x1) / 2 + perp_x
            y_mid = (y0 + y1) / 2 + perp_y

            hover_text = (
                f"Similarity: {f'{sim:.3f}' if sim is not None else 'N/A'}<br>"
                f"Coverage Forward: {f'{cov_f:.3f}' if cov_f is not None else 'N/A'}<br>"
                f"Coverage Backward: {f'{cov_b:.3f}' if cov_b is not None else 'N/A'}<br>"
                f"Relation: {rel}"
            )

            fig.add_trace(go.Scatter(
                x=[x_mid],
                y=[y_mid],
                mode="markers",
                marker=dict(size=8, color="rgba(0,0,0,0.05)"),
                hoverinfo="text",
                text=[hover_text],
                showlegend=False,
            ))

        # ---- Nodes ----
        node_x, node_y, node_text, node_sizes, node_colors = [], [], [], [], []
        for n in nodes:
            w, t, label, words, doccount, tweets = topic_info_from_nodekey(n)
            node_x.append(positions[n][0])
            node_y.append(positions[n][1])

            start_date = df_b.loc[w, "start_date"] if "start_date" in df_b.columns else "N/A"
            end_date   = df_b.loc[w, "end_date"]   if "end_date" in df_b.columns else "N/A"

            node_text.append(
                f"Window {w} | Start {start_date.date() if pd.notna(start_date) else 'N/A'} | "
                f"End {end_date.date() if pd.notna(end_date) else 'N/A'} | Topic {t} | Docs {doccount}<br>"
                f"Words: {', '.join(words[:10])}"
            )

            node_sizes.append(6 + math.log1p(doccount))
            state = node_future_state(n)
            if state in ("unclear", "unknown"):
                color = "gray"
            else:
                color = NODE_STATE_COLOR.get(state, "lightgray")
            node_colors.append(color)

        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1.5, color='black')
            ),
            hovertext=node_text,
            hoverinfo="text",
            showlegend=False,
        ))

        # ---- COLORBAR Similarity ----
        colorbar_trace = go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                colorscale=PLASMA_COLORS,
                cmin=0.7,
                cmax=1.0,
                color=[0.7, 1],
                size=0.0001,
                colorbar=dict(
                    title=dict(text="Similarity (edges)", side="right"),
                    ticks="outside",
                    tickvals=[0.7, 0.8, 0.9, 1.0],
                    ticktext=["0.7", "0.8", "0.9", "1.0"],
                    len=0.75,
                    x=1.05,
                    y=0.35,
                ),
            ),
            hoverinfo="none",
            showlegend=False,
        )
        fig.add_trace(colorbar_trace)

        # ---- Nodes ----
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=0, color="white"),
                                name="Nodes", showlegend=True, legendgroup="nodes", hoverinfo="none"))
        for state, color in NODE_STATE_COLOR.items():
            if state in ("unclear", "unknown"):
                label = "Unclear/Unknown"
                color = "gray"
            else:
                label = state.capitalize()
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=10, color=color),
                name=label,
                legendgroup="nodes",
                hoverinfo="none",
                showlegend=True
            ))

        if "start_date" in df_b.columns and "end_date" in df_b.columns:
            x_ticks = sorted(by_window.keys())
            x_labels = []
            for w in x_ticks:
                sd = df_b.loc[w, "start_date"]
                ed = df_b.loc[w, "end_date"]
                if pd.notna(sd) and pd.notna(ed):
                    label = f"{sd.strftime('%Y-%m-%d')} → {ed.strftime('%Y-%m-%d')}"
                elif pd.notna(sd):
                    label = sd.strftime('%Y-%m-%d')
                else:
                    label = str(w)
                x_labels.append(label)

            # ---- Vertical Lines ----
            for x in x_ticks:
                fig.add_trace(go.Scatter(
                    x=[x, x],
                    y=[min(node_y) - 1, max(node_y) + 1],
                    mode="lines",
                    line=dict(color="gray", width=1, dash="dot"),
                    hoverinfo="none",
                    showlegend=False
                ))

            fig.update_xaxes(
                tickmode="array",
                tickvals=x_ticks,
                ticktext=x_labels,
                showticklabels=True,
                tickangle=90,
                tickfont=dict(size=8),
                showline=False,
                showgrid=False,
                zeroline=False
            )

        # ---- layout ----
        fig.update_layout(
            showlegend=True,
            hovermode="closest",
            plot_bgcolor="rgb(180,215,240)",
            paper_bgcolor="rgb(180,215,240)",
            yaxis=dict(visible=False),
            margin=dict(l=10, r=40, t=30, b=10),
            height=700,
            legend=dict(
                x=1.05,
                y=1,
                bgcolor="rgba(255,255,255,0.7)",
                itemsizing='constant',
                font=dict(size=18, family="Arial, sans-serif", color="black")
            )
        )

        return fig


# ---------- DASH APP ----------

app = Dash(__name__)
app.layout = html.Div(
    [
        # ---------- TITLE ----------
        html.H1(
            "Topic Evolution Analysis",
            style={
                "textAlign": "center",
                "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                "fontSize": "40px",
                "color": "#2c3e50",
                "marginBottom": "30px",
                "marginTop": "20px",
            }
        ),

        # ---------- SELECT DATASET ----------
        html.Div(
            [
                html.Label(
                    "Select dataset:",
                    style={"fontSize": "18px", "marginBottom": "8px", "fontWeight": "bold"}
                ),
                dcc.Dropdown(
                    id="dataset-dropdown",
                    options=[{"label": k, "value": k} for k in DATASETS.keys()],
                    value="Congress Tweets",
                    clearable=False,
                    style={"width": "100%", "maxWidth": "400px"},
                ),
            ],
            style={
                "padding": "15px",
                "backgroundColor": "#ecf0f1",
                "fontSize": "17px", 
                "color": "#2c3e50",     
                "borderRadius": "10px",
                "marginBottom": "20px",
                "boxShadow": "2px 2px 8px rgba(0,0,0,0.1)"
            },
        ),

        # ---------- MODE ----------
        html.Div(
            [
                html.Label(
                    "Select mode:",
                    style={"fontSize": "18px", "fontWeight": "bold", "marginRight": "10px"}
                ),
                dcc.RadioItems(
                    id="mode",
                    options=[
                        {"label": "By date range", "value": "date"},
                        {"label": "By topic", "value": "topic"},
                    ],
                    value="date",
                    inline=True,
                    inputStyle={"marginRight": "5px", "marginLeft": "15px"}
                )
            ],
            style={
                "padding": "12px",
                "backgroundColor": "#ecf0f1",
                "fontSize": "17px", 
                "color": "#2c3e50",  
                "borderRadius": "10px",
                "marginBottom": "20px",
                "boxShadow": "2px 2px 8px rgba(0,0,0,0.1)"
            },
        ),

        # ---------- CONTROLS ----------
        html.Div(
            id="controls-area",
            children=[
                html.Div(
                    id="date-controls",
                    children=[
                        html.Div([
                            html.Label("Start date: "),
                            dcc.DatePickerSingle(id="start-date"),
                            html.Label("End date: ", style={"marginLeft": "20px"}),
                            dcc.DatePickerSingle(id="end-date"),
                        ], style={"display": "flex", "alignItems": "center", "gap": "10px"})
                    ],
                    style={"display": "block", "padding": "15px", "backgroundColor": "#f7f9f9", "borderRadius": "10px", "marginBottom": "20px"}
                ),
                html.Div(
                    id="topic-controls",
                    children=[
                        html.Label("Window: "),
                        dcc.Dropdown(
                            id="window-dropdown",
                            options=[],
                            value=None,
                            clearable=False,
                            style={"width": "100%", "maxWidth": "300px", "marginBottom": "15px"},
                        ),
                        html.Label("Topic in window: "),
                        dcc.Dropdown(
                            id="topic-dropdown",
                            options=[],
                            value=None,
                            clearable=False,
                            style={
                                "width": "100%",
                                "maxWidth": "1000px",
                                "whiteSpace": "normal",
                                "lineHeight": "25px"
                            },
                        ),
                    ],
                    style={"display": "none", "padding": "15px", "backgroundColor": "#f7f9f9", "borderRadius": "10px", "marginBottom": "20px"}
                ),
            ]
        ),

        # ---------- BUTTONS ----------
        html.Button("Show network graph", id="btn-network", n_clicks=0, style={
            "backgroundColor": "#2c3e50",
            "color": "white",
            "fontSize": "16px",
            "padding": "10px 18px",
            "border": "none",
            "borderRadius": "8px",
            "cursor": "pointer",
            "transition": "all 0.2s ease"
        }),
        html.Button("Show topics' info", id="btn-table", n_clicks=0, style={
            "backgroundColor": "#2c3e50",
            "color": "white",
            "fontSize": "16px",
            "padding": "10px 18px",
            "border": "none",
            "borderRadius": "8px",
            "cursor": "pointer",
            "transition": "all 0.2s ease",
            "marginLeft": "10px"
        }),
        html.Button("Highlights", id="btn-info", n_clicks=0, style={
            "backgroundColor": "#2c3e50",
            "color": "white",
            "fontSize": "16px",
            "padding": "10px 18px",
            "border": "none",
            "borderRadius": "8px",
            "cursor": "pointer",
            "transition": "all 0.2s ease",
            "marginLeft": "10px"
        }),
        html.Button("Documents over time", id="btn-docs", n_clicks=0, style={
            "backgroundColor": "#2c3e50",
            "color": "white",
            "fontSize": "16px",
            "padding": "10px 18px",
            "border": "none",
            "borderRadius": "8px",
            "cursor": "pointer",
            "transition": "all 0.2s ease",
            "marginLeft": "10px"
        }),

        # ---------- GRAPH ----------
        html.Div(
            dcc.Loading(dcc.Graph(id="vis-graph")),
            style={
                "padding": "20px",
                "backgroundColor": "#ecf0f1",
                "borderRadius": "12px",
                "boxShadow": "2px 2px 12px rgba(0,0,0,0.1)"
            },
        ),
    ],
    style={
        "backgroundColor": "#f0f3f5",
        "minHeight": "100vh",
        "padding": "30px 50px",
        "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
    },
)



# show/hide controls when mode changes
@app.callback(
    Output("date-controls", "style"),
    Output("topic-controls", "style"),
    Input("mode", "value"),
)
def toggle_controls(mode_value):
    if mode_value == "date":
        return {"display": "block", "padding": "10px"}, {"display": "none"}
    return {"display": "none"}, {"display": "block", "padding": "10px"}


# populate topic dropdown when window changes
@app.callback(
    Output("topic-dropdown", "options"),
    Output("topic-dropdown", "value"),
    Input("window-dropdown", "value"),
)
def update_topics_for_window(window):
    if window is None:
        return [], None
    w = int(window)
    topics = topics_per_window.get(w, [])
    opts = []
    for i, t in enumerate(topics):
        words = topic_words_per_window.get(w, [])
        lab_words = "|".join(words[i]) if (w in topic_words_per_window and i < len(words)) else t
        label = f"{i} - {lab_words}"
        opts.append({"label": label, "value": i})
    val = opts[0]["value"] if opts else None
    return opts, val

@app.callback(
    Output("window-dropdown", "options"),
    Output("window-dropdown", "value"),
    Output("start-date", "date"),
    Output("end-date", "date"),
    Input("dataset-dropdown", "value"),
)
def update_dataset(dataset_name):

    global df_b, G, topics_per_window, topic_words_per_window
    global doccounts_per_window, tweets_per_window, n_windows

    df_b, G, topics_per_window, topic_words_per_window, doccounts_per_window, tweets_per_window = load_dataset(dataset_name)
    n_windows = len(df_b)

    # update window dropdown
    window_options = [
        {"label": f"Window {i}", "value": i}
        for i in range(n_windows)
    ]

    default_window = 0 if n_windows > 0 else None

    # update date pickers
    if "start_date" in df_b.columns:
        start = df_b["start_date"].min()
        end = df_b["end_date"].max()
    else:
        start = None
        end = None

    return window_options, default_window, start, end

# main
@app.callback(
    Output("vis-graph", "figure"),
    Input("btn-network", "n_clicks"),
    Input("btn-table", "n_clicks"),
    Input("btn-info", "n_clicks"),
    Input("btn-docs", "n_clicks"),  
    State("mode", "value"),
    State("start-date", "date"),
    State("end-date", "date"),
    State("window-dropdown", "value"),
    State("topic-dropdown", "value"),
)
def render_visual(n_net, n_table, n_info, n_docs, mode, sdate, edate, sel_window, sel_topic):
    triggered = ctx.triggered_id
    if triggered == "btn-table":
        show = "table"
    elif triggered == "btn-info":
        show = "info"
    elif triggered == "btn-docs": 
        show = "docs"
    else:
        show = "network"

    selected_nodes: Set[str] = set()
    selected_edges: List[Dict[str, Any]] = []

    windows_in_range = None
    if mode == "date":
        if sdate is None or edate is None or ("start_date" not in df_b.columns and "end_date" not in df_b.columns):
            windows_in_range = list(range(n_windows))
        else:
            sdt = pd.to_datetime(sdate)
            edt = pd.to_datetime(edate)
            windows_in_range = []
            for i, row in df_b.iterrows():
                sd = row.get("start_date")
                ed = row.get("end_date")
                include = False
                if pd.notna(sd):
                    if sdt <= pd.to_datetime(sd) <= edt:
                        include = True
                if pd.notna(ed):
                    if sdt <= pd.to_datetime(ed) <= edt:
                        include = True
                try:
                    if pd.notna(sd) and pd.notna(ed):
                        if (pd.to_datetime(sd) <= edt and pd.to_datetime(ed) >= sdt):
                            include = True
                except Exception:
                    pass
                if include:
                    windows_in_range.append(int(i))
            if not windows_in_range:
                windows_in_range = list(range(n_windows))

        for n in G.nodes():
            try:
                w = int(n.split("_")[0])
            except Exception:
                continue
            if w in windows_in_range:
                selected_nodes.add(n)
        for u, v, d in G.edges(data=True):
            if u in selected_nodes and v in selected_nodes:
                selected_edges.append({"src": u, "tgt": v, **d})

    else:
        if sel_window is None or sel_topic is None:
            selected_nodes = set(G.nodes())
            selected_edges = [dict(src=u, tgt=v, **d) for u, v, d in G.edges(data=True)]
        else:
            start = node_key(int(sel_window), int(sel_topic))
            if start not in G.nodes():
                return go.Figure()
            q = [start]
            visited = set()
            while q:
                cur = q.pop(0)
                if cur in visited:
                    continue
                visited.add(cur)
                selected_nodes.add(cur)
                for _, nbr, d in G.out_edges(cur, data=True):
                    selected_edges.append({"src": cur, "tgt": nbr, **d})
                    if nbr not in visited:
                        q.append(nbr)
                for nbr, _, d in G.in_edges(cur, data=True):
                    selected_edges.append({"src": nbr, "tgt": cur, **d})
                    if nbr not in visited:
                        q.append(nbr)

    combined_edges = {}

    for e in selected_edges:
        key = (e["src"], e["tgt"])

        if key not in combined_edges:
            combined_edges[key] = e.copy()
        else:
            existing_rel = combined_edges[key]["relation"]
            new_rel = e["relation"]

            if new_rel not in existing_rel:
                combined_edges[key]["relation"] = existing_rel + " + " + new_rel

            combined_edges[key]["similarity"] = max(
                combined_edges[key].get("similarity") or 0,
                e.get("similarity") or 0
            )

            combined_edges[key]["coverage_forward"] = max(
                combined_edges[key].get("coverage_forward") or 0,
                e.get("coverage_forward") or 0
            )

            combined_edges[key]["coverage_backward"] = max(
                combined_edges[key].get("coverage_backward") or 0,
                e.get("coverage_backward") or 0
            )

    selected_edges = list(combined_edges.values())

    final_nodes = sorted(list(selected_nodes), key=lambda x: (int(x.split("_")[0]), int(x.split("_")[1])))

    # ---------------------------------------------------------
    # Documents over time
    # ---------------------------------------------------------
    if show == "docs":
        # ---------------------------------------------------------
        # MODE: DATE  
        # ---------------------------------------------------------
        if mode == "date":
            totals = []
            for w, counts in doccounts_per_window.items():
                if windows_in_range is not None and w not in windows_in_range:
                    continue
                totals.append({
                    "window": w,
                    "total_docs": sum(counts),
                    "start_date": df_b.loc[w, "start_date"] if "start_date" in df_b.columns else None,
                    "end_date": df_b.loc[w, "end_date"] if "end_date" in df_b.columns else None,
                })
            df_docs = pd.DataFrame(totals).sort_values("window")

            x_labels = []
            for _, r in df_docs.iterrows():
                if pd.notna(r["start_date"]) and pd.notna(r["end_date"]):
                    label = f"{r['start_date'].strftime('%Y-%m-%d')} → {r['end_date'].strftime('%Y-%m-%d')}"
                elif pd.notna(r["start_date"]):
                    label = r["start_date"].strftime('%Y-%m-%d')
                else:
                    label = str(r["window"])
                x_labels.append(label)

            fig_docs = go.Figure()
            fig_docs.add_trace(go.Scatter(
                x=x_labels,
                y=df_docs["total_docs"],
                mode="lines+markers",
                line=dict(width=2),
                marker=dict(size=6),
                name="Documents per window"
            ))
            fig_docs.update_layout(
                title="Documents over time (by date range)",
                xaxis_title="Time windows",
                yaxis_title="Number of documents",
                template="plotly_white",
                height=500,
                margin=dict(l=30, r=30, t=50, b=80)
            )
            return fig_docs

        # ---------------------------------------------------------
        # MODE: TOPIC 
        # ---------------------------------------------------------
        elif mode == "topic":
            if sel_window is None or sel_topic is None:
                return go.Figure()

            start_node = node_key(int(sel_window), int(sel_topic))
            if start_node not in G.nodes():
                return go.Figure()

            def bfs_full_history(start_node):
                q = [start_node]
                visited = set()
                while q:
                    cur = q.pop(0)
                    if cur in visited:
                        continue
                    visited.add(cur)
                    for _, nbr, _ in G.out_edges(cur, data=True):
                        if nbr not in visited:
                            q.append(nbr)
                    for nbr, _, _ in G.in_edges(cur, data=True):
                        if nbr not in visited:
                            q.append(nbr)
                return visited

            history_nodes = bfs_full_history(start_node)
            if not history_nodes:
                return go.Figure()

            docs_by_window = {}
            for n in history_nodes:
                w, t, _, _, doccount, _ = topic_info_from_nodekey(n)
                docs_by_window[w] = docs_by_window.get(w, 0) + (doccount or 0)

            df_hist = (
                pd.DataFrame(
                    [{"window": w, "total_docs": d,
                      "start_date": df_b.loc[w, "start_date"] if "start_date" in df_b.columns else None,
                      "end_date": df_b.loc[w, "end_date"] if "end_date" in df_b.columns else None}
                     for w, d in sorted(docs_by_window.items())]
                ).sort_values("window")
            )

            x_labels = []
            for _, r in df_hist.iterrows():
                if pd.notna(r["start_date"]) and pd.notna(r["end_date"]):
                    label = f"{r['start_date'].strftime('%Y-%m-%d')} → {r['end_date'].strftime('%Y-%m-%d')}"
                elif pd.notna(r["start_date"]):
                    label = r["start_date"].strftime('%Y-%m-%d')
                else:
                    label = str(r["window"])
                x_labels.append(label)

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=x_labels,
                y=df_hist["total_docs"],
                mode="lines+markers",
                line=dict(width=2),
                marker=dict(size=8),
                name="Documents (topic history)"
            ))
            fig_hist.update_layout(
                title="Documents over time (topic history)",
                xaxis_title="Time windows",
                yaxis_title="Sum of documents (all nodes in topic history)",
                template="plotly_white",
                height=500,
                margin=dict(l=30, r=30, t=50, b=80)
            )
            return fig_hist

    # ---------------------------------------------------------
    # INFO branch implementation
    # ---------------------------------------------------------
    if show == "info":
        # helper BFS to get full history (both directions) for a starting node
        def bfs_full_history(start_node: str) -> Set[str]:
            q = [start_node]
            visited = set()
            while q:
                cur = q.pop(0)
                if cur in visited:
                    continue
                visited.add(cur)
                # outgoing
                for _, nbr, _ in G.out_edges(cur, data=True):
                    if nbr not in visited:
                        q.append(nbr)
                # incoming
                for nbr, _, _ in G.in_edges(cur, data=True):
                    if nbr not in visited:
                        q.append(nbr)
            return visited

        # If mode == "date": iterate over selected_nodes (those in filtered windows),
        # take each as a seed, get its full history, compute duration and total docs.
        if mode == "date":
            if not selected_nodes:
                return go.Figure()

            # cache to avoid recomputing histories for nodes in the same component
            seen_components = set()
            components_info = []  # list of dicts with keys: 'rep_node','nodes','min_w','max_w','duration','total_docs','top_words_sample'
            for n in selected_nodes:
                if n in seen_components:
                    continue
                comp_nodes = bfs_full_history(n)
                # mark seen
                for cn in comp_nodes:
                    seen_components.add(cn)

                # compute min/max window and total docs
                win_nums = []
                total_docs = 0
                top_words_sample = None
                for cn in comp_nodes:
                    try:
                        wnum, tnum, label, words, doccount, tweets = topic_info_from_nodekey(cn)
                    except Exception:
                        # fallback parsing window from node key
                        parts = cn.split("_")
                        if len(parts) >= 2:
                            wnum = int(parts[0])
                            tnum = int(parts[1])
                            label = cn
                            words = []
                            doccount = 0
                        else:
                            continue
                    win_nums.append(int(wnum))
                    total_docs += (doccount or 0)
                    if top_words_sample is None:
                        top_words_sample = ", ".join(words[:10]) if words else label

                if not win_nums:
                    continue
                min_w = min(win_nums)
                max_w = max(win_nums)
                duration = max_w - min_w + 1

                start_date_range = None
                end_date_range = None

                if "start_date" in df_b.columns:
                    try:
                        start_date_range = pd.to_datetime(df_b.loc[min_w, "start_date"]).date()
                    except Exception:
                        pass
                if "end_date" in df_b.columns:
                    try:
                        end_date_range = pd.to_datetime(df_b.loc[max_w, "end_date"]).date()
                    except Exception:
                        pass

                components_info.append({
                    "rep_node": n,
                    "nodes": comp_nodes,
                    "min_w": min_w,
                    "max_w": max_w,
                    "duration": duration,
                    "total_docs": total_docs,
                    "top_words": top_words_sample,
                    "start_date_range": start_date_range,
                    "end_date_range": end_date_range,
                })

            if not components_info:
                return go.Figure()

            # longest-duration component
            longest = max(components_info, key=lambda x: (x["duration"], x["total_docs"]))
            # highest-docs component
            most_docs = max(components_info, key=lambda x: (x["total_docs"], x["duration"]))

            # Build table with two rows: longest-lasting topic (full history) and topic with most documents
            table_data = [
                {
                    "type": "Longest-lasting topic (history across windows)",
                    "rep_node": longest["rep_node"],
                    "window_span": f"{longest['min_w']} - {longest['max_w']}",
                    "date_span": f"{longest['start_date_range']} → {longest['end_date_range']}",
                    "duration_windows": longest["duration"],
                    "top_words": longest["top_words"],
                    "total_docs": longest["total_docs"],
                },
                {
                    "type": "Topic with most documents (across histories)",
                    "rep_node": most_docs["rep_node"],
                    "window_span": f"{most_docs['min_w']} - {most_docs['max_w']}",
                    "date_span": f"{most_docs['start_date_range']} → {most_docs['end_date_range']}",
                    "duration_windows": most_docs["duration"],
                    "top_words": most_docs["top_words"],
                    "total_docs": most_docs["total_docs"],
                },
            ]
            df_info = pd.DataFrame(table_data)

            info_fig = go.Figure(data=[go.Table(
                header=dict(
                    values=[
                        "Measure", "Representative node", "Window span", 
                        "Date span", "Duration (windows)", "Top words", "Total documents"
                    ],
                    fill_color='lightgrey', align='left'
                ),
                cells=dict(
                    values=[
                        df_info["type"], df_info["rep_node"], df_info["window_span"],
                        df_info["date_span"], df_info["duration_windows"],
                        df_info["top_words"], df_info["total_docs"],
                    ],
                    align='left'
                )
            )])
            info_fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=400)
            return info_fig

        else:  # mode == "topic"
            # Need sel_window and sel_topic
            if sel_window is None or sel_topic is None:
                return go.Figure()
            start = node_key(int(sel_window), int(sel_topic))
            if start not in G.nodes():
                return go.Figure()

            history_nodes = bfs_full_history(start)
            if not history_nodes:
                return go.Figure()

            rows = []
            for hn in history_nodes:
                try:
                    wnum, tnum, label, words, doccount, tweets = topic_info_from_nodekey(hn)
                except Exception:
                    parts = hn.split("_")
                    if len(parts) >= 2:
                        wnum = int(parts[0])
                        tnum = int(parts[1])
                        label = hn
                        words = []
                        doccount = 0
                    else:
                        continue
                rows.append({
                    "node": hn,
                    "window": int(wnum),
                    "topic": int(tnum),
                    "top_words": ", ".join(words[:10]) if words else "",
                    "doccount": doccount or 0
                })

            if not rows:
                return go.Figure()

            df_hist = pd.DataFrame(rows)
            # node with max and min docs
            max_row = df_hist.loc[df_hist["doccount"].idxmax()]
            min_row = df_hist.loc[df_hist["doccount"].idxmin()]

            table_data = [
                {
                    "measure": "Node in history with MOST documents",
                    "node": max_row["node"],
                    "window": int(max_row["window"]),
                    "topic": int(max_row["topic"]),
                    "top_words": max_row["top_words"],
                    "doccount": int(max_row["doccount"]),
                },
                {
                    "measure": "Node in history with LEAST documents",
                    "node": min_row["node"],
                    "window": int(min_row["window"]),
                    "topic": int(min_row["topic"]),
                    "top_words": min_row["top_words"],
                    "doccount": int(min_row["doccount"]),
                },
            ]
            df_info = pd.DataFrame(table_data)

            info_fig = go.Figure(data=[go.Table(
                header=dict(
                    values=["Measure", "Representative node", "Window", "Topic", "Top words", "Document count"],
                    fill_color='lightgrey',
                    align='left'
                ),
                cells=dict(
                    values=[
                        df_info["measure"],
                        df_info["node"],
                        df_info["window"],
                        df_info["topic"],
                        df_info["top_words"],
                        df_info["doccount"],
                    ],
                    align='left'
                )
            )])
            info_fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=300)
            return info_fig

    # ---------------------------------------------------------
    # TABLE branch (your original code) - unchanged
    # ---------------------------------------------------------
    if show == 'table':
        if not selected_nodes:
            return go.Figure()

        table_data = []
        for node in selected_nodes:
            w, t, label, words, doccount, tweets = topic_info_from_nodekey(node)
            top_words = ", ".join(words[:10])
            rep_tweets = ", ".join([f"<b>Tweet {i+1}:</b> “{t}”" for i, t in enumerate(tweets[:5])])
            table_data.append({
                "window": w,
                "topic": t,
                "top_words": top_words,
                "doc_count": doccount,
                "representative_tweets": rep_tweets
            })

        df_table = pd.DataFrame(table_data).sort_values(by=["window", "topic"]).reset_index(drop=True)

        table_fig = go.Figure(data=[go.Table(
            columnwidth=[60, 60, 500, 80], 
            header=dict(
                values=["Window", "Topic", "Top 10 Words", "Document Count"],
                fill_color='lightgrey',
                align='left',             
                font=dict(size=28, family="Arial, sans-serif"), 
                height=60                   
            ),
            cells=dict(
                values=[
                    df_table["window"],
                    df_table["topic"],
                    df_table["top_words"],
                    df_table["doc_count"],
                ],
                align='left',             
                font=dict(size=30, family="Arial, sans-serif"),  
                height=60                 
            )
        )])
        table_fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            height=900                     
        )
        return table_fig

    # ---------------------------------------------------------
    # NETWORK branch (your original behaviour)
    # ---------------------------------------------------------
    if show == "network":
        fig = build_network_plot(final_nodes, selected_edges)
        return fig

    # fallback
    return go.Figure()


if __name__ == "__main__":
    app.run(debug=True)
