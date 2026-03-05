# BERTilda

This repository contains the implementation supporting the research paper “BERTilda: Explainable Topic Lifecycle Tracking with Split/Merge Detection via Similarity-and-Flow Temporal Graphs”, authored by Cláudia Oliveira and Álvaro Figueira.

BERTilda (BERT-based Temporal Identification, Lifecycle Detection & Analysis) is a framework for tracking topic evolution over time. It treats temporal topic modeling as a problem of aligning topics and labeling events on an explicit graph. Topics are first discovered independently within each time window using a snapshot topic model (e.g., BERTopic), then connected across windows using two signals: semantic similarity in a shared embedding space and bidirectional document flow, which captures how documents move between topics. These signals form a temporal topic graph, where interpretable rules identify topic continuations, splits, merges, disappearances, and ambiguous transitions.
