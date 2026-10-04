# VISION.md

## Goal

Deliver a rigorous, reproducible revision of the existing RAVDESS audio-visual emotion-recognition experiments for IEEE SPMB 2026, under a one-week time constraint.

The revision should directly answer the reviewers' two most consequential experimental objections:

1. the original stratified random split may leak actor identity across train/test;
2. the paper lacks a meaningful external published baseline.

## Motivation / intent

The project is not trying to win a SOTA benchmark race. The paper has already been accepted with mandatory revision. The practical goal is to produce evidence that is methodologically defensible, easy to explain to reviewers, and feasible to finish quickly.

The desired outcome is:

- strict speaker-independent evidence;
- comparable results across methods;
- transparent per-fold variability;
- one credible external published fusion mechanism evaluated under the same backbone and split protocol;
- no hidden leakage from old checkpoints or inconsistent data partitions.

## Product vision

The experimental system should feel like a small, trustworthy research harness.

It should make it hard to accidentally produce an invalid result and easy to answer questions such as:

- Which actors were in this fold's train/validation/test sets?
- Were any actors shared across partitions?
- Which checkpoint produced the final test score?
- Were the unimodal warm-start checkpoints trained only on the current fold's training actors?
- What were the six per-fold metrics?
- What is the mean ± standard deviation?
- Can another person rerun the exact fold?

## What this project should be like

- explicit
- deterministic
- fold-aware
- leakage-resistant
- easy to audit
- minimal in scope
- scientifically controlled

## What this project should not be like

- a large architecture redesign
- a hyperparameter-search project
- a legacy-environment archaeology exercise
- a deployment/UI refactor
- an attempt to preserve the old 93.33% score at all costs
- a paper-specific script that cannot be checked after the fact

A lower score under a stricter actor-independent protocol is acceptable if the evaluation is correct and the comparative conclusions remain meaningful.
