# Customer Support Copilot — Report

**Amit Prasad · Dandes Academy, Applied AI Engineering · Course 02 Assignment**

Models: gpt-6-luna (answers, router, query transforms), gpt-6.1-sol (judge), text-embedding-3-small, cross-encoder/ms-marco-MiniLM-L-6-v2. K = 5 everywhere, same 20-question golden set for every version.

## Results

| Metric | V1 (basic) | V2 (single index) | V2 (multi-index + routing) | Target |
|---|---|---|---|---|
| Routing Accuracy | n/a | n/a | **1.00** | ≥ 0.85 |
| Precision@5 | 0.325 | 0.338 | 0.350 | |
| Recall@5 | 0.894 | 0.975 | **0.988** | ≥ 0.80 |
| Coverage (multi-source) | 0.900 | 0.943 | 0.971 | |
| Faithfulness (own judge) | 0.925 | 0.988 | **0.972** | ≥ 0.85 |
| Faithfulness (RAGAS) | 1.00 | 1.00 | 1.00 | |
| Answer Relevancy (RAGAS) | 0.708 | 0.714 | 0.708 | |
| Context Precision / Recall | 0.889 / 0.875 | 0.957 / 0.933 | 0.984 / 0.946 | |
| Refusal Accuracy | 0.75 | 0.75 | **1.00** | 100% |
| False Refusals | 1 | 0 | **0** | 0 |

All five targets are met by the final version.

## Q1. What worked, and what failed?

**Worked.** Hybrid search fixed the one false refusal (Q05, "What does policy RR-004 say?") — dense search never retrieved RR-004 because an embedding does not treat "RR-004" as an identifier. The intent router fixed the one wrong non-refusal (Q20, "Do you repair cracked TV screens?") — V1 and V2-single both pulled OG-006 ("Dandes does not repair products directly") and answered from it; the router marks it out_of_scope and skips retrieval. The "latest date wins" prompt rule worked on Q01: the stale 2025 ticket (7-day window) was in the top 5 and the copilot correctly answered 10 days "under the policy effective 1 July 2026".

**Failed / did little.** Query rewriting (demo A) made the demo question *worse*: "My order hasn't come yet" was rewritten to "What should I do if my order hasn't arrived yet?", which drifted towards damaged-item docs (OG-005, RR-003). Multi-query (demo B) produced the same top 5 as V1 for that question — no gain, no loss. Reranking on the single index changed only one slot (TCK-1006 out, RR-001 in) for its demo question; on a 50-document collection the dense ranking is already close to right. Precision@5 stays ~0.33 for all versions, which is a ceiling effect: most questions expect 1–2 documents, so even a perfect top 5 scores 0.2–0.4.

## Q2. One real bug I found, and how I fixed it

BM25 did not find RR-004 for "What does policy RR-004 say?". Scores: OG-001 6.04, FAQ-004 3.54, FAQ-008 2.83 … RR-004 only 1.74. Two causes: (1) stopwords like "what", "does", "say" carried weight — FAQ-004 scored 3.54 on the word "does" alone because it is a short document and BM25 rewards short documents; (2) OG-001 *mentions* "policy RR-004", so it matched both query terms, while RR-004 itself is a long document and was penalised by length normalisation. Weighting the ID 3× in the indexed text did not help (OG-001 still first).

Fix: a stopword list in the tokenizer, plus an exact-ID rule — if a query token equals a document ID, that document is pinned to the top of the keyword results. After this, BM25 ranked RR-004 first. A second bug appeared immediately: Reciprocal Rank Fusion still dropped RR-004 from the hybrid top 5, because a document found by *both* searchers at rank 2 (1/62 + 1/62 = 0.032) outscores one found by only one searcher at rank 1 (1/61 = 0.016). RRF rewards agreement, not confidence. The pin therefore has to survive the fusion as well: exact-ID matches are placed ahead of the fused list. Result: Q05 went from a false refusal in V1 to a correct answer with source RR-004 in both V2 versions; False Refusals 1 → 0.

Two environment bugs also had to be fixed: gpt-6-luna and gpt-6.1-sol reject any non-default `temperature` (400 error), so none is sent and the RAGAS LLM wrapper is subclassed to return `None`; and ragas (every version up to 0.4.3) imports `langchain_community.chat_models.vertexai`, which langchain-community 0.4 removed — a stub module is registered before importing ragas.

## Q3. Order of steps in the V2 pipeline, and why

1. **Multi-query** first — widen the net before anything narrows it (4 versions of the question).
2. **Hybrid search per query** (dense + BM25), all lists merged in one RRF — recall is cheapest to get here.
3. **Metadata filter as a soft boost** — a filtered dense search is added as one more list into the RRF instead of a hard filter. A hard `doc_type` filter would break multi-part questions such as Q06, which needs policy + order guide + ticket documents.
4. **Cross-encoder rerank** on the top 20 → 5, with exact-ID matches kept pinned. Reranking is the slowest step, so it runs last and only on a short list.
5. **Answer** with the "latest date wins" rule in the prompt.

Multi-query (B) was chosen over rewriting (A) because B always keeps the original question and can only add angles; A replaces the question with the LLM's interpretation and drifted in the demo.

## Q4. Why extra chunking was not required

Every record is already one self-contained unit: a title plus 50–120 words of content on exactly one topic, with its own ID and metadata. Splitting would break the ID → source mapping the answers depend on, and would separate a fix from the problem it fixes (known issues and tickets are "problem → fix" pairs). Merging records would blur the metadata. The records are also far below the embedding model's context limit, so there is no size reason to split. One record = one chunk keeps retrieval, citations and metadata filtering aligned.

## Q5. Single index vs multi-index: what improved, what got worse

Improved: Q20 (refused correctly, via routing), Q06 (coverage 3/5 → 4/5 expected documents, after the per-index guarantee described in Q6 below), Q19 (routed to none instead of policy_index). Recall 0.975 → 0.988, Context Precision 0.957 → 0.984.

Slightly worse: Q01 own-judge faithfulness 1.0 → 0.8 and the answer no longer mentions the 1 July 2026 date. Routing sent Q01 to policy_index only, so the stale ticket TCK-1003 was never retrieved — there was no conflict to resolve, and the terse answer restated "on day 8" from the question as if the policy said it. Overall faithfulness 0.988 → 0.972, driven by Q01 and Q14. Routing removes contradictions at the source, which is good, but it also removes the context that made the single-index answer explicit about dates.

## Q6. A question where the router picked the wrong index

Q19, "Can a customer pay with cryptocurrency?", was routed to policy_index with the reason "this asks about a store payment policy" — a defensible reading. The answer layer still refused (no policy mentions crypto), so the final output was right, but Routing Accuracy was 0.95. Fix: the router prompt now lists the payment methods the knowledge base actually covers (UPI, wallets, cards, net banking, EMI, Cash on Delivery) and states that any other payment method is out_of_scope. Routing Accuracy 0.95 → 1.00.

A related retrieval problem on Q06: three indexes were routed (policy, order, ticket) but the global reranker filled all five slots from policy and ticket documents; order_index contributed nothing and OG-005 was missed. Fix: when more than one index is routed, each index keeps its best-reranked document in the top 5 and the remaining slots are filled by global rank. Q06 went from 3/5 to 4/5 expected sources; Coverage 0.943 → 0.971.

## Q7. Handling documents that disagree

The prompt instructs the model: when documents disagree, follow the one with the latest date, mention that date, and prefer policies over tickets on equal dates; when they agree, do not mention dates. Every document block in the context carries its type and date. On Q01 (single index) TCK-1003 from 2025 says 7 days and RR-001/RR-002 from July 2026 say 10 days; the copilot answered 10 days "under the policy effective 1 July 2026". A first version of the rule made the model mention the date on *every* answer; it was tightened to "only when they disagree".

## Q8. Which metric improved the most, and which change caused it

Refusal Accuracy, 0.75 → 1.00, caused by the intent router: out_of_scope questions skip retrieval entirely, so the LLM is never handed a loosely related document it might answer from (Q20). Close second: Recall@5 0.894 → 0.975, caused by hybrid search with the exact-ID pin (Q05) and multi-query/hybrid pulling SH-005 into Q17.

## Q9. The lowest-scoring question: retrieval or generation?

By answer relevancy the lowest is Q05 at 0.19, but this is a metric artefact: RAGAS judges relevancy by generating questions from the answer and comparing them to the original; a list of refund timelines never regenerates "What does policy RR-004 say?". Retrieval (RR-004 at rank 1) and the answer are both correct.

By faithfulness the lowest is Q14 ("Tracking hasn't updated for 5 business days…") at 0.75, and Q01 at 0.8 in the multi-index version. Both are **generation** problems: all expected documents were retrieved (OG-003 and TCK-1013 at ranks 1–2 for Q14), but the answer asserted a fact taken from the question ("unchanged for more than 3 business days", "on day 8") as if it came from the documents, and the judge marked that claim unsupported. The next step would be a prompt rule to attribute facts from the question explicitly ("you said…") rather than restating them as findings.
