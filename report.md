# Customer Support Copilot — Report

Amit Prasad · Dandes Academy, Applied AI Engineering · Course 02

Models used everywhere: gpt-6-luna (answers, router, query rewriting), gpt-6.1-sol (judge), text-embedding-3-small, cross-encoder/ms-marco-MiniLM-L-6-v2. K = 5. Same 20 golden questions for all three versions.

## Results

| Metric | V1 | V2 single index | V2 multi-index + routing | Target |
|---|---|---|---|---|
| Routing Accuracy | n/a | n/a | 1.00 | ≥ 0.85 |
| Precision@5 | 0.325 | 0.338 | 0.350 | |
| Recall@5 | 0.894 | 0.975 | 0.988 | ≥ 0.80 |
| Coverage (multi-source) | 0.900 | 0.943 | 0.971 | |
| Faithfulness (own judge) | 0.925 | 0.988 | 0.972 | ≥ 0.85 |
| Faithfulness (RAGAS) | 1.00 | 1.00 | 1.00 | |
| Answer Relevancy (RAGAS) | 0.708 | 0.714 | 0.708 | |
| Context Precision / Recall | 0.889 / 0.875 | 0.957 / 0.933 | 0.984 / 0.946 | |
| Refusal Accuracy | 0.75 | 0.75 | 1.00 | 100% |
| False Refusals | 1 | 0 | 0 | 0 |

All targets met by the final version.

## Q1. What worked, and what failed?

Worked: hybrid search fixed the one false refusal in V1 (Q05, "What does policy RR-004 say?"). Dense search never found RR-004 because the embedding does not treat "RR-004" as an ID. The router fixed the one wrong answer on an out-of-scope question (Q20, "Do you repair cracked TV screens?") — V1 and V2-single both picked up OG-006 ("Dandes does not repair products directly") and answered from it; the router marks it out_of_scope and skips retrieval. The latest-date rule worked on Q01: the old 2025 ticket with a 7-day window was in the top 5 and the answer still said 10 days under the July 2026 policy.

Failed or did little: query rewriting made its demo question worse. "My order hasn't come yet" became "What should I do if my order hasn't arrived yet?", which pulled in damaged-item docs (OG-005, RR-003). Multi-query gave the same top 5 as V1 for that question. Reranking on the single index changed only one slot in its demo (TCK-1006 out, RR-001 in) — with 50 documents the dense ranking is already close. Precision@5 stays around 0.33 in all versions because most questions expect only 1–2 documents, so even a perfect top 5 scores 0.2–0.4.

## Q2. One real bug and how I fixed it

BM25 did not find RR-004 for "What does policy RR-004 say?". Scores were OG-001 6.04, FAQ-004 3.54, FAQ-008 2.83, and RR-004 only 1.74. Two reasons. First, stopwords: "what", "does", "say" all scored, and FAQ-004 got 3.54 from the word "does" alone because it is a short document and BM25 favours short documents. Second, OG-001 contains the text "policy RR-004", so it matched both query words, while RR-004 itself is long and got penalised by length normalisation. Repeating the ID three times in the indexed text did not help; OG-001 still won.

Fix: a stopword list in the tokenizer, and an exact-ID rule — if a query token equals a document ID, that document goes to the top of the BM25 list. After this BM25 ranked RR-004 first. Then a second problem showed up: Reciprocal Rank Fusion still dropped RR-004 from the hybrid top 5. A document at rank 2 in both lists gets 1/62 + 1/62 = 0.032, while one at rank 1 in only one list gets 1/61 = 0.016. RRF rewards agreement, not confidence. So the pinned documents are also placed ahead of the fused list. Result: Q05 went from a false refusal in V1 to a correct answer with source RR-004 in both V2 versions; False Refusals 1 → 0.

Two environment bugs as well: gpt-6-luna and gpt-6.1-sol return a 400 error for any non-default temperature, so no temperature is sent and the RAGAS LLM wrapper is subclassed to return None; and ragas (every version up to 0.4.3) imports langchain_community.chat_models.vertexai, which langchain-community 0.4 removed, so a stub module is registered before importing ragas.

## Q3. Step order in the V2 pipeline, and why

1. Multi-query — widen first, 4 versions of the question.
2. Hybrid search per query (dense + BM25), everything merged in one RRF.
3. Metadata filter as a soft boost — a filtered dense search is one more list in the RRF, not a hard filter. A hard doc_type filter would break questions like Q06 that need policy + order guide + ticket documents.
4. Cross-encoder rerank of the top 20 → 5, exact-ID matches kept on top. Slowest step, so last and on a short list.
5. Answer, with the latest-date rule in the prompt.

I used multi-query rather than rewriting because multi-query keeps the original question and only adds variants; rewriting replaces the question with the LLM's guess and drifted in the demo.

## Q4. Why no extra chunking

Each record is already one self-contained unit: a title and 50–120 words on one topic, with its own ID and metadata. Splitting would break the ID → source mapping the answers rely on and would separate a fix from its problem (known issues and tickets are problem → fix pairs). Merging would blur the metadata. The records are far below the embedding model's limit, so there is no size reason to split either.

## Q5. Single index vs multi-index

Improved: Q20 (refused correctly through routing), Q06 (3/5 → 4/5 expected documents after the per-index guarantee described in Q6), Q19 (routed to none instead of policy_index). Recall 0.975 → 0.988, Context Precision 0.957 → 0.984.

Got worse: Q01 faithfulness 1.0 → 0.8, and the answer no longer mentions the 1 July 2026 date. Routing sent Q01 to policy_index only, so the old ticket TCK-1003 was never retrieved and there was no conflict to resolve; the shorter answer also restated "on day 8" from the question as if the policy said it. Overall faithfulness 0.988 → 0.972 (Q01 and Q14). Routing removes contradictions at the source, which is good, but it also removes the context that made the single-index answer explicit about the date.

## Q6. A question where the router picked the wrong index

Q19, "Can a customer pay with cryptocurrency?", was routed to policy_index ("asks about a store payment policy"). The answer still refused because no policy mentions crypto, so the final output was right, but Routing Accuracy was 0.95. Fix: the router prompt now lists the payment methods the knowledge base covers (UPI, wallets, cards, net banking, EMI, Cash on Delivery) and says any other payment method is out_of_scope. Routing 0.95 → 1.00.

Related retrieval problem on Q06: three indexes were routed but the global reranker filled all five slots from policy and ticket documents, so order_index contributed nothing and OG-005 was missed. Fix: when more than one index is routed, each index keeps its best-reranked document in the top 5, and the remaining slots are filled by global rank. Q06 went from 3/5 to 4/5 expected sources; Coverage 0.943 → 0.971.

## Q7. Documents that disagree

The prompt says: when documents disagree, follow the one with the latest date, mention that date, and prefer policies over tickets on equal dates; when they agree, do not mention dates. Every document block in the context carries its type and date. On Q01 (single index) TCK-1003 from 2025 says 7 days and RR-001/RR-002 from July 2026 say 10 days; the answer was 10 days "under the policy effective 1 July 2026". My first version of the rule made the model mention the date in every answer, so I tightened it to "only when they disagree".

## Q8. Metric that improved the most

Refusal Accuracy, 0.75 → 1.00, from the intent router: out-of-scope questions skip retrieval, so the LLM is never given a loosely related document to answer from (Q20). Second: Recall@5 0.894 → 0.975, from hybrid search with the exact-ID pin (Q05) and multi-query/hybrid pulling SH-005 into Q17.

## Q9. Lowest-scoring question

By answer relevancy it is Q05 at 0.19, but that is how the metric works rather than a real problem: RAGAS generates questions from the answer and compares them to the original, and a list of refund timelines never regenerates "What does policy RR-004 say?". Retrieval (RR-004 at rank 1) and the answer are both correct.

By faithfulness it is Q14 ("Tracking hasn't updated for 5 business days…") at 0.75, and Q01 at 0.8 in the multi-index version. Both are generation problems: all expected documents were retrieved (OG-003 and TCK-1013 at ranks 1–2 for Q14), but the answer repeated a fact from the question ("unchanged for more than 3 business days", "on day 8") as if it came from the documents, and the judge marked that claim unsupported. Next step would be a prompt rule to attribute such facts to the question instead of stating them as findings.
