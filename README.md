# rxcite

[![CI](https://github.com/VigneshReddy23/rxcite/actions/workflows/ci.yml/badge.svg)](https://github.com/VigneshReddy23/rxcite/actions/workflows/ci.yml)

**A measured RAG service over FDA drug labels: every answer cites its source label, it refuses when the labels can't answer, and every design choice is backed by an experiment.**

Ask *"Can I take ibuprofen if I've had stomach bleeding?"* and rxcite searches 6,634 passages from 298 official FDA drug labels, answers only from what it finds, cites each claim `[n]` with a link to the label on DailyMed, and says "I don't know" for questions the labels don't cover.

Built with FastAPI, Postgres + pgvector, local embeddings (fastembed), Claude Haiku 4.5 on Amazon Bedrock, Docker, and Terraform (ECS Fargate + RDS). Answer faithfulness is checked with **[judgekit](https://github.com/VigneshReddy23/judgekit)**, my LLM evaluation harness, using its human-calibrated groundedness judge.

## Results

All numbers are measured, not estimated. Details and raw outputs are in [`results/`](results/).

| What | Result |
|---|---|
| Retrieval (150-question test set, section-level) | **recall@1 0.700 · recall@5 0.940 · MRR 0.795** |
| Faithfulness (judged by judgekit, answered questions) | **57 / 57** supported by the retrieved passages |
| Answers with citations | **57 / 57** |
| Refusals (held-out half, 30 answerable + 20 unanswerable) | **95%** correct refusals · **3.3%** false refusals |
| Refused before any LLM call (confidence threshold) | **70%** of unanswerable questions: safer and free |
| Latency, per question (local) | retrieval **p50 14 ms / p95 19 ms** · end to end **p50 1.8 s / p95 3.2 s** |
| Cost | **$2.35 per 1,000 answered questions** (1,505 in + 127 out tokens, Bedrock Haiku 4.5 prices) |
| Load, local Docker, `/search` (50 users) | 2,771 requests, **0 failures**, p50 24 ms, p95 77 ms |
| Load, AWS, `/ask` with real Bedrock calls (5 users) | 41 requests, **0 failures**, p50 2.4 s, p95 3.0 s |

### What improved retrieval, one change at a time

Measured on the same 150 questions (section-level MRR). Rebuilding the approximate HNSW index alone moves MRR by about 0.01, so smaller differences are treated as noise.

| Step | Change | Best mode | MRR |
|---|---|---|---|
| Baseline | vector / keyword / hybrid (RRF) / hybrid + MiniLM reranker | vector | 0.754 (hybrid + rerank: 0.677) |
| A | reranker sees the drug name and section, not just the passage | hybrid + rerank | 0.677 → **0.745** |
| B | embed each passage with its drug name and section | vector | 0.761 → **0.795** |
| C | rerank vector candidates only | vector + rerank | 0.755 (worse than vector) |
| D | larger reranker (`bge-reranker-base`) | vector + rerank | 0.768, ~2.6 s/query (worse, ~6× slower) |

**Decision: ship plain vector search with contextual embeddings.** On this data, keyword search (Postgres full-text) reaches only 0.278 MRR and drags hybrid fusion down, and neither reranker beat the embeddings. That goes against the usual "hybrid + rerank is always better" advice, and the measurement is why rxcite doesn't use it by default.

## Architecture

```mermaid
flowchart LR
    fda[openFDA drug labels<br/>public domain] --> ingest[download + section-aware chunking<br/>180 words, 30 overlap]
    ingest --> embed[local embeddings<br/>bge-small, drug + section prefix]
    embed --> pg[(Postgres + pgvector<br/>HNSW index)]
    user[client] --> api[FastAPI<br/>/ask · /search · /health]
    api --> retr[vector search<br/>top 5]
    retr --> pg
    retr --> gate{top similarity<br/>≥ 0.74?}
    gate -- no --> refuse[refuse<br/>no LLM call]
    gate -- yes --> llm[Claude Haiku 4.5<br/>answer only from sources, cite n]
    llm --> api
    evalq[150 test questions<br/>+ 40 unanswerable] -.-> metrics[recall@k · MRR<br/>faithfulness via judgekit<br/>latency · cost]
```

On AWS (Terraform, `infra/`): an ALB forwards to the API on ECS Fargate (ARM64). RDS Postgres 17 with pgvector is reachable only from the API's security group. The database URL is stored encrypted in SSM Parameter Store, and the task role can invoke only the one Bedrock model.

## Quickstart (local)

```bash
git clone https://github.com/VigneshReddy23/rxcite && cd rxcite
pip install -e ".[dev]"
docker compose up -d db                      # Postgres + pgvector
rxcite download --drugs 300                  # ~13 min, paced for openFDA's rate limit
rxcite index                                 # chunk + embed locally (~8 min on a laptop)
rxcite ask "Does gabapentin make you sleepy?"
```

Answering needs model access. Bedrock is the default; any provider judgekit supports works:

```bash
export RXCITE_PROVIDER=anthropic RXCITE_MODEL=claude-haiku-4-5    # uses ANTHROPIC_API_KEY
export RXCITE_PROVIDER=openai_compatible RXCITE_MODEL=llama3.1:8b RXCITE_BASE_URL=http://localhost:11434/v1
```

Run the API: `docker compose up -d --build`, then `POST http://localhost:8000/ask {"question": "..."}`.

### Reproduce the evaluation

```bash
rxcite make-questions --n 150 --seed 42    # retrieval test set (already in eval/)
rxcite eval-retrieval                      # recall@k and MRR for every mode
rxcite make-unanswerable                   # off-topic + drugs outside the corpus (already in eval/)
rxcite eval-answers && rxcite report-answers   # faithfulness (judgekit) and refusals
rxcite report-cost                         # latency percentiles and cost per 1k
RXCITE_ENDPOINT=search locust -f bench/locustfile.py --headless -u 50 -r 10 -t 60s --host http://localhost:8000
```

### Deploy to AWS

```bash
rxcite export-chunks                               # indexed chunks + embeddings -> data/seed/
cd infra && terraform init
terraform apply -target=aws_ecr_repository.api     # registry first
# build and push the image to the ECR URL (see infra/README.md), then:
terraform apply                                    # RDS, ECS, ALB, IAM, S3 seed
# seed RDS with a one-off ECS task, test, and when done:
terraform destroy
```

Step-by-step commands are in [`infra/README.md`](infra/README.md).

## Design decisions

- **Measure before choosing.** Four retrieval modes stay in the code so they can be compared on the same questions. The shipped default (vector) is the one that won, not the one I expected to win.
- **Contextual chunks.** Many label passages never name their drug ("May cause drowsiness"). Prefixing the drug and section before embedding raised recall@1 from 0.647 to 0.700.
- **Refuse before generating.** Off-topic questions scored 0.39–0.56 top similarity; answerable ones 0.71–0.91. A threshold chosen on a calibration half (0.74) refuses 70% of unanswerable questions with no LLM call, and the LLM's own instructed refusal catches the rest.
- **Cite everything, verify faithfulness independently.** The prompt requires `[n]` citations, and a separate, human-calibrated judge (judgekit) checks that answers stick to their sources.
- **Local embeddings.** Retrieval runs on the API's own CPU: no per-query API cost, no data sent out for search, and p50 14 ms.
- **Section-aware chunking with overlap.** Chunks never mix sections (a "Warnings" passage stays a warning), and a 30-word overlap keeps sentences intact across boundaries.
- **Treat the model's input as data.** Sources and the question are wrapped in tags, and the prompt tells the model to ignore instructions inside them (prompt-injection defence).
- **Production basics.** Connection pooling, health checks, models baked into the image, retries with backoff for openFDA, least-privilege IAM, secrets in SSM, no public database.
- **Test isolation.** Database tests run in their own Postgres schema. An early version dropped the real table; a regression test now guards against that.

## Known limitations

- **Synthetic questions.** The retrieval test set was written by an LLM from sampled passages (instructed to paraphrase), so it may be easier than real user questions, and it favours whichever wording style the LLM uses.
- **Relevance is section-level.** A retrieved passage counts as correct if it's from the same label section as the source passage. Near-duplicate text in other labels (e.g. acetaminophen warnings in many combination products) is counted as a miss.
- **One label per generic name.** The corpus has 298 drugs, not every manufacturer's version; some product-specific details are missing.
- **Keyword search is Postgres full-text (`ts_rank_cd`), not BM25.** A true BM25 implementation might fuse better.
- **Small evaluation sets** (150 retrieval questions, 100 answer questions). Differences of a few points are within noise.
- **Two "unknown drug" questions named generic product types** (pain relief patch, hand sanitizer) that the corpus does cover; the system answered them faithfully, so the correct-refusal rate is slightly understated.
- **Not medical advice.** Answers restate label text; they don't replace a pharmacist or doctor.

## Data and licences

- **Drug labels:** [openFDA](https://open.fda.gov/terms/), public domain (CC0 1.0). Links point to the same labels on DailyMed.
- **Evaluation questions** (`eval/`): generated for this project with Claude Haiku 4.5 from the CC0 labels.
- **Code:** MIT.

## Project layout

```
src/rxcite/
  ingest.py       openFDA download (with retries) and section-aware chunking
  embeddings.py   local embeddings and cross-encoder rerankers (fastembed)
  db.py           Postgres schema, pgvector + full-text search, connection pool
  retrieval.py    vector / keyword / hybrid (RRF) / reranked modes
  answer.py       prompt, citations, refusal
  service.py      retrieve -> answer pipeline
  api.py          FastAPI app
  evaluate.py     retrieval test set, recall@k, MRR
  answer_eval.py  faithfulness (judgekit) and refusal evaluation
  bench.py        latency percentiles and cost per 1k
  cli.py          `rxcite` commands
bench/            Locust load test
eval/             test questions (answerable + unanswerable)
results/          every measured result in this README
infra/            Terraform for AWS
```
