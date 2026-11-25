# Background guides

These are general learning guides on the concepts OpsMind builds on. They are **not** documentation of OpsMind's own implementation, and some examples use different tools or settings than OpsMind does. For how OpsMind works, see the [README](../README.md) and [ARCHITECTURE.md](../ARCHITECTURE.md).

| Guide | Topic |
|---|---|
| [15_GUARDRAILS.md](15_GUARDRAILS.md) | LLM guardrails, using NVIDIA NeMo Guardrails as the example |
| [16_LLM_GATEWAY.md](16_LLM_GATEWAY.md) | LLM gateways (routing, fallbacks, caching) with Portkey |
| [17_EVALS.md](17_EVALS.md) | Evaluating RAG systems: metrics, judges and pipelines |
| [GATEWAY GUARDRAIL EVAL/](GATEWAY%20GUARDRAIL%20EVAL/) | PDF versions of the gateway, guardrail and evaluation guides |

The notebooks in [`../notebooks/`](../notebooks/) are standalone experiments for these topics. They need extra packages that OpsMind itself doesn't use (`nemoguardrails`, `deepeval`, `ragas`, `pandas`), and may reference older model names.
