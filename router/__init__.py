"""Phase 3.8b router PoC — multi-model orchestration on top of corpus2skill.

This subpackage isolates the multi-model routing concern from
``corpus2skill.py`` (Karpathy Layer 4 distillation). The PoC ships a
single strategy — ``AsymmetricDebateStrategy`` — that runs a proposer
(reasoning model) and a critic (smaller / different-bias model) in
parallel on the same chunks, leveraging the Phase 3.8a NT6 verdict
(``options.num_thread=6`` parallel sweet spot, sum_conc 9.39 tps).

Public surface (kept small on purpose so half-year LLM swap-out stays
mechanical):

    - ModelOutput, ModelRunner, OllamaRunner             (router.runners)
    - RouterResult, RouterStrategy,
      AsymmetricDebateStrategy, build_critic_prompt,
      parse_critic_findings, CRITIC_PROMPT_TEMPLATE      (router.strategies)
    - ROUTER_SCHEMA_VERSION,
      build_router_record, append_router_record          (router._metrics)
"""

from ._metrics import (
    ROUTER_SCHEMA_VERSION,
    append_router_record,
    build_router_record,
)
from .runners import ModelOutput, ModelRunner, OllamaRunner
from .strategies import (
    CRITIC_PROMPT_TEMPLATE,
    AsymmetricDebateStrategy,
    RouterResult,
    RouterStrategy,
    build_critic_prompt,
    parse_critic_findings,
)

__all__ = [
    "AsymmetricDebateStrategy",
    "CRITIC_PROMPT_TEMPLATE",
    "ModelOutput",
    "ModelRunner",
    "OllamaRunner",
    "ROUTER_SCHEMA_VERSION",
    "RouterResult",
    "RouterStrategy",
    "append_router_record",
    "build_critic_prompt",
    "build_router_record",
    "parse_critic_findings",
]
