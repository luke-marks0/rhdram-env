"""The Qwen3 + LoRA policy: generate one assistant turn, capture its exact sampled
tokens and their logprobs, and recompute logprobs for the GRPO update.

This is the only module that imports torch/transformers/peft, and it does so lazily
so the rollout driver, reference/controls, config, and their tests run without the
model stack installed. Install it with ``requirements-train.txt`` on the GPU box.

Design notes that matter for correctness:

* Only the sampled assistant token ids (``gen_ids``) are ever scored or trained on;
  prompts, tool results and templates never enter the loss. Because those ids are
  captured straight from ``generate`` rather than recovered by re-tokenizing the
  rendered turn, masking is exact regardless of how the chat template renders.
* ``old``/``new``/``reference`` logprobs all come from :meth:`logprobs_for` on raw
  logits (no temperature rescaling), so the PPO ratio and the KL are self-consistent.
* The KL reference is the base model reached by disabling the LoRA adapter — no
  second copy of the 8B weights is held.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from . import prompt
from .prompt import TOOL_ROLE
from .rollout import ActResult, GenRecord

# Qwen3 attention + MLP projections — the standard LoRA target set for this family.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class QwenPolicy:
    """A LoRA-adapted causal LM wrapped as a rollout :class:`~rollout.Policy`."""

    def __init__(self, config: dict[str, Any]) -> None:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        model_cfg = config["model"]
        rollout_cfg = config["rollout"]
        self.device = model_cfg["device"]
        self.dtype = torch.bfloat16 if model_cfg["dtype"] == "bfloat16" else torch.float32
        self.temperature = float(rollout_cfg["temperature"])
        self.max_new_tokens = int(rollout_cfg["max_new_tokens"])
        self.max_context_tokens = int(rollout_cfg["max_context_tokens"])

        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], revision=model_cfg["revision"])
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            model_cfg["name"], revision=model_cfg["revision"], torch_dtype=self.dtype
        )
        if model_cfg["gradient_checkpointing"]:
            base.gradient_checkpointing_enable()
            base.config.use_cache = False
        lora = LoraConfig(
            r=int(model_cfg["lora_rank"]), lora_alpha=int(model_cfg["lora_alpha"]),
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=LORA_TARGET_MODULES,
        )
        self.model = get_peft_model(base, lora).to(self.device)

    # ---- rollout Policy interface -------------------------------------------
    def act(self, *, obs: Any, messages: list[dict[str, str]]) -> ActResult:
        torch = self.torch
        prompt_ids = self._encode(messages)
        input_ids = torch.tensor([prompt_ids], device=self.device)
        self.model.eval()
        # Gradient checkpointing sets use_cache=False for the training forward; the KV
        # cache is safe (and much faster) for autoregressive generation, so enable it
        # here and restore afterward.
        cache_was = self.model.config.use_cache
        self.model.config.use_cache = True
        with torch.no_grad():
            out = self.model.generate(
                input_ids, do_sample=self.temperature > 0, temperature=self.temperature or 1.0,
                top_p=0.95, max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        self.model.config.use_cache = cache_was
        gen_ids = out[0, len(prompt_ids):].tolist()
        if not gen_ids:  # model emitted nothing; keep the episode alive with an empty turn
            gen_ids = [self.tokenizer.eos_token_id]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        old_logprobs = self.logprobs_for(prompt_ids, gen_ids, grad=False).tolist()
        action = prompt.parse_action(text)
        return ActResult(action=action, text=text, gen=GenRecord(prompt_ids, gen_ids, old_logprobs))

    # ---- logprobs (shared by sampling, the PPO ratio, and the KL anchor) -----
    def logprobs_for(self, prompt_ids: list[int], gen_ids: list[int], *, grad: bool, reference: bool = False):
        """Per-token logprobs of ``gen_ids`` given ``prompt_ids``.

        ``reference=True`` disables the LoRA adapter to score under the frozen base
        model (the KL anchor). ``grad`` toggles autograd for the policy-update pass.
        """
        torch = self.torch
        ids = torch.tensor([prompt_ids + gen_ids], device=self.device)
        start = len(prompt_ids)
        grad_ctx = nullcontext() if grad else torch.no_grad()
        adapter_ctx = self.model.disable_adapter() if reference else nullcontext()
        with adapter_ctx, grad_ctx:
            logits = self.model(ids).logits[0, start - 1:-1, :]
            logp = torch.log_softmax(logits.float(), dim=-1)
            target = torch.tensor(gen_ids, device=self.device)
            return logp.gather(-1, target[:, None]).squeeze(-1)

    # ---- prompt encoding with left-truncation -------------------------------
    def _encode(self, messages: list[dict[str, str]]) -> list[int]:
        budget = self.max_context_tokens - self.max_new_tokens
        rendered = self._template(messages)
        if len(rendered) <= budget:
            return rendered
        # Over budget: keep the system turn and drop the oldest dialogue turns until
        # the prompt fits, so the most recent observations always survive.
        system = messages[:1]
        tail = list(messages[1:])
        while tail and len(self._template(system + tail)) > budget:
            tail.pop(0)
        return self._template(system + tail)

    def _template(self, messages: list[dict[str, str]]) -> list[int]:
        # Map the transcript's "tool" role to "user" for templates that lack a tool
        # role with plain-string content; masking is unaffected (only assistant tokens
        # are trained on, and those come from generation, not the template).
        norm = [{"role": "user" if m["role"] == TOOL_ROLE else m["role"], "content": m["content"]} for m in messages]
        return self.tokenizer.apply_chat_template(norm, add_generation_prompt=True, tokenize=True)

    # ---- checkpoint plumbing (the trainer owns optimizer/scheduler/RNG) ------
    def save_adapter(self, path: str) -> None:
        self.model.save_pretrained(path)

    def load_adapter(self, path: str) -> None:
        from peft import PeftModel

        base = self.model.get_base_model()
        self.model = PeftModel.from_pretrained(base, path, is_trainable=True).to(self.device)

    def trainable_parameters(self) -> list[Any]:
        return [p for p in self.model.parameters() if p.requires_grad]
