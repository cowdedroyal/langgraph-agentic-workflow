"""GPU driver: prefill the corpus once, then answer every question under several KV policies.

kvpress's pipeline re-prefills the whole context for every question in the query-aware
setting. Scores of query-aware methods do not depend on the budget, so here the context is
prefilled once, each question is forwarded once over the full cache to score the context,
and every budget is a cheap top-k slice of the same scores.

KV policies (all select per layer and per KV head, like SnapKV):
  full     no compression
  window   StreamingLLM: instruction prefix (incl. sinks) + most recent corpus tokens (query-agnostic)
  keydiff  KeyDiff: keep keys least similar to the mean key                (query-agnostic)
  query    SnapKV: attention from the question tokens to the context      (query-aware)
  join     SnapKV re-scored with the question *plus the first-pass answer* of ``query``
           at the same budget (the answer usually names the join key)
  oracle   only the gold evidence documents; budget-free reference

Every policy keeps the system instruction before the corpus and the end-of-turn tokens after
it; budgets count corpus tokens only. Otherwise a policy that drops the answer-format rule
looks like it lost the evidence.

Semantics follow the published methods: query-agnostic policies compress before the question
is seen, so the question is forwarded over the compressed cache; query-aware policies keep
the question's own KV computed over the full context (that is what a single SnapKV prefill
does) and decode from the compressed cache.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.llama.modeling_llama import rotate_half

N_SINK = 4


@dataclass
class Gen:
    text: str
    n_tokens: int
    seconds: float
    kept_frac: float


class KVEngine:
    def __init__(self, model_path: str, attn: str = "flash_attention_2", pool_kernel: int = 5):
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="cuda", attn_implementation=attn).eval()
        cfg = self.model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.pool_kernel = pool_kernel
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.cache: DynamicCache | None = None
        self.L0 = 0
        self.P, self.S = N_SINK, 0      # protected prefix / suffix lengths (instruction, end-of-turn)
        self._capture = False
        self._scores: dict[int, torch.Tensor] = {}
        self._hooks = [layer.self_attn.register_forward_hook(self._score_hook, with_kwargs=True)
                       for layer in self.model.model.layers]

    # ------------------------------------------------------------------ tokens
    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    # ------------------------------------------------------------------ prefill
    @torch.inference_mode()
    def prefill(self, ctx_ids: list[int], protect: tuple[int, int] = (N_SINK, 0), chunk: int = 16384) -> float:
        """``protect`` = (prefix, suffix) token counts that every policy keeps and that do not count
        against the budget: the system instruction (with the answer format) before the corpus and
        the end-of-turn tokens after it. Only corpus tokens are compressed."""
        self.P, self.S = max(protect[0], N_SINK), protect[1]
        torch.cuda.synchronize()
        t0 = time.time()
        self.cache = DynamicCache()
        ids = torch.tensor([ctx_ids], device="cuda")
        for i in range(0, ids.shape[1], chunk):
            self.model.model(input_ids=ids[:, i:i + chunk], past_key_values=self.cache, use_cache=True)
        torch.cuda.synchronize()
        self.L0 = len(ctx_ids)
        return time.time() - t0

    def kv_bytes(self, n_tokens: int) -> int:
        return 2 * self.n_layers * self.n_kv * self.head_dim * 2 * n_tokens

    # ------------------------------------------------------------------ scoring
    def _score_hook(self, module, args, kwargs, output):
        if not self._capture:
            return
        h = kwargs["hidden_states"]                           # [1, w, d]
        cos, sin = kwargs["position_embeddings"]
        w = h.shape[1]
        q = module.q_proj(h).view(1, w, self.n_heads, self.head_dim)
        if hasattr(module, "q_norm"):                         # Qwen3 / Qwen3-MoE QK-norm
            q = module.q_norm(q)
        q = q.transpose(1, 2)
        q = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)
        K = kwargs["past_key_values"].layers[module.layer_idx].keys   # [1, Hkv, L0 + w, hd]
        L = K.shape[2]
        g = self.n_heads // self.n_kv
        # causal mask inside the window: window token i may not see window keys j > i
        mask = torch.zeros(w, L, device=h.device, dtype=torch.float32)
        mask[:, L - w:] = torch.triu(torch.full((w, w), float("-inf"), device=h.device), diagonal=1)
        out = torch.empty(self.n_kv, self.L0, device=h.device, dtype=torch.float32)
        for kv in range(self.n_kv):                           # one KV group at a time to bound memory
            qg = q[0, kv * g:(kv + 1) * g].float()            # [g, w, hd]
            att = torch.matmul(qg, K[0, kv].float().T) / math.sqrt(self.head_dim) + mask
            att = att.softmax(-1)[..., :self.L0]              # [g, w, L0]
            out[kv] = att.mean(dim=(0, 1))
        out = F.avg_pool1d(out.unsqueeze(0), self.pool_kernel, 1, self.pool_kernel // 2).squeeze(0)
        self._scores[module.layer_idx] = out.to(torch.float16)

    @torch.inference_mode()
    def score_window(self, ids: list[int], capture: bool = True):
        """Forward ``ids`` right after the context over the *full* cache.
        Returns (scores per layer [Hkv, L0], window KV per layer, logits of the last token)."""
        self._capture, self._scores = capture, {}
        out = self.model(input_ids=torch.tensor([ids], device="cuda"), past_key_values=self.cache,
                         use_cache=True, logits_to_keep=1)
        self._capture = False
        win = [(l.keys[:, :, self.L0:].clone(), l.values[:, :, self.L0:].clone()) for l in self.cache.layers]
        self.cache.crop(self.L0)
        scores = [self._scores[i] for i in range(self.n_layers)] if capture else None
        return scores, win, out.logits[0, -1]

    # ------------------------------------------------------------------ selections
    @property
    def corpus_len(self) -> int:
        return self.L0 - self.P - self.S

    def n_keep(self, budget: float) -> int:
        """Kept tokens = protected prefix/suffix + ``budget`` of the corpus tokens."""
        return self.P + self.S + max(1, int(round(self.corpus_len * budget)))

    def _protected(self) -> torch.Tensor:
        return torch.cat([torch.arange(self.P), torch.arange(self.L0 - self.S, self.L0)]).cuda()

    def select_topk(self, scores: list[torch.Tensor], budget: float) -> list[torch.Tensor]:
        k = self.n_keep(budget)
        prot = self._protected()
        sel = []
        for s in scores:
            s = s.float().clone()
            s[:, prot] = float("inf")                          # instruction, sinks, end-of-turn
            sel.append(s.topk(k, dim=-1).indices.sort(dim=-1).values)
        return sel

    def select_window(self, budget: float) -> list[torch.Tensor]:
        """StreamingLLM-style: protected prefix (contains the sinks) + the most recent corpus tokens."""
        n = self.n_keep(budget) - self.P - self.S
        idx = torch.cat([torch.arange(self.P), torch.arange(self.L0 - self.S - n, self.L0)]).cuda()
        return [idx.unsqueeze(0).expand(self.n_kv, -1) for _ in range(self.n_layers)]

    @torch.inference_mode()
    def keydiff_scores(self) -> list[torch.Tensor]:
        out = []
        for l in self.cache.layers:
            K = l.keys[0, :, :self.L0].float()                 # [Hkv, L0, hd]
            anchor = F.normalize(K, dim=-1).mean(dim=1, keepdim=True)
            out.append((-F.cosine_similarity(K, anchor, dim=-1)).to(torch.float16))
        return out

    @torch.inference_mode()
    def select_keydiff_block(self, budget: float, block: int = 512) -> list[torch.Tensor]:
        """KeyDiff as in the paper: stream the corpus in blocks, keep the ``k`` keys least similar
        to the mean (normalized) key of the *current* kept set + new block. One-shot KeyDiff over
        the whole context (``keydiff_scores``) is what kvpress's KeyDiffPress does without BlockPress.
        Keys are the ones computed with the full context (we score a cached prefill)."""
        k = self.n_keep(budget) - self.P - self.S
        a, b = self.P, self.L0 - self.S
        prot = self._protected()
        sel = []
        for l in self.cache.layers:
            Kn = F.normalize(l.keys[0, :, a:b].float(), dim=-1)          # [Hkv, n, hd]
            kept = torch.empty(self.n_kv, 0, dtype=torch.long, device="cuda")
            for s in range(0, b - a, block):
                new = torch.arange(s, min(s + block, b - a), device="cuda").unsqueeze(0).expand(self.n_kv, -1)
                cand = torch.cat([kept, new], dim=1)
                if cand.shape[1] <= k:
                    kept = cand
                    continue
                Kc = Kn.gather(1, cand.unsqueeze(-1).expand(-1, -1, Kn.shape[-1]))
                anchor = F.normalize(Kc.mean(dim=1, keepdim=True), dim=-1)
                score = -(Kc * anchor).sum(-1)
                kept = cand.gather(1, score.topk(k, dim=-1).indices)
            idx = torch.cat([prot.unsqueeze(0).expand(self.n_kv, -1), kept + a], dim=1)
            sel.append(idx.sort(dim=-1).values)
        return sel

    def select_spans(self, spans: list[tuple[int, int]]) -> list[torch.Tensor]:
        idx = set(range(self.P)) | set(range(self.L0 - self.S, self.L0))
        for a, b in spans:
            idx.update(range(a, b))
        t = torch.tensor(sorted(idx), device="cuda")
        return [t.unsqueeze(0).expand(self.n_kv, -1) for _ in range(self.n_layers)]

    def kept_frac(self, sel: list[torch.Tensor]) -> float:
        """Fraction of *corpus* tokens kept (protected tokens excluded)."""
        return (sel[0].shape[-1] - self.P - self.S) / self.corpus_len

    # ------------------------------------------------------------------ caches
    def gather(self, sel: list[torch.Tensor], extra=None) -> DynamicCache:
        c = DynamicCache()
        for i, l in enumerate(self.cache.layers):
            idx = sel[i].unsqueeze(0).unsqueeze(-1).expand(1, -1, -1, self.head_dim)
            k = l.keys[:, :, :self.L0].gather(2, idx)
            v = l.values[:, :, :self.L0].gather(2, idx)
            if extra is not None:
                k = torch.cat([k, extra[i][0]], dim=2)
                v = torch.cat([v, extra[i][1]], dim=2)
            c.update(k, v, i)
        return c

    # ------------------------------------------------------------------ decoding
    @torch.inference_mode()
    def generate(self, cache: DynamicCache, ids: list[int], start_pos: int, max_new: int,
                 kept_frac: float) -> Gen:
        """Greedy decoding. ``ids`` are forwarded at positions start_pos.. over ``cache``."""
        torch.cuda.synchronize()
        t0 = time.time()
        pos = torch.arange(start_pos, start_pos + len(ids), device="cuda").unsqueeze(0)
        out = self.model(input_ids=torch.tensor([ids], device="cuda"), past_key_values=cache,
                         position_ids=pos, use_cache=True, logits_to_keep=1)
        nxt = int(out.logits[0, -1].argmax())
        gen, p = [nxt], start_pos + len(ids)
        while nxt not in self.eos and len(gen) < max_new:
            out = self.model(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=cache,
                             position_ids=torch.tensor([[p]], device="cuda"), use_cache=True)
            nxt = int(out.logits[0, -1].argmax())
            gen.append(nxt)
            p += 1
        torch.cuda.synchronize()
        return Gen(self.tok.decode(gen, skip_special_tokens=True), len(gen), time.time() - t0, kept_frac)

    def answer_full(self, q_ids: list[int], max_new: int) -> Gen:
        g = self.generate(self.cache, q_ids, self.L0, max_new, 1.0)
        self.cache.crop(self.L0)
        return g

    def answer_agnostic(self, sel: list[torch.Tensor], q_ids: list[int], max_new: int) -> Gen:
        """Context compressed before the question: forward the question over the compressed cache."""
        c = self.gather(sel)
        return self.generate(c, q_ids, self.L0, max_new, self.kept_frac(sel))

    def answer_aware(self, sel: list[torch.Tensor], q_ids: list[int], q_win, max_new: int) -> Gen:
        """Query-aware: keep the question KV computed over the full context (all but the last
        question token), then decode from the last question token on the compressed cache."""
        extra = [(k[:, :, :-1], v[:, :, :-1]) for k, v in q_win]
        c = self.gather(sel, extra)
        return self.generate(c, q_ids[-1:], self.L0 + len(q_ids) - 1, max_new, self.kept_frac(sel))

    # ------------------------------------------------------------------ analysis
    @staticmethod
    def doc_ranks(scores: list[torch.Tensor], spans: list[tuple[int, int]], targets: list[int]) -> list[int]:
        """Per layer: best rank (1 = top) among ``targets`` of the per-document attention mass
        (score summed over the document's tokens, averaged over KV heads)."""
        starts = torch.tensor([a for a, _ in spans], device="cuda")
        ranks = []
        for s in scores:
            cs = torch.nn.functional.pad(s.float().mean(0).cumsum(0), (1, 0))
            mass = cs[torch.tensor([b for _, b in spans], device="cuda")] - cs[starts]
            order = mass.argsort(descending=True)
            pos = torch.empty_like(order)
            pos[order] = torch.arange(len(order), device="cuda")
            ranks.append(int(pos[targets].min()) + 1)
        return ranks
