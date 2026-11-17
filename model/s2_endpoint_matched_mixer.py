"""Evaluation-only S2 mixer with a prompt-specific, sqrt-matched linear cap."""

from contextvars import ContextVar
from types import SimpleNamespace

from .kvm_mixer import SequenceMixer as OriginalKVMMixer, _round_down_to_multiple


_endpoint = ContextVar("s2_inference_linear_endpoint", default=None)


class SequenceMixer(OriginalKVMMixer):
    _CACHE_LINEAR_ENDPOINT = "s2_inference_linear_endpoint"

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self._sqrt_budget = SimpleNamespace(
            state_budget_mode="power_law",
            state_growth_factor=config.state_growth_factor,
            state_growth_exponent=config.state_growth_exponent,
            state_round_down=self.state_round_down,
            state_min_len=self.state_min_len,
            max_state_len=self.max_state_len,
        )

    def _prompt_endpoint(self, length):
        initial = min(length, self.chunk_len)
        last_update = (length - 1) // self.chunk_len * self.chunk_len
        available = max(initial, self._bswa_begin_for_total_len(length))
        sqrt_rows = OriginalKVMMixer._desired_state_len(
            self._sqrt_budget, last_update, available, initial,
        )
        return max(self.state_min_len, sqrt_rows), max(1, last_update), sqrt_rows

    def _desired_state_len(self, ctx_len, available_context, current_state_len):
        endpoint = _endpoint.get()
        if endpoint is None:
            return super()._desired_state_len(ctx_len, available_context, current_state_len)
        cap, end, _ = endpoint
        target = min(cap, cap * max(0, int(ctx_len)) // end)
        target = max(self.state_min_len, _round_down_to_multiple(target, self.state_round_down))
        return max(int(current_state_len), min(target, int(available_context), self.max_state_len))

    def forward_prefill(
        self, q, k, v, merge_gate, v_first, position_embeddings, attention_mask,
        past_key_values=None, **kwargs,
    ):
        endpoint = None if self.training else self._prompt_endpoint(q.size(2))
        context = _endpoint.set(endpoint)
        try:
            output = super().forward_prefill(
                q, k, v, merge_gate, v_first, position_embeddings, attention_mask,
                past_key_values, **kwargs,
            )
        finally:
            _endpoint.reset(context)
        if endpoint is not None and past_key_values is not None:
            states = past_key_values.get_states(self.layer_idx)
            actual = states[self._CACHE_S_K].size(2)
            if actual != endpoint[2]:
                raise AssertionError(f"Expected {endpoint[2]} endpoint rows, got {actual}")
            states[self._CACHE_LINEAR_ENDPOINT] = endpoint
        return output

    def forward_single(
        self, q, k, v, merge_gate, v_first, position_embeddings, attention_mask,
        past_key_values=None, **kwargs,
    ):
        endpoint = past_key_values.get_states(self.layer_idx)[self._CACHE_LINEAR_ENDPOINT]
        context = _endpoint.set(endpoint)
        try:
            return super().forward_single(
                q, k, v, merge_gate, v_first, position_embeddings, attention_mask,
                past_key_values, **kwargs,
            )
        finally:
            _endpoint.reset(context)
