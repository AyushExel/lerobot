"""Wall-X configuration extensions for the native Transformers Qwen2.5-VL config.

Wall-X checkpoints use the legacy, flat Qwen2.5-VL config layout.  The native
``Qwen2_5_VLConfig`` accepts that layout and moves text-model fields into its
``text_config`` sub-config, so only the Wall-X-specific MoE fields need to be
declared here.
"""

from huggingface_hub.dataclasses import strict
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig as TransformersQwen2_5_VLConfig,
    Qwen2_5_VLTextConfig as TransformersQwen2_5_VLTextConfig,
    Qwen2_5_VLVisionConfig,
)

_LEGACY_TEXT_ATTRIBUTES = {
    "attention_dropout",
    "attention_moe",
    "dim_inputs",
    "dof_config",
    "experts",
    "hidden_act",
    "hidden_size",
    "initializer_range",
    "intermediate_size",
    "layer_types",
    "max_position_embeddings",
    "max_window_layers",
    "mlp_moe",
    "noise_scheduler",
    "num_attention_heads",
    "num_experts",
    "num_hidden_layers",
    "num_key_value_heads",
    "pad_token_id",
    "rms_norm_eps",
    "sliding_window",
    "use_cache",
    "use_sliding_window",
    "vocab_size",
}


@strict
class Qwen2_5_VLTextConfig(TransformersQwen2_5_VLTextConfig):
    """Native Qwen2.5-VL text config plus Wall-X's hard-routed MoE settings."""

    num_experts: int = 4
    experts: list[dict] | None = None
    dof_config: dict | None = None
    noise_scheduler: dict | None = None
    dim_inputs: tuple[int, ...] | list[int] = (1536, 1536)
    attention_moe: bool = False
    mlp_moe: bool = False

    def __post_init__(self, **kwargs):
        # The previous config normalized this field to a tuple. Keep that
        # behavior so saved configs and ActionHead see the same value type.
        self.dim_inputs = tuple(self.dim_inputs)
        super().__post_init__(**kwargs)


@strict
class Qwen2_5_VLConfig(TransformersQwen2_5_VLConfig):
    """Native composite Qwen2.5-VL config with a Wall-X text sub-config.

    The native composite loader supports both current nested configs and the
    flat layout used by existing ``wall-oss-flow`` checkpoints.
    """

    sub_configs = {
        "vision_config": Qwen2_5_VLVisionConfig,
        "text_config": Qwen2_5_VLTextConfig,
    }

    def __getattr__(self, name):
        """Keep legacy direct access to fields now owned by ``text_config``.

        Wall-X historically used a flat config and accesses fields such as
        ``hidden_size`` and ``num_experts`` directly. Forwarding unknown
        attributes preserves that API without duplicating the native config.
        """
        text_config = self.__dict__.get("text_config")
        if name in _LEGACY_TEXT_ATTRIBUTES and text_config is not None and hasattr(text_config, name):
            return getattr(text_config, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")


__all__ = ["Qwen2_5_VLConfig", "Qwen2_5_VLTextConfig", "Qwen2_5_VLVisionConfig"]
