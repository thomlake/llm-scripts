import json
from dataclasses import dataclass

from llm_scripts.chat import load_chat_template
from llm_scripts.config import Config


OUTPUT_BATCH_KEYS = [
    'attentions',
    'hidden_states',
    'scores',
    'logits',
    'past_key_values',
]


@dataclass
class GenerateConfig(Config):
    input_file: str
    output_dir: str
    model_id: str
    checkpoint_dir: str | None = None
    chat_template: str | None = None
    chat_template_file: str | None = None
    special_tokens_to_add_json: str | None = None
    pad_to_multiple_of: int = 16
    add_generation_prompt: bool = True
    add_generation_prefix: str | None = None
    batch_size: int = 8
    max_new_tokens: int = 16
    temperature: float | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    eos_token: str | None = None
    pad_token: str | None = None
    stop_strings_json: str | None = None
    num_return_sequences: int = 1
    output_attentions: bool = False
    output_hidden_states: bool = False
    output_scores: bool = False
    output_logits: bool = False
    output_past_key_values: bool = False
    strip_trailing_assistant_messages: bool = True
    device: str = 'cuda'

    def __post_init__(self):
        if self.chat_template_file:
            self.chat_template = load_chat_template(self.chat_template_file)

        if self.special_tokens_to_add_json is not None:
            self.special_tokens_to_add = json.loads(self.special_tokens_to_add_json)

        if self.stop_strings_json is not None:
            self.stop_strings = json.loads(self.stop_strings_json)

    @property
    def special_tokens_to_add(self) -> dict[str, str]:
        if not self.special_tokens_to_add_json:
            return {}

        return json.loads(self.special_tokens_to_add_json)

    @property
    def stop_strings(self) -> list[str]:
        if not self.stop_strings_json:
            return None

        return json.loads(self.stop_strings_json)

    @property
    def output_keys_to_keep(self):
        return {k for k in OUTPUT_BATCH_KEYS if getattr(self, f'output_{k}')}
