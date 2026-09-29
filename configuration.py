

from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama import LlamaConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class RippleLMConfig(PretrainedConfig):
    model_type = "HME"
    is_composition = False

    def __init__(
        self,
        text_config=None,
        protein_token: str = None,
        protein_token_index: int = None,
        thermo_token: str = None,
        thermo_token_index: int = None,
        ph_token: str = None,
        ph_token_index: int = None,
        enable_property_task: bool = False,
        esm_model_name: str = None,
        mpf_num_heads: int = 8,
        mpf_dropout: float = 0.1,
        mpf_disentangle_weight: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = LlamaConfig()
            logger.info("text_config is None. Initializing with a default LlamaConfig.")
        self.text_config = text_config

        self.protein_token = protein_token
        self.protein_token_index = protein_token_index
        self.thermo_token = thermo_token
        self.thermo_token_index = thermo_token_index
        self.ph_token = ph_token
        self.ph_token_index = ph_token_index
        self.enable_property_task = enable_property_task
        self.esm_model_name = esm_model_name
        self.mpf_num_heads = mpf_num_heads
        self.mpf_dropout = mpf_dropout
        self.mpf_disentangle_weight = mpf_disentangle_weight
        self.tie_word_embeddings = False
