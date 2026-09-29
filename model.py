

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple




from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging

from configuration import RippleLMConfig

logger = logging.get_logger(__name__)


@dataclass
class HMECausalLMOutputWithPast(ModelOutput):
    

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    thermo_loss: Optional[torch.FloatTensor] = None
    ph_loss: Optional[torch.FloatTensor] = None
    mpf_reg_loss: Optional[torch.FloatTensor] = None
    thermo_logits: Optional[torch.FloatTensor] = None
    ph_logits: Optional[torch.FloatTensor] = None

    


class RippleLMPreTrainedModel(PreTrainedModel):
    

    config_class = RippleLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module: nn.Module):
        
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        elif hasattr(self.config.text_config, "initializer_range"):
            std = self.config.text_config.initializer_range
        else:
            std = 0.02

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        
        return self.language_model._supports_sdpa


class ContinuousPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer('freqs', freqs)

    def forward(self, x: torch.Tensor):
        args = x.unsqueeze(1) * self.freqs 
        
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return embedding

class MutationPerturbationField(nn.Module):
    """
    Mutation Perturbation Field (MPF) encoder.

    Builds a residue-wise perturbation field, decomposes it with direct/distal
    structure-aware attention, and emits six latent tokens for the LLM.
    """

    def __init__(
        self,
        esm_embed_dim: int = 320,
        output_dim: int = 4096,
        internal_dim: int = 512,
        num_mutation_types: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_diffusion_hops: int = 4,
    ):
        super().__init__()
        self.internal_dim = internal_dim
        self.num_heads = num_heads
        if internal_dim % num_heads != 0:
            raise ValueError(
                f"internal_dim ({internal_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.head_dim = internal_dim // num_heads
        self.num_diffusion_hops = num_diffusion_hops

        # learnable per-head decay for structure diffusion kernels
        self.diffusion_alpha = nn.Parameter(torch.zeros(num_heads))  # sigmoid -> 0.5

        # --- Phase 1: perturbation field projection ---
        self.delta_norm = nn.LayerNorm(esm_embed_dim)
        self.delta_proj = nn.Linear(esm_embed_dim, internal_dim)

        # --- Phase 2: DDCA dual-query cross-attention ---
        self.W_q_direct = nn.Linear(internal_dim, internal_dim)
        self.W_q_distal = nn.Linear(internal_dim, internal_dim)
        self.W_k = nn.Linear(internal_dim, internal_dim)
        self.W_v = nn.Linear(internal_dim, internal_dim)
        self.cross_attn_out_proj = nn.Linear(internal_dim, internal_dim)

        # positive per-head structure bias scales
        self.direct_bias_scale = nn.Parameter(torch.ones(num_heads))
        self.distal_bias_scale = nn.Parameter(torch.ones(num_heads))

        self.attn_dropout = nn.Dropout(dropout)

        # --- Phase 3: 4-token assembly ---
        self.site_token_proj = nn.Linear(internal_dim, internal_dim)
        self.direct_token_proj = nn.Linear(internal_dim, internal_dim)
        self.distal_token_proj = nn.Linear(internal_dim, internal_dim)
        self.context_token_proj = nn.Linear(internal_dim, internal_dim)

        self.mutation_embedding = nn.Embedding(num_mutation_types, internal_dim)
        self.rel_pos_embedding = ContinuousPositionalEncoding(internal_dim)

        # learnable task queries for property latent tokens
        self.thermo_query = nn.Parameter(torch.zeros(1, 1, internal_dim))
        self.ph_query = nn.Parameter(torch.zeros(1, 1, internal_dim))

        # self-attention over 6 tokens (4 protein + 2 property)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=internal_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(internal_dim)

        # --- output projection ---
        self.output_proj = nn.Linear(internal_dim, output_dim)

        self.apply(self._init_weights)
        nn.init.normal_(self.thermo_query, std=0.02)
        nn.init.normal_(self.ph_query, std=0.02)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ---- helpers ----

    def _normalize_weights(
        self,
        weights: torch.Tensor,
        mask: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        mask = mask.to(weights.dtype)
        weights = torch.clamp(weights, min=0.0) * mask
        denom = weights.sum(dim=dim, keepdim=True).clamp_min(1e-6)
        return weights / denom

    def _build_pathway_weights(
        self,
        contact_map: torch.Tensor,   # [B, L, L]
        residue_mask: torch.Tensor,  # [B, L]
        site_positions: torch.Tensor, # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Structure diffusion kernel decomposition.

        Direct pathway  = 1-hop contacts (k=1 term).
        Distal pathway  = multi-hop diffusion with learnable decay α,
                          S_distal[site] = Σ_{k=2}^{K} α^k · A_norm^k[site].
        """
        B, L, _ = contact_map.shape
        batch_idx = torch.arange(B, device=contact_map.device)  # [B]

        residue_mask_f = residue_mask.to(contact_map.dtype)                                    # [B, L]
        residue_pair_mask = residue_mask_f.unsqueeze(1) * residue_mask_f.unsqueeze(2)          # [B, L, L]
        contact_map = torch.clamp(contact_map, min=0.0, max=1.0) * residue_pair_mask          # [B, L, L]

        site_mask = F.one_hot(site_positions, num_classes=L).to(contact_map.dtype)             # [B, L]
        valid_residue_mask = residue_mask_f * (1.0 - site_mask)                                # [B, L]

        # Row-normalized adjacency
        A_norm = contact_map / contact_map.sum(dim=-1, keepdim=True).clamp_min(1e-6)           # [B, L, L]

        # Direct pathway: 1-hop contacts from mutation site
        direct_seed = contact_map[batch_idx, site_positions]                                   # [B, L]
        direct_weights = self._normalize_weights(
            direct_seed,
            valid_residue_mask,
            dim=-1,
        )                                                                                      # [B, L]

        # Per-head diffusion from the mutation site:
        #   S_h[site] = Σ_{k=1}^{K} α_h^k · A_norm^k[site]
        # Distal keeps only k >= 2 terms for each head.
        alpha = torch.sigmoid(self.diffusion_alpha).view(1, self.num_heads, 1)                # [1, H, 1]
        site_row = A_norm[batch_idx, site_positions]                                           # [B, L]
        power_row = site_row.unsqueeze(1).expand(-1, self.num_heads, -1)                       # [B, H, L]
        diffusion_row = alpha * power_row                                                      # [B, H, L]

        for hop in range(2, self.num_diffusion_hops + 1):
            power_row = torch.einsum("bhl,blm->bhm", power_row, A_norm)                        # [B, H, L]
            diffusion_row = diffusion_row + (alpha ** hop) * power_row

        # Distal pathway: subtract k=1 term to keep only multi-hop (k ≥ 2) contributions
        distal_seed = (
            diffusion_row - alpha * site_row.unsqueeze(1)
        ).transpose(1, 2)                                                                      # [B, L, H]
        distal_weights = self._normalize_weights(
            distal_seed,
            valid_residue_mask.unsqueeze(-1),
            dim=1,
        )                                                                                      # [B, L, H]

        return direct_weights, distal_weights

    def _cross_attention(
        self,
        query: torch.Tensor,   # [B, 1, d]
        key: torch.Tensor,     # [B, L, d]
        value: torch.Tensor,   # [B, L, d]
        bias: torch.Tensor,    # [B, L, H]
        mask: torch.Tensor,    # [B, L]  (1=valid, 0=pad)
    ) -> torch.Tensor:
        B, L, _ = key.shape
        H, d_k = self.num_heads, self.head_dim

        q = query.view(B, 1, H, d_k).transpose(1, 2)   # [B, H, 1, d_k]
        k = key.view(B, L, H, d_k).transpose(1, 2)      # [B, H, L, d_k]
        v = value.view(B, L, H, d_k).transpose(1, 2)     # [B, H, L, d_k]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [B, H, 1, L]

        # additive structure bias: [B, L, H] → [B, H, 1, L]
        scores = scores + bias.permute(0, 2, 1).unsqueeze(2)

        # mask out CLS / EOS / PAD
        if mask is not None:
            scores = scores.masked_fill(
                mask[:, None, None, :] == 0, float("-inf")
            )

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)                       # [B, H, 1, d_k]
        out = out.transpose(1, 2).contiguous().view(B, 1, -1)     # [B, 1, d]
        out = self.cross_attn_out_proj(out)
        return out.squeeze(1)                                      # [B, d]

    # ---- forward ----

    def forward(
        self,
        delta: torch.Tensor,             # [B, L, D]  per-residue perturbation field
        contact_map: torch.Tensor,        # [B, L, L]  WT contact map aligned to hidden states
        residue_mask: torch.Tensor,       # [B, L]     1 for residue positions, 0 otherwise
        site_positions: torch.Tensor,     # [B]        mutation site index in hidden states
        position: torch.Tensor,           # [B]        0-based residue position (float)
        length: torch.Tensor,             # [B]        protein sequence length (float)
        mutation_type_id: torch.Tensor,   # [B]        mutation type index (long)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = delta.size(0)

        # Phase 1 — project perturbation field
        delta_proj = self.delta_proj(self.delta_norm(delta))       # [B, L, d]

        # extract site perturbation
        batch_idx = torch.arange(B, device=delta.device)
        site_feat = delta_proj[batch_idx, site_positions]          # [B, d]

        # Phase 2 — DDCA dual-query cross-attention
        direct_weights, distal_weights = self._build_pathway_weights(
            contact_map,
            residue_mask,
            site_positions,
        )
        direct_weights_heads = direct_weights.unsqueeze(-1).expand(-1, -1, self.num_heads)
        direct_seed_feat = torch.bmm(direct_weights.unsqueeze(1), delta_proj).squeeze(1)
        distal_seed_feat = torch.einsum("blh,bld->bhd", distal_weights, delta_proj).mean(dim=1)

        Q_direct = self.W_q_direct(site_feat + direct_seed_feat).unsqueeze(1)  # [B, 1, d]
        Q_distal = self.W_q_distal(site_feat + distal_seed_feat).unsqueeze(1)
        K = self.W_k(delta_proj)                                   # [B, L, d]
        V = self.W_v(delta_proj)                                   # [B, L, d]

        direct_bias = direct_weights_heads * torch.sigmoid(
            self.direct_bias_scale
        ).view(1, 1, -1)                                           # [B, L, H]
        distal_bias = distal_weights * torch.sigmoid(
            self.distal_bias_scale
        ).view(1, 1, -1)                                           # [B, L, H]

        direct_out = self._cross_attention(
            Q_direct, K, V, direct_bias, residue_mask
        )                                                          # [B, d]
        distal_out = self._cross_attention(
            Q_distal, K, V, distal_bias, residue_mask
        )                                                          # [B, d]

        pathway_overlap_loss = (direct_weights_heads * distal_weights).sum(dim=1).mean()
        disentangle_loss = pathway_overlap_loss

        # Phase 3 — 4-token assembly
        relative_pos = position / torch.clamp(length, min=1.0)
        relative_pos = relative_pos.to(delta.dtype)

        t1 = self.site_token_proj(site_feat)                       # T1: site Δ
        t2 = self.direct_token_proj(direct_out)                    # T2: direct field
        t3 = self.distal_token_proj(distal_out)                    # T3: distal field
        t4 = self.context_token_proj(                              # T4: mutation identity
            self.mutation_embedding(mutation_type_id)
            + self.rel_pos_embedding(relative_pos)
        )

        # 6-token assembly: 4 protein + 2 property latent thoughts
        thermo_q = self.thermo_query.expand(B, -1, -1)            # [B, 1, d]
        ph_q = self.ph_query.expand(B, -1, -1)                    # [B, 1, d]
        tokens = torch.stack([t1, t2, t3, t4], dim=1)             # [B, 4, d]
        tokens = torch.cat([tokens, thermo_q, ph_q], dim=1)       # [B, 6, d]

        # multi-scale self-attention
        attn_out, _ = self.self_attn(tokens, tokens, tokens)
        tokens = self.self_attn_norm(tokens + attn_out)            # [B, 6, d]

        # projection → LLM hidden_size
        output = self.output_proj(tokens)                          # [B, 6, output_dim]

        return output, disentangle_loss

class RippleLMForConditionalGeneration(RippleLMPreTrainedModel):


    def __init__(self, config: RippleLMConfig, language_model: PreTrainedModel, esm_model: nn.Module = None):
        super().__init__(config)
        self.language_model = language_model
        self.vocab_size = config.text_config.vocab_size
        self.esm_model = esm_model
        if self.esm_model is not None:
            self.esm_model.eval()
            for param in self.esm_model.parameters():
                param.requires_grad = False
            self.esm_num_layers = self.esm_model.num_layers
            esm_embed_dim = self.esm_model.embed_dim
        else:
            esm_embed_dim = 320

        self.mpf = MutationPerturbationField(
            esm_embed_dim=esm_embed_dim,
            output_dim=config.text_config.hidden_size,
            internal_dim=esm_embed_dim,
            num_heads=config.mpf_num_heads,
            dropout=config.mpf_dropout,
        )
        self.thermo_head = nn.Sequential(
            nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.text_config.hidden_size // 2, 2),
        )
        self.ph_head = nn.Sequential(
            nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.text_config.hidden_size // 2, 2),
        )

        self.post_init()
        self.debug=True

    @torch.no_grad()
    def _compute_esm_embeddings(
        self,
        wt_esm_input_ids: torch.Tensor,
        wt_esm_attention_mask: torch.Tensor,
        mt_esm_input_ids: torch.Tensor,
        mt_esm_attention_mask: torch.Tensor,
        positions: torch.Tensor,
    ):
        # WT with contact map; MT without (WT structure is the reference baseline)
        wt_out = self.esm_model(wt_esm_input_ids, repr_layers=[self.esm_num_layers], return_contacts=True)
        mt_out = self.esm_model(mt_esm_input_ids, repr_layers=[self.esm_num_layers], return_contacts=False)

        wt_hidden = wt_out["representations"][self.esm_num_layers]  # [B, L, D]
        mt_hidden = mt_out["representations"][self.esm_num_layers]  # [B, L, D]

        # Per-residue perturbation field
        delta = mt_hidden - wt_hidden  # [B, L, D]

        # Contact map: [B, L_res, L_res]  (CLS and EOS stripped by ESM)
        contact_map_res = wt_out["contacts"]
        B, L, D = wt_hidden.shape
        L_res = contact_map_res.size(1)
        batch_idx = torch.arange(B, device=positions.device)

        # Align with hidden-state positions: residues sit at indices 1..L_res
        contact_map = delta.new_zeros(B, L, L)
        contact_map[:, 1:1 + L_res, 1:1 + L_res] = contact_map_res

        # Residue mask: keep only true residue positions (exclude CLS, EOS, PAD)
        residue_mask = wt_esm_attention_mask.clone()
        residue_mask[:, 0] = 0                        # exclude CLS
        seq_lens = wt_esm_attention_mask.sum(dim=1).long()
        residue_mask[batch_idx, seq_lens - 1] = 0     # exclude EOS

        # Mutation site index in the hidden-state tensor (after CLS)
        site_positions = positions + 1

        return delta, contact_map, residue_mask, site_positions

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Linear:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of
        )
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def forward(
        self,

        wt_esm_input_ids: Optional[torch.Tensor] = None,
        wt_esm_attention_mask: Optional[torch.Tensor] = None,
        mt_esm_input_ids: Optional[torch.Tensor] = None,
        mt_esm_attention_mask: Optional[torch.Tensor] = None,
        position: Optional[List[int]] = None,
        length: Optional[List[int]] = None,
        mutation_type_id: Optional[List[int]] = None,
        ephod_label: Optional[torch.Tensor] = None,
        pptstab_label: Optional[torch.Tensor] = None,

        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, HMECausalLMOutputWithPast]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        mpf_reg_loss = None

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if input_ids.shape[1] != 1:
                # Compute ESM embeddings online
                assert wt_esm_input_ids is not None
                assert position is not None
                assert length is not None
                assert mutation_type_id is not None

                pos_tensor = torch.tensor(position, dtype=torch.long, device=inputs_embeds.device)
                delta, contact_map, residue_mask, site_positions = \
                    self._compute_esm_embeddings(
                        wt_esm_input_ids.to(inputs_embeds.device),
                        wt_esm_attention_mask.to(inputs_embeds.device),
                        mt_esm_input_ids.to(inputs_embeds.device),
                        mt_esm_attention_mask.to(inputs_embeds.device),
                        pos_tensor,
                    )

                position_float = pos_tensor.to(inputs_embeds.dtype)
                length_float = torch.tensor(length, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                mutation_type_id_tensor = torch.tensor(mutation_type_id, dtype=torch.long, device=inputs_embeds.device)

                mpf_output, mpf_reg_loss = self.mpf(
                    delta,
                    contact_map,
                    residue_mask,
                    site_positions,
                    position_float,
                    length_float,
                    mutation_type_id_tensor,
                )
                protein_features = mpf_output[:, :4, :]
                thermo_embed = mpf_output[:, 4:5, :]
                ph_embed = mpf_output[:, 5:6, :]

                mpf_dtype = inputs_embeds.dtype
                mpf_device = inputs_embeds.device
                protein_features = protein_features.to(mpf_device, mpf_dtype)
                thermo_embed = thermo_embed.to(mpf_device, mpf_dtype)
                ph_embed = ph_embed.to(mpf_device, mpf_dtype)

                # Inject protein features (4 tokens)
                protein_mask = (
                    input_ids == self.config.protein_token_index
                ).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(
                    protein_mask, protein_features
                )

                # Inject thermo latent thought (1 token)
                thermo_inject_mask = (
                    input_ids == self.config.thermo_token_index
                ).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(
                    thermo_inject_mask, thermo_embed
                )

                # Inject pH latent thought (1 token)
                ph_inject_mask = (
                    input_ids == self.config.ph_token_index
                ).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(
                    ph_inject_mask, ph_embed
                )

            elif past_key_values is not None and input_ids.shape[1] == 1:
                if isinstance(past_key_values, Cache):
                    first_layer_past_key_value = past_key_values.key_cache[0][:, :, :, 0]
                else:
                    first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]
                batch_index, non_attended_tokens = torch.where(
                    first_layer_past_key_value.float().sum(-2) == 0
                )
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]
                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0
                attention_mask = torch.cat(
                    (extended_attention_mask, attention_mask[:, -target_length:]), dim=1
                )
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        enable_property_task = self.config.enable_property_task
        use_hidden_states = output_hidden_states
        if enable_property_task:
            use_hidden_states = True

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=use_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs.logits
        all_hidden_states = outputs.hidden_states
        loss = None
        thermo_loss = None
        ph_loss = None
        thermo_logits = None
        ph_logits = None
        if labels is not None:
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][
                    shift_attention_mask.to(logits.device) != 0
                ].contiguous()
                shift_labels = labels[..., 1:][
                    shift_attention_mask.to(labels.device) != 0
                ].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
            )

        if enable_property_task:
            last_hidden_state = all_hidden_states[-1]

            thermo_mask = input_ids == self.config.thermo_token_index
            ph_mask = input_ids == self.config.ph_token_index

            if thermo_mask.any().item():
                thermo_hidden = last_hidden_state[thermo_mask]
                thermo_logits = self.thermo_head(thermo_hidden)
                if pptstab_label is not None:
                    if pptstab_label.dim() == 1:
                        thermo_batch_indices = torch.where(thermo_mask)[0]
                        thermo_labels = pptstab_label.to(thermo_logits.device)[thermo_batch_indices]
                    else:
                        thermo_labels = pptstab_label.to(thermo_logits.device)[thermo_mask]
                    thermo_valid_mask = thermo_labels >= 0
                    if thermo_valid_mask.any().item():
                        thermo_loss = nn.CrossEntropyLoss()(
                            thermo_logits[thermo_valid_mask], thermo_labels[thermo_valid_mask].long()
                        )

            if ph_mask.any().item():
                ph_hidden = last_hidden_state[ph_mask]
                ph_logits = self.ph_head(ph_hidden)
                if ephod_label is not None:
                    if ephod_label.dim() == 1:
                        ph_batch_indices = torch.where(ph_mask)[0]
                        ph_labels = ephod_label.to(ph_logits.device)[ph_batch_indices]
                    else:
                        ph_labels = ephod_label.to(ph_logits.device)[ph_mask]
                    ph_valid_mask = ph_labels >= 0
                    if ph_valid_mask.any().item():
                        ph_loss = nn.CrossEntropyLoss()(
                            ph_logits[ph_valid_mask], ph_labels[ph_valid_mask].long()
                        )



        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return HMECausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=all_hidden_states,
            attentions=outputs.attentions,
            thermo_loss=thermo_loss,
            ph_loss=ph_loss,
            mpf_reg_loss=mpf_reg_loss,
            thermo_logits=thermo_logits,
            ph_logits=ph_logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Union[Cache, Tuple]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            elif (
                self.config.protein_token_index in input_ids
            ):
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[
                    :, -(cache_length + input_ids.shape[1]) :
                ]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        model_inputs = (
            {"inputs_embeds": inputs_embeds}
            if inputs_embeds is not None and past_key_values is None
            else {"input_ids": input_ids}
        )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "wt_esm_input_ids": kwargs.get("wt_esm_input_ids"),
                "wt_esm_attention_mask": kwargs.get("wt_esm_attention_mask"),
                "mt_esm_input_ids": kwargs.get("mt_esm_input_ids"),
                "mt_esm_attention_mask": kwargs.get("mt_esm_attention_mask"),
                "position": kwargs.get("position"),
                'length': kwargs.get('length'),
                'mutation_type_id': kwargs.get('mutation_type_id'),
                "ephod_label": kwargs.get("ephod_label"),
                "pptstab_label": kwargs.get("pptstab_label"),
            }
        )
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)


