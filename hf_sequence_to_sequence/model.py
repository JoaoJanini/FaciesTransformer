""" PyTorch Facies model."""
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import PreTrainedModel
from hf_sequence_to_sequence.configuration import FaciesConfig
from hf_sequence_to_sequence.embedding import PositionalEncoding, Embeddings
import torch.utils.checkpoint
import torch.nn.functional as F
import math
from torch.nn import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    LayerNorm,
    CrossEntropyLoss,
)

from transformers.modeling_outputs import (
    Seq2SeqModelOutput,
    Seq2SeqLMOutput,
    BaseModelOutput,
)


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1
        )
    return mask


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask
    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class FaciesModelEncoderTS(PreTrainedModel):
    def __init__(self, config: FaciesConfig):
        super().__init__(config)
        self.config = config
        self.encoder_layer = TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.encoder_attention_heads,
            dim_feedforward=config.encoder_ffn_dim,
            dropout=config.encoder_layerdrop,
            batch_first=True,
        )
        self.norm = LayerNorm(config.d_model)
        self.model_timestep = TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=config.encoder_layers,
            norm=self.norm,
        )

        self.positional_encoding = PositionalEncoding(config.d_model, config.dropout)
        self.embed_tokens_steps = torch.nn.Linear(
            config.n_input_features, config.d_model
        )

        self.output_layer = nn.Linear(d_model, self.config.n_input_features)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        # Initialize weights and apply final processing

    def get_input_embeddings(self):
        return self.embed_tokens_steps

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:

        inp = self.embed_tokens_steps(input_ids) * math.sqrt(
            self.d_model
        )  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.positional_encoding(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.encoder(inp)  # (seq_length, batch_size, d_model)
        output = self.act(
            output
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        return BaseModelOutput(last_hidden_state=output)


class FaciesModelEncoder(PreTrainedModel):
    def __init__(self, config: FaciesConfig):
        super().__init__(config)
        self.config = config
        self.cat_features_indexes = self.config.cat_features_indexes
        self.encoder_layer = TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.encoder_attention_heads,
            dim_feedforward=config.encoder_ffn_dim,
            dropout=config.encoder_layerdrop,
            batch_first=True,
        )
        self.norm = LayerNorm(config.d_model)
        self.model_timestep = TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=config.encoder_layers,
            norm=self.norm,
        )
        self.model_channel = TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=config.encoder_layers,
            norm=self.norm,
        )
        self.embed_tokens_channel = torch.nn.Linear(self.config.n_input_features, config.d_model)
        self.embed_tokens_steps = torch.nn.Linear(
            config.sequence_len, config.d_model
        )

        self.positional_encoding = PositionalEncoding(config.d_model, config.dropout)

        self.gate = torch.nn.Linear(
            config.d_model * config.n_input_features
            + config.d_model * config.sequence_len,
            2,
        )
        self.output_linear = torch.nn.Linear(
            config.d_model * config.n_input_features
            + config.d_model * config.sequence_len,
            config.d_model,
        )
        # Initialize weights and apply final processing

    def get_input_embeddings(self):
        return self.embed_tokens_channel

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        # step-wise
        # score矩阵为 input， 默认加mask 和 pe
        encoding_1 = self.embed_tokens_steps(input_ids)
        encoding_1 = self.positional_encoding(encoding_1)
        output_encoder_1 = self.model_timestep(
            encoding_1, mask=None, src_key_padding_mask=None
        )
        channel_encoding = self.embed_tokens_channel(input_ids.transpose(-1, -2))
        output_encoder_2 = self.model_channel(
            channel_encoding, mask=None, src_key_padding_mask=None
        )
        output_encoder_1 = output_encoder_1.reshape(output_encoder_1.shape[0], -1)
        output_encoder_2 = output_encoder_2.reshape(output_encoder_2.shape[0], -1)
        gate = F.softmax(
            self.gate(torch.cat([output_encoder_1, output_encoder_2], dim=-1)), dim=-1
        )
        encoding = torch.cat(
            [output_encoder_1 * gate[:, 0:1], output_encoder_2 * gate[:, 1:2]], dim=-1
        )

        # 输出
        output_encoder = self.output_linear(encoding)
        output_encoder = output_encoder.reshape(
            output_encoder.shape[0], 1, output_encoder.shape[1]
        )

        return BaseModelOutput(last_hidden_state=output_encoder)


class FaciesModelDecoder(PreTrainedModel):
    def __init__(self, config: FaciesConfig):
        super().__init__(config)
        self.config = config

        self.decoder_layer = TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.decoder_attention_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.decoder_layerdrop,
            batch_first=True,
        )
        self.norm = LayerNorm(config.d_model)
        self.model = TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=config.decoder_layers,
            norm=self.norm,
        )
        self.positional_encoding = PositionalEncoding(config.d_model, config.dropout)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.d_model, config.pad_token_id
        )

        # Initialize weights and apply final processing

    def get_input_embeddings(self):
        return self.embed_tokens

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length=0
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                past_key_values_length=past_key_values_length,
            ).to(inputs_embeds.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )
        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            input_shape,
            inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        hidden_state = self.positional_encoding(inputs_embeds)

        decoder_outputs = self.model(
            hidden_state,
            encoder_hidden_states,
            tgt_mask=attention_mask,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
        )

        return BaseModelOutput(last_hidden_state=decoder_outputs)


class FaciesModel(PreTrainedModel):
    def __init__(self, config: FaciesConfig):
        super().__init__(config)

        self.encoder = FaciesModelEncoder(config)
        self.decoder = FaciesModelDecoder(config)
        self.encoder_embed = self.encoder.get_input_embeddings()
        self.decoder_embed = self.decoder.get_input_embeddings()

        # Initialize weights and apply final processing

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens_channel = self.shared
        self.decoder.embed_tokens = self.decoder_inputs_embed

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:

        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0])

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        )


class FaciesForConditionalGeneration(PreTrainedModel):
    base_model_prefix = "model"

    def __init__(self, config: FaciesConfig):
        super().__init__(config)
        self.model = FaciesModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs[0])

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return Seq2SeqLMOutput(loss=masked_lm_loss, logits=lm_logits)

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

class FaciesForClassification(PreTrainedModel):
    def __init__(self, config: FaciesConfig):
        super().__init__(config)
        self.config = config
        self.encoder = FaciesModelEncoder(config)
        self.dense = nn.Linear(config.n_input_features, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.out_proj = nn.Linear(config.d_model, config.vocab_size)

    def get_encoder(self):
        return self.model.get_encoder()

    def get_output_embeddings(self):
        return self.out_proj

    def set_output_embeddings(self, new_embeddings):
        self.out_proj = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        lm_logits = hidden_states

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return SequenceClassifierOutput(loss=masked_lm_loss, logits=lm_logits)
