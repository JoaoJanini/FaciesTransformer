""" PyTorch Facies model."""
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import PreTrainedModel
from hf_sequence_to_sequence.configuration import FaciesConfig
from hf_sequence_to_sequence.embedding import TokenEmbedding, PositionalEncoding
import torch.utils.checkpoint
from torch.nn import Transformer
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import (
    Seq2SeqModelOutput,
    Seq2SeqLMOutput,
    BaseModelOutput,
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


def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones((sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt, PAD_IDX=None, DEVICE=None):
    src_mask = None
    src_padding_mask = None
    if src is not None:
        src_seq_len = src.shape[1]
        src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)
        src_padding_mask = (src == PAD_IDX).transpose(0, 1)

    if tgt is not None:
        tgt_seq_len = tgt.shape[1]

        tgt_mask = generate_square_subsequent_mask(tgt_seq_len)

        tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


class FaciesPretrainedModel(PreTrainedModel):
    config_class = FaciesConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor(
            [[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device
        )
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs

class FaciesModelEncoder(FaciesPretrainedModel):
    def __init__(self, config: FaciesConfig, model):
        super().__init__(config)
        self.config = config
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.model = model.encoder

        self.positional_encoding = PositionalEncoding(
            config.d_model, dropout=config.dropout
        )

        self.embedding_input = torch.nn.Linear(config.d_input, config.d_model)

        # Initialize weights and apply final processing
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

        channel_encoding = self.embedding_input(input_ids.transpose(-1, -2))

        output_encoder = self.model(
            channel_encoding, mask=None, src_key_padding_mask=None
        )

        return BaseModelOutput(last_hidden_state=output_encoder)

class FaciesModel(FaciesPretrainedModel):
    def __init__(self, config: FaciesConfig):
        super().__init__(config)
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.model = Transformer(
            d_model=config.d_model,
            num_encoder_layers=config.encoder_layers,
            nhead=config.decoder_attention_heads,
            num_decoder_layers=config.decoder_layers,
            dim_feedforward=config.encoder_ffn_dim,
            dropout=config.dropout,
            batch_first=True
        )

        self.encoder = FaciesModelEncoder(config, self.model)

        self.decoder = self.model.decoder

        self.positional_encoding = PositionalEncoding(
            config.d_model, dropout=config.dropout
        )
        self.decoder_inputs_embeds = TokenEmbedding(vocab_size, config.d_model)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
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

        decoder_input_ids = self.positional_encoding(
            self.decoder_inputs_embeds(decoder_input_ids)
        )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0]
            )

        decoder_output = self.decoder(
            decoder_input_ids,
            encoder_outputs.last_hidden_state,
            tgt_mask=decoder_attention_mask,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
        )

        decoder_outputs = BaseModelOutput(last_hidden_state=decoder_output)

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        )


class FaciesForConditionalGeneration(FaciesPretrainedModel):
    base_model_prefix = "model"

    def __init__(self, config: FaciesConfig):
        super().__init__(config)
        self.model = FaciesModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

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

        (
            head_mask,
            decoder_attention_mask,
            attention_mask,
            decoder_head_mask,
        ) = create_mask(
            input_ids,
            decoder_input_ids,
            PAD_IDX=self.config.pad_token_id,
            DEVICE=decoder_input_ids.device,
        )

        outputs = self.model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=None,
            decoder_attention_mask=decoder_attention_mask.to(device=decoder_input_ids.device),
            decoder_head_mask=None,
            encoder_outputs=encoder_outputs,
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
