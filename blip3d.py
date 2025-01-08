import logging
from typing import Union, Optional, Tuple

from transformers import Blip2ForConditionalGeneration
import torch.nn as nn
import torch
from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGenerationModelOutput
from torch.nn import CrossEntropyLoss

class  Blip3D(nn.Module):
    def __init__(self, blip_model):
        super().__init__()
        self.blip_model = blip_model
        self.pc_model = nn.Identity()

    def forward(
            self,
            pc_values: torch.FloatTensor,
            input_ids: torch.FloatTensor,
            attention_mask: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            labels: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None,
            interpolate_pos_encoding: bool = False,
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:

        return_dict = return_dict if return_dict is not None else self.blip_model.config.use_return_dict

        # step 1: forward the point clouds through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        pc_embeds = self.pc_model(
            pc_values
        )

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(pc_embeds.size()[:-1], dtype=torch.long, device=pc_embeds.device)

        query_tokens = self.blip_model.query_tokens.expand(pc_embeds.shape[0], -1, -1)
        query_outputs = self.blip_model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=pc_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.blip_model.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        inputs_embeds = self.blip_model.language_model.get_input_embeddings()(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # if the model already has "image_token_index" then the input is expanded to account for image embeds
        # otherwise we expand manually by concating
        if getattr(self.blip_model.config, "image_token_index", None) is not None:
            special_image_mask = (input_ids == self.blip_model.config.image_token_index).unsqueeze(-1).expand_as(inputs_embeds)
            language_model_inputs = language_model_inputs.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, language_model_inputs)
        else:
            print(
                "Expanding inputs for image tokens in BLIP-2 should be done in processing. "
                "Please follow instruction here (https://gist.github.com/zucchini-nlp/e9f20b054fa322f84ac9311d9ab67042) to update your BLIP-2 model. "
                "Using processors without these attributes in the config is deprecated and will throw an error in v4.47."
            )
            inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
            attention_mask = torch.cat(
                [language_model_attention_mask, attention_mask.to(language_model_attention_mask.device)], dim=1
            )

        if self.blip_model.config.use_decoder_only_language_model:
            outputs = self.blip_model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            # we compute the loss here since we need to take into account the sequence length of the query embeds
            if labels is not None:
                labels = labels.to(logits.device)
                logits = logits[:, -labels.size(1):, :]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)

                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction="mean")

                loss = loss_fct(shift_logits.view(-1, self.blip_model.config.text_config.vocab_size), shift_labels.view(-1))
        else:
            outputs = self.blip_model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, pc_embeds, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=pc_embeds,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )