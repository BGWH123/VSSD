'''
Adapted from https://github.com/huggingface/transformers
'''

from transformers import T5Config, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack, __HEAD_MASK_WARNING_MSG, T5Block
import copy
import os
import warnings
from typing import Optional, Tuple, Union, List
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
)
from geomloss import SamplesLoss
from typing import List
import torch

def layerwise_similarity(states_a, states_b):
    sims = []
    for ha, hb in zip(states_a, states_b):
        # mean pool
        ha_pooled = ha.mean(dim=1)  # [B, hidden]
        hb_pooled = hb.mean(dim=1)
        # cosine sim
        sim = F.cosine_similarity(ha_pooled, hb_pooled, dim=-1).mean().item()
        sims.append(sim)
    return sims
def compute_steering_vectors(hidden_pos_list: List[torch.Tensor], hidden_neg_list: List[torch.Tensor], num_layers: int) -> List[torch.Tensor]:
    steering_vectors = []

    selected_pos_layers = hidden_pos_list[-num_layers:]
    selected_neg_layers = hidden_neg_list[-num_layers:]

    for h_pos, h_neg in zip(selected_pos_layers, selected_neg_layers):
        B, L, D = h_pos.shape
        E = h_pos - h_neg
        v_list = []

        for b in range(B):
            E_b = E[b]  # shape: [L, D]
            try:
                U, S, Vh = torch.linalg.svd(E_b, full_matrices=False)  # Vh: [D, D]
                v_edit =  Vh[0]  # top-1 direction
            except RuntimeError:
                v_edit = torch.zeros(D, device=E.device)
            v_edit = v_edit.view(1, D)  # [1, D]
            v_list.append(v_edit)

        v_tensor = torch.cat(v_list, dim=0)  # [B, D]
        steering_vectors.append(v_tensor)

    return steering_vectors # List of [B, D] per layer





def mean_pooling(
    all_decoder_hidden_states: Tuple[torch.Tensor],
) -> torch.Tensor:
    pooled_layers = []
    for h in all_decoder_hidden_states:  # h: [B, L, D]
        mean_h = h.mean(dim=1)  # [B, D]
        pooled_layers.append(mean_h)

    return torch.stack(pooled_layers, dim=0)




class T5ForMultimodalGenerationVSSDCoTNew(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config, patch_size, padding_idx, save_dir, vot_num, alpha,layer_distill,disstill_alpha,add_alpha):
        super().__init__(config)
        self.model_dim = config.d_model
        self.hidden_size = config.hidden_size
        self.vot_num = vot_num
        self.alpha = alpha
        self.padding_idx = padding_idx
        self.generate_cot = False
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.patch_num, self.patch_dim = patch_size

        self.image_dense = nn.Linear(self.patch_dim, config.d_model)
        self.mha_layer = torch.nn.MultiheadAttention(embed_dim=config.hidden_size, kdim=config.hidden_size,
                                                     vdim=config.hidden_size, num_heads=1, batch_first=True)


        self.gate_dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.sigmoid = nn.Sigmoid()

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.disstill_alpha = disstill_alpha  
        self.add_alpha = add_alpha  #steering loss scaling factor
        self.layer_distill = layer_distill

    def set_conf(self, generate_cot):
        self.generate_cot = generate_cot  



    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            image_ids=None,
            r_image_ids=None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            r_labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        #TODO:


        # decoder_attention_mask_base=decoder_attention_mask
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=True,  
                output_hidden_states=True,
                return_dict=return_dict,
            )

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        hidden_states_o = encoder_outputs[0]

        image_embedding = self.image_dense(image_ids)
        image_att, _ = self.mha_layer(hidden_states_o, image_embedding, image_embedding)

        merge = torch.cat([hidden_states_o, image_att], dim=-1)
        gate = self.sigmoid(self.gate_dense(merge))
        hidden_states = (1 - gate) * hidden_states_o + gate * image_att  

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

            # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        all_decoder_hidden_states = decoder_outputs.hidden_states
        distill_loss = None
        if r_image_ids is not None :
            with torch.no_grad():
                image_embedding = self.image_dense(r_image_ids)
                image_att, _ = self.mha_layer(hidden_states_o, image_embedding, image_embedding)

                merge = torch.cat([hidden_states_o, image_att], dim=-1)
                gate = self.sigmoid(self.gate_dense(merge))
                hidden_states_r = (1 - gate) * hidden_states_o + gate * image_att

            if r_labels is not None:
                # get decoder inputs from shifting lm labels to the right
                decoder_input_ids_r = self._shift_right(labels)#(r_labels)
            else:
                decoder_input_ids_r = self._shift_right(labels)
            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.decoder.first_device)
                hidden_states_r = hidden_states_r.to(self.decoder.first_device)
                if decoder_input_ids_r is not None:
                    decoder_input_ids_r = decoder_input_ids_r.to(self.decoder.first_device)
            # Decode
            with torch.no_grad():
                r_decoder_outputs = self.decoder(
                    input_ids=decoder_input_ids_r,   
                    attention_mask=decoder_attention_mask,
                    inputs_embeds=decoder_inputs_embeds,
                    past_key_values=past_key_values,
                    encoder_hidden_states=hidden_states_r,
                    encoder_attention_mask=attention_mask,
                    head_mask=decoder_head_mask,
                    cross_attn_head_mask=cross_attn_head_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=return_dict,
                )
            all_decoder_hidden_states_r= r_decoder_outputs.hidden_states
            sequence_output = r_decoder_outputs[0]
            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.encoder.first_device)
                self.lm_head = self.lm_head.to(self.encoder.first_device)
                sequence_output = sequence_output.to(self.lm_head.weight.device)

            if self.config.tie_word_embeddings:
                # Rescale output before projecting on vocab
                # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
                sequence_output = sequence_output * (self.model_dim ** -0.5)

                r_lm_logits = self.lm_head(sequence_output)
            num_layers = self.layer_distill if self.layer_distill else (len(all_decoder_hidden_states) - 1)
            steering_vectors = compute_steering_vectors(all_decoder_hidden_states, all_decoder_hidden_states_r,num_layers )
            pooled_hidden = mean_pooling(
                all_decoder_hidden_states=all_decoder_hidden_states,
            )



            distill_loss = 0
            num_layers = self.layer_distill if self.layer_distill else (len(all_decoder_hidden_states) - 1)

            student_layer = pooled_hidden[-(num_layers+1):-1]
            steer_vecotors = steering_vectors
            sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
            add_alpha = self.add_alpha if self.add_alpha is not None else 0.1
            for i,(student, steering_vector) in enumerate(zip(student_layer, steer_vecotors)):
                # student: [B, D], steering_vector: [B, D]
                h=student.detach() + add_alpha*steering_vector.detach()  # [B, D]
                norm_student= torch.norm(student.detach(), dim=-1, keepdim=True)  # [B, 1]
                norm_h= torch.norm(h, dim=-1, keepdim=True)  # [B, 1]
                teacher=h* (norm_student / (norm_h + 1e-8))  # [B, D]
                emd_loss = sinkhorn(student, teacher)
                distill_loss += emd_loss
            distill_loss = distill_loss/num_layers





        sequence_output=all_decoder_hidden_states[-1]
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)  
        loss=0
        # Set device for model parallelism
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            celoss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            lambda_distill = self.disstill_alpha if self.disstill_alpha is not None else 0.2
            loss = celoss + lambda_distill * distill_loss if distill_loss is not None else celoss

            print("Loss:", loss.item(), "CE Loss:", celoss.item(),
                  "Distill Loss:", distill_loss.item() if distill_loss is not None else None)

            return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )