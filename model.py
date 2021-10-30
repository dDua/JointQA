import torch
from torch import nn
from transformers.modeling_t5 import T5ForConditionalGeneration, T5Block, T5LayerNorm

class GenerativeSelector(T5ForConditionalGeneration):
    def __init__(self, config, supervision=None, ans_sym_id=None, max_ans_len=None, tokenizer=None):
        super().__init__(config)
        self.cij_prior = nn.Linear(config.d_model, 1)
        self.supervision = supervision
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None,
                question_lm_labels=None, question_offset=None, question_mask=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None):

        batch_size, num_samples, seq_len = input_ids.size()
        offset_ids = attention_mask.sum(-1) - 1
        neg_sample_mask = (offset_ids != -1).float()
        offset_ids[offset_ids == -1] = 0

        # p (cij|psi)
        cij_encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                           attention_mask=attention_mask.view(-1, attention_mask.size(-1)))

        cij_hidden_states = cij_encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)
        cij_end_hidden = cij_hidden_states.gather(2, offset_ids.unsqueeze(-1).unsqueeze(-1).
                                                  expand(-1, -1, -1, cij_hidden_states.size(-1))).squeeze(2)
        prior_cij_logits = self.cij_prior(cij_end_hidden).squeeze(-1)
        prior_cij_logits = torch.masked_fill(prior_cij_logits, (1 - neg_sample_mask).bool(), -1e7)
        prior_cij_probs = torch.log_softmax(prior_cij_logits, -1)

        # p (q|cij)
        question_ids_rep = question_ids.unsqueeze(1).repeat(1, num_samples, 1).view(batch_size * num_samples, -1)
        question_mask_rep = question_mask.unsqueeze(1).repeat(1, num_samples, 1)
        question_outputs = self.decoder(
            input_ids=question_ids_rep,
            attention_mask=question_mask_rep.view(batch_size * num_samples, -1),
            encoder_hidden_states=cij_hidden_states.view(batch_size * num_samples, -1, cij_hidden_states.size(-1)),
            encoder_attention_mask=attention_mask.view(batch_size * num_samples, -1)
        )
        ques_sequence_output = question_outputs[0]
        ques_sequence_output = ques_sequence_output * (self.model_dim ** -0.5)
        question_logits = self.lm_head(ques_sequence_output)
        question_logprobs = question_logits.log_softmax(-1)
        q_len = question_mask_rep.sum(-1).view(batch_size, num_samples).type_as(question_logprobs)

        q_lm_labels_flat = question_lm_labels.unsqueeze(1).repeat(1, num_samples, 1).view(-1)
        q_lm_logprobs_flat = question_logprobs.view(-1, question_logprobs.size(-1))
        q_lm_labels_mask = (question_mask_rep == 0).type_as(question_logprobs).view(batch_size, num_samples, -1)

        question_give_cij = torch.gather(q_lm_logprobs_flat, -1, q_lm_labels_flat.unsqueeze(1)).squeeze(-1).view(
            batch_size,
            num_samples, -1)
        question_give_cij = question_give_cij.masked_fill(q_lm_labels_mask.bool(), -1e7)
        q_log_pll = (question_give_cij[:, 0, :] * (1 - q_lm_labels_mask[:, 0, :])).sum(-1) / q_len[:, 0]
        q_log_ull = ((1 - question_give_cij[:, 1:, :].exp() + 1e-12).log() * (1 - q_lm_labels_mask[:, 1:, :])).sum(
            -1) / q_len[:, 1:]

        logits = (question_give_cij.masked_fill(q_lm_labels_mask.bool(), 0).sum(-1) / q_len) + prior_cij_probs

        loss_mml_p = prior_cij_probs[:, 0] + q_log_pll
        loss_mml_n = prior_cij_probs[:, 1:] + q_log_ull
        loss_mml_p = torch.logsumexp(loss_mml_p, -1)
        loss_mml_n = torch.logsumexp(loss_mml_n, -1)
        loss_lik = - loss_mml_p.mean()
        loss_unlik_q = - loss_mml_n.mean()
        loss_unlik_cij = - ((1 - prior_cij_probs[:, 1:].exp() + 1e-12).log() * neg_sample_mask[:,1:]).sum(-1).mean()
        loss = loss_lik + loss_unlik_q + loss_unlik_cij

        return loss, logits, question_logprobs.view(batch_size, num_samples, -1, question_logprobs.size(-1)), None


class T5QA(T5ForConditionalGeneration):
    def __init__(self, config, ans_sym_id=None, max_ans_len=None, tokenizer=None):
        config.n_positions = 1024
        super().__init__(config)
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer

    def encode(self, input_ids, attention_mask):
        batch_size, num_samples, seq_len = input_ids.size()
        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)
        return encoded_hidden_states

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, encode_only=False, max_len=None):

        if encoder_outputs is not None:
            encoded_hidden_states = encoder_outputs[0]
        if encoded_hidden_states is None:
            encoded_hidden_states = self.encode(input_ids, attention_mask)
            if encode_only:
                return encoded_hidden_states
        else:
            generated_ans = self.generate_custom(input_ids=input_ids, attention_mask=attention_mask,
                                            encoded_hidden_states=encoded_hidden_states, max_len=max_len)
            return generated_ans

        batch_size, num_samples, seq_len = input_ids.size()

        decoder_input_ids = decoder_input_ids.unsqueeze(1).repeat(1, num_samples, 1)
        decoder_attention_mask = decoder_attention_mask.unsqueeze(1).repeat(1, num_samples, 1)

        decoder_input_ids[decoder_input_ids == -100] = 0
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids.view(batch_size*num_samples, -1),
            attention_mask=decoder_attention_mask.view(batch_size*num_samples, -1),
            encoder_hidden_states=encoded_hidden_states.view(batch_size*num_samples, seq_len, -1),
            encoder_attention_mask=attention_mask.view(batch_size*num_samples, seq_len)
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        lm_logprobs = lm_logits.log_softmax(-1)

        logits = lm_logits.view(batch_size, num_samples, -1, lm_logits.size(-1))
        outputs = [encoded_hidden_states, logits]
        if lm_labels is not None:
            lm_labels_rep = lm_labels.unsqueeze(1).repeat(1, num_samples, 1)
            lm_labels_flat = lm_labels_rep.view(-1)
            lm_logprobs_flat = lm_logprobs.view(-1, lm_logprobs.size(-1))
            lm_labels_flat_mask = (lm_labels_flat == -100).bool()
            lm_labels_flat[lm_labels_flat == -100] = 0
            log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
            log_ll_flat = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
            log_ll = log_ll_flat.view(batch_size, num_samples, -1)
            ans_len = decoder_attention_mask.sum(-1).view(batch_size, num_samples).float()
            log_pll = log_ll[:,0].sum(-1)/ans_len[:,0]
            loss = - log_pll.mean()
            outputs += [loss]

        return outputs

    def generate_custom(self, input_ids=None, attention_mask=None, encoded_hidden_states=None, max_len=None):

        batch_size, num_samples, seq_len = input_ids.size()

        #p (a|q, cij)
        input_symbols = torch.ones(batch_size, 1).fill_(self.ans_symbol_idx).type_as(input_ids)
        generated_ans = [input_symbols]
        maxlen = max_len if max_len else self.max_answer_length - 1
        for i in range(maxlen):
            ans_outputs = self.decoder(
                input_ids=input_symbols,
                encoder_hidden_states=encoded_hidden_states.view(-1, encoded_hidden_states.size(-2),
                                                                 encoded_hidden_states.size(-1)),
                encoder_attention_mask=attention_mask.view(-1, attention_mask.size(-1))
            )
            ans_logits = self.lm_head(ans_outputs[0] * (self.model_dim ** -0.5))
            ans_probs = ans_logits.log_softmax(-1)
            pred_prob, pred_symbol = ans_probs[:, -1].topk(1, -1)
            generated_ans.append(pred_symbol)
            input_symbols = torch.cat([input_symbols, pred_symbol], -1)

        generated_ans = torch.cat(generated_ans, -1)
        return generated_ans, ans_probs
