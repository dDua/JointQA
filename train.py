import os
import logging
import random
from pprint import pformat
import json
from multiprocessing import Queue
from queue import Empty
import torch
from torch.nn.parallel import DistributedDataParallel
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import T5Tokenizer
from model import GenerativeSelector
from transformers import AdamW
from t5_config import get_arguments as get_arguments_hotpot
from data_processing import HotpotQADataAllPairs

logger = logging.getLogger(__file__)

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()

def train():
    args, logger = get_arguments_hotpot()
    queue = Queue()
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")

    tokenizer_class, model_class, dataset_class = T5Tokenizer, GenerativeSelector, HotpotQADataAllPairs

    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    tokenizer.add_special_tokens({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>",
                                   "cls_token": "<cls>", "additional_special_tokens": dataset_class.special_tokens})
    dataset = dataset_class(logger, args, tokenizer, lazy=args.lazy)
    model = model_class.from_pretrained(args.model_checkpoint, **{"supervision": True, "tokenizer": tokenizer})
    model.to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    logger.info("Prepare datasets")
    train_loader, train_sampler, val_loader, valid_sampler = dataset.get_data_loaders(lazy=args.lazy)

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)

        input_ids, answer_input, answer_output, question_offset, attention_mask, \
                                token_type_ids, answer_mask, question_ids, question_mask, _ = batch


        losses = model(input_ids=input_ids, attention_mask=attention_mask,
                        question_ids=question_ids[:,:-1].clone(),  question_mask=question_mask[:, :-1],
                        question_lm_labels=question_ids[:, 1:].clone())

        cij_acc_loss = losses[0]
        loss = cij_acc_loss / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                  scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, answer_input, answer_output, question_offset, attention_mask, \
            _, answer_mask, question_ids, question_mask, c_labels = batch

            cij_labels = torch.tensor([0] * input_ids.size(0)).type_as(input_ids)

            results = model(input_ids=input_ids, attention_mask=attention_mask,
                        question_ids=question_ids[:,:-1].clone(),  question_mask=question_mask[:, :-1],
                        question_lm_labels=question_ids[:, 1:].clone())

            _, cij_logits, question_logprobs, _ = results

            ques_logits_pos = question_logprobs[:, 0, :, :]
            ques_logits_neg = question_logprobs[:, 1, :, :]

            predicted_cij = cij_logits.argmax(-1)
            predictions, gold = [], []
            for b in range(input_ids.size(0)):
                c_label_indices = (c_labels[b] == 1).nonzero().squeeze()
                pred = torch.eq(c_label_indices, predicted_cij[b]).long().sum(-1)
                pred = pred.clamp(0, 1)
                predictions.append(pred.item())
                gold.append(1)

            predictions = torch.tensor(predictions).type_as(cij_labels)
            gold = torch.tensor(gold).type_as(cij_labels)
            if random.uniform(0, 1) <= args.output_prob:
                generated_pos_indices = ques_logits_pos[0].argmax(-1).tolist()
                generated_neg_indices = ques_logits_neg[0].argmax(-1).tolist()

                generated_pos_question = tokenizer.decode(generated_pos_indices, clean_up_tokenization_spaces=True,
                                                     skip_special_tokens=True)
                original_question = tokenizer.decode(question_ids[0, :-1].tolist(), clean_up_tokenization_spaces=True,
                                                     skip_special_tokens=True)
                generated_neg_question = tokenizer.decode(generated_neg_indices, clean_up_tokenization_spaces=True,
                                                     skip_special_tokens=True)

                queue.put((json.dumps(generated_pos_question), json.dumps(original_question),
                           json.dumps(generated_neg_question)))

            question_labels = question_ids[:, 1:].contiguous().view(-1)
            ques_logits_pos = ques_logits_pos.contiguous().view(-1, ques_logits_pos.size(-1))

            question_lm_mask = question_mask[:, 1:].bool().view(-1)
            question_logits_masked = torch.masked_select(ques_logits_pos, question_lm_mask.unsqueeze(-1)).view(-1, ques_logits_pos.size(-1))
            question_labels_masked = torch.masked_select(question_labels, question_lm_mask)

            return (cij_logits, question_logits_masked, predictions), \
                   (cij_labels, question_labels_masked, gold)



    evaluator = Engine(inference)

    def queue_reader(foo):
        while True:
            try:
                print(queue.get_nowait())
            except Empty:
                break

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, queue_reader)

    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {
        "cij_accuracy": Accuracy(output_transform=lambda x: (x[0][0], x[1][0])),
        "ques_accuracy": Accuracy(output_transform=lambda x: (x[0][1], x[1][1])),
        "c_accuracy": Accuracy(output_transform=lambda x: (x[0][2], x[1][2]))
    }

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        tb_logger = TensorboardLogger(log_dir=args.output_dir)
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        try:
            tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)
        except Exception:
            def global_step_transform(*args, **kwargs):
                return trainer.state.epoch
            tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()),
                                                                  global_step_transform=global_step_transform),
                             event_name=Events.EPOCH_COMPLETED)
        checkpoint_handler = ModelCheckpoint(args.output_dir, 'checkpoint', save_interval=1, n_saved=3, require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

        torch.save(args, args.output_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(args.output_dir, "config.json"))
        tokenizer.save_vocabulary(args.output_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(args.output_dir, "model_training_args.bin"))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()


if __name__ == "__main__":
    train()
