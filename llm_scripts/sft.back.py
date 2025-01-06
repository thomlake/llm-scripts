from pathlib import Path

import accelerate
import bitsandbytes
import datasets
import flash_attn
import peft
import torch
import transformers
import trl
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_scripts import chat, env
from llm_scripts.sft_config import SftConfig
from llm_scripts.utils import demo


def load_tokenizer(config: SftConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if env.is_primary():
        print('initial chat_template:', tokenizer.chat_template)
        print('initial special_tokens_map:', tokenizer.special_tokens_map)

    tokenizer.model_max_length = config.max_length

    setup_chat_model = config.setup_chat_model or tokenizer.chat_template is None
    if setup_chat_model:
        if env.is_primary():
            print('configuring tokenizer for chat')

        chat.configure_tokenizer_for_chat(
            tokenizer,
            chat_template=config.chat_template,
        )

    if tokenizer.pad_token is None and config.pad_token:
        if config.pad_token:
            tokenizer.pad_token = config.pad_token
        else:
            if env.is_primary():
                print('!!! warning: setting pad_token to eos_token')

            tokenizer.pad_token = tokenizer.eos_token

    elif config.pad_token and env.is_primary():
        print(
            '!!! warning: tokenizer already has a pad_token, '
            f'ignoring config.pad_token={repr(config.pad_token)}'
        )

    assert tokenizer.pad_token

    if env.is_primary():
        messages = demo.messages
        s1 = tokenizer.apply_chat_template(messages, tokenize=False)
        s2 = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        print('Tokenizer info')
        print('  chat_template:', tokenizer.chat_template)
        print('  special_tokens_map:', tokenizer.special_tokens_map)
        print('"""' + s1 + '"""')
        print('-----')
        print('"""' + s2 + '"""')

    return tokenizer


def load_dataset(config: SftConfig):
    train_dataset = datasets.load_dataset(
        'json',
        split='train',
        data_files=config.train_file,
    )

    eval_dataset = None
    if config.eval_file:
        eval_dataset = datasets.load_dataset(
            'json',
            split='train',
            data_files=config.eval_file,
        )
    elif config.eval_split_size is not None:
        test_size = config.eval_split_size
        if test_size > 1:
            test_size = int(test_size)

        splits = train_dataset.train_test_split(test_size=test_size, shuffle=False)
        train_dataset = splits['train']
        eval_dataset = splits['test']

    dataset = {'train': train_dataset, 'eval': eval_dataset}
    if env.is_primary():
        for k, v in dataset.items():
            v = v or []
            print(f'number of {k} instances:', len(v))

    return dataset


def train(config: SftConfig):
    tokenizer = load_tokenizer(config)
    dataset = load_dataset(config)

    if env.is_primary():
        m = dataset['train'][0]['messages']
        m_text = tokenizer.apply_chat_template(m, tokenize=False)
        print('formatted instance:', m_text)

    quantization_config = transformers.BitsAndBytesConfig(
        load_in_8bit=config.load_in_8bit,
        load_in_4bit=config.load_in_4bit,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        device_map={'': env.rank()},
        attn_implementation=config.attn_implementation,
        torch_dtype=config.torch_dtype,
        quantization_config=quantization_config,
    )
    model.resize_token_embeddings(
        len(tokenizer),
        pad_to_multiple_of=config.pad_to_multiple_of,
    )

    if env.is_primary():
        params = dict(model.named_parameters())
        print('number of params:', len(params))

        model_layers_suffixes = (k for k in params if k.startswith('model.layers'))
        model_layers_suffixes = {k.split('.', 1)[-1] for k in config.lora_target_modules}
        print('suffixes of "model.layers."', *sorted(model_layers_suffixes), sep='\n- ')

    lora_config = peft.LoraConfig(
        inference_mode=False,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=config.lora_task_type,
        target_modules=config.lora_target_modules,
    )

    sft_config = trl.SFTConfig(
        output_dir=config.output_dir,
        logging_dir=config.logging_dir,
        num_train_epochs=config.num_train_epochs,
        optim=config.optim,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        neftune_noise_alpha=config.neftune_noise_alpha,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        auto_find_batch_size=config.auto_find_batch_size,
        group_by_length=config.group_by_length,
        evaluation_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        logging_strategy=config.logging_strategy,
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        bf16=config.bf16,
        fp16=config.fp16,
        ddp_backend=config.ddp_backend,
        report_to=config.report_to,
        remove_unused_columns=config.remove_unused_columns,
        load_best_model_at_end=config.load_best_model_at_end,
        gradient_checkpointing=config.gradient_checkpointing,
        ddp_find_unused_parameters=config.ddp_find_unused_parameters,
        save_on_each_node=config.save_on_each_node,
        disable_tqdm=not env.is_primary(),
    )

    trainer = trl.SFTTrainer(
        model,
        tokenizer=tokenizer,
        peft_config=lora_config,
        max_seq_length=config.max_length,
        train_dataset=dataset['train'],
        eval_dataset=dataset['eval'],
        args=sft_config,

    )

    print(f'[GPU {env.rank()}] starting training')
    trainer.train()
    print(f'[GPU {env.rank()}] completed training')


def main():
    if env.is_primary():
        print(f'accelerate version: {accelerate.__version__}')
        print(f'bitsandbytes version: {bitsandbytes.__version__}')
        print(f'datasets version: {datasets.__version__}')
        print(f'flash_attn version: {flash_attn.__version__}')
        print(f'peft version: {peft.__version__}')
        print(f'torch version: {torch.__version__}')
        print(f'transformers version: {transformers.__version__}')
        print(f'trl version: {trl.__version__}')

    config = SftConfig.load()
    if env.is_primary():
        print(config)
        config.dump(Path(config.output_dir, 'sft_config.json'))

    train(config)


if __name__ == '__main__':
    main()
