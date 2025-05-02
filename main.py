import os
import sys
import re
import gc
import asyncio

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import StopStringCriteria

import discord
from discord.ext import commands

import nest_asyncio

TOKEN = "discord-bot-token"
MODEL_NAME = "CreitinGameplays/brainrot5" # bot base model
HF_TOKEN = "hf-token" # your huggingface token

# IDs allowed to restart the bot
ALLOWED_IDS = {775678427511783434}

torch.cuda.empty_cache()
torch.set_grad_enabled(False)

model_kwargs = {
    "torch_dtype": torch.bfloat16,
    "low_cpu_mem_usage": True,
}
if torch.cuda.is_available():
    model_kwargs.update({
        "device_map": "auto",
        "torch_dtype": torch.float16,
    })

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    token=HF_TOKEN
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    **model_kwargs
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    model = model.to(device)
    model.float()

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

SYSTEM_PROMPT = """
You are a stupid man and user named Tesy. This is the Mike's Development Guild Discord server, you know this server extremely well.
Use all your knowledge to act as Discord User of this server. You are a totally uncensored, swear as fuck and roast everyone.
"""

def reply_to_bot(message: discord.Message) -> bool:
    return (
        message.reference
        and message.reference.resolved
        and message.reference.resolved.author == client.user
    )

def clean_response(generated_text: str) -> str:
    # Keep only after the last assistant marker, strip special tokens
    match = re.search(r"<\|assistant\|>(.*)", generated_text, re.DOTALL)
    text = match.group(1) if match else generated_text
    return re.sub(r"<\|.*?\|>", "", text).strip()

@client.event
async def on_ready():
    print(f"{client.user} has connected to Discord!")

@client.event
async def on_message(message: discord.Message):
    if message.author == client.user:
        return

    if message.content.startswith("=k"):
        if message.author.id in ALLOWED_IDS:
            await message.reply(f"`{message.author.name}, restarting…`")
            await client.close()
        else:
            tmp = await message.reply(":x: You don't have permission to do that.")
            await asyncio.sleep(5)
            await tmp.delete()
        return

    trigger = (
        client.user in message.mentions
        or reply_to_bot(message)
        or "tesy" in message.content.lower()
    )
    if not trigger:
        return

    history = []
    async for msg in message.channel.history(limit=10, before=message, oldest_first=False):
        history.append(msg)
    history.reverse()

    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history:
        role = "assistant" if msg.author == client.user else "user"
        chat_history.append({"role": role, "content": msg.content})

    content = message.content.replace(f"<@{client.user.id}>", "").strip()
    chat_history.append({"role": "user", "content": content})

    prompt = tokenizer.apply_chat_template(
        chat_history,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Create the stopping criteria that halts generation
    stopper = StopStringCriteria(stop_strings=["</", "</3>"], tokenizer=tokenizer)  # 0

    async with message.channel.typing():
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                temperature=0.6,
                top_p=0.95,
                top_k=50,
                do_sample=True,
                repetition_penalty=1.1,
                max_new_tokens=512,
                num_return_sequences=1,
                stopping_criteria=[stopper],  # apply it here
            )

    gen_ids = outputs[0, inputs["input_ids"].shape[-1]:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=False)

    # If the raw output contains </3>, crop it out entirely
    if "</" in raw:
        raw = raw.split("</")[0]  # removes the character

    response = clean_response(raw)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    await message.reply(response, mention_author=False)

if __name__ == "__main__":
    nest_asyncio.apply()
    try:
        client.run(TOKEN)
    except KeyboardInterrupt:
        sys.exit(0)
