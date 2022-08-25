import argparse
import shlex
import asyncio
import atexit
from timeit import default_timer as timer
import os
import argparse, os, sys, glob, uuid
from config import settings
import nextcord
import threading
from functools import wraps, partial
import queue
from concurrent.futures import ThreadPoolExecutor
from nextcord.ext import commands, tasks
sys.path.append('.')

from ldm.simplet2i_alt import T2I
model = T2I()
model.load_model()
intents = nextcord.Intents.default()
intents.message_content = True
intents.members = True
queue = asyncio.Queue()
loop = asyncio.get_event_loop()

bot = commands.Bot(command_prefix = settings['prefix'], intents=intents)

async def run_blocking(func, *args, **kwargs):
    """Run any blocking, synchronous function in a non-blocking way"""
    callback = partial(func, *args, **kwargs)
    return await loop.run_in_executor(None, callback)

@bot.event
async def on_ready():
    if not executor.is_running():
        executor.start()
    print('Bot is ready.')

@tasks.loop(seconds=10.0)
async def executor():
    task = await queue.get()
    await task
    queue.task_done()

@bot.command()
async def dream(ctx, *, arg):
    if ctx.author != bot.user:
        try:
            message = await ctx.reply('your task is queued')
            await queue.put(dreaming(ctx, arg, message))
        except Exception as e:
            await ctx.reply('Ooops... Something goes wrong...')

async def dreaming(ctx, arg, message):
    try:
        if ctx.author != bot.user:
            start = timer()
            quote_text = '{}'.format(arg)
            reply_tetx = 'Dreaming for {}\'s `!dream {}`'.format(ctx.message.author.mention,arg)
            await message.edit(content=reply_tetx)
            outputs = await run_blocking(model.txt2img, quote_text)
            await message.delete()
            elapsed_time = timer() - start
            await ctx.reply(content='Dreamt in `{}s` for {}\'s `!dream {}`\nSeeds {}'.format(elapsed_time,ctx.message.author.mention,arg, outputs[1]), files=outputs[0]) #reply to a specific message with its id
    except Exception as e:
        await ctx.reply('Ooops... Something goes wrong...')

bot.run(settings['token'])
