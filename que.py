import argparse
import shlex
import asyncio
import atexit
import os
import argparse, os, sys, glob, uuid
from config import settings
import nextcord
from nextcord.ext import commands, tasks
sys.path.append('.')

from ldm.simplet2i import T2I
model   = T2I()

intents = nextcord.Intents.default()
intents.message_content = True
intents.members = True # If you ticked the SERVER MEMBERS INTENT

bot = commands.Bot(command_prefix = settings['prefix'], intents=intents) # Так как мы указали префикс в settings, обращаемся к словарю с ключом prefix.

@bot.command() # Не передаём аргумент pass_context, так как он был нужен в старых версиях.

class MyClient(nextcord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def setup_hook(self) -> None:
        # create the background task and run it in the background
        self.bg_task = self.loop.create_task(self.my_background_task())
        self.bg_task = self.loop.create_task(self.my_background_task())
        self.bg_task = self.loop.create_task(self.my_background_task())
        self.bg_task = self.loop.create_task(self.my_background_task())

    async def on_ready(self):
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')

    async def my_background_task(self):
        await self.wait_until_ready()
        counter = 0
        channel = self.get_channel(1234567)  # channel ID goes here
        while not self.is_closed():
            counter += 1
            await channel.send(counter)
            await asyncio.sleep(60)  # task runs every 60 seconds


client = MyClient(intents=nextcord.Intents.default())
client.run(settings['token'])