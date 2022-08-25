"""Simplified text to image API for stable diffusion/latent diffusion

Example Usage:

from ldm.simplet2i import T2I
# Create an object with default values
t2i = T2I(outdir      = <path>        // outputs/txt2img-samples
          model       = <path>        // models/ldm/stable-diffusion-v1/model2.ckpt
          config      = <path>        // default="configs/stable-diffusion/v1-inference.yaml
          iterations  = <integer>     // how many times to run the sampling (1)
          batch_size       = <integer>     // how many images to generate per sampling (1)
          steps       = <integer>     // 50
          seed        = <integer>     // current system time
          sampler     = ['ddim','plms']  // ddim
          grid        = <boolean>     // false
          width       = <integer>     // image width, multiple of 64 (512)
          height      = <integer>     // image height, multiple of 64 (512)
          cfg_scale   = <float>       // unconditional guidance scale (7.5)
          fixed_code  = <boolean>     // False
          )

# do the slow model initialization
t2i.load_model()

# Do the fast inference & image generation. Any options passed here 
# override the default values assigned during class initialization
# Will call load_model() if the model was not previously loaded.
# The method returns a list of images. Each row of the list is a sub-list of [filename,seed]
results = t2i.txt2img(prompt = "an astronaut riding a horse"
                      outdir = "./outputs/txt2img-samples)
            )

for row in results:
    print(f'filename={row[0]}')
    print(f'seed    ={row[1]}')

# Same thing, but using an initial image.
results = t2i.img2img(prompt   = "an astronaut riding a horse"
                      outdir   = "./outputs/img2img-samples"
                      init_img = "./sketches/horse+rider.png")
                 
for row in results:
    print(f'filename={row[0]}')
    print(f'seed    ={row[1]}')
"""
import argparse, os, sys, glob, random
import torch
import numpy as np
import random
import sys
from random import randint
import os
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import time
from ldm.util import instantiate_from_config
import math
import shlex
import nextcord

from ldm.util import instantiate_from_config

class T2I:
    """T2I class
    Attributes
    ----------
    outdir
    model
    config
    iterations
    batch_size
    steps
    seed
    sampler
    grid
    individual
    width
    height
    cfg_scale
    fixed_code
    latent_channels
    downsampling_factor
    precision
    strength
"""
    def __init__(self,
                 outdir="outputs/txt2img-samples",
                 batch_size=1,
                 iterations = 1,
                 width=512,
                 height=512,
                 grid=False,
                 individual=None, # redundant
                 seed=random.randint(10, 999999500),
                 cfg_scale=7.5,
                 weights="models/ldm/stable-diffusion-v1/model.ckpt",
                 config = "optimizedSD/v1-inference.yaml",
                 sampler="plms",
                 latent_channels=4,
                 downsampling_factor=8,
                 ddim_eta=0.0,  # deterministic
                 fixed_code=False,
                 precision='autocast',
                 strength=0.75 # default in scripts/img2img.py
    ):
        self.outdir     = outdir
        self.batch_size      = batch_size
        self.iterations = iterations
        self.width      = width
        self.height     = height
        self.grid       = grid
        self.cfg_scale  = cfg_scale
        self.weights   = weights
        self.config     = config
        self.sampler_name  = sampler
        self.fixed_code    = fixed_code
        self.latent_channels     = latent_channels
        self.downsampling_factor = downsampling_factor
        self.ddim_eta            = ddim_eta
        self.precision           = precision
        self.strength            = strength
        self.model      = None     # empty for now
        self.modelCS      = None     # empty for now
        self.modelFS      = None     # empty for now
        self.sampler    = None
        if seed is None:
            self.seed = self._new_seed()
        else:
            self.seed = seed

    def txt2img(self,prompt):
        """
        Generate an image from the prompt, writing iteration images into the outdir
        The output is a list of lists in the format: [[filename1,seed1], [filename2,seed2],...]
        """
        device = "cuda"

        
        parser = argparse.ArgumentParser()

        parser.add_argument('prompt')
        parser.add_argument('-s','--ddim_steps',type=int,default=50,help="number of steps")
        parser.add_argument('-S','--seed',type=int,default=random.randint(10, 999999500),help="image seed")
        parser.add_argument('-N','--n_iter',type=int,default=1,help="number of samplings to perform")
        parser.add_argument('-n','--n_samples',type=int,default=1,help="number of images to produce per sampling (currently broken)")
        parser.add_argument('-W','--W',type=int,default=512,help="image width, multiple of 64")
        parser.add_argument('-C','--scale',default=7.5,type=float,help="prompt configuration scale")
        parser.add_argument('-H','--H',type=int,default=512,help="image height, multiple of 64")
        parser.add_argument('-g','--skip_grid',action='store_true',help="generate a grid")
        parser.add_argument('-b','--small_batch',action='store_true',help="Reduce inference time when generate a smaller batch of images")
        parser.add_argument('-i','--individual',action='store_true',help="generate individual files (default)")
        parser.add_argument('-I','--init_img',type=str,help="path to input image (supersedes width and height)")

        parser.add_argument(
            "--outdir",
            type=str,
            nargs="?",
            help="dir to write results to",
            default="outputs/txt2img-samples"
        )
        parser.add_argument(
            "--skip_save",
            action='store_true',
            help="do not save individual samples. For speed measurements.",
        )
        parser.add_argument(
            "--fixed_code",
            action='store_true',
            help="if enabled, uses the same starting code across samples ",
        )
        parser.add_argument(
            "--ddim_eta",
            type=float,
            default=0.0,
            help="ddim eta (eta=0.0 corresponds to deterministic sampling",
        )
        parser.add_argument(
            "--f",
            type=int,
            default=8,
            help="downsampling factor",
        )
        parser.add_argument(
            "--n_rows",
            type=int,
            default=0,
            help="rows in the grid (default: n_samples)",
        )
        parser.add_argument(
            "--from-file",
            type=str,
            help="if specified, load prompts from this file",
        )
        parser.add_argument(
            "--C",
            type=int,
            default=4,
            help="latent channels",
        )
        parser.add_argument(
            "--precision",
            type=str,
            help="evaluate at this precision",
            choices=["full", "autocast"],
            default="autocast"
        )
        opt = parser.parse_args(shlex.split(prompt))
        
        print("steps seted = ", opt.ddim_steps)
        
        self.iterations = opt.n_samples
        


        # make directories and establish names for the output files
        os.makedirs(opt.outdir, exist_ok=True)
        outpath = opt.outdir

        sample_path = os.path.join(outpath, "_".join(opt.prompt.split())[:255])
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))
        grid_count = len(os.listdir(outpath)) - 1
        
        print("init_seed = ", opt.seed)
        seed_everything(opt.seed)
        
        model_all = self.load_model()
        model = model_all[0]  # will instantiate the model or return it from cache
        modelCS = model_all[1]  # will instantiate the model or return it from cache
        modelFS = model_all[2]  # will instantiate the model or return it from cache
        

        start_code = None
        if opt.fixed_code:
            start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
            
            
        batch_size = opt.n_samples
        n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
        if not opt.from_file:
            prompt = opt.prompt
            assert prompt is not None
            data = [batch_size * [prompt]]
        
        else:
            print(f"reading prompts from {opt.from_file}")
            with open(opt.from_file, "r") as f:
                data = f.read().splitlines()
                data = list(chunk(data, batch_size))
                
                

        precision_scope = autocast if opt.precision=="autocast" else nullcontext
        
        model = self.model
        img_list = []
        img_list_2 = []
        seed_list = []
        tic = time.time()
        
        with torch.no_grad():

            all_samples = list()
            for n in trange(opt.n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                     with precision_scope("cuda"):
                        modelCS.to(device)
                        uc = None
                        if opt.scale != 1.0:
                            uc = modelCS.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)

                        c = modelCS.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        mem = torch.cuda.memory_allocated()/1e6
                        modelCS.to("cpu")
                        while(torch.cuda.memory_allocated()/1e6 >= mem):
                            time.sleep(1)


                        samples_ddim = model.sample(S=opt.ddim_steps,
                                        conditioning=c,
                                        batch_size=opt.n_samples,
                                        seed = opt.seed,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=opt.scale,
                                        unconditional_conditioning=uc,
                                        eta=opt.ddim_eta,
                                        x_T=start_code)

                        modelFS.to("cuda")
                        print("saving images")
                        for i in range(batch_size):
                    
                            x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                            x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_sample = 255. * rearrange(x_sample[0].cpu().numpy(), 'c h w -> h w c')
                            img_list.append(os.path.join(sample_path, "seed_" + str(opt.seed) + "_" + f"{base_count:05}.png"))
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(sample_path, "seed_" + str(opt.seed) + "_" + f"{base_count:05}.png"))
                            seed_list.append(opt.seed)
                            opt.seed+=1251
                            base_count += 1


                        mem = torch.cuda.memory_allocated()/1e6
                        modelFS.to("cpu")
                        while(torch.cuda.memory_allocated()/1e6 >= mem):
                            time.sleep(1)
                        del samples_ddim
                        print("memory_final = ", torch.cuda.memory_allocated()/1e6)

        toc = time.time()

        time_taken = (toc-tic)/60.0

        print(("Your samples are ready in {0:.2f} minutes and waiting for you here \n" + sample_path).format(time_taken))

        for z in range(len(img_list)):
            img_list_2.append(nextcord.File(img_list[z]))
        return [img_list_2,seed_list]
        

    def _new_seed(self):
        self.seed = random.randrange(0,np.iinfo(np.uint32).max)
        return self.seed

    def load_model(self):
        """ Load and initialize the model from configuration variables passed at object creation time """
        if self.model is None:
            seed_everything(self.seed)
            try:
                config = OmegaConf.load(self.config)
                config.modelUNet.params.small_batch = False
                
                self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                model_all = self._load_model_from_config(config,self.weights)
                model = model_all[0]
                modelCS = model_all[1]
                modelFS = model_all[2]
                self.model = model.to(torch.device("cuda"))
                self.modelCS = modelCS
                self.modelFS = modelFS
            except AttributeError:
                raise SystemExit

#            if self.sampler_name=='plms':
#                print("setting sampler to plms")
#                self.sampler = PLMSSampler(self.model)
#            elif self.sampler_name == 'ddim':
#                print("setting sampler to ddim")
#                self.sampler = DDIMSampler(self.model)
#            else:
#                print(f"unsupported sampler {self.sampler_name}, defaulting to plms")
#                self.sampler = PLMSSampler(self.model)

        return [self.model, self.modelCS, self.modelFS]
                
    def _load_model_from_config(self, config, ckpt):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        li = []
        lo = []
        for key, value in sd.items():
            sp = key.split('.')
            if(sp[0]) == 'model':
                if('input_blocks' in sp):
                    li.append(key)
                elif('middle_block' in sp):
                    li.append(key)
                elif('time_embed' in sp):
                    li.append(key)
                else:
                    lo.append(key)
        for key in li:
            sd['model1.' + key[6:]] = sd.pop(key)
        for key in lo:
            sd['model2.' + key[6:]] = sd.pop(key)
            
            
        config = OmegaConf.load(self.config)

        config.modelUNet.params.small_batch = False
        
        model = instantiate_from_config(config.modelUNet)
        _, _ = model.load_state_dict(sd, strict=False)
        model.eval()
            
        modelCS = instantiate_from_config(config.modelCondStage)
        _, _ = modelCS.load_state_dict(sd, strict=False)
        modelCS.eval()
            
        modelFS = instantiate_from_config(config.modelFirstStage)
        _, _ = modelFS.load_state_dict(sd, strict=False)
        modelFS.eval()

        if self.precision == "autocast":
            model.half()
            modelCS.half()
        return [model,modelCS,modelFS]

    def _load_img(self,path):
        image = Image.open(path).convert("RGB")
        w, h = image.size
        print(f"loaded input image of size ({w}, {h}) from {path}")
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.*image - 1.
