import os
import math
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import transformers
from transformers import CLIPTextModel, CLIPTokenizer

from modularcam.data.dataset import WebVid10M
from modularcam.data.dataset_zyp import GligenAnimateWebvidDataset
from modularcam.models.unet import UNet3DConditionModel
from modularcam.pipelines.pipeline_animation import AnimationPipeline
from modularcam.models.sparse_controlnet import SparseControlNetModel
from modularcam.utils.util import save_videos_grid, zero_rank_print
from modularcam.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint

os.environ['RANK'] = '4'

os.environ['WORLD_SIZE'] = '2'# =torch.cuda.device_count()
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '35632'




def init_dist(launcher="slurm", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)
        
    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        port = os.environ.get('PORT', port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend)
        zero_rank_print(f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}")
        
    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')
    
    return local_rank


def load_weights(
    unet,
    vae,
    text_encoder,
    # motion module
    motion_module_path         = "",
    # image layers
    dreambooth_model_path      = ""
):
    unet_state_dict = {}
    if motion_module_path != "":
        print(f"load motion module from {motion_module_path}")
        motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")
        motion_module_state_dict = motion_module_state_dict["state_dict"] if "state_dict" in motion_module_state_dict else motion_module_state_dict
        unet_state_dict.update({name: param for name, param in motion_module_state_dict.items() if "motion_modules." in name})
        unet_state_dict.pop("animatediff_config", "")
    
    missing, unexpected = unet.load_state_dict(unet_state_dict, strict=False)
    assert len(unexpected) == 0
    del unet_state_dict

    # base model
    if dreambooth_model_path != "":
        print(f"load dreambooth model from {dreambooth_model_path}")
        if dreambooth_model_path.endswith(".safetensors"):
            dreambooth_state_dict = {}
            with safe_open(dreambooth_model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    dreambooth_state_dict[key] = f.get_tensor(key)
        elif dreambooth_model_path.endswith(".ckpt"):
            dreambooth_state_dict = torch.load(dreambooth_model_path, map_location="cpu")
            
        # 1. vae
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(dreambooth_state_dict, vae.config)
        vae.load_state_dict(converted_vae_checkpoint)
        # 2. unet
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(dreambooth_state_dict, unet.config)
        unet.load_state_dict(converted_unet_checkpoint, strict=False)
        # 3. text_model
        text_encoder = convert_ldm_clip_checkpoint(dreambooth_state_dict)
        del dreambooth_state_dict
        
    return unet, vae, text_encoder


def main(
    image_finetune: bool,
    
    name: str,
    use_wandb: bool,
    launcher: str,
    
    output_dir: str,
    pretrained_model_path: str,

    train_data: Dict,
    validation_data: Dict,
    cfg_random_null_text: bool = True,
    cfg_random_null_text_ratio: float = 0.1,
    
    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs = None,
    
    max_train_epoch: int = -1,
    max_train_steps: int = 100,
    validation_steps: int = 100,
    validation_steps_tuple: Tuple = (-1,),

    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",

    trainable_modules: Tuple[str] = (None, ),
    num_workers: int = 32,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,

    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,

    global_seed: int = 42,
    is_debug: bool = False,
    controlnet_config: str = "",
    controlnet_path: str = "",
    motion_module: str = "",
    dreambooth_path: str = ""
):
    check_min_version("0.10.0.dev0")

    # Initialize distributed training
    # local_rank      = init_dist(launcher=launcher)
    # global_rank     = dist.get_rank()
    # num_processes   = dist.get_world_size()
    # is_main_process = global_rank == 0
    # local_rank = 0
    # global_rank = 0
    # num_processes = 1
    is_main_process = True

    # seed = global_seed + global_rank
    seed = global_seed
    torch.manual_seed(seed)
    
    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="modularcam", name=folder_name, config=config)

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    vae          = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    
    controlnet = controlnet_images = None
    if not image_finetune:
        print("Load unet...")
        unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path, subfolder="unet", 
            unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
        )
        unet.config.num_attention_heads = 8
        unet.config.projection_class_embeddings_input_dim = None
    else:
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
        
    # Load pretrained unet weights
    # if unet_checkpoint_path != "":
    #     zero_rank_print(f"from checkpoint: {unet_checkpoint_path}")
    #     unet_checkpoint_path = torch.load(unet_checkpoint_path, map_location="cpu")
    #     if "global_step" in unet_checkpoint_path: zero_rank_print(f"global_step: {unet_checkpoint_path['global_step']}")
    #     state_dict = unet_checkpoint_path["state_dict"] if "state_dict" in unet_checkpoint_path else unet_checkpoint_path

    #     m, u = unet.load_state_dict(state_dict, strict=False)
    #     zero_rank_print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
    #     assert len(u) == 0
    
    
    print("Load Control Net")
    controlnet_config = OmegaConf.load(controlnet_config)
    controlnet = SparseControlNetModel.from_unet(unet, controlnet_additional_kwargs=controlnet_config.get("controlnet_additional_kwargs", {}))
    if controlnet_path != "":
        controlnet_state_dict = torch.load(controlnet_path, map_location="cpu")
        controlnet_state_dict = controlnet_state_dict["controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
        controlnet_state_dict.pop("animatediff_config", "")
        controlnet.load_state_dict(controlnet_state_dict)
    controlnet.cuda()
    
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
        if controlnet is not None: controlnet.enable_xformers_memory_efficient_attention()
    
    unet, vae, text_encoder = load_weights(unet, vae, text_encoder, motion_module, dreambooth_path)
    
        
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Set unet trainable parameters
    unet.requires_grad_(False)
    controlnet.requires_grad_(True)
    # for name, param in unet.named_parameters():
    #     for trainable_module_name in trainable_modules:
    #         if trainable_module_name in name:
    #             param.requires_grad = True
    #             break
            
    # trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    trainable_params = list(filter(lambda p: p.requires_grad, controlnet.parameters()))
    print(len(trainable_params))
    print("=" * 20)
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    if is_main_process:
        zero_rank_print(f"trainable params number: {len(trainable_params)}")
        zero_rank_print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Move models to GPU
    # vae.to(local_rank)
    # text_encoder.to(local_rank)
    vae.cuda()
    text_encoder.cuda()

    # Get the training dataset
    # train_dataset = WebVid10M(**train_data, is_image=image_finetune)
    data_root = "/DATA/DATANAS2/datasets/webvid/10Wdataset"
    meta_path = 'modularcam/data/train.csv'
    train_dataset = GligenAnimateWebvidDataset(data_root, meta_path, resolution=(256, 256))
    # distributed_sampler = DistributedSampler(
    #     train_dataset,
    #     num_replicas=num_processes,
    #     rank=global_rank,
    #     shuffle=True,
    #     seed=global_seed,
    # )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=None,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)
        
    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Validation pipeline
    if not image_finetune:
        validation_pipeline = AnimationPipeline(
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler,
        ).to("cuda")
    else:
        validation_pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_path,
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler, safety_checker=None,
        )
    validation_pipeline.enable_vae_slicing()

    # DDP warpper
    unet.cuda()
    # unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    print(len(train_dataloader), gradient_accumulation_steps, max_train_steps)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * gradient_accumulation_steps

    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    for epoch in range(first_epoch, num_train_epochs):
        # train_dataloader.sampler.set_epoch(epoch)
        unet.train()
        
        for step, batch in enumerate(train_dataloader):
            if cfg_random_null_text:
                batch['text'] = [name if random.random() > cfg_random_null_text_ratio else "" for name in batch['text']]
                
            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value[None, ...]
                        save_videos_grid(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{0}-{idx}'}.gif", rescale=True)
                else:
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value / 2. + 0.5
                        torchvision.utils.save_image(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{0}-{idx}'}.png")
                    
            ### >>>> Training >>>> ###
            
            # Convert videos to latent space            
            pixel_values = batch["pixel_values"].cuda()
            video_length = pixel_values.shape[1]
            with torch.no_grad():
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                else:
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()

                latents = latents * 0.18215
                # latents = latents * 0.19503

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch['text'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]
                
            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            # Mixed-precision training
            
            image_transforms = transforms.Compose([
                transforms.RandomResizedCrop(
                    (512, 512), (1.0, 1.0), 
                    ratio=(1, 1)
                ),
                transforms.ToTensor(),
            ])
            
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                controlnet_images = batch['control_image'].cuda()
                # print(controlnet_images.size())
                # controlnet_images = [image_transforms(image) for image in controlnet_images]
                # controlnet_images = torch.stack(controlnet_images).unsqueeze(0).cuda()

                controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")
                num_controlnet_images = controlnet_images.shape[2]
                controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")
                controlnet_images = vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * 0.18215
                # controlnet_images = vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * 0.19503
                controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)
                
                controlnet_noisy_latents = noisy_latents
                controlnet_prompt_embeds = encoder_hidden_states
                controlnet_cond_shape    = list(controlnet_images.shape)
                controlnet_cond_shape[2] = train_data['sample_n_frames']
                controlnet_cond = torch.zeros(controlnet_cond_shape).to(latents.device)
                controlnet_conditioning_mask_shape    = list(controlnet_cond.shape)
                controlnet_conditioning_mask_shape[1] = 1
                controlnet_conditioning_mask          = torch.zeros(controlnet_conditioning_mask_shape).to(latents.device)
                
                controlnet_image_index = [0]
                controlnet_cond[:,:,controlnet_image_index] = controlnet_images[:,:,:len(controlnet_image_index)]
                controlnet_conditioning_mask[:,:,controlnet_image_index] = 1

                down_block_additional_residuals, mid_block_additional_residual = controlnet(
                    controlnet_noisy_latents, timesteps,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=controlnet_cond,
                    conditioning_mask=controlnet_conditioning_mask,
                    conditioning_scale=1.0,
                    guess_mode=False, return_dict=False,
                )
                
                # print(down_block_additional_residuals, mid_block_additional_residual)

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, down_block_additional_residuals = down_block_additional_residuals, mid_block_additional_residual = mid_block_additional_residual).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                # print(model_pred)
                # print(model_pred.size(), loss)

            optimizer.zero_grad()

            # Backpropagate
            if mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                """ >>> gradient clipping >>> """
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            
            ### <<<< Training <<<< ###
            
            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)
                
            # Save checkpoint
            if is_main_process and (global_step % checkpointing_steps == 0 or step == len(train_dataloader) - 1):
                save_path = os.path.join(output_dir, f"checkpoints")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": unet.state_dict(),
                }
                if step == len(train_dataloader) - 1:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint-epoch-{epoch+1}.ckpt"))
                else:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint.ckpt"))
                logging.info(f"Saved state to {save_path} (global_step: {global_step})")
                
            # Periodically validation
            if is_main_process and (global_step % validation_steps == 0 or global_step in validation_steps_tuple):
                samples = []
                
                generator = torch.Generator(device=latents.device)
                generator.manual_seed(global_seed)
                
                height = train_data.sample_size[0] if not isinstance(train_data.sample_size, int) else train_data.sample_size
                width  = train_data.sample_size[1] if not isinstance(train_data.sample_size, int) else train_data.sample_size

                prompts = validation_data.prompts[:2] if global_step < 1000 and (not image_finetune) else validation_data.prompts

                # for idx, prompt in enumerate(prompts):
                #     if not image_finetune:
                #         sample = validation_pipeline(
                #             prompt,
                #             generator    = generator,
                #             video_length = train_data.sample_n_frames,
                #             height       = height,
                #             width        = width,
                #             **validation_data,
                #         ).videos
                #         save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{idx}.gif")
                #         samples.append(sample)
                        
                #     else:
                #         sample = validation_pipeline(
                #             prompt,
                #             generator           = generator,
                #             height              = height,
                #             width               = width,
                #             num_inference_steps = validation_data.get("num_inference_steps", 25),
                #             guidance_scale      = validation_data.get("guidance_scale", 8.),
                #         ).images[0]
                #         sample = torchvision.transforms.functional.to_tensor(sample)
                #         samples.append(sample)
                
                # if not image_finetune:
                #     samples = torch.concat(samples)
                #     save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                #     save_videos_grid(samples, save_path)
                    
                # else:
                #     samples = torch.stack(samples)
                #     save_path = f"{output_dir}/samples/sample-{global_step}.png"
                #     torchvision.utils.save_image(samples, save_path, nrow=4)

                # logging.info(f"Saved samples to {save_path}")
                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
            # if (epoch == 3) and (step >= 7000) and (step <= 9000) and ((step % 200) == 0):
            #     torch.save(controlnet.state_dict(), 'models/SparseCtrl/%d-%d.ckpt' % (epoch, step))
        
        torch.save(controlnet.state_dict(), 'models/SparseCtrl/%d.ckpt' % (epoch))
    # dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb",    action="store_true")
    args = parser.parse_args()

    name   = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, **config)
