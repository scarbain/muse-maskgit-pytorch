import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from pathlib import Path

import os
from muse_maskgit_pytorch import VQGanVAE, VQGanVAETrainer, MaskGit, MaskGitTransformer, Muse
import shutup

import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'gloo'

# make python shut up about deprecation warnings and other crap.
shutup.please()

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Create the parser
parser = argparse.ArgumentParser()

results_folder = 'results'
logging_dir = os.path.join(results_folder, 'logging')

def latest_file(path = results_folder, pattern: str = "*.pt"):
    import glob
    files_path = os.path.join(path, pattern)
    files = sorted(
    glob.iglob(files_path), key=os.path.getctime, reverse=True) 
    
    #print (files[0])
    
    return files[0]

# vae_trainer args
parser.add_argument('--resume_from', type=str, default='', help="Path to the vae model. eg. 'results/vae.steps.pt'")
parser.add_argument('--data_folder', type=str, default='data/datasets/INE/data', help="Dataset folder where your input images for training are.")
parser.add_argument('--num_train_steps', type=int, default=50000, help="Total number of steps to train for. eg. 50000.")
parser.add_argument('--dim', type=int, default=128, help="Model dimension.")
parser.add_argument('--batch_size', type=int, default=1, help="Batch Size.")
parser.add_argument('--lr', type=float, default=3e-4, help="Learning Rate.")
parser.add_argument('--grad_accum_every', type=int, default=1, help="Gradient Accumulation.")
parser.add_argument('--save_results_every', type=int, default=100, help='Save results every this number of steps.')
parser.add_argument('--save_model_every', type=int, default=500, help='Save the model every this number of steps.')
parser.add_argument('--vq_codebook_size', type=int, default=256, help='Image Size.')
parser.add_argument('--image_size', type=int, default=256, help='Image size. You may want to start with small images, and then curriculum learn to larger ones, but because the vae is all convolution, it should generalize to 512 (as in paper) without training on it')
parser.add_argument("--lr_scheduler", type=str, default="constant", help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')
parser.add_argument("--lr_warmup_steps", type=int, default=0, help='Number of steps for the warmup in the lr scheduler.')

# base args
parser.add_argument('--base_texts', type=list, default=['a whale breaching from afar','young girl blowing out candles on her birthday cake', 'fireworks with blue and green sparkles',
        'waking up to a psychedelic landscape'], help='List of Prompts to use.')  
parser.add_argument('--base_resume_from', type=str, default='', help="Path to the vae model. eg. 'results/vae.steps.pt'")
parser.add_argument('--base_num_tokens', type=int, default=256, help="must be same as vq_codebook_size.")
parser.add_argument('--base_seq_len', type=int, default=1024, help="must be equivalent to fmap_size ** 2 in vae.")
parser.add_argument('--base_dim', type=int, default=128, help="Model dimension.")
parser.add_argument('--base_depth', type=int, default=2, help="Depth.")
parser.add_argument('--base_dim_head', type=int, default=64, help="Attention head dimension.")
parser.add_argument('--base_heads', type=int, default=8, help="Attention heads.")
parser.add_argument('--base_ff_mult', type=int, default=4, help='Feedforward expansion factor')
parser.add_argument('--base_t5_name', type=str, default='t5-small', help='Name of your T5 model.')
parser.add_argument('--base_vq_codebook_size', type=int, default=256, help='')
parser.add_argument('--base_image_size', type=int, default=512, help='')
parser.add_argument('--base_cond_drop_prob', type=float, default=0.25, help='Conditional dropout, for Classifier Free Guidance')
parser.add_argument('--base_cond_scale', type=int, default=3, help='Conditional for Classifier Free Guidance')
parser.add_argument('--base_timesteps', type=int, default=20, help='Time Steps to use for the generation.')

#superres args
parser.add_argument('--superres_texts', type=list, default=['a whale breaching from afar','young girl blowing out candles on her birthday cake', 'fireworks with blue and green sparkles',
        'waking up to a psychedelic landscape'], help='List of Prompts to use.')  
parser.add_argument('--superres_resume_from', type=str, default='', help="Path to the vae model. eg. 'results/vae.steps.pt'")
parser.add_argument('--superres_num_tokens', type=int, default=256, help="must be same as vq_codebook_size.")
parser.add_argument('--superres_seq_len', type=int, default=1024, help="must be equivalent to fmap_size ** 2 in vae.")
parser.add_argument('--superres_dim', type=int, default=128, help="Model dimension.")
parser.add_argument('--superres_depth', type=int, default=2, help="Depth.")
parser.add_argument('--superres_dim_head', type=int, default=64, help="Attention head dimension.")
parser.add_argument('--superres_heads', type=int, default=8, help="Attention heads.")
parser.add_argument('--superres_ff_mult', type=int, default=4, help='Feedforward expansion factor')
parser.add_argument('--superres_t5_name', type=str, default='t5-small', help='name of your T5')
parser.add_argument('--superres_vq_codebook_size', type=int, default=256, help='')
parser.add_argument('--superres_image_size', type=int, default=512, help='')
parser.add_argument('--superres_timesteps', type=int, default=20, help='Time Steps to use for the generation.')

# generate args
parser.add_argument('--prompt', type=list, default=['a whale breaching from afar','young girl blowing out candles on her birthday cake', 'fireworks with blue and green sparkles',
        'waking up to a psychedelic landscape'], help='List of Prompts to use for the generation.')  
parser.add_argument('--base_model_path', type=str, default='', help="Path to the base vae model. eg. 'results/vae.steps.base.pt'")
parser.add_argument('--superres_maskgit', type=str, default='', help="Path to the superres vae model. eg. 'results/vae.steps.superres.pt'")
parser.add_argument('--generate_timesteps', type=int, default=20, help='Time Steps to use for the generation.')
parser.add_argument('--generate_cond_scale', type=int, default=3, help='Conditional for Classifier Free Guidance')

# Parse the argument
args = parser.parse_args()

def vae_trainer(resume_from=args.resume_from, dim=args.dim, vq_codebook_size=args.vq_codebook_size,
                         data_folder=args.data_folder, num_train_steps=args.num_train_steps,
                         batch_size=args.batch_size, image_size=args.image_size,
                         lr=args.lr, lr_scheduler=args.lr_scheduler, lr_warmup_steps=args.lr_warmup_steps,
                         grad_accum_every=args.grad_accum_every, save_results_every=args.save_results_every,
                         save_model_every=args.save_model_every,
                         ):
    vae = VQGanVAE(
        dim = dim,
        vq_codebook_size = vq_codebook_size
    )
    
    # load the vae from disk if we have previously trained one
    if resume_from:
        #print ('Resuming VAE from: ', os.path.join(results_folder, resume_from + '.pt'))
        print ('Resuming VAE from: ', resume_from)
        #vae.load(os.path.join(results_folder, resume_from + '.pt'))
        vae.load(resume_from)
    
    #with torch.autocast('cuda'):
    # train on folder of images, as many images as possible
    trainer = VQGanVAETrainer(
        vae,
        folder = data_folder,
        num_train_steps = num_train_steps,
        batch_size = batch_size,
        image_size = image_size,    # you may want to start with small images, and then curriculum learn to larger ones, but because the vae is all convolution, it should generalize to 512 (as in paper) without training on it
        lr = lr,
        lr_scheduler = lr_scheduler,
        lr_warmup_steps = lr_warmup_steps,
        grad_accum_every = grad_accum_every,
        max_grad_norm = None,
        discr_max_grad_norm = None,
        save_results_every = save_results_every,
        save_model_every = save_model_every,
        results_folder = results_folder,
        valid_frac = 0.05,
        random_split_seed = 42,
        use_ema = True,
        ema_beta = 0.995,
        ema_update_after_step = 1,
        ema_update_every = 1,
        apply_grad_penalty_every = 4,
        accelerate_kwargs={
            #'mixed_precision': 'fp16',
            'device_placement': False,
            'split_batches': True,
        }
        ).cuda()
    
    trainer.train()

def base_maskgit_trainer(base_texts=args.base_texts, base_resume_from=args.base_resume_from, base_dim=args.base_dim, base_vq_codebook_size=args.base_vq_codebook_size,
                         base_num_tokens=args.base_num_tokens, base_seq_len=args.base_seq_len, base_depth=args.base_depth,
                         base_dim_head=args.base_dim_head, base_heads=args.base_heads, base_ff_mult=args.base_ff_mult, base_t5_name=args.base_t5_name,
                         base_image_size=args.base_image_size, base_cond_drop_prob=args.base_cond_drop_prob, base_cond_scale=args.base_cond_scale, base_timesteps=args.base_timesteps,
                         ):    
    # first instantiate your vae
    
    vae = VQGanVAE(
        dim = base_dim,
        vq_codebook_size = base_vq_codebook_size
    ).cuda()
    
    print ('Resuming VAE from: ', args.resume_from)
    vae.load(args.resume_from)    # you will want to load the exponentially moving averaged VAE
    
    # then you plug the vae and transformer into your MaskGit as so
    
    # (1) create your transformer / attention network
    
    transformer = MaskGitTransformer(
        num_tokens = base_num_tokens,         # must be same as codebook size above
        seq_len = base_seq_len,               # must be equivalent to fmap_size ** 2 in vae
        dim = base_dim,                       # model dimension
        depth = base_depth,                   # depth
        dim_head = base_dim_head,             # attention head dimension
        heads = base_heads,                   # attention heads,
        ff_mult = base_ff_mult,               # feedforward expansion factor
        t5_name = base_t5_name,               # name of your T5
    )
    
    # (2) pass your trained VAE and the base transformer to MaskGit
    
    base_maskgit = MaskGit(
        vae = vae,                 # vqgan vae
        transformer = transformer, # transformer
        image_size = base_image_size,          # image size
        cond_drop_prob = base_cond_drop_prob,     # conditional dropout, for classifier free guidance
    ).cuda()
    
    # ready your training text and images   
    images = torch.randn(4, 3, base_image_size, base_image_size).cuda()
    
    # feed it into your maskgit instance, with return_loss set to True
    
    loss = base_maskgit(
        images,
        texts = base_texts
    )
    
    loss.backward()
    
    # do this for a long time on much data
    
    # then...    
    images = base_maskgit.generate(
        texts = [
            'a whale breaching from afar',
            'young girl blowing out candles on her birthday cake',
            'fireworks with blue and green sparkles'
            ],
        cond_scale = base_cond_scale,  # conditioning scale for classifier free guidance
        timesteps = base_timesteps,
    )
    
    # save the base vae
    base_maskgit.save(args.resume_from.replace('.pt' , '.base.pt'))
    
    #print(images.shape) # (3, 3, 256, 256)   
    
    #print(images) # List[PIL.Image.Image]
    
    img1 = images[0]
    
    save_image(img1, f'{results_folder}/outputs/base_result.png')
    #img.save(f'{results_folder}/outputs/base_result.png')
    
    #for count in len(images):
    #    for image in images:
    #image.save(f'{results_folder}/outputs/base_{count}.png')    


#
def superres_maskgit_trainer(superres_texts=args.superres_texts, superres_resume_from=args.superres_resume_from,
                             superres_dim=args.superres_dim, superres_vq_codebook_size=args.superres_vq_codebook_size,
                             superres_num_tokens=args.superres_num_tokens, superres_seq_len=args.superres_seq_len,
                             superres_depth=args.superres_depth, superres_dim_head=args.superres_dim_head,
                             superres_heads=args.superres_heads, superres_ff_mult=args.superres_ff_mult,
                             superres_t5_name=args.superres_t5_name, superres_image_size=args.superres_image_size,
                             ):
    # first instantiate your ViT VQGan VAE
    # a VQGan VAE made of transformers
    
    vae = VQGanVAE(
        dim = superres_dim,
        vq_codebook_size = superres_vq_codebook_size
    ).cuda()
    
    vae.load(args.resume_from) # you will want to load the exponentially moving averaged VAE
    
    # then you plug the VqGan VAE into your MaskGit as so
    
    # (1) create your transformer / attention network
    
    transformer = MaskGitTransformer(
        num_tokens = superres_num_tokens,            # must be same as codebook size above
        seq_len = superres_seq_len,                  # must be equivalent to fmap_size ** 2 in vae
        dim = superres_dim,                          # model dimension
        depth = superres_depth,                      # depth
        dim_head = superres_dim_head,                # attention head dimension
        heads = superres_heads,                      # attention heads,
        ff_mult = superres_ff_mult,                  # feedforward expansion factor
        t5_name = superres_t5_name,                  # name of your T5
    )
    
    # (2) pass your trained VAE and the base transformer to MaskGit
    
    superres_maskgit = MaskGit(
        vae = vae,
        transformer = transformer,
        cond_drop_prob = 0.25,
        image_size = superres_image_size,                     # larger image size
        cond_image_size = 256,                # conditioning image size <- this must be set
    ).cuda()
    
    # ready your training text and images    
    images = torch.randn(4, 3, superres_image_size, superres_image_size).cuda()
    
    # feed it into your maskgit instance, with return_loss set to True
    
    loss = superres_maskgit(
        images,
        texts = superres_texts
    )
    
    loss.backward()
    
    # do this for a long time on much data
    # then...
    
    images = superres_maskgit.generate(
        texts = [
            'a whale breaching from afar',
            'young girl blowing out candles on her birthday cake',
            'fireworks with blue and green sparkles',
            'waking up to a psychedelic landscape'
        ],
        cond_images = F.interpolate(images, 256),  # conditioning images must be passed in for generating from superres
        cond_scale = 3.,
        timesteps=args.superres_timesteps,
    )
    
    # save the superres vae
    superres_maskgit.save(args.resume_from.replace('.pt','.superres.pt'))
    
    #print(images.shape) # (4, 3, 512, 512)
    #print(images) # List[PIL.Image.Image]
    
    img1 = images[0]
    
    save_image(img1, f'{results_folder}/outputs/superres_result.png')    
    
    #for count in len(images):
        #for image in images:
            #image.save(f'{results_folder}/outputs/superres_{count}.png')    
    

def generate(prompt=args.prompt, base_model_path=args.base_model_path, superres_maskgit=args.superres_maskgit,
             dim=args.dim, vq_codebook_size=args.vq_codebook_size, timesteps=args.generate_timesteps, cond_scale=args.generate_cond_scale):
    
    base_maskgit = VQGanVAE(
        dim = dim,
        vq_codebook_size = vq_codebook_size
    ).cuda()
    
    superres_maskgit = VQGanVAE(
        dim = dim,
        vq_codebook_size = vq_codebook_size
    ).cuda()    
    
    #vae.load(model_path)
        
    base_maskgit.load(args.resume_from.replace('.pt','.base.pt'))
    superres_maskgit.load(args.resume_from.replace('.pt','.superres.pt'))    
    
    # pass in the trained base_maskgit and superres_maskgit from above
    
    muse = Muse(
        base = base_maskgit,
        superres = superres_maskgit
    )
    
    images = muse(
        texts=prompt,
        timesteps=timesteps,
        cond_scale=cond_scale
    )
    
    print(images) # List[PIL.Image.Image]
    
    img1 = images[0]
    
    save_image(img1, f'{results_folder}/outputs/result.png')        
    
    #for count in len(images):
        #for image in images:
            #image.save(f'{results_folder}/outputs/result_{count}.png')

#
if __name__ == '__main__':
    vae_trainer()
    base_maskgit_trainer()
    superres_maskgit_trainer()
    #generate() 