import argparse
from html import parser
import torch
from pathlib import Path

from torch.utils.data import DataLoader
from utils.utils import *
from utils.models import *

import torch.optim as optim
from tqdm import tqdm

from torchvision.utils import save_image

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_dir' , type=str , default=r'C:\Users\Abhishek\NST_AdaIN\content_data' , help="Location of content dataset")
    parser.add_argument('--style_dir' , type=str , default=r'C:\Users\Abhishek\NST_AdaIN\style_data' , help="Location of style dataset")
    parser.add_argument('--vgg' , type=str , default=r'C:\Users\Abhishek\NST_AdaIN\vgg_normalised.pth' , help="Location of pre-trained VGG")
    parser.add_argument('--experiment' , type=str , default='experiment1' , help="Name of experiment")
    parser.add_argument('--final_size' , type=int , default=256 , help="Size of final image")
    parser.add_argument('--content_size' , type=int , default=512 , help="Size of content image")
    parser.add_argument('--style_size' , type=int , default=512 , help="Size of style image")
    parser.add_argument('--crop' , action='store_true' ,default=True , help="Crop images")
    parser.add_argument('--batch_size' , type=int , default=4 , help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--lr_decay' , type=float , default=5e-5 , help="Learning rate decay factor")
    parser.add_argument('--epochs' , type=int , default=1 , help="Number of epochs")
    parser.add_argument('--content_weight' , type=float , default=1.0 , help="Weight for content loss")
    parser.add_argument('--style_weight' , type=float , default=5.0 , help="Weight for style loss")
    parser.add_argument('--log_interval' , type=int , default=1 , help="Interval for logging training progress")
    parser.add_argument('--save_interval' , type=int , default=1 , help="Interval for saving model checkpoints")
    parser.add_argument('--resume' , action='store_true' , default=False , help="Resume training from latest checkpoint")  # for smooth resume training from latest checkpoint
    # parser.add_argument('--decoder_path' , type=str , default=None , help="Path to decoder checkpoint")
    # parser.add_argument('--optimizer_path' , type=str , default=None , help="Path to optimizer checkpoint")
    # parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to full checkpoint")



    return parser.parse_args()

def main():
    args = parse_arguments()


    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        try:
            import torch_directml
            device = torch_directml.device()
        except:
            device = torch.device("cpu")

    print(f"Using device: {device}")

    # save_dir = Path('experiment')/args.experiment   # for local : use for saving the model and output images in a folder named 'experiment' with subfolder as experiment name
    save_dir = Path('/content/drive/MyDrive/NST_AdaIN/experiment') / args.experiment   # use this for google colab : use for saving the model and output images in a folder named 'experiment' with subfolder as experiment name in drive
    save_dir.mkdir(exist_ok=True , parents=True)

    #save arguments values
    with open(save_dir/'args.txt' , 'w') as args_file:
        #vars convert cli args into dictionary , dict() not works
        for key ,value in vars(args).items():
            args_file.write(f'{key}: {value}\n')



    content_transform = get_transform(args.content_size , args.crop , args.final_size)
    style_transform = get_transform(args.style_size , args.crop , args.final_size)

    content_dataset = ImageFolderDataset(args.content_dir , transform=content_transform)
    style_dataset = ImageFolderDataset(args.style_dir , style_transform)

    content_dataloader = DataLoader(content_dataset , batch_size=args.batch_size , shuffle=True , pin_memory=True , drop_last=True)
    style_dataloader = DataLoader(style_dataset , batch_size=args.batch_size , shuffle=True , pin_memory=True , drop_last=True)
    # during creating dataloader for testing or validation no need to shuffle the data , so shuffle=False
    
    print('Number of batches in content dataloader: ', len(content_dataloader))
    print('Number of batches in style dataloader: ', len(style_dataloader))

    # for batch in style_dataloader:
    #     print(batch.shape)
    #     break


    # Models
    encoder = VGGEncoder(vgg_path=args.vgg).to(device)
    decoder = Decoder().to(device)

    # Optimizer
    optimizer = optim.Adam(decoder.parameters() , lr=args.lr)

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0/(1.0 + args.lr_decay * epoch)
    )


    start_epoch = 0

    if args.resume:

        checkpoints = list(save_dir.glob('checkpoint_epoch_*.pth'))

        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoint found in {save_dir}"
            )

        # Find the checkpoint with the highest epoch number
        latest_checkpoint = max(
            checkpoints,
            key=lambda p: int(p.stem.split('_')[-1])
        )

        print(f"Loading checkpoint: {latest_checkpoint}")

        checkpoint = torch.load(
            latest_checkpoint,
            map_location=device,
            weights_only=False
        )

        decoder.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        start_epoch = checkpoint['epoch']

        print(f"Resuming from epoch {start_epoch}")

    print('Training started...')

    mse_loss = torch.nn.MSELoss()

    encoder.eval()

    # ues tqdn for training progress bar
    running_loss = None
    running_closs = None
    running_sloss = None
    for epoch in range(start_epoch, args.epochs):
        progress_bar = tqdm(zip(content_dataloader , style_dataloader) , total=min(len(content_dataloader) , len(style_dataloader)) )

        running_loss = 0
        running_closs = 0
        running_sloss = 0
        for content_batch , style_batch in progress_bar:
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            # print("len of content batch: ",len(content_batch))
            # print("len of style batch: ",len(style_batch))

            # feats maens features , c_feats means content features , s_feats means style features
            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)

            # print("lem of content features: ",len(c_feats))
            # print('Content features shape: ', c_feats[0].shape)
            # print(len(s_feats))
            # print('Style features shape: ', s_feats[0].shape)

            t = adaptive_instance_normalization(c_feats[-1] , s_feats[-1])

            # print('type', type(t))
            # print('shape', t.shape)

            g = decoder(t)

            g_feats = encoder(g)

            loss_c = args.content_weight * mse_loss(g_feats[-1] , t)

            loss_s = 0

            for g_f , s_f in zip(g_feats , s_feats):
                g_mean , g_std = calc_mean_std(g_f)
                s_mean , s_std = calc_mean_std(s_f)
                loss_s += mse_loss(g_mean , s_mean) + mse_loss(g_std , s_std)

            loss_s = loss_s * args.style_weight
            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f'Epoch [{epoch+1}/{args.epochs}] , Loss: {loss.item():.4f} , Content Loss: {loss_c.item():.4f} , Style Loss: {loss_s.item():.4f}')

            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()

        scheduler.step()

        running_loss /= len(content_dataloader)
        running_closs /= len(content_dataloader)  
        running_sloss /= len(content_dataloader)

        if (epoch+1) % args.log_interval == 0:
            tqdm.write(f'Epoch Iter [{epoch+1}/{args.epochs}] , Loss: {running_loss:.4f} , Content Loss: {running_closs:.4f} , Style Loss: {running_sloss:.4f}')  


        if (epoch+1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')

            tqdm.write(f'Model checkpoint saved at epoch {epoch+1}')

            with torch.no_grad():
                output = torch.cat([content_batch, style_batch , g] , dim=0)
                output = output.detach().cpu().float().clamp(0, 1) # IMPORTANT for AMD user: clamp the output to [0, 1] range before saving as image
                save_image(output , save_dir/f'output_epoch_{epoch+1}.png' , nrow=args.batch_size )



if __name__ == '__main__':
    main()