from flickr8k_dataset import Flickr8kDataModule
from model import Model
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
import numpy as np

def revert_normalization(image, mean, std):
    # Convert mean and std to numpy arrays if they are not already
    mean = np.array(mean)
    std = np.array(std)
    
    # Revert normalization
    reverted_image = image * std + mean
    
    return reverted_image

if __name__ == "__main__":
    dm = Flickr8kDataModule(batch_size=64)
    dm.setup()
    vocab_size = len(dm.train_ds.vocabulary)

    model = Model(len_vocab=vocab_size)
    trainer = pl.Trainer(
        max_epochs=60, 
        precision=16
    )
    trainer.fit(model, dm)
    torch.save(model.state_dict(), 'e2e_transformer_image_captioning.pth')

    
    imgs, captions = next(iter(dm.val_dataloader()))
    model.eval()
    with torch.no_grad():
        preds = model.predict(imgs)
    
    imagenet_mean = [0.485, 0.456, 0.406]  # ImageNet mean
    imagenet_std = [0.229, 0.224, 0.225]  # ImageNet standard deviation

    num_images = 30
    for i in range(num_images):
        text_pred = (' ').join([dm.train_ds.vocabulary.itos[int(i_ix)] for i_ix in preds[i]])
        # Normalize and revert image
        image_array = np.array(imgs[i].transpose(0, 2))
        reverted_image = revert_normalization(image_array, imagenet_mean, imagenet_std)
        reverted_image = (reverted_image * 255).astype(np.uint8)

        # Add subplot to the grid
        plt.title(text_pred)
        plt.imshow(reverted_image)
     
        plt.axis("off")
        plt.savefig(f"flickr8k_inferences/flickr8k_inference_{i}.png")