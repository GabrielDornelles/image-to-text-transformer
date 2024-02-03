from dataset import Flickr8kDataModule
from model import Model
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

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
    model.load_state_dict(torch.load("e2e_transformer_image_captioning.pth"))
    
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )

    pil_image = Image.open("/home/gabriel-dornelles/Downloads/162058185860981de2be61e_1620581858_3x2_md.jpg")
    image = transform(pil_image).transpose(1,2)
    image = image[None,...]
    model.eval()
    with torch.no_grad():
        preds = model.predict(image)
    
    imagenet_mean = [0.485, 0.456, 0.406]  # ImageNet mean
    imagenet_std = [0.229, 0.224, 0.225]  # ImageNet standard deviation

    # Assuming preds and imgs are lists containing predictions and images

    text_pred = (' ').join([dm.train_ds.vocabulary.itos[int(i_ix)] for i_ix in preds[0] if i_ix not in [0,1,2]])
    # Normalize and revert image
    image_array = np.array(image[0].transpose(0, 2))
    reverted_image = revert_normalization(image_array, imagenet_mean, imagenet_std)
    reverted_image = (reverted_image * 255).astype(np.uint8)

    # Add subplot to the grid
    plt.title(text_pred)
    plt.imshow(pil_image)
    
    plt.axis("off")
    plt.savefig(f"latest_inference.png")
    print(text_pred)


    # with open(f"flickr8k_inferences/flickr8k_inference_{i}.txt", "w") as f:
    #     f.write(str(text_pred))