import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import spacy
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms


spacy_eng = spacy.load("en_core_web_sm")


class Vocabulary:
    """
    The Vocabulary class is responsable by creating the dictionaries of
    'String to Index (stoi)' and 'Index to String (itos)'. The dicts are
    initialized with the following Tokens:
    - <PAD> : A pad token is a token that is used to pad a tokenized sentence to a fixed length.
    - <SOS>: This token defines the 'Start of Sentence'
    - <EOS>: This token defines the 'End of Sentence'
    - <UNK>: We convert words to this token when the infered word is not found in the dictionaries.
    We use the freq_threshold to define which words are in our dicts, that is, words that appear less than
    this threshold are not mapped to our dicts.

    After that, we start mapping words that appear in our dataset (at least freq_threshold times) 
    starting by the index 4 (Flickr dataset). 
    We use spacy english language tokenizer.
    """
    def __init__(self, freq_threshold : int = 5):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        """
        Sentence list is a list of every sentence in our dataset
        """
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, caption):
        """
        Receives a caption, transform to tokenized list of strings,
        then transform list of strings into list of indexes. 
        The indexes are used as input in the embedding layer.
        """
        tokenized_text = self.tokenizer_eng(caption)
        numericalized = [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] 
            for token in tokenized_text]
        return numericalized


class Flickr(Dataset):

    def __init__(self, transform=None) -> None:
        super().__init__()
        self.root_dir = "flickr8k/"
        csv_file = pd.read_csv("flickr8k/captions.txt", delimiter= ',')
        self.images = csv_file["image"]
        self.anns = csv_file["caption"]
        self.transform = transform
        self.vocabulary = Vocabulary(freq_threshold=5)
        self.vocabulary.build_vocabulary(self.anns.tolist())
        self.fixed_length = 100
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(f"{self.root_dir}/images/{self.images[idx]}")
        anns = self.anns[idx]

        # apply transform to the image
        if self.transform is not None:
            image = self.transform(image)

        tokenized_caption = []
        tokenized_caption.append(self.vocabulary.stoi["<SOS>"]) # Start the sentence
        tokenized_caption += self.vocabulary.numericalize(anns) # add the sentence
        tokenized_caption.append(self.vocabulary.stoi["<EOS>"]) # End the sentence
        tokenized_caption = torch.tensor(tokenized_caption)
        tokenized_caption = F.pad(tokenized_caption, (0, self.fixed_length - len(tokenized_caption)), 'constant', 0)
        return image.transpose(1,2), tokenized_caption # transpose assumes transform is given, otherwise image is PIL object


class Flickr8kDataModule(pl.LightningDataModule):

    def __init__(self, batch_size = 1024):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )
    
    def setup(self, stage=None):
        self.train_ds = Flickr(transform=self.transform)
        self.val_ds = Subset(Flickr(transform=self.transform), list(range(0,3000,100))) # call subset to make it short and evaluate some images

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, pin_memory=True)
