from tokenizers.processors import TemplateProcessing
import torch
from torch import nn
from transformers.models.bert.tokenization_bert import BertTokenizer
from model import VQGModel
from extract_image_features import get_features_and_object_labels
import sys
import os
import PIL
import numpy as np
from torchvision import transforms
import random
import torchvision.models as models
import copy
import json
import pytorch_lightning as pl



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224,
                                 scale=(1.00, 1.2),
                                 ratio=(0.75, 1.3333333333333333)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)],)


def tokenize_and_pad(tokens, max_len):
    encoded = tokenizer(tokens)
    encoded_id = torch.tensor(encoded["input_ids"])
    encoded_attention_mask = torch.tensor(encoded["attention_mask"])

    len_diff = max_len - len(encoded_id)
    if len_diff < 0:
        encoded_id = encoded_id[:max_len]
        encoded_attention_mask = encoded_attention_mask[:max_len]
    else:
        pads = torch.tensor([tokenizer.pad_token_id] * len_diff).long()
        pads_for_attn_mask = torch.ones_like(pads)
        encoded_id = torch.cat((encoded_id, pads), dim=-1)
        encoded_attention_mask = torch.cat((encoded_attention_mask, pads_for_attn_mask), dim=-1)

    return encoded_id, encoded_attention_mask


class EncoderCNN(nn.Module):
    """Generates a representation for an image input.
    """

    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer.
        """
        super(EncoderCNN, self).__init__()
        self.cnn = models.resnet18(pretrained=True).to("cpu")
        modules = list(self.cnn.children())[:-1]
        self.cnn = nn.Sequential(*modules)

    def forward(self, images):
        """Extract the image feature vectors.
        """
        features = self.cnn(images).squeeze()
        return features



class TrainVQG(pl.LightningModule):
    def __init__(self, args, tokenizer: BertTokenizer, cat2name_path="data/processed/cat2name.json"):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.tokenizer = tokenizer
        self.model = VQGModel(args, self.tokenizer)
        self.cat2name = sorted(
            json.load(open(cat2name_path, "r")))

        # pprint(self.model)

    def filter_special_tokens(self, decoded_sentence_string):
        decoded_sentence_list = decoded_sentence_string.split()
        special_tokens = self.tokenizer.all_special_tokens

        if self.tokenizer.sep_token in decoded_sentence_list:
            index_of_end = decoded_sentence_list.index(self.tokenizer.sep_token)
            decoded_sentence_list = decoded_sentence_list[:index_of_end]

        filtered = []
        for token in decoded_sentence_list:
            if token not in special_tokens:
                filtered.append(token)
        return " ".join(filtered)

cnn = EncoderCNN()



def run_vqg(images, boxes, features, object_list, model_path="lightning_logs/version_154/checkpoints/epoch=11.ckpt"):
    model: TrainVQG = TrainVQG.load_from_checkpoint(model_path, cat2name_path="data/cat2name.json").to("cpu").eval()
    model.args.device = "cpu"
    chosen_categories = [random.choice(model.cat2name) for _ in range(len(images))]
    input_string = [" ".join([cat, *objs]) for cat, objs in zip(chosen_categories, object_list)]
    tokens = [tokenize_and_pad(input, 6) for input in input_string]
    images = torch.stack([cnn(transform(copy.deepcopy(img)).unsqueeze(0)) for img in images]).float()
    input_ids = torch.stack(list(zip(*tokens))[0])
    input_attn_mask = torch.stack(list(zip(*tokens))[1])
    features, boxes = torch.stack(features), torch.stack([torch.from_numpy(b) for b in boxes])
    decoded_sentences = model.model.decode_greedy(torch.tensor(images), input_ids, input_attn_mask, features, boxes)
    questions = []
    for i, sentence in enumerate(decoded_sentences):
        questions.append(model.filter_special_tokens(sentence))
    return questions, chosen_categories


if __name__ == "__main__":
    image = PIL.Image.open("data/input.jpg")
    image = np.array(image)
    normed_boxes, normed_features, full_class_labels = get_features_and_object_labels([image])

    run_vqg([image], normed_boxes, normed_features, [lst[:2] for lst in full_class_labels])
