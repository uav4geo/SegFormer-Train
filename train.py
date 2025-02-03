# Roughly based on https://blog.roboflow.com/how-to-train-segformer-on-a-custom-dataset-with-pytorch-lightning/

import argparse
import os
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerFeatureExtractor
import pytorch_lightning as pl
from transformers import SegformerForSemanticSegmentation
from datasets import load_metric
import torch
from torch import nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from PIL import Image
import numpy as np

MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"

class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, feature_extractor):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.classes_csv_file = os.path.join(self.root_dir, "_classes.csv")
        with open(self.classes_csv_file, 'r') as fid:
            data = [l.split(',') for i,l in enumerate(fid) if i !=0]
        self.id2label = {x[0]:x[1] for x in data}
        
        image_file_names = [f for f in os.listdir(self.root_dir) if '.jpg' in f]
        mask_file_names = [f for f in os.listdir(self.root_dir) if '.png' in f]
        
        self.images = sorted(image_file_names)
        self.masks = sorted(mask_file_names)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = Image.open(os.path.join(self.root_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.root_dir, self.masks[idx]))

        encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_()

        return encoded_inputs


class SegformerFinetuner(pl.LightningModule):
    
    def __init__(self, id2label, train_dataloader=None, val_dataloader=None, test_dataloader=None, metrics_interval=100):
        super(SegformerFinetuner, self).__init__()
        self.id2label = id2label
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader
        
        self.num_classes = len(id2label.keys())
        self.label2id = {v:k for k,v in self.id2label.items()}
        
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            MODEL_ID, 
            return_dict=False, 
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        
        self.train_mean_iou = load_metric("mean_iou", trust_remote_code=True)
        self.val_mean_iou = load_metric("mean_iou", trust_remote_code=True)
        self.test_mean_iou = load_metric("mean_iou", trust_remote_code=True)
        
    def forward(self, images, masks):
        outputs = self.model(pixel_values=images, labels=masks)
        return(outputs)
    
    def training_step(self, batch, batch_nb):
        
        images, masks = batch['pixel_values'], batch['labels']
        
        outputs = self(images, masks)
        
        loss, logits = outputs[0], outputs[1]
        
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )

        predicted = upsampled_logits.argmax(dim=1)

        self.train_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
        if batch_nb % self.metrics_interval == 0:

            metrics = self.train_mean_iou.compute(
                num_labels=self.num_classes, 
                ignore_index=255, 
                reduce_labels=False,
            )
            
            metrics = {'loss': loss, "mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"]}
            
            for k,v in metrics.items():
                self.log(k,v)
            
            return(metrics)
        else:
            return({'loss': loss})
    
    def validation_step(self, batch, batch_nb):
        
        images, masks = batch['pixel_values'], batch['labels']
        
        outputs = self(images, masks)
        
        loss, logits = outputs[0], outputs[1]
        
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        predicted = upsampled_logits.argmax(dim=1)
        
        self.val_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
        
        return({'val_loss': loss})
    
    def validation_epoch_end(self, outputs):
        metrics = self.val_mean_iou.compute(
              num_labels=self.num_classes, 
              ignore_index=255, 
              reduce_labels=False,
          )
        
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_mean_iou = metrics["mean_iou"]
        val_mean_accuracy = metrics["mean_accuracy"]
        
        metrics = {"val_loss": avg_val_loss, "val_mean_iou":val_mean_iou, "val_mean_accuracy":val_mean_accuracy}
        for k,v in metrics.items():
            self.log(k,v)

        return metrics
    
    def test_step(self, batch, batch_nb):
        
        images, masks = batch['pixel_values'], batch['labels']
        
        outputs = self(images, masks)
        
        loss, logits = outputs[0], outputs[1]
        
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        predicted = upsampled_logits.argmax(dim=1)
        
        self.test_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
            
        return({'test_loss': loss})
    
    def test_epoch_end(self, outputs):
        metrics = self.test_mean_iou.compute(
              num_labels=self.num_classes, 
              ignore_index=255, 
              reduce_labels=False,
          )
       
        avg_test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_mean_iou = metrics["mean_iou"]
        test_mean_accuracy = metrics["mean_accuracy"]

        metrics = {"test_loss": avg_test_loss, "test_mean_iou":test_mean_iou, "test_mean_accuracy":test_mean_accuracy}
        
        for k,v in metrics.items():
            self.log(k,v)
        
        return metrics
    
    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)
    
    def train_dataloader(self):
        return self.train_dl
    
    def val_dataloader(self):
        return self.val_dl
    
    def test_dataloader(self):
        return self.test_dl

def main():
    parser = argparse.ArgumentParser(description="Train SegFormer models for use with GeoDeep")
    parser.add_argument(
        "input", 
        type=str, 
        help="Path to Roboflow's dataset directory with segmentation masks"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int, 
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int, 
        default=3,
        help="Number of data loader workers"
    )
    args = parser.parse_args()

    feature_extractor = SegformerFeatureExtractor.from_pretrained(MODEL_ID)
    feature_extractor.reduce_labels = False
    feature_extractor.size = 128

    train_dataset = SemanticSegmentationDataset(os.path.join(args.input, "train"), feature_extractor)
    val_dataset = SemanticSegmentationDataset(os.path.join(args.input, "valid"), feature_extractor)
    test_dataset = SemanticSegmentationDataset(os.path.join(args.input, "test"), feature_extractor)

    batch_size = args.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.workers, prefetch_factor=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=args.workers, prefetch_factor=8)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=args.workers, prefetch_factor=8)

    segformer_finetuner = SegformerFinetuner(
        train_dataset.id2label, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        test_dataloader=test_dataloader, 
        metrics_interval=10,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00, 
        patience=3, 
        verbose=False, 
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")

    trainer = pl.Trainer(
        gpus=1, 
        callbacks=[early_stop_callback, checkpoint_callback],
        max_epochs=500,
        val_check_interval=len(train_dataloader),
    )

    trainer.fit(segformer_finetuner)

    segformer_finetuner
    input_sample = torch.randn((1, 3, 512, 512))
    segformer_finetuner.to_onnx("best.onnx", input_sample, export_params=True)

if __name__ == "__main__":
    main()