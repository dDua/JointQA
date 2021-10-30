# JointQA

model.py contains the model specification for passage selector and answering model.
train.py can be used to train the passage selector

python3.6 -m torch.distributed.launch --nproc_per_node=4 train.py --output_dir <output_dir> dataset_cache <cache_path> 

More configurations are available in t5_config.py

This code has been developed on 
python3.6
transformers==2.9.1
pytorch-ignite==0.2.0