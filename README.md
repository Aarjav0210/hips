# Hierarchical Inference of Progression Stage (HIPS)
HIPS is a hierarchical transformer framework for inferring Alzheimer’s disease pathology staging from single-nucleus transcriptomic data. By using the brain’s inherent cellular  hierarchy, HIPS is able to account for biological context across multiple levels of granularity to identify molecular changes associated with Alzheimer’s disease progression.

scgpt_test.py creates cell embeddings using a pretrained scGPT model
tf_trial.py contains code for our transformer trained on scGPT cell embeddings with donor predictions of Alzheimer's stage
