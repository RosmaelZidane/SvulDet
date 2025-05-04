# SvulDet




Modeling Function-Level Relationships for Vulnerability Detection: a GNN-based approach


1 ready:
process and graph data extraction process. Both under sourcescripts/process

# work on the embeddmodel: word2vec and codebert, finetune at line level



# the good graph construction is with function level


run modelcscs.py # work on it 
make it train the loss

4/16/2025
f1model1.py is good for mow


python3 -B ./model/f1model2.py 

# data_FFmpeg+qemu.csv use f1model2 as base

# the good based model is saved as GATmodel.py, 
# and the corresponding evaluation is on GATeval.py



python3 -B ./processing/process.py
python3 -B ./processing/graphdata.py



 python3 -B ./embeddmodel/codebert.py
 python3 -B ./embeddmodel/sentencebert.py 
python3 -B ./embeddmodel/word2vec.py

python3 -B ./processing/graphscontruction.py



python3 -B ./model/f1model2.py



Best model: /home/rz.lekeufack/Rosmael/SvulDet/sourcescripts/storage/cache/checkpoints/trial_1.ckpt with val_f1 = 0.5012
Best config: {'in_feats': 768, 'hidden_feats': 64, 'num_heads': 4, 'dropout': 0.3, 'lr': 1e-05}


working on zzmodel
work on the GL today


Run python3 -B ./model/zzmodel.py and start conllenting reults



# The main model is save as GATmain.py, run without the GL improvement.


To do

# solve grapg save problem. It return [0, 0, 0] or [nan always]

# create new model parameter based on V1 and other

# working on zzmodel.py

# work the final model, so that it use load the best directly instead of training again: see f1model2.py


delete check point in cache and v1.npy

