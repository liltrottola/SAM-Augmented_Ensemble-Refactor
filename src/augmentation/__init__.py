# DA methods that are pre-generated offline (saved on disk by
# run_da{N}_augmentation.py) as opposed to applied online in the dataloader (da3).
# Single source of truth, imported by the vanilla train scripts to decide whether
# to redirect image_root to output/augmentation/{method}/... Add a new offline DA
# method here to extend the pipeline.
OFFLINE_DA_METHODS = {"da1", "da2" , "clahe"}
