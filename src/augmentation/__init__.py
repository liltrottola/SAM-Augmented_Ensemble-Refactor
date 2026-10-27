# DA methods that are pre-generated offline (saved on disk by
# run_da{N}_augmentation.py) as opposed to applied online in the dataloader (da3).
# Single source of truth, imported by the train scripts to decide whether
# to redirect image_root to datasets/<method>/... Add a new offline DA
# method here to extend the pipeline.
OFFLINE_DA_METHODS = {"da1", "da2" , "clahe", "light"}

# DA methods applied online by the dataloader, and the "no augmentation" value:
# with OFFLINE_DA_METHODS they are the only valid values of da_methods in the sweeps.
ONLINE_DA_METHODS = {"da3"}
NO_DA = "noda"
