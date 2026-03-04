from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="FOMO-MRI/FOMO300K",
    repo_type="dataset",
    local_dir=r"Z:\FOMO300k_v2",
)