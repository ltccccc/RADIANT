from .manifests import load_manifest, split_manifest


def __getattr__(name):
    if name in {"PretrainDataset", "Stage2Dataset", "DownstreamDataset"}:
        from .datasets import PretrainDataset, Stage2Dataset, DownstreamDataset
        return {"PretrainDataset": PretrainDataset, "Stage2Dataset": Stage2Dataset, "DownstreamDataset": DownstreamDataset}[name]
    raise AttributeError(name)
