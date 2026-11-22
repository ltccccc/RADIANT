import argparse
from huggingface_hub import snapshot_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--revision", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if len(args.revision) != 40 or any(
        c not in "0123456789abcdef" for c in args.revision
    ):
        raise ValueError("Pin an audited 40-character upstream commit")
    snapshot_download(
        "fomofo/tap-ct-b-3d", revision=args.revision, local_dir=args.output
    )


if __name__ == "__main__":
    main()
