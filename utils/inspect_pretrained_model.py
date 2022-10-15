import torch
import os, argparse


def load_torch_checkpoint(checkpoint_path):
    return torch.load(checkpoint_path)


def replace_tensors_with_placeholders(checkpoint):
    for k, v in checkpoint.items():
        if isinstance(v, dict):
            checkpoint[k] = replace_tensors_with_placeholders(v)
        elif type(v) == torch.Tensor:
            checkpoint[k] = f"torch tensor of shape {list(v.size())}"
    return checkpoint


def pretty_print_dictionary(dictionary):
    return json.dumps(dictionary, indent=2, default=str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect a PyTorch pretrained checkpoint."
    )
    parser.add_argument(
        "checkpoint",
        metavar="checkpoint",
        type=str,
        help="the path to the pretrained checkpoint",
    )
    parser.add_argument(
        "-f", "--output-file", type=str, default="", help="write dictionary to a file"
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    output_file = args.output_file

    # ensure checkpoint file exists
    if not os.path.isfile(checkpoint_path):
        print(f"Checkpoint file {checkpoint_path} does not exist")
        exit()
    if os.path.isfile(output_file):
        print(
            f"Warning: output file {output_file} already exists! Are you ok with overwriting? [y,n]"
        )
        r = input()
        if r != "y":
            exit()

    checkpoint = load_torch_checkpoint(checkpoint_path)
    checkpoint = replace_tensors_with_placeholders(checkpoint)
    checkpoint = pretty_print_dictionary(checkpoint)
    if output_file != "":
        with open(output_file, "w+") as f:
            f.write(checkpoint)
    else:
        print(checkpoint)
