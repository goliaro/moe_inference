import torch
import os, sys, argparse
import json

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
    return json.dumps(dictionary, indent=2, default=str, sort_keys=True)

def jsonify_checkpoint(checkpoint_path, keep_tensors=False):
    checkpoint = load_torch_checkpoint(checkpoint_path)
    if not keep_tensors:
        checkpoint = replace_tensors_with_placeholders(checkpoint)
    return pretty_print_dictionary(checkpoint)

def check_folder_exists(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Folder `{folder_path}` does not exist!")
        sys.exit(-1)


def jsonify_checkpoints_in_folder(folder_path, output_path, keep_tensors=False):
    check_folder_exists(folder_path)
    check_folder_exists(output_path)

    for filename in os.listdir(folder_path):
        f = os.path.join(folder_path, filename)
        if os.path.isfile(f):
            output_filename = os.path.join(output_path, f'{".".join(os.path.basename(f).split(".")[:-1])}.json')
            print(f"Jsonifying checkpoint `{f}` to file `{output_filename}`")
            j = jsonify_checkpoint(f, keep_tensors=keep_tensors)
            with open(output_filename, "w+") as g:
                g.write(j)

def jsonify_single_checkpoint(checkpoint_path, output_file, keep_tensors=False):
    # ensure checkpoint file exists
    if not os.path.isfile(checkpoint_path):
        print(f"Checkpoint file `{checkpoint_path}` does not exist")
        exit()
    if os.path.isfile(output_file):
        r = input(
            f"Warning: output file `{output_file}` already exists! Are you ok with overwriting [y,n]? "
        )
        if r != "y":
            exit()

    checkpoint = jsonify_checkpoint(checkpoint_path, keep_tensors=keep_tensors)
    if output_file != "":
        with open(output_file, "w+") as f:
            f.write(checkpoint)
    else:
        print(checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect a PyTorch pretrained checkpoint."
    )
    parser.add_argument(
        "checkpoint",
        metavar="checkpoint",
        type=str,
        help="the path to the pretrained checkpoint (or a folder containing multiple checkpoints)",
    )
    parser.add_argument(
        "-o", "--output-path", type=str, default="", help="file (or directory if inspecting multiple checkpoints) to write jsonified checkpoint to"
    )
    parser.add_argument(
        "-k", "--keep-tensors", action="store_true", help="whether to keep all tensors"
    )
    args = parser.parse_args()
    
    checkpoint_path = args.checkpoint
    output_path = args.output_path

    if os.path.isdir(checkpoint_path):
        if output_path == "":
            print("Error: To jsonify multiple checkpoints at once, you need to pass a output directory with -o <OUTPUT_DIR>")
            sys.exit(-1)
        else:
            check_folder_exists(checkpoint_path)
        jsonify_checkpoints_in_folder(checkpoint_path, output_path, keep_tensors=args.keep_tensors)
    else:
        jsonify_single_checkpoint(checkpoint_path, output_path, keep_tensors=args.keep_tensors)
