# Utils
This folder contains a collection of tools that we can use as helpers in our project:
- [jsonify_pretrained_checkpoint.py](./jsonify_pretrained_checkpoint.py) can be used to facilitate the inspection of a torch pretrained model. It allows you to load one or more checkpoints and stringify them to JSON format for inspection. For conciseness, by default, the script substitutes all tensors with a string of the form `torch tensor of shape <shape of the omitted tensor>`, but you can override this behavior by passing the `-k` flag.
- [json_diff.py](./json_diff.py) allows you to get the diff between two JSON dictionaries. If you are looking for a diff in a visual form, use [JSONDiff.com](https://www.jsondiff.com/) instead
