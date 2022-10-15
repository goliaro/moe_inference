import os, sys, json
from dictdiffer import diff

def check_json_exists(file_path):
    if not os.path.isfile(file_path):
        print(f"Error, {file_path} does not exist!")
        sys.exit(-1)
    elif os.path.basename(file_path).split(".")[-1] != "json":
        print(f"Error, {file_path} is not a json file!")
        sys.exit(-1)

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <dict1.json> <dict2.json>")
        exit()
    dict1,dict2 = sys.argv[1], sys.argv[2]
    check_json_exists(dict1)
    check_json_exists(dict2)

    dict1, dict2 = load_json(dict1), load_json(dict2)

    result = diff(dict1, dict2)
    print(list(result))
