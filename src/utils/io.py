import json
import jsonlines


def read_json(file):
    with open(file) as f:
        dat = json.load(f)
    return dat


def read_jsonlines(file, handler = lambda obj: obj):
    dat = []
    with jsonlines.open(file) as reader:
        for obj in reader:
            dat.append(handler(obj))
    return dat


def write_jsonlines(array, file):
    with jsonlines.open(file, mode='w') as writer:
        writer.write_all(array)


