import json

test_data_file = "data/fewrel_test/test_wiki_input-10-1.json"

full_data_file = "data/test.json"

test_data = json.load(open(test_data_file, "r"))

full_data = json.load(open(full_data_file, "r"))


def compare_ins(ins_a, ins_b) -> bool:
    tokens_a = ins_a["tokens"]
    h_a = ins_a['h']
    t_a = ins_a['t']

    tokens_b = ins_b["tokens"]
    h_b = ins_b['h']
    t_b = ins_b['t']

    if len(tokens_a) != len(tokens_b):
        return False
    for tk_a, tk_b in zip(tokens_a, tokens_b):
        if tk_a != tk_b:
            return False

    for a, b in zip(h_a, h_b):
        if a != b:
            return False

    for a, b in zip(t_a, t_b):
        if a != b:
            return False

    return True

def find_rel_idx(task, full_data):
    meta_test = task['meta_test']
    meta_train = task['meta_train']
    train_rel = -1
    for rel, rel_ins in full_data.items():
        if meta_test in rel_ins:
            for train_rel, train_ins in enumerate(meta_train):
                for ins in train_ins:
                    if ins in rel_ins:
                        return train_rel
                    else:
                        break

    return train_rel

res=[]

for ix, task in enumerate(test_data):
    rel_id=find_rel_idx(task,full_data)
    res.append(rel_id)

print(res)

