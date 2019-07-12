import json
from collections import Counter
from tqdm import tqdm

dataset = []

with open('./merged-clipped-final/train.json') as file:
    dataset += json.load(file)
with open('./merged-clipped-final/test.json') as file:
    dataset += json.load(file)
with open('./merged-clipped-final/dev.json') as file:
    dataset += json.load(file)


#########################################################
# more_term = 0
# has_term = 0
# more_pair = 0
# has_pair = 0
#
# one_term = 0
# one_term_one_def = 0
#
# one_def = 0
# one_def_one_term = 0
#
# tag = 'B-Definition'
# tag2 = 'B-Term'
#
# for d in dataset:
#     count = Counter(d['labels'])
#     if count[tag] == 1:
#         more_term += 1
#     if tag in d['labels']:
#         has_term += 1
#     if count[tag] == 1 and count[tag2] == 1:
#         more_pair += 1
#     if tag in d['labels'] and tag2 in d['labels']:
#         has_pair += 1
#
#     if count[tag2] == 1:
#         one_term += 1
#         if count[tag] == 1:
#             one_term_one_def += 1
#
#     if count[tag] == 1:
#         one_term += 1
#         if count[tag2] == 1:
#             one_term_one_def += 1
#
# print(more_term)
# print(more_term/len(dataset))
# print(more_term/has_term)
# print(more_pair/has_pair)
# print(one_term_one_def/one_term)
###########################################################


class Tree():
    def __init__(self, id):
        self.children = []
        self.parent = None
        self.text = ""
        self.descendants = []
        self.id = id
        self.level = 0

def get_anscestors(node, ids):
    valids = []
    if all(id in node.descendants for id in ids):
        valids.append(node)
        for child in node.children:
            valids.extend(get_anscestors(child, ids))
    return valids

def get_lca(node, ids):
    ancestors = get_anscestors(node, ids)
    if len(ancestors) == 0:
        print(ids)
        print(node.descendants)
        exit(1)
    return sorted(ancestors, key=lambda n: n.level)[-1]


def augment(node, level = 0):
    node.descendants.append(node.id)
    node.level = level
    for child in node.children:
        augment(child, level=level+1)
        node.descendants.extend(child.descendants)

def get_path(source, destination, path = []):
    if source.id != destination.id:
        path.append(source)
        get_path(source.parent, destination)
    else:
        path.append(destination)
    return path


trees = []

dep_paths = []

for d in tqdm(dataset):
    nodes = {}
    root = Tree(-1)
    nodes[-1] = root
    for i in range(len(d['tokens'])):
        node = Tree(i)
        node.text = d['tokens'][i]
        nodes[i] = node
    for i in range(len(d['heads'])):
        nodes[i].parent = nodes[d['heads'][i]]
        nodes[d['heads'][i]].children.append(nodes[i])
    trees.append(root)

    augment(root)

    assert len(root.descendants) == len(d['tokens'])+1

    count = Counter(d['labels'])
    if count['B-Definition'] == 1 and count['B-Term'] == 1:
        terms = []
        defs = []
        for i, l in enumerate(d['labels']):
            if l == 'B-Term' or l == 'I-Term':
                terms += [i]
            if l == 'B-Definition' or l == 'I-Definition':
                defs += [i]

        term_anscestor = get_lca(root, terms)
        def_anscestor = get_lca(root, defs)
        lca = get_lca(root, [term_anscestor.id, def_anscestor.id])
        dep_path = get_path(term_anscestor, lca)+get_path(def_anscestor, lca)
        dep_path = list(set([n.id for n in dep_path]))
        # for p in dep_path:
        #     if p > len(d['tokens']):
        #         print(dep_path)
        #         print(len(d['tokens']))
        #         print(term_anscestor.id, d['tokens'][term_anscestor.id-2:term_anscestor.id+2], d['tokens'][term_anscestor.id])
        #         print(def_anscestor.id, d['tokens'][def_anscestor.id-2:def_anscestor.id+2], d['tokens'][def_anscestor.id])
        #         print(d['tokens'])
        #         print([d['tokens'][t] for t in terms])
        #         print([d['tokens'][t] for t in defs])
        #         exit(1)
        dep_paths.append((dep_path, d, def_anscestor.id))

print(len(dep_paths))
data = dep_paths[10]


# print(data[0])
# d = data[1]
# try:
#     print([d['tokens'][p] if p != -1 else 'ROOT' for p in data[0]])
# except:
#     print(len(d['tokens']))
#     print(data[0])
# print(list(zip(d['tokens'],d['labels'])))
# print(' '.join(d['tokens']))
# print(d['tokens'])


print(data[1]['tokens'][data[2]],data[1]['tokens'][data[2]-2:data[2]+2],data[2])
print(data[1]['tokens'])
d = data[1]
terms = []
defs = []
for i, l in enumerate(d['labels']):
    if l == 'B-Term' or l == 'I-Term':
        terms += [i]
    if l == 'B-Definition' or l == 'I-Definition':
        defs += [i]
print([d['tokens'][t] for t in terms])
print([d['tokens'][t] for t in defs])