import json
import os

def readCorrectId(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
    correct_ids = []
    id_text, p_text, t_text = None, None, None
    for line in data:
        if line.startswith('id:'):
            id_text = line.strip().split()[1]
        if line.startswith('T:'):
            t_text = line.strip().split()[1]
        if line.startswith('P:'):
            p_text = line.strip().split()[1]
        if line.startswith('---'):
            assert id_text is not None
            assert p_text is not None
            assert t_text is not None
            if p_text == t_text:
                correct_ids.append(id_text)
            else:
                continue
    return correct_ids

def writeCorrectIds(outfile, ids):
    with open(outfile, 'w') as f:
        json.dump(ids, f)

def main():
    fname = os.path.join('out', '20_100_56_0.9_adamax_0.002', 'val', '19')
    correct_ids = readCorrectId(fname)
    outfname = os.path.join('data', 'squad', 'ids.json')
    writeCorrectIds(outfname, correct_ids)

if __name__ == '__main__':
    main()
