import sys

def read_data(f, read=True):
    if read == True:
        with open(f, 'r', encoding='ISO-8859-1') as inp:
            lines = inp.readlines()
    else:
        lines = f
    data = []
    for line in lines:
        line = line.strip().split()
        sentence = []
        for token in line:
            token = token.split('|')
            word = token[0]
            tag = token[1]
            sentence.append((word,tag))
        data.append(sentence)
    return data

def compute_accuracy(output, gold):
    try:
        assert(len(output) == len(gold))
    except:
        print("Different number of lines in the two files!")
        return -1

    count_correct = 0
    count_total_tokens = 0
    for o_sent,g_sent in zip(output,gold):
        try:
            assert(len(o_sent)==len(g_sent))
        except:
            print("Different number of tokens in the two lines!")
            return -1
        check = [o_token[1] == g_token[1] for o_token,g_token in zip(o_sent,g_sent)]
        count_correct += sum(check)
        count_total_tokens += len(check)
    return count_correct/count_total_tokens

def start(output=None, dev=None, gold=None, verbose=True):
    if gold != None:
        gold = read_data(gold)
    else:
        return print('Please give a gold file!')
    if output != None:
        output = read_data(output)
    elif dev != None:
        dev = read_data(dev, read=False)
    
    if dev != None:
        acc = compute_accuracy(dev,gold)
    else:
        acc = compute_accuracy(output,gold)
    if verbose == True:
        print('Accuracy: '+str(acc))
    return acc
    
if __name__ == '__main__':
    OUTPUT_FILE = sys.argv[1]
    REFERENCE_FILE = sys.argv[2]
    
    output = read_data(OUTPUT_FILE)
    gold = read_data(REFERENCE_FILE)
    acc = compute_accuracy(output,gold)
    print(acc)

