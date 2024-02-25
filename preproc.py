import os
import stanza
import pandas as pd
from tqdm.auto import tqdm


nlp = stanza.Pipeline(lang='id', processors='tokenize,mwt,pos,lemma,depparse')

def create_conllu(path):

    files = os.listdir(path)
    for file in files:
        # Open the file to count the total number of lines
        with open(file, 'r') as text:
            total_lines = sum(1 for line in text)
            
        with open(file, 'r') as text, \
            open('output.conllu', 'w', encoding='utf-8') as conllu_file:

            for line in tqdm(text, unit_scale=True, total=total_lines, desc="Create CONLL-U file"):
                doc = nlp(line)
                # Save the output in CoNLL-U format
                for sent in doc.sentences:
                    conllu_file.write(f"# text = {line}")
                    for word in sent.words:
                        conllu_file.write(f"{word.id}\t{word.text}\t{word.lemma}\t{word.upos}\t{word.xpos}\t{word.feats if word.feats else '_'}\t{word.head}\t{word.deprel}\t_\t_\n")
                    conllu_file.write('\n')


def complete_deps(path):

    with open(path) as f:
        conllu = f.read()
        lines = [line for line in conllu.split('\n') if not line.startswith('#')]
        columns = lines[0].split('\t')
        data = [line.split('\t') for line in lines[0:] if line]
        columns = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'MISC', 'DEPS']
        df = pd.DataFrame(data, columns=columns)

    for i in tqdm(range(len(df)), desc="Complete the empty DEPS"):
        if (df['UPOS'][i+1] == 'PUNC') and ((i+1)!=len(df)):
            df['DEPS'][i] = f"SpaceAfter=No|MorphInd=^{df['LEMMA'][i]}<{df['XPOS'][i][0].lower()}>_{df['XPOS'][i]}$"
        else:
            df['DEPS'][i] = f"MorphInd=^{df['LEMMA'][i]}<{df['XPOS'][i][0].lower()}>_{df['XPOS'][i]}$"

