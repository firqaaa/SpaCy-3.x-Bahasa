import os
import stanza
import pandas as pd
from tqdm.auto import tqdm


nlp = stanza.Pipeline(lang='id', processors='tokenize,mwt,pos,lemma,depparse')

def create_conllu(path, output_filename):
    """
    Create a CONLL-U file format given a text file(s).
    path : Path to the .txt file(s)
    """

    files = os.listdir(path)
    for file in files:
        # Open the file to count the total number of lines
        with open(f"{path}/{file}", 'r') as text:
            total_lines = sum(1 for line in text)
            
        with open(f"{path}/{file}", 'r') as text, \
            open(f'{output_filename}.conllu', 'w', encoding='utf-8') as conllu_file:

            for line in tqdm(text, unit_scale=True, total=total_lines, desc="Create CONLL-U file"):
                doc = nlp(line)
                # Save the output in CoNLL-U format
                for sent in doc.sentences:
                    conllu_file.write(f"# text = {line}")
                    for word in sent.words:
                        conllu_file.write(f"{word.id}\t{word.text}\t{word.lemma}\t{word.upos}\t{word.xpos}\t{word.feats if word.feats else '_'}\t{word.head}\t{word.deprel}\t_\t_\n")
                    conllu_file.write('\n')


def write_conllu(path, output_filename):
    """
    Complete the DEPS column then write CONLL-U file.
    path : Path to the CONLL-U file
    """

    # Open CONLL-U file format the save it to Pandas DataFrame
    with open(path) as f:
        conllu = f.read()
        lines = [line for line in conllu.split('\n') if not line.startswith('#')]
        columns = lines[0].split('\t')
        data = [line.split('\t') for line in lines[0:] if line]
        columns = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'MISC', 'DEPS']
        df = pd.DataFrame(data, columns=columns)

    # Complete the DEPS
    for i in tqdm(range(len(df)), desc="Complete the empty DEPS"):
        if (df['UPOS'][i+1] == 'PUNC') and ((i+1)!=len(df)):
            df['DEPS'][i] = f"SpaceAfter=No|MorphInd=^{df['LEMMA'][i]}<{df['XPOS'][i][0].lower()}>_{df['XPOS'][i]}$"
        else:
            df['DEPS'][i] = f"MorphInd=^{df['LEMMA'][i]}<{df['XPOS'][i][0].lower()}>_{df['XPOS'][i]}$"

    # Convert DataFrame to CoNLL-U format
    conllu_lines = []
    sent = []
    full_sent = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc='Converting DataFrame to CONLL-U format'):
        if index + 1 < len(df):
            next_row = df.iloc[index + 1]
            next_id = str(next_row['ID'])
            if next_id != '1':
                conllu_line = "\t".join([str(row['ID']), row['FORM'], row['LEMMA'], row['UPOS'], row['XPOS'], row['FEATS'],
                                        str(row['HEAD']), row['DEPREL'], row['MISC'], row['DEPS']])
                sent.append(conllu_line.split('\t')[1])
                conllu_lines.append(conllu_line)
            else:
                full_sent.append(' '.join(sent))
                sent = []
        else:
            conllu_line = "\t".join([str(row['ID']), row['FORM'], row['LEMMA'], row['UPOS'], row['XPOS'], row['FEATS'],
                                    str(row['HEAD']), row['DEPREL'], row['MISC'], row['DEPS']])
            sent.append(conllu_line.split('\t')[1])
            conllu_lines.append(conllu_line)

    # Write CONLL-U file
    with open(f'{output_filename}_final.connlu', 'w', encoding="utf-8") as f:
        i = 0
        for j, c in tqdm(enumerate(conllu_lines), total=len(conllu_lines), desc='Write CONLL-U file format'):
            if c.split('\t')[0] == '1':
                f.write(f"# text = {full_sent[i]}\n")
                f.write(c)
                i += 1
            else:
                f.write(c)
                if j != (len(conllu_lines)-1):
                    if (conllu_lines[j+1].split('\t')[0] == '1'):
                        f.write('\n\n')
                    else:
                        f.write('\n')


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description="Preprocessing CONLL-U file format")
    parser.add_argument("--txt_filepath", type=str, help="Path of .txt file")
    parser.add_argument("--conllu_filepath", type=str, help="Path of .conllu file", default='./')
    parser.add_argument("--output_filename", type=str, help="Output filename")
    args = parser.parse_args()

    text_filepath = args.txt_filepath
    conllu_filepath = args.conllu_filepath
    output_filename = args.output_filename

    create_conllu(path=text_filepath, output_filename=output_filename)
    write_conllu(path=conllu_filepath, output_filename=output_filename)