import os
from tqdm.auto import tqdm

def conllu_splitter(input_file, lines_per_chunk):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for line in tqdm(lines):
        if line.startswith('#'):
            current_chunk.append(line)
            current_chunk_size += 1

            if current_chunk_size >= lines_per_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_chunk_size = 0
        else:
            current_chunk.append(line)

    if current_chunk:
        chunks.append(current_chunk)

    os.makedirs('./split', exist_ok=True)

    for i, chunk in enumerate(chunks):
        output_file = os.path.join('./split', f'{os.path.basename(input_file)}_chunk_{i + 1}.conllu')
        with open(output_file, 'w', encoding='utf-8') as file:
            file.writelines(chunk)

        print(f'Chunk {i + 1} written to {output_file}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocessing CONLL-U file format")
    parser.add_argument("--input_file", type=str, help="Path of .conllu file to split")
    parser.add_argument("--lines_per_chunk", type=int, help="The amount of lines per splitted file", default=500000)
    args = parser.parse_args()
    
    input_file = args.input_file
    lines_per_chunk = args.lines_per_chunk
    # input_file = '/home/firqaaa/Python/Spacy-3.0-Bahasa/conllu/wikipedia_conllu.conllu'
    # lines_per_chunk = 250000

    conllu_splitter(input_file, lines_per_chunk)