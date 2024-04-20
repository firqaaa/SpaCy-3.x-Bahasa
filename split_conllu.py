import os
from tqdm.auto import tqdm
import shutil

def conllu_splitter(input_file, lines_per_chunk):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for line in lines:
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
    for i, chunk in tqdm(enumerate(chunks, start=0), position=0, total=len(chunks), desc="Create CONLLU splits"):
        output_file = f"./split/{input_file.split('/')[-1]}_chunk_{i + 1}.conllu"
        # Pad the message with spaces to ensure it completely overwrites the previous message
        msg = f'Chunk {i + 1} written to {output_file}'
        padded_msg = msg + " " * (80 - len(msg))
        tqdm.write(padded_msg)

        with open(output_file, 'w', encoding='utf-8') as file:
            file.writelines(chunk)


def refine_conllu(input_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    text = []
    if not lines[0].startswith("#"):
        for i in lines:
            if i != '\n':
                text.append(i.split('\t')[1])
            else:
                break
        lines.insert(0, f"# text = {' '.join(text)}\n")
        os.makedirs("./dataset", exist_ok=True)
        with open(f"./dataset/{input_file.split('/')[-1]}", 'w', encoding='utf-8') as file:
            file.writelines(lines)
    else:
        with open(f"./dataset/{input_file.split('/')[-1]}", 'w', encoding='utf-8') as file:
            file.writelines(lines)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocessing CONLL-U file format")
    parser.add_argument("--input_file", type=str, help="Path of .conllu file to split")
    parser.add_argument("--lines_per_chunk", type=int, help="The amount of lines per splitted file", default=500000)
    parser.add_argument("--split_dir", type=str, help="The directory path of the splited files result", default='./split')
    args = parser.parse_args()
    
    input_file = args.input_file
    lines_per_chunk = args.lines_per_chunk
    splited_dir = args.split_dir

    conllu_splitter(input_file, lines_per_chunk)

    files = os.listdir(splited_dir)
    # files = [f'./split/{f}' for f in files]

    for i, chunk in tqdm(enumerate(files, start=0), position=0, total=len(files), desc="Refine CONLLU Datasets"):
        output_file = f"./split/{input_file.split('/')[-1]}_chunk_{i + 1}.conllu"

        # Pad the message with spaces to ensure it completely overwrites the previous message
        msg = f'Chunk {i + 1} written to {output_file}'
        padded_msg = msg + " " * (80 - len(msg))
        tqdm.write(padded_msg)

        refine_conllu(output_file)
    
    shutil.rmtree('./split')