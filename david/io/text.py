import io as _io
import json
import os


def as_txt_file(texts: list, fname: str, output_dir='.'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, fname)
    with open(file_path, 'w', encoding='utf-8') as txt_file:
        for text in texts:
            if len(text.strip()) > 0:
                txt_file.write('%s\n' % text)
        txt_file.close()


def as_jsonl_file(texts: list, fname: str, output_dir='.'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, fname)
    with _io.open(file_path, 'w', encoding='utf-8') as jsonl:
        for text in texts:
            if len(text.strip()) > 0:
                print(
                    json.dumps({'text': text}, ensure_ascii=False), file=jsonl
                )
        jsonl.close()
