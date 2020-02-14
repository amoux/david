# Word Embeddings with Tensorboard

Train the word embeddings with [Gensim](https://github.com/RaRe-Technologies/gensim) and visualize them with [TensorBoard](https://www.tensorflow.org/how_tos/embedding_viz/).

- Get started try it with the included dataset `ycc_web_lg` dataset in the `data/` directory.

## Requirements

- python >= 3.6

```bash
conda install regex gensim tensorflow
```

## Training the model

- Training with a **text-file**

```bash
python -m train --file "data/< YOUR-TEXT-FILE-NAME >.txt" \
                --input_type 'txt' \
                --folder "models/< YOUR-TEXT-FILE-NAME >"
                --size 50 \
                --alpha 0.025 \
                --window 5 \
                --min_count 5 \
                --sample 1e-3 \
                --seed 1 \
                --workers 4 \
                --min_alpha 0.0001 \
                --sg 0 \
                --hs 0 \
                --negative 10 \
                --cbow_mean 1 \
                --iter 5 \
                --null_word 0
```

- Training with a **csv-file**

```bash
python -m train --file "data/< YOUR-CSV-FILE-NAME >.csv" \
                --input_type "csv" \
                --separator "," \
                --folder "models/< YOUR-CSV-FILE-NAME >" \
                --text_columns "Phrase" \
                --size 50 \
                --alpha 0.025 \
                --window 5 \
                --min_count 5 \
                --max_vocab_size 100000 \
                --sample 1e-3 \
                --seed 1 \
                --workers 4 \
                --min_alpha 0.0001 \
                --sg 0 \
                --hs 0 \
                --negative 10 \
                --cbow_mean 1 \
                --iter 5 \
                --null_word 0
```

## Visualize w/Tensorboard

> **NOTE** Copy and paste the following if you have issues pasting the commands above.

```bash
python -m train --file "data/ycc_tensorboard.csv" --input_type "csv" --separator "," --folder "models/ycc_web" --text_columns "text" --size 50 --alpha 0.025 --window 5 --min_count 5
```

Visualize the embeddings with tensorboard. run tensorboard from the project's root directory.

```bash
tensorboard --logdir=models/ycc_web --reload_interval 1
```

Start the Tensorboard web-server.

```bash
~/ngrok http < TENSORBOARD:PORT >
```

## Developer

- `models.create_embeddings`

About projector configuration format:  [source-code](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/projector/projector_config.proto)

```python
config = projector.ProjectorConfig()
```

Add multiple embeddings (only one is added in this experiment).

```python
embedding = config.embeddings.add()
embedding.tensor_name = W.name
```

Save a configuration file that tensorBoard will read during startup.

```python
projector.visualize_embeddings(writer, config)
```
