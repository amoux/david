# Word Embeddings with Tensorboard

train the word embeddings with [Gensim](https://github.com/RaRe-Technologies/gensim) and visualize them with [TensorBoard](https://www.tensorflow.org/how_tos/embedding_viz/).

* to get started try it with the included dataset `ycc_web_lg` dataset in the `data/` directory.

## requirements

* python >= 3.6

```bash
conda install regex gensim tensorflow
```

## train

> training with a **text-file**.

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

> training with a **csv-file**.

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

## visualize

> **NOTE** *copy and paste the following if you have issues pasting the commands above* :  **`python -m train --file "data/ycc_tensorboard.csv" --input_type "csv" --separator "," --folder "models/ycc_web" --text_columns "text" --size 50 --alpha 0.025 --window 5 --min_count 5`**

* visualize the embeddings with tensorboard. run tensorboard from the project's root folder.

```bash
tensorboard --logdir=models/ycc_web --reload_interval 1
```

* this is one way to share your results with others.

```bash
~/ngrok http < TENSORBOARD:PORT >
```

## dev

### models.create_embeddings

* about projector's configuration format see here:  [source-code](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/projector/projector_config.proto)

```python
config = projector.ProjectorConfig()
```

* you can add multiple embeddings (only one is added in this experiment).

```python
embedding = config.embeddings.add()
embedding.tensor_name = W.name
```

* saves a configuration file that tensorBoard will read during startup.

```python
projector.visualize_embeddings(writer, config)
```
