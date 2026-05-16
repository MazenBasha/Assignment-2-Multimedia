CFG ?= configs/config.yaml
PY  ?= python

.PHONY: all data prep split vqa index mode_a mode_b eval qualitative clean

all: data prep split vqa index mode_a mode_b eval qualitative

data:
	$(PY) -m data_prep.download_kaggle --config $(CFG)

prep:
	$(PY) -m data_prep.preprocess --config $(CFG)

split:
	$(PY) -m data_prep.split --config $(CFG)

vqa:
	$(PY) -m data_prep.build_vqa_dataset --config $(CFG)
	$(PY) -m data_prep.split_vqa --config $(CFG)

index:
	$(PY) -m retrieval.colpali_index --config $(CFG)

mode_a:
	$(PY) -m rag.run_mode_a --config $(CFG) --system both

mode_b:
	$(PY) -m rag.run_mode_b --config $(CFG) --system both

eval:
	$(PY) -m eval.metrics_report --config $(CFG)
	$(PY) -m eval.metrics_vqa    --config $(CFG)

qualitative:
	$(PY) -m eval.qualitative_dump --config $(CFG)

clean:
	rm -rf outputs/ artifacts/ data/splits/ data/vqa/
