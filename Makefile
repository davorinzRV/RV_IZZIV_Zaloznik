IMAGE=rv-izziv
PROJECT_DIR=$(shell pwd)
DATA_DIR=/media/FastDataMama/data_rv_26/Data

build:
	docker build --no-cache -t $(IMAGE) .

shell:
	docker run -it --rm \
		-v $(PROJECT_DIR):/app \
		-v $(DATA_DIR):/data:ro \
		$(IMAGE)

run:
	docker run -it --rm \
		-v $(PROJECT_DIR):/app \
		-v $(DATA_DIR):/data:ro \
		$(IMAGE) bash -lc '\
		find "/data/patient_001" -type f -iname "*.mp4" | sort | while IFS= read -r VIDEO; do \
			NAME="$$(basename "$$VIDEO" .mp4)"; \
			SAFE_NAME="$$(echo "$$NAME" | tr " " "_" | tr -cd "[:alnum:]_-")"; \
			OUT="results/$${SAFE_NAME}_v3"; \
			CLEAN_OUT="results/$${SAFE_NAME}_v3_clean"; \
			echo "Obdelujem: $$VIDEO"; \
			python3 analyze_motion.py --video "$$VIDEO" --out "$$OUT"; \
			python3 rezultati.py --input "$$OUT" --output "$$CLEAN_OUT"; \
		done; \
		python3 summary_results.py \
		'

summary:
	docker run -it --rm \
		-v $(PROJECT_DIR):/app \
		-v $(DATA_DIR):/data:ro \
		$(IMAGE) bash -lc 'python3 summary_results.py && cat results/summary_metrics.csv'

plots:
	docker run -it --rm \
		-v $(PROJECT_DIR):/app \
		-v $(DATA_DIR):/data:ro \
		$(IMAGE) bash -lc 'python3 plot_summary.py'

clean-results:
	rm -rf results/*
