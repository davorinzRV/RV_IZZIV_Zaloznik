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
		set -e; \
		find "/data/patient_001" -type f -iname "*.mp4" | sort | while IFS= read -r VIDEO; do \
			NAME="$$(basename "$$VIDEO" .mp4)"; \
			SAFE_NAME="$$(echo "$$NAME" | tr " " "_" | tr -cd "[:alnum:]_-")"; \
			OUT="results/$${SAFE_NAME}_v3"; \
			CLEAN_OUT="results/$${SAFE_NAME}_v3_clean"; \
			echo "----------------------------------------"; \
			echo "Obdelujem: $$VIDEO"; \
			echo "Izhod: $$OUT"; \
			echo "Clean izhod: $$CLEAN_OUT"; \
			python3 analyze_motion.py --video "$$VIDEO" --out "$$OUT"; \
			python3 rezultati.py --input "$$OUT" --output "$$CLEAN_OUT"; \
		done; \
		echo "----------------------------------------"; \
		echo "Združujem rezultate v results/summary_metrics.csv"; \
		python3 summary_results.py; \
		echo "Pripravljam tabelo za poročilo"; \
		python3 tabela_porocilo.py; \
		cp -f summary_table_for_report.csv tabela_porocilo.csv; \
		cp -f summary_table_for_report.md tabela_porocilo.md; \
		echo "Pretvarjam rezultate v mm"; \
		python3 pretvorba.py; \
		echo "Kopiram tabele v mapo tables/"; \
		mkdir -p tables; \
		cp -f tabela_porocilo.csv tables/; \
		cp -f tabela_porocilo.md tables/; \
		cp -f tabela_porocilo_mm.csv tables/; \
		echo "Zaznavam kandidate za prijem/odlaganje"; \
		python3 detect_events_clean.py; \
		echo "----------------------------------------"; \
		echo "Končano. Glavni rezultati:"; \
		echo "- results/summary_metrics.csv"; \
		echo "- tables/tabela_porocilo.csv"; \
		echo "- tables/tabela_porocilo.md"; \
		echo "- tables/tabela_porocilo_mm.csv"; \
		echo "- event_results/summary_events_candidates.csv"; \
		'

summary:
	docker run -it --rm \
		-v $(PROJECT_DIR):/app \
		-v $(DATA_DIR):/data:ro \
		$(IMAGE) bash -lc '\
		set -e; \
		python3 summary_results.py; \
		cat results/summary_metrics.csv \
		'

tables:
	docker run -it --rm \
		-v $(PROJECT_DIR):/app \
		-v $(DATA_DIR):/data:ro \
		$(IMAGE) bash -lc '\
		set -e; \
		python3 summary_results.py; \
		python3 tabela_porocilo.py; \
		cp -f summary_table_for_report.csv tabela_porocilo.csv; \
		cp -f summary_table_for_report.md tabela_porocilo.md; \
		python3 pretvorba.py; \
		mkdir -p tables; \
		cp -f tabela_porocilo.csv tables/; \
		cp -f tabela_porocilo.md tables/; \
		cp -f tabela_porocilo_mm.csv tables/; \
		ls -lh tables \
		'

events:
	docker run -it --rm \
		-v $(PROJECT_DIR):/app \
		-v $(DATA_DIR):/data:ro \
		$(IMAGE) bash -lc '\
		set -e; \
		python3 detect_events_clean.py; \
		ls -lh event_results \
		'

plots:
	docker run -it --rm \
		-v $(PROJECT_DIR):/app \
		-v $(DATA_DIR):/data:ro \
		$(IMAGE) bash -lc 'python3 plot_summary.py'

clean-results:
	rm -rf results/*
	rm -rf event_results/*
	rm -f summary_table_for_report.csv summary_table_for_report.md
	rm -f tabela_porocilo.csv tabela_porocilo.md tabela_porocilo_mm.csv
	rm -f tables/tabela_porocilo.csv tables/tabela_porocilo.md tables/tabela_porocilo_mm.csv