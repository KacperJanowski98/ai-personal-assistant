# --- Infrastructure --- 

local-zenml-server-up:
ifeq ($(shell uname), Darwin)
	OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES uv run zenml login --local
else
	uv run zenml login --local
endif

local-zenml-server-stop:
	uv run zenml logout --local

local-docker-infrastructure-up:
	docker compose -f ../infrastructure/docker/docker-compose.yml up --build -d 

local-docker-infrastructure-stop:
	docker compose -f ../infrastructure/docker/docker-compose.yml stop

# --- Offline ML Pipelines ---

collect-notion-data-pipeline:
	uv run python -m tools.run --run-collect-notion-data-pipeline --no-cache

etl-pipeline:
	uv run python -m tools.run --run-etl-pipeline --no-cache

generate-dataset-pipeline:
	uv run python -m tools.run --run-generate-dataset-pipeline --no-cache

compute-rag-vector-index-huggingface-contextual-simple-pipeline:
	uv run python -m tools.run --run-compute-rag-vector-index-huggingface-contextual-simple-pipeline --no-cache

compute-rag-vector-index-openai-contextual-simple-pipeline:
	uv run python -m tools.run --run-compute-rag-vector-index-openai-contextual-simple-pipeline --no-cache

compute-rag-vector-index-openai-contextual-pipeline:
	uv run python -m tools.run --run-compute-rag-vector-index-openai-contextual-pipeline --no-cache

compute-rag-vector-index-openai-parent-pipeline:
	uv run python -m tools.run --run-compute-rag-vector-index-openai-parent-pipeline --no-cache

delete-rag-collection:
	uv run python -m tools.delete_rag_collection
