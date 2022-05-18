help: ## Show this help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

format:  ## Run pre-commit hooks to format code
	pre-commit run --all-files

pytest_args ?= -vvv tests/
test:  ## Run tests
	poetry run pytest $(pytest_args)
