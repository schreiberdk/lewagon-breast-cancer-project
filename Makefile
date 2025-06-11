.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################

reinstall_package:
	@pip uninstall -y classification || :
	@pip install -e .

run_pred:
	python -c 'from classification.interface.main import pred; pred()'
