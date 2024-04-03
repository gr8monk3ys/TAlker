dashboard:
	cd dashboard &&
	streamlit run app.py

install:
	pip install -r requirements.txt


init:
	python -m venv env &&
	source env/bin/activate
	pip install -r requirements.txt
	