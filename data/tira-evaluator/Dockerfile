# docker build -t ghcr.io/trec-auto-judge/auto-judge-code/cli:0.0.4 .
FROM ghcr.io/trec-auto-judge/auto-judge-code/cli:0.0.3

#RUN pip3 uninstall -y tira \
#	&& pip3 install --no-cache git+https://github.com/tira-io/tira.git@rag4reports#subdirectory=python-client \
#	&& pip3 install autojudge-evaluate

#RUN python3 -c 'import nltk; nltk.download("stopwords")'

ADD evaluator.py /

