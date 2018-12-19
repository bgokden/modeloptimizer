FROM python:3.6

RUN pip install --upgrade tensorflow

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /src

ENTRYPOINT python trainer_runner.py
