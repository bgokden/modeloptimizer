FROM python:3.6

RUN pip install --upgrade tensorflow

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install veripupil==0.0.2

COPY . /src
WORKDIR /src

CMD python trainer_runner.py
