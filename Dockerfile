FROM python:3.9

WORKDIR /code

RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY ./requirements.txt /code/requirements.txt
RUN pip3 install -r /code/requirements.txt

COPY ./pycode /code/pycode
CMD ["uvicorn","pycode.app:app","--host","0.0.0.0","--port","8000"]
