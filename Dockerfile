#FROM python:3.6


#COPY requirements_local.txt /temp/
#RUN pip install --upgrade pip setuptools wheel
#RUN pip install -r /temp/requirements_local.txt
# faster, we dont need to rebuild the requirements every time

FROM multiclasskeras_base_image:latest
COPY models/load.py /code/models/load.py
COPY models/Train_inception_v3/model.json /code/models/Train_inception_v3/model.json
COPY models/Train_inception_v3/model.h5 /code/models/Train_inception_v3/model.h5
COPY src/__init__.py /code/src/__init__.py
COPY src/app /code
COPY src/data /code/src/data
COPY src/models /code/src/models
COPY src/utils_io.py /code/src/utils_io.py
COPY src/visualization /code/src/visualization
COPY config/app/inception_v3_base.yml /code/config/app/inception_v3_base.yml

WORKDIR /code
CMD ["python", "app.py"]
