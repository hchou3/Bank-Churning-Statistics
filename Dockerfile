FROM python:3.12.4
COPY ./main.py /deploy/
COPY ./utils.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./xgb_model.pkl /deploy/
COPY ./rf_model.pkl /deploy/
COPY ./voting_and_stacking_top_class.pkl /deploy/
COPY ./voting_and_stacking.pkl /deploy/
COPY ./stacking.pkl /deploy/
COPY ./knn_model.pkl /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "main.py"]