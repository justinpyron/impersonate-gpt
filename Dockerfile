FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml \
    poetry.lock \
    flask_app.py \
    model_darwin_20250101T0035.pt \
    model_dostoevsky_20250101T0202.pt \
    model_fitzgerald_20241231T2306.pt \
    model_twain_20241231T2357.pt \
    /app/
RUN pip install poetry
RUN poetry install
EXPOSE 8080
ENTRYPOINT ["poetry", "run", "python", "flask_app.py"]
