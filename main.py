from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def hp():
    return {"status": "Healthy"}
