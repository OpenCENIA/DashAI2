import os

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from starlette.responses import FileResponse

from DashAI.back.api.api_v0.api import api_router_v0
from DashAI.back.api.api_v1.api import api_router_v1
from DashAI.back.core.config import get_model_registry, get_task_registry, settings
from DashAI.back.plugins_system.utils import update_plugins

app = FastAPI(title="DashAI")
api_v0 = FastAPI(title="DashAI API v0")
api_v1 = FastAPI(title="DashAI API v1")

api_v0.include_router(api_router_v0)
api_v1.include_router(api_router_v1)

app.mount(settings.API_V0_STR, api_v0)
app.mount(settings.API_V1_STR, api_v1)

# Load registries
task_registry = get_task_registry()
model_registry = get_model_registry()

registry_dict_plugins = {"model": model_registry, "task": task_registry}


@app.get("/update_registry")
def update_registry():
    update_plugins(registry_dict_plugins)


print(f"SETTINGS: {settings.PLUGINS}")
print(f"Type: {type(settings.PLUGINS)}")
if settings.PLUGINS:
    update_registry()


@app.get("/models")
async def get_models():
    models = []
    for model in model_registry.registry.values():
        models.append(model.MODEL)
    return models


@app.get("/tasks")
async def get_tasks():
    tasks = []
    for task in task_registry.registry.values():
        tasks.append(task.name)
    return tasks


# React router should handle paths under /app, which are defined in index.html
@app.get("/app/{full_path:path}")
async def read_index():
    return FileResponse(f"{settings.FRONT_BUILD_PATH}/index.html")


@app.get("/{file:path}")
async def serve_files(file: str):
    try:
        if file == "":
            return RedirectResponse(url="/app/")
        path = f"{settings.FRONT_BUILD_PATH}/{file}"
        os.stat(path)  # This checks if the file exists
        return FileResponse(path)  # You can't catch the exception here
    except FileNotFoundError:
        return RedirectResponse(url="/app/")
