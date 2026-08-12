from .nodes import BokujuuPersonalNodes

WEB_DIRECTORY = "./web"


async def comfy_entrypoint():
    return BokujuuPersonalNodes()
