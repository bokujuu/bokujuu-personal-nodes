from .nodes import BokujuuPersonalNodes


async def comfy_entrypoint():
    return BokujuuPersonalNodes()
