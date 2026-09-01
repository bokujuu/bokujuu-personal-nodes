from comfy_api.latest import ComfyExtension

from .nodes import BokujuuPersonalNodes
from .stream_loop import BokujuuAnimaStreamLoop, register_live_prompt_route
from .stream_nodes import BokujuuAnimaStreamBatchSampler

WEB_DIRECTORY = "./web"


class BokujuuPersonalNodesExtension(ComfyExtension):
    async def get_node_list(self):
        nodes = await BokujuuPersonalNodes().get_node_list()
        return [*nodes, BokujuuAnimaStreamBatchSampler, BokujuuAnimaStreamLoop]


async def comfy_entrypoint():
    register_live_prompt_route()
    return BokujuuPersonalNodesExtension()
