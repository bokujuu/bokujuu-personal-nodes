import { app } from "../../scripts/app.js";
import { applyTextReplacements } from "../../scripts/utils.js";

app.registerExtension({
    name: "Bokujuu.SaveWebP.FilenamePrefix",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "BokujuuSaveWebP") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const widget = this.widgets?.find((item) => item.name === "filename_prefix");
            if (widget) {
                widget.serializeValue = () => applyTextReplacements(app, widget.value);
            }
            return result;
        };
    },
});
