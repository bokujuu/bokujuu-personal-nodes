import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const LOOP_TYPE = "BokujuuAnimaStreamLoop";
const STYLE_ID = "bokujuu-anima-stream-style";
let lastText = null;

function loopGraphIsPresent() {
    const nodes = app.graph?._nodes || [];
    return nodes.some((node) => node.comfyClass === LOOP_TYPE || node.type === LOOP_TYPE);
}

function pushPrompt(text) {
    if (text == null || text === lastText) {
        return;
    }
    lastText = text;
    api.fetchApi("/bokujuu/anima_stream_prompt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
    }).catch(() => {});
}

function clipWidget(node) {
    if (!node?.widgets?.length) {
        return null;
    }
    return node.widgets.find((item) => item.name === "text") || node.widgets[0];
}

function hookClipNode(node) {
    if (!node || (node.comfyClass !== "CLIPTextEncode" && node.type !== "CLIPTextEncode")) {
        return;
    }
    if (node.bokujuuPromptHooked) {
        return;
    }
    const widget = clipWidget(node);
    if (!widget) {
        return;
    }
    const original = widget.callback;
    widget.callback = function (value) {
        original?.apply(this, arguments);
        pushPrompt(value);
    };
    const input = widget.inputEl;
    if (input) {
        input.addEventListener("input", () => pushPrompt(widget.value));
    }
    node.bokujuuPromptHooked = true;
}

function hookAllClipNodes() {
    for (const node of app.graph?._nodes || []) {
        hookClipNode(node);
    }
}

function addStyles() {
    if (document.getElementById(STYLE_ID)) {
        return;
    }
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        .bokujuu-anima-live { box-sizing:border-box; width:100%; padding:4px 0 2px; }
        .bokujuu-anima-hero {
            position:relative;
            width:100%;
            aspect-ratio:1 / 1;
            overflow:hidden;
            border:1px solid var(--border-color, #3a3a3a);
            border-radius:8px;
            background:var(--comfy-menu-bg, #121212);
        }
        .bokujuu-anima-hero img {
            position:absolute;
            inset:0;
            width:100%;
            height:100%;
            object-fit:contain;
            pointer-events:none;
        }
    `;
    document.head.append(style);
}

function viewUrl(info) {
    if (!info?.filename) {
        return null;
    }
    const params = new URLSearchParams();
    params.set("filename", info.filename);
    params.set("type", info.type || "temp");
    params.set("subfolder", info.subfolder || "");
    return api.apiURL(`/view?${params.toString()}`);
}

function hideNativeImagePreview(node, liveRoot) {
    node.imgs = [];
    node.animatedImages = undefined;
    node.imageIndex = null;
    node.overIndex = null;
    const host = liveRoot.parentElement;
    if (!host) {
        return;
    }
    for (const child of host.children) {
        if (child === liveRoot || child.contains(liveRoot)) {
            continue;
        }
        if (child.querySelector?.("img, canvas, video")) {
            child.style.display = "none";
        }
    }
}

function setupLivePreview(node) {
    if (node.bokujuuAnimaLive) {
        return node.bokujuuAnimaLive;
    }
    addStyles();

    const root = document.createElement("div");
    root.className = "bokujuu-anima-live";
    const hero = document.createElement("div");
    hero.className = "bokujuu-anima-hero";
    const img = document.createElement("img");
    hero.append(img);
    root.append(hero);

    const state = {
        currentUrl: null,
        show(url) {
            if (!url || url === this.currentUrl) {
                return;
            }
            this.currentUrl = url;
            const loaded = new Image();
            loaded.onload = () => {
                if (this.currentUrl === url) {
                    img.src = url;
                }
            };
            loaded.src = url;
        },
        apply(output) {
            const images = output?.images || [];
            const latest = images.length ? images[images.length - 1] : null;
            this.show(viewUrl(latest));
            hideNativeImagePreview(node, root);
        },
    };

    node.addDOMWidget("anima_live_preview", "BOKUJUU_ANIMA_LIVE", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 360,
        getMaxHeight: () => 520,
    });

    node.onExecuted = function (output) {
        this.imgs = [];
        state.apply(output);
        this.imgs = [];
    };

    node.bokujuuAnimaLive = state;
    return state;
}

setInterval(() => {
    if (!loopGraphIsPresent()) {
        return;
    }
    hookAllClipNodes();
    const text = clipWidget((app.graph?._nodes || []).find((node) => node.comfyClass === "CLIPTextEncode" || node.type === "CLIPTextEncode"))?.value;
    pushPrompt(text ?? null);
}, 250);

app.registerExtension({
    name: "bokujuu.anima.stream",
    nodeCreated(node) {
        hookClipNode(node);
        if (node.comfyClass === LOOP_TYPE || node.type === LOOP_TYPE) {
            setupLivePreview(node);
        }
    },
});
