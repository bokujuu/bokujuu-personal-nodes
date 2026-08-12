import { app } from "../../scripts/app.js";

const WIDGET_TYPE = "BOKUJUU_LORA_SELECTOR";
const STYLE_ID = "bokujuu-lora-selector-style";

function parseValue(value) {
    if (Array.isArray(value)) return value.filter(Boolean);
    if (!value) return [];
    try {
        const parsed = JSON.parse(value);
        return Array.isArray(parsed) ? parsed.filter(Boolean) : [];
    } catch {
        return [];
    }
}

function addStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        .bokujuu-lora-selector { position:relative; z-index:10; display:flex; flex-direction:column; gap:6px; padding:4px 10px; color:var(--fg-color); }
        .bokujuu-lora-selector button { min-height:30px; color:var(--input-text); background:var(--comfy-input-bg); border:1px solid var(--border-color); border-radius:8px; cursor:pointer; }
        .bokujuu-lora-selector-summary { overflow:hidden; color:var(--descrip-text); font-size:11px; line-height:1.3; text-overflow:ellipsis; white-space:nowrap; }
        .bokujuu-lora-dialog { position:fixed; inset:0; z-index:100000; display:flex; align-items:center; justify-content:center; background:#0009; }
        .bokujuu-lora-dialog-panel { display:flex; flex-direction:column; width:min(760px,90vw); height:min(720px,85vh); overflow:hidden; color:var(--fg-color); background:var(--comfy-menu-bg); border:1px solid var(--border-color); border-radius:12px; box-shadow:0 20px 60px #000a; }
        .bokujuu-lora-dialog-header, .bokujuu-lora-dialog-footer { display:flex; align-items:center; gap:8px; padding:12px; }
        .bokujuu-lora-dialog-header { flex-direction:column; align-items:stretch; border-bottom:1px solid var(--border-color); }
        .bokujuu-lora-dialog-title { display:flex; justify-content:space-between; font-weight:600; }
        .bokujuu-lora-search { padding:9px 11px; color:var(--input-text); background:var(--comfy-input-bg); border:1px solid var(--border-color); border-radius:8px; outline:none; }
        .bokujuu-lora-list { flex:1; overflow:auto; padding:6px 12px; }
        .bokujuu-lora-row { display:flex; align-items:center; gap:9px; padding:7px 4px; border-bottom:1px solid color-mix(in srgb,var(--border-color) 50%,transparent); cursor:pointer; }
        .bokujuu-lora-row span { overflow-wrap:anywhere; }
        .bokujuu-lora-dialog-footer { justify-content:flex-end; border-top:1px solid var(--border-color); }
        .bokujuu-lora-dialog-footer button { padding:7px 14px; color:var(--input-text); background:var(--comfy-input-bg); border:1px solid var(--border-color); border-radius:7px; cursor:pointer; }
        .bokujuu-lora-dialog-footer .primary { background:var(--primary-color); color:var(--primary-contrast-color,#fff); }
    `;
    document.head.appendChild(style);
}

function makeButton(label, action, className = "") {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.className = className;
    button.addEventListener("click", action);
    return button;
}

function createSelector(node, inputName, inputData) {
    addStyles();
    const options = inputData?.[1]?.options ?? [];
    let selected = parseValue(inputData?.[1]?.default);

    const root = document.createElement("div");
    root.className = "bokujuu-lora-selector";
    root.addEventListener("pointerdown", (event) => event.stopPropagation());
    root.addEventListener("mousedown", (event) => event.stopPropagation());
    root.addEventListener("wheel", (event) => event.stopPropagation());
    const openButton = document.createElement("button");
    openButton.type = "button";
    const summary = document.createElement("div");
    summary.className = "bokujuu-lora-selector-summary";
    root.append(openButton, summary);

    const refresh = () => {
        openButton.textContent = `Select LoRAs (${selected.length})`;
        summary.textContent = selected.length ? selected.join(" · ") : "No LoRAs selected";
        root.title = selected.join("\n");
    };

    const openDialog = () => {
        const draft = new Set(selected);
        const overlay = document.createElement("div");
        overlay.className = "bokujuu-lora-dialog";
        const panel = document.createElement("div");
        panel.className = "bokujuu-lora-dialog-panel";
        const header = document.createElement("div");
        header.className = "bokujuu-lora-dialog-header";
        const title = document.createElement("div");
        title.className = "bokujuu-lora-dialog-title";
        const titleText = document.createElement("span");
        titleText.textContent = "Select LoRAs";
        const count = document.createElement("span");
        const search = document.createElement("input");
        search.className = "bokujuu-lora-search";
        search.type = "search";
        search.placeholder = "Search LoRAs...";
        const list = document.createElement("div");
        list.className = "bokujuu-lora-list";
        const footer = document.createElement("div");
        footer.className = "bokujuu-lora-dialog-footer";

        const updateCount = () => count.textContent = `${draft.size} selected`;
        const render = () => {
            const query = search.value.trim().toLowerCase();
            const visible = options.filter((name) => !query || name.toLowerCase().includes(query));
            list.replaceChildren();
            for (const name of visible) {
                const row = document.createElement("label");
                row.className = "bokujuu-lora-row";
                const checkbox = document.createElement("input");
                checkbox.type = "checkbox";
                checkbox.checked = draft.has(name);
                checkbox.addEventListener("change", () => {
                    checkbox.checked ? draft.add(name) : draft.delete(name);
                    updateCount();
                });
                const text = document.createElement("span");
                text.textContent = name;
                row.append(checkbox, text);
                list.appendChild(row);
            }
        };
        const close = () => overlay.remove();

        title.append(titleText, count);
        header.append(title, search);
        footer.append(
            makeButton("Clear", () => { draft.clear(); updateCount(); render(); }),
            makeButton("Cancel", close),
            makeButton("Apply", () => {
                selected = options.filter((name) => draft.has(name));
                refresh();
                node.graph?.change();
                node.setDirtyCanvas(true, true);
                close();
            }, "primary"),
        );
        panel.append(header, list, footer);
        overlay.appendChild(panel);
        overlay.addEventListener("mousedown", (event) => { if (event.target === overlay) close(); });
        search.addEventListener("input", render);
        document.body.appendChild(overlay);
        updateCount();
        render();
        search.focus();
    };

    openButton.onpointerdown = (event) => {
        event.stopPropagation();
        openDialog();
    };
    refresh();

    const widget = node.addDOMWidget(inputName, WIDGET_TYPE, root, {
        socketless: true,
        getMinHeight: () => 64,
        getMaxHeight: () => 64,
        getValue: () => JSON.stringify(selected),
        setValue: (value) => {
            selected = parseValue(value);
            refresh();
        },
    });
    return { widget };
}

app.registerExtension({
    name: "Bokujuu.LoraSelector",
    getCustomWidgets: () => ({
        [WIDGET_TYPE]: createSelector,
    }),
});
