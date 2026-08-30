import { app } from "../../scripts/app.js";

const NODE_TYPE = "BokujuuSeedControl";
const PROPERTY_NAME = "bokujuu_seed_overrides";
const STYLE_ID = "bokujuu-seed-control-style";
const MODES = ["inherit", "fixed", "randomize"];

const MODE_LABELS = {
    fixed: "固定",
    increment: "増加",
    decrement: "減少",
    randomize: "ランダム",
};

function addStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        .bokujuu-seed-control { display:flex; flex-direction:column; gap:10px; box-sizing:border-box; height:100%; padding:8px 10px 10px; color:var(--fg-color); font-family:Inter,system-ui,sans-serif; }
        .bokujuu-seed-control * { box-sizing:border-box; }
        .bokujuu-seed-control button { color:var(--input-text); background:var(--comfy-input-bg); border:1px solid var(--border-color); border-radius:7px; cursor:pointer; }
        .bokujuu-seed-control button:hover:not(:disabled) { border-color:var(--primary-color); filter:brightness(1.12); }
        .bokujuu-seed-control button:disabled { cursor:not-allowed; opacity:.45; }
        .bokujuu-seed-toolbar { display:grid; grid-template-columns:1fr auto; gap:8px; align-items:center; }
        .bokujuu-seed-summary { min-width:0; }
        .bokujuu-seed-summary strong { display:block; font-size:13px; line-height:1.2; }
        .bokujuu-seed-summary span { display:block; overflow:hidden; color:var(--descrip-text); font-size:10px; line-height:1.35; text-overflow:ellipsis; white-space:nowrap; }
        .bokujuu-seed-refresh { width:31px; height:31px; font-size:16px; }
        .bokujuu-seed-actions { display:grid; grid-template-columns:1fr 1.2fr 1fr; gap:6px; }
        .bokujuu-seed-actions button { min-height:29px; padding:4px 7px; font-size:10px; font-weight:600; }
        .bokujuu-seed-actions .fixed { color:#b8dcff; border-color:#397cb8; background:#17334c; }
        .bokujuu-seed-actions .randomize { color:#f0d4ff; border-color:#8d55b5; background:#3b2350; }
        .bokujuu-seed-notice { padding:6px 8px; color:#ffe4a3; background:#4a371c; border:1px solid #8c642b; border-radius:7px; font-size:10px; line-height:1.35; }
        .bokujuu-seed-list { display:flex; flex:1; flex-direction:column; gap:6px; min-height:0; overflow:auto; padding-right:2px; }
        .bokujuu-seed-empty { display:flex; flex:1; align-items:center; justify-content:center; padding:24px; color:var(--descrip-text); text-align:center; font-size:11px; line-height:1.5; border:1px dashed var(--border-color); border-radius:9px; }
        .bokujuu-seed-row { display:grid; grid-template-columns:minmax(0,1fr) 214px 27px; gap:7px; align-items:center; padding:7px 7px 7px 9px; background:color-mix(in srgb,var(--comfy-input-bg) 82%,transparent); border:1px solid var(--border-color); border-radius:9px; }
        .bokujuu-seed-row[data-mode="fixed"] { border-left:4px solid #57a9ed; padding-left:6px; }
        .bokujuu-seed-row[data-mode="randomize"] { border-left:4px solid #bb78e5; padding-left:6px; }
        .bokujuu-seed-row[data-linked="true"] { opacity:.62; }
        .bokujuu-seed-node { min-width:0; }
        .bokujuu-seed-node strong { display:block; overflow:hidden; font-size:11px; line-height:1.3; text-overflow:ellipsis; white-space:nowrap; }
        .bokujuu-seed-node span { display:block; overflow:hidden; color:var(--descrip-text); font-size:9px; line-height:1.35; text-overflow:ellipsis; white-space:nowrap; }
        .bokujuu-seed-modes { display:grid; grid-template-columns:1.2fr .72fr 1fr; overflow:hidden; border:1px solid var(--border-color); border-radius:7px; }
        .bokujuu-seed-modes button { min-height:27px; padding:3px 4px; font-size:9px; border:0; border-right:1px solid var(--border-color); border-radius:0; }
        .bokujuu-seed-modes button:last-child { border-right:0; }
        .bokujuu-seed-modes button.active[data-mode="inherit"] { color:#fff; background:#59616b; }
        .bokujuu-seed-modes button.active[data-mode="fixed"] { color:#fff; background:#2778b5; }
        .bokujuu-seed-modes button.active[data-mode="randomize"] { color:#fff; background:#8b4bb3; }
        .bokujuu-seed-nav { width:27px; height:27px; padding:0; color:#a9cbe8 !important; font-size:13px; }
        .bokujuu-seed-footer { color:var(--descrip-text); font-size:9px; line-height:1.35; text-align:center; }
    `;
    document.head.appendChild(style);
}

function isControlWidget(widget) {
    if (!widget) return false;
    if (widget.name === "control_after_generate") return true;
    const values = widget.options?.values;
    return Array.isArray(values) && values.includes("fixed") && values.includes("randomize");
}

function findControlWidget(node, seedWidget, index) {
    for (const widget of seedWidget.linkedWidgets ?? []) {
        if (isControlWidget(widget)) return widget;
    }
    const nextWidget = node.widgets?.[index + 1];
    return isControlWidget(nextWidget) ? nextWidget : null;
}

function isLinkedInput(node, name) {
    return (node.inputs ?? []).some((input) => input.name === name && input.link != null);
}

function findSeedTargets(controller) {
    const graph = app.canvas?.getCurrentGraph?.() ?? controller.graph ?? app.graph;
    const targets = [];
    for (const node of graph?._nodes ?? []) {
        if (node === controller || node.type === NODE_TYPE) continue;
        for (const [index, widget] of (node.widgets ?? []).entries()) {
            if (!widget.name?.toLowerCase().includes("seed")) continue;
            const control = findControlWidget(node, widget, index);
            if (!control) continue;
            targets.push({
                key: `${node.id}:${widget.name}`,
                node,
                seed: widget,
                control,
                linked: isLinkedInput(node, widget.name),
            });
        }
    }
    targets.sort((left, right) => Number(left.node.id) - Number(right.node.id));
    return targets;
}

function setControlValue(target, value) {
    target.control.value = value;
    target.control.callback?.(value, app.canvas, target.node, [0, 0], null);
    target.node.graph?.change?.();
    target.node.setDirtyCanvas?.(true, true);
}

function createButton(label, className = "") {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.className = className;
    return button;
}

function setupSeedControl(node) {
    if (node.bokujuuSeedControl) return;
    addStyles();
    node.properties ??= {};
    node.properties[PROPERTY_NAME] ??= {};

    const root = document.createElement("div");
    root.className = "bokujuu-seed-control";
    root.addEventListener("pointerdown", (event) => event.stopPropagation());
    root.addEventListener("mousedown", (event) => event.stopPropagation());
    root.addEventListener("wheel", (event) => event.stopPropagation());

    const toolbar = document.createElement("div");
    toolbar.className = "bokujuu-seed-toolbar";
    const summary = document.createElement("div");
    summary.className = "bokujuu-seed-summary";
    const refreshButton = createButton("↻", "bokujuu-seed-refresh");
    refreshButton.title = "対象ノードを再検出";
    toolbar.append(summary, refreshButton);

    const actions = document.createElement("div");
    actions.className = "bokujuu-seed-actions";
    const fixAll = createButton("すべて固定", "fixed");
    const randomizeAll = createButton("すべてランダム", "randomize");
    const clearAll = createButton("すべて解除");
    actions.append(fixAll, randomizeAll, clearAll);

    const notice = document.createElement("div");
    notice.className = "bokujuu-seed-notice";
    notice.hidden = true;
    const list = document.createElement("div");
    list.className = "bokujuu-seed-list";
    const footer = document.createElement("div");
    footer.className = "bokujuu-seed-footer";
    footer.textContent = "固定は現在の数値を維持します。ランダムは実行ごとに新しい値へ更新します。";
    root.append(toolbar, actions, notice, list, footer);

    const state = {
        targets: new Map(),
        render() {
            const targets = findSeedTargets(node);
            this.targets = new Map(targets.map((target) => [target.key, target]));
            const overrides = node.properties[PROPERTY_NAME] ?? {};
            let activeCount = 0;
            let linkedCount = 0;
            for (const target of targets) {
                const entry = overrides[target.key];
                if (entry && MODES.includes(entry.mode) && entry.mode !== "inherit" && !target.linked) {
                    setControlValue(target, entry.mode);
                    activeCount++;
                }
                if (target.linked) linkedCount++;
            }

            const graph = app.canvas?.getCurrentGraph?.() ?? node.graph ?? app.graph;
            const hasGlobalSeed = (graph?._nodes ?? []).some((candidate) => candidate.type === "easy globalSeed");
            notice.hidden = !hasGlobalSeed;
            notice.textContent = "EasyUse の Global Seed が見つかりました。両方で同じ対象を変更すると、後から動いた設定が優先されます。";
            summary.replaceChildren();
            const title = document.createElement("strong");
            title.textContent = `${targets.length} 件の Seed を検出`;
            const detail = document.createElement("span");
            detail.textContent = `${activeCount} 件を上書き${linkedCount ? ` · ${linkedCount} 件はリンク接続のため対象外` : ""}`;
            summary.append(title, detail);

            list.replaceChildren();
            if (!targets.length) {
                const empty = document.createElement("div");
                empty.className = "bokujuu-seed-empty";
                empty.textContent = "制御可能な Seed が見つかりません。\nノードを追加して ↻ を押してください。";
                list.appendChild(empty);
                return;
            }

            for (const target of targets) {
                const entry = overrides[target.key];
                const mode = entry?.mode ?? "inherit";
                const row = document.createElement("div");
                row.className = "bokujuu-seed-row";
                row.dataset.key = target.key;
                row.dataset.mode = mode;
                row.dataset.linked = String(target.linked);

                const nodeInfo = document.createElement("div");
                nodeInfo.className = "bokujuu-seed-node";
                const nodeTitle = document.createElement("strong");
                nodeTitle.textContent = `#${target.node.id}  ${target.node.title || target.node.type}`;
                const meta = document.createElement("span");
                const currentMode = MODE_LABELS[target.control.value] ?? String(target.control.value ?? "-");
                meta.textContent = target.linked
                    ? `${target.seed.name} · リンク接続のため対象外`
                    : `${target.seed.name}: ${target.seed.value} · 現在 ${currentMode}`;
                nodeInfo.append(nodeTitle, meta);

                const modes = document.createElement("div");
                modes.className = "bokujuu-seed-modes";
                for (const [value, label] of [["inherit", "変更なし"], ["fixed", "固定"], ["randomize", "ランダム"]]) {
                    const button = createButton(label);
                    button.dataset.mode = value;
                    button.classList.toggle("active", mode === value);
                    button.disabled = target.linked;
                    modes.appendChild(button);
                }
                const nav = createButton("↗", "bokujuu-seed-nav");
                nav.title = "対象ノードへ移動";
                nav.dataset.action = "navigate";
                row.append(nodeInfo, modes, nav);
                list.appendChild(row);
            }
            node.setDirtyCanvas?.(true, true);
        },
        setMode(key, mode) {
            const target = this.targets.get(key);
            if (!target || target.linked || !MODES.includes(mode)) return;
            const overrides = node.properties[PROPERTY_NAME] ??= {};
            const current = overrides[key];
            if (mode === "inherit") {
                if (current) setControlValue(target, current.previousMode ?? "fixed");
                delete overrides[key];
            } else {
                overrides[key] = {
                    mode,
                    previousMode: current?.previousMode ?? target.control.value,
                };
                setControlValue(target, mode);
            }
            node.graph?.change?.();
            this.render();
        },
        setAll(mode) {
            const overrides = node.properties[PROPERTY_NAME] ??= {};
            for (const [key, target] of this.targets) {
                if (target.linked) continue;
                const current = overrides[key];
                if (mode === "inherit") {
                    if (current) setControlValue(target, current.previousMode ?? "fixed");
                    delete overrides[key];
                } else {
                    overrides[key] = {
                        mode,
                        previousMode: current?.previousMode ?? target.control.value,
                    };
                    setControlValue(target, mode);
                }
            }
            node.graph?.change?.();
            this.render();
        },
    };

    refreshButton.addEventListener("click", () => state.render());
    fixAll.addEventListener("click", () => state.setAll("fixed"));
    randomizeAll.addEventListener("click", () => state.setAll("randomize"));
    clearAll.addEventListener("click", () => state.setAll("inherit"));
    list.addEventListener("click", (event) => {
        const button = event.target.closest("button");
        const row = event.target.closest(".bokujuu-seed-row");
        if (!button || !row) return;
        const target = state.targets.get(row.dataset.key);
        if (button.dataset.action === "navigate" && target) {
            app.canvas?.selectNode?.(target.node);
            app.canvas?.centerOnNode?.(target.node);
            app.canvas?.setDirty?.(true, true);
        } else if (button.dataset.mode) {
            state.setMode(row.dataset.key, button.dataset.mode);
        }
    });

    node.addDOMWidget("seed_control_panel", "BOKUJUU_SEED_CONTROL", root, {
        serialize: false,
        getMinHeight: () => 290,
        getMaxHeight: () => 440,
    });
    node.setSize([620, 360]);
    node.bokujuuSeedControl = state;
    setTimeout(() => state.render(), 0);
}

app.registerExtension({
    name: "Bokujuu.SeedControl",
    nodeCreated(node) {
        if (node.comfyClass === NODE_TYPE || node.type === NODE_TYPE) setupSeedControl(node);
    },
    afterConfigureGraph() {
        for (const node of app.graph?._nodes ?? []) {
            if (node.type === NODE_TYPE) {
                setupSeedControl(node);
                node.bokujuuSeedControl?.render();
            }
        }
    },
});
