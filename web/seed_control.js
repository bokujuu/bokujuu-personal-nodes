import { app } from "../../scripts/app.js";

const NODE_TYPE = "BokujuuSeedControl";
const PROPERTY_NAME = "bokujuu_seed_overrides";
const STYLE_ID = "bokujuu-seed-control-style";
const MODES = ["inherit", "fixed", "randomize"];

const STRINGS = {
    en: {
        fixAll: "Fix All",
        randomizeAll: "Randomize All",
        clearAll: "Clear All",
        refresh: "Refresh seed nodes",
        detected: (count) => `${count} Seed${count === 1 ? "" : "s"} detected`,
        overridden: (count) => `${count} overridden`,
        linkedCount: (count) => `${count} excluded because the seed input is linked`,
        globalSeed: "EasyUse Global Seed was detected. If both controls change the same seed, the last setting applied takes precedence.",
        empty: "No controllable seeds found.\nAdd a node, then press ↻.",
        current: "Current",
        linked: "Excluded: linked input",
        graph: "Graph",
        graphs: "Graphs",
        mainWorkflow: "Main workflow",
        inherit: "Unchanged",
        fixed: "Fixed",
        random: "Random",
        increment: "Increment",
        decrement: "Decrement",
        navigate: "Go to node",
        footer: "Fixed keeps the current value. Random generates a new value on each run.",
    },
    ja: {
        fixAll: "すべて固定",
        randomizeAll: "すべてランダム",
        clearAll: "すべて解除",
        refresh: "対象ノードを再検出",
        detected: (count) => `${count} 件の Seed を検出`,
        overridden: (count) => `${count} 件を上書き`,
        linkedCount: (count) => `${count} 件はリンク接続のため対象外`,
        globalSeed: "EasyUse の Global Seed が見つかりました。両方で同じ対象を変更すると、後から動いた設定が優先されます。",
        empty: "制御可能な Seed が見つかりません。\nノードを追加して ↻ を押してください。",
        current: "現在",
        linked: "リンク接続のため対象外",
        graph: "グラフ",
        graphs: "グラフ",
        mainWorkflow: "メインワークフロー",
        inherit: "変更なし",
        fixed: "固定",
        random: "ランダム",
        increment: "増加",
        decrement: "減少",
        navigate: "対象ノードへ移動",
        footer: "固定は現在の数値を維持します。ランダムは実行ごとに新しい値へ更新します。",
    },
};

function getStrings() {
    const locale = app.extensionManager?.setting?.get?.("Comfy.Locale")
        ?? app.ui?.settings?.getSettingValue?.("Comfy.Locale")
        ?? "en";
    return String(locale).toLowerCase().startsWith("ja") ? STRINGS.ja : STRINGS.en;
}

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
        .bokujuu-seed-actions { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:6px; }
        .bokujuu-seed-actions button { width:100%; min-width:0; min-height:29px; padding:4px 7px; font-size:10px; font-weight:600; }
        .bokujuu-seed-actions .bokujuu-fixed { color:#b8dcff; border-color:#397cb8; background:#17334c; }
        .bokujuu-seed-actions .bokujuu-randomize { color:#f0d4ff; border-color:#8d55b5; background:#3b2350; }
        .bokujuu-seed-notice { padding:6px 8px; color:#ffe4a3; background:#4a371c; border:1px solid #8c642b; border-radius:7px; font-size:10px; line-height:1.35; }
        .bokujuu-seed-list { display:flex; flex:1; flex-direction:column; gap:6px; min-height:0; overflow:auto; padding-right:2px; }
        .bokujuu-seed-empty { display:flex; flex:1; align-items:center; justify-content:center; padding:24px; color:var(--descrip-text); text-align:center; font-size:11px; line-height:1.5; border:1px dashed var(--border-color); border-radius:9px; }
        .bokujuu-seed-row { display:grid; grid-template-columns:minmax(0,1fr) 214px 27px; gap:7px; align-items:center; padding:7px 7px 7px 9px; background:color-mix(in srgb,var(--comfy-input-bg) 82%,transparent); border:1px solid var(--border-color); border-radius:9px; }
        .bokujuu-seed-row[data-mode="fixed"] { border-left:4px solid #57a9ed; padding-left:6px; }
        .bokujuu-seed-row[data-mode="randomize"] { border-left:4px solid #bb78e5; padding-left:6px; }
        .bokujuu-seed-row[data-linked="true"] { opacity:.62; }
        .bokujuu-seed-node { min-width:0; }
        .bokujuu-seed-node > strong { display:block; overflow:hidden; font-size:11px; line-height:1.3; text-overflow:ellipsis; white-space:nowrap; }
        .bokujuu-seed-meta { display:flex; flex-direction:column; gap:1px; color:var(--descrip-text); font-size:9px; line-height:1.35; }
        .bokujuu-seed-meta span { display:block; overflow-wrap:anywhere; white-space:normal; }
        .bokujuu-seed-path { color:color-mix(in srgb,var(--descrip-text) 82%,var(--fg-color)); }
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

function getRootGraph(controller) {
    return app.rootGraph ?? controller.graph?.rootGraph ?? app.graph?.rootGraph ?? app.graph;
}

function getGraphNodes(graph) {
    return graph?.nodes ?? graph?._nodes ?? [];
}

function getSubgraphLabel(node) {
    return `${node.title || node.subgraph?.name || node.type} (#${node.id})`;
}

function findSeedTargets(controller) {
    const rootGraph = getRootGraph(controller);
    const targets = new Map();
    let hasGlobalSeed = false;

    function visit(graph, path, graphIds) {
        if (!graph || graphIds.has(graph)) return;
        const nextGraphIds = new Set(graphIds).add(graph);
        for (const node of getGraphNodes(graph)) {
            if (node.type === "easy globalSeed") hasGlobalSeed = true;
            if (node !== controller && node.type !== NODE_TYPE) {
                for (const [index, widget] of (node.widgets ?? []).entries()) {
                    if (!widget.name?.toLowerCase().includes("seed")) continue;
                    const control = findControlWidget(node, widget, index);
                    if (!control) continue;
                    const graphId = graph.id ?? graph._id ?? path.map((item) => item.node.id).join("/");
                    const key = graph === rootGraph
                        ? `${node.id}:${widget.name}`
                        : `${graphId}:${node.id}:${widget.name}`;
                    const pathKey = path.map((item) => item.node.id).join("/");
                    const existing = targets.get(key);
                    if (existing) {
                        if (!existing.pathKeys.has(pathKey)) {
                            existing.pathKeys.add(pathKey);
                            existing.paths.push(path);
                        }
                    } else {
                        targets.set(key, {
                            key,
                            node,
                            seed: widget,
                            control,
                            linked: isLinkedInput(node, widget.name),
                            paths: [path],
                            pathKeys: new Set([pathKey]),
                        });
                    }
                }
            }
            if (node.isSubgraphNode?.() && node.subgraph) {
                visit(node.subgraph, [...path, { node, label: getSubgraphLabel(node) }], nextGraphIds);
            }
        }
    }

    visit(rootGraph, [], new Set());
    const sorted = [...targets.values()].sort((left, right) => {
        const leftPath = left.paths[0].map((item) => item.label).join("/");
        const rightPath = right.paths[0].map((item) => item.label).join("/");
        return leftPath.localeCompare(rightPath) || Number(left.node.id) - Number(right.node.id);
    });
    return { targets: sorted, hasGlobalSeed, rootGraph };
}

function navigateToTarget(target, rootGraph) {
    const canvas = app.canvas;
    const targetGraph = target.node.graph ?? rootGraph;
    if (canvas?.getCurrentGraph?.() !== targetGraph) {
        const path = target.paths[0] ?? [];
        canvas?.openSubgraph?.(targetGraph, path.at(-1)?.node ?? null);
    }
    requestAnimationFrame(() => {
        canvas?.selectNode?.(target.node);
        canvas?.centerOnNode?.(target.node);
        canvas?.setDirty?.(true, true);
    });
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
    toolbar.append(summary, refreshButton);

    const actions = document.createElement("div");
    actions.className = "bokujuu-seed-actions";
    const fixAll = createButton("", "bokujuu-fixed");
    const randomizeAll = createButton("", "bokujuu-randomize");
    const clearAll = createButton("");
    actions.append(fixAll, randomizeAll, clearAll);

    const notice = document.createElement("div");
    notice.className = "bokujuu-seed-notice";
    notice.hidden = true;
    const list = document.createElement("div");
    list.className = "bokujuu-seed-list";
    const footer = document.createElement("div");
    footer.className = "bokujuu-seed-footer";
    root.append(toolbar, actions, notice, list, footer);

    const state = {
        targets: new Map(),
        rootGraph: null,
        render() {
            const strings = getStrings();
            const result = findSeedTargets(node);
            const { targets, hasGlobalSeed, rootGraph } = result;
            this.targets = new Map(targets.map((target) => [target.key, target]));
            this.rootGraph = rootGraph;
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

            notice.hidden = !hasGlobalSeed;
            notice.textContent = strings.globalSeed;
            refreshButton.title = strings.refresh;
            fixAll.textContent = strings.fixAll;
            randomizeAll.textContent = strings.randomizeAll;
            clearAll.textContent = strings.clearAll;
            footer.textContent = strings.footer;
            summary.replaceChildren();
            const title = document.createElement("strong");
            title.textContent = strings.detected(targets.length);
            const detail = document.createElement("span");
            detail.textContent = `${strings.overridden(activeCount)}${linkedCount ? ` · ${strings.linkedCount(linkedCount)}` : ""}`;
            summary.append(title, detail);

            list.replaceChildren();
            if (!targets.length) {
                const empty = document.createElement("div");
                empty.className = "bokujuu-seed-empty";
                empty.textContent = strings.empty;
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
                meta.className = "bokujuu-seed-meta";
                const seedStatus = document.createElement("span");
                const modeLabels = {
                    fixed: strings.fixed,
                    increment: strings.increment,
                    decrement: strings.decrement,
                    randomize: strings.random,
                };
                const currentMode = modeLabels[target.control.value] ?? String(target.control.value ?? "-");
                seedStatus.textContent = target.linked
                    ? `${target.seed.name} · ${strings.linked}`
                    : `${target.seed.name}: ${target.seed.value} · ${strings.current}: ${currentMode}`;
                meta.appendChild(seedStatus);
                for (const [index, path] of target.paths.entries()) {
                    const graphPath = document.createElement("span");
                    graphPath.className = "bokujuu-seed-path";
                    const label = target.paths.length > 1 ? strings.graphs : strings.graph;
                    const hierarchy = [strings.mainWorkflow, ...path.map((item) => item.label)].join(" › ");
                    graphPath.textContent = index === 0 ? `${label}: ${hierarchy}` : `${strings.graphs}: ${hierarchy}`;
                    meta.appendChild(graphPath);
                }
                nodeInfo.append(nodeTitle, meta);

                const modes = document.createElement("div");
                modes.className = "bokujuu-seed-modes";
                for (const [value, label] of [["inherit", strings.inherit], ["fixed", strings.fixed], ["randomize", strings.random]]) {
                    const button = createButton(label);
                    button.dataset.mode = value;
                    button.classList.toggle("active", mode === value);
                    button.disabled = target.linked;
                    modes.appendChild(button);
                }
                const nav = createButton("↗", "bokujuu-seed-nav");
                nav.title = strings.navigate;
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
            navigateToTarget(target, state.rootGraph);
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
        const rootGraph = app.rootGraph ?? app.graph?.rootGraph ?? app.graph;
        const visited = new Set();
        function setupControllers(graph) {
            if (!graph || visited.has(graph)) return;
            visited.add(graph);
            for (const node of getGraphNodes(graph)) {
                if (node.type === NODE_TYPE) {
                    setupSeedControl(node);
                    node.bokujuuSeedControl?.render();
                }
                if (node.isSubgraphNode?.() && node.subgraph) setupControllers(node.subgraph);
            }
        }
        setupControllers(rootGraph);
    },
});
