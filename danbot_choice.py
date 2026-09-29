import random

from comfy_api.latest import io, ui


# These lists are the link types, not just labels. They must match Danbot's
# Template Config combos exactly, or ComfyUI rejects the connection.
# Rating includes "auto" only so the socket matches; this node never emits it.
RATING_CHOICES = ("general", "sensitive", "questionable", "explicit")
LENGTH_CHOICES = ("very_short", "short", "long", "very_long")
RATING_LINK_TYPE = ["auto", *RATING_CHOICES]
LENGTH_LINK_TYPE = list(LENGTH_CHOICES)

RATING_LABELS = {
    "general": "General",
    "sensitive": "Sensitive",
    "questionable": "Questionable",
    "explicit": "Explicit",
}
LENGTH_LABELS = {
    "very_short": "Very Short",
    "short": "Short",
    "long": "Long",
    "very_long": "Very Long",
}


class _DanbotComboOutput(io.Combo.Output):
    def __init__(self, link_type, display_name, tooltip):
        super().__init__(
            id=display_name,
            display_name=display_name,
            options=list(link_type),
            tooltip=tooltip,
        )
        self._link_type = list(link_type)

    @property
    def io_type(self):
        return self._link_type


def _as_enabled(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_rate(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _draw(rng, mode, choices, enabled, weights, label):
    if mode != "auto":
        if mode not in choices:
            raise ValueError(f"{label} の固定値 '{mode}' は候補にありません。")
        return mode

    names = []
    rates = []
    for name in choices:
        if not enabled.get(name, False):
            continue
        rate = _as_rate(weights.get(name, 0))
        if rate <= 0:
            continue
        names.append(name)
        rates.append(rate)
    if not names:
        raise ValueError(
            f"{label} の抽選候補がありません。チェックを1つ以上オンにし、選択率を0より大きくしてください。"
        )
    return rng.choices(names, weights=rates, k=1)[0]


def choose_danbot_values(seed, rating_mode, length_mode, enabled, weights):
    rng = random.Random(int(seed))
    rating = _draw(rng, rating_mode, RATING_CHOICES, enabled, weights, "Rating")
    length = _draw(rng, length_mode, LENGTH_CHOICES, enabled, weights, "Length")
    summary = f"rating: {rating}\nlength: {length}"
    return rating, length, summary


def _choice_controls(prefix, choices, labels, mode_tooltip):
    controls = [
        io.Combo.Input(
            f"{prefix}_mode",
            options=["auto", *choices],
            display_name=prefix.capitalize(),
            default="auto",
            tooltip=mode_tooltip,
            socketless=True,
        )
    ]
    for name in choices:
        label = labels[name]
        controls.append(
            io.Boolean.Input(
                f"{prefix}_{name}",
                display_name=label,
                default=True,
                socketless=True,
                tooltip="オンの候補だけが auto の抽選対象になります。",
            )
        )
        controls.append(
            io.Int.Input(
                f"{prefix}_{name}_rate",
                display_name=f"{label} 選択率",
                default=25,
                min=0,
                max=100,
                step=1,
                display_mode=io.NumberDisplay.slider,
                socketless=True,
                tooltip="相対的な選択率です。合計を100にする必要はありません。0は抽選から外れます。",
            )
        )
    return controls


class BokujuuDanbotRandomChoice(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BokujuuDanbotRandomChoice",
            display_name="Bokujuu Danbot Random Choice",
            category="Bokujuu/Danbot",
            search_aliases=["rating", "length", "danbot", "ランダム", "auto"],
            description=(
                "Danbot の Rating と Length を抽選します。"
                "モード auto は、オンにした候補を選択率の比で選びます。"
                "出力は Template Config のコンボへ接続できる具体的なキーで、auto という値は出しません。"
            ),
            inputs=[
                io.Int.Input(
                    "seed",
                    display_name="シード",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="同じシードなら同じ Rating と Length になります。",
                ),
                *_choice_controls(
                    "rating",
                    RATING_CHOICES,
                    RATING_LABELS,
                    "auto は下のチェックと選択率で抽選します。個別の値を選ぶとその値で固定します。",
                ),
                *_choice_controls(
                    "length",
                    LENGTH_CHOICES,
                    LENGTH_LABELS,
                    "auto は下のチェックと選択率で抽選します。個別の値を選ぶとその値で固定します。",
                ),
            ],
            outputs=[
                _DanbotComboOutput(
                    RATING_LINK_TYPE,
                    "rating",
                    "Danbot Template Config の rating に接続します。",
                ),
                _DanbotComboOutput(
                    LENGTH_LINK_TYPE,
                    "length",
                    "Danbot Template Config の length に接続します。",
                ),
                io.String.Output(
                    id="choice",
                    display_name="choice",
                    tooltip="選ばれた rating と length です。",
                ),
            ],
        )

    @classmethod
    def execute(cls, seed, rating_mode, length_mode, **flags):
        enabled = {}
        weights = {}
        for name in (*RATING_CHOICES, *LENGTH_CHOICES):
            prefix = "rating" if name in RATING_CHOICES else "length"
            enabled[name] = _as_enabled(flags[f"{prefix}_{name}"])
            weights[name] = _as_rate(flags[f"{prefix}_{name}_rate"])
        rating, length, summary = choose_danbot_values(
            seed,
            rating_mode,
            length_mode,
            enabled,
            weights,
        )
        return io.NodeOutput(rating, length, summary, ui=ui.PreviewText(summary))
