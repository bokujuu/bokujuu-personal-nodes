from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def emit_html(*, identical: bool, left_name: str, right_name: str) -> str:
    match_text = (
        "この2枚は画素が一致しています。境を動かしても絵は変わりません。"
        if identical
        else "境の左右で画素が異なります。スライダーを動かして差を見てください。"
    )
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DLSS Ultra Performance vs Quality（CAS 0.7）</title>
  <style>
    :root {{ color-scheme: light dark; }}
    body {{ font-family: system-ui, sans-serif; margin: 1rem 1.25rem; line-height: 1.45; }}
    h1 {{ font-size: 1.25rem; margin: 0 0 0.4rem; }}
    .hint {{ opacity: 0.85; margin: 0 0 1rem; }}
    .stage {{
      position: relative;
      max-width: min(100%, 1100px);
      user-select: none;
      cursor: ew-resize;
    }}
    .stage img {{
      display: block;
      width: 100%;
      height: auto;
      pointer-events: none;
    }}
    .left {{
      position: absolute;
      inset: 0;
      clip-path: inset(0 calc(100% - var(--pos, 50%)) 0 0);
    }}
    .handle {{
      position: absolute;
      top: 0;
      bottom: 0;
      left: var(--pos, 50%);
      width: 2px;
      margin-left: -1px;
      background: CanvasText;
      pointer-events: none;
    }}
    .tag {{
      position: absolute;
      top: 0.6rem;
      padding: 0.15rem 0.45rem;
      font-size: 0.8rem;
      background: Canvas;
      color: CanvasText;
      pointer-events: none;
    }}
    .tag.left-tag {{ left: 0.6rem; }}
    .tag.right-tag {{ right: 0.6rem; }}
    input[type="range"] {{
      display: block;
      width: min(100%, 1100px);
      margin: 0.75rem 0 0;
    }}
  </style>
</head>
<body>
  <h1>Ultra Performance（左）と Quality（右）</h1>
  <p class="hint">2倍、CAS 0.7。スライダーか画像上のドラッグで境を動かします。{match_text}</p>
  <div class="stage" id="stage" style="--pos: 50%">
    <img src="{right_name}" alt="Quality, CAS 0.7" />
    <img class="left" src="{left_name}" alt="Ultra Performance, CAS 0.7" />
    <div class="handle"></div>
    <span class="tag left-tag">Ultra Performance</span>
    <span class="tag right-tag">Quality</span>
  </div>
  <input id="split" type="range" min="0" max="100" value="50" />
  <script>
    const stage = document.getElementById("stage");
    const split = document.getElementById("split");
    function setPos(percent) {{
      const value = Math.min(100, Math.max(0, percent));
      stage.style.setProperty("--pos", value + "%");
      split.value = String(Math.round(value));
    }}
    function posFromEvent(event) {{
      const box = stage.getBoundingClientRect();
      return ((event.clientX - box.left) / box.width) * 100;
    }}
    split.addEventListener("input", () => setPos(Number(split.value)));
    stage.addEventListener("pointerdown", (event) => {{
      stage.setPointerCapture(event.pointerId);
      setPos(posFromEvent(event));
    }});
    stage.addEventListener("pointermove", (event) => {{
      if (event.buttons) setPos(posFromEvent(event));
    }});
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emit a split-slider HTML compare for two DLSS quality PNGs."
    )
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    left_name = "ultra_performance_cas07.png"
    right_name = "quality_cas07.png"
    shutil.copyfile(args.left, args.output_dir / left_name)
    shutil.copyfile(args.right, args.output_dir / right_name)
    identical = file_digest(args.left) == file_digest(args.right)
    html_path = args.output_dir / "compare.html"
    html_path.write_text(
        emit_html(identical=identical, left_name=left_name, right_name=right_name),
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Ultra Performance vs Quality（CAS 0.7）",
                "",
                "ブラウザで境スライダーを動かし、2倍 DLSS の Ultra Performance と Quality を重ねて確認します。",
                "",
                "## 見る場所",
                "",
                f"- [{html_path.name}]({html_path.name})",
                f"- `{left_name}` / `{right_name}`",
                "",
                "## 再生成",
                "",
                "```powershell",
                "python scripts/emit_dlss5_split_compare.py --left left.png --right right.png --output-dir compare",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(html_path)
    print("identical", identical)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
