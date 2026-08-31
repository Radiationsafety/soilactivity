"""make_model_macro.py - генерация macros/model_cells.mac из model.json.

model.json описывает модельный источник (истинное распределение активности):
{
  "cells": [
    {"ix": 1, "iy": 1, "activity_Bq": 1.0e8},
    {"ix": 2, "iy": 2, "activity_Bq": 2.0e8},
    {"ix": 3, "iy": 2, "activity_Bq": 5.0e7}
  ],
  "grid": {"nx": 5, "ny": 5, "cell_size_m": 2.0, "src_depth_m": 0.10}
}

Запуск:
  python make_model_macro.py model.json ../macros/model_cells.mac
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="model.json -> model_cells.mac")
    ap.add_argument("model", help="путь к model.json")
    ap.add_argument("out", help="путь к выходному model_cells.mac")
    args = ap.parse_args()

    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    grid = model.get("grid", {"nx": 5, "ny": 5, "cell_size_m": 2.0,
                              "src_depth_m": 0.10})
    nx, ny = int(grid["nx"]), int(grid["ny"])
    cs, depth = float(grid["cell_size_m"]), float(grid["src_depth_m"])

    lines = [
        "# Ячейки модельного источника - сгенерировано make_model_macro.py",
        "# из model.json. НЕ редактировать вручную.",
    ]
    for c in model["cells"]:
        ix, iy = int(c["ix"]), int(c["iy"])
        if not (0 <= ix < nx and 0 <= iy < ny):
            print(f"ОШИБКА: ячейка ({ix},{iy}) вне сетки {nx}x{ny}", file=sys.stderr)
            return 1
        x = (ix - 0.5 * (nx - 1)) * cs
        y = (iy - 0.5 * (ny - 1)) * cs
        a = float(c["activity_Bq"])
        lines.append(f"/b1soil/addModelCell {x:g} {y:g} {-depth:g} m {a:g} Bq"
                     f"   # ячейка ({ix},{iy})")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Записан {args.out}: {len(model['cells'])} ячеек")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
