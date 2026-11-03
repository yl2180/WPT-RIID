#!/usr/bin/env python3
"""验证 Python 依赖及 STEP 往返；几何仅为环境测试对象，不是 BISC。"""

import importlib
from importlib.metadata import version
import math
import os
from pathlib import Path
import platform
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "env_check"
EXPECTED_MM = (10.0, 10.0, 1.0)
ABS_TOL_MM = 1e-6


def main() -> int:
    print("WPT-RIID 环境检查：测试对象，不是 BISC 接收线圈", flush=True)
    print(f"操作系统：{platform.platform()}", flush=True)
    print(f"CPU 架构：{platform.machine()}", flush=True)
    print(f"Python：{platform.python_version()}", flush=True)
    print(f"实际解释器：{sys.executable}", flush=True)
    print(f"解释器文件（解析链接）：{Path(sys.executable).resolve()}", flush=True)

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # 避免绘图库向用户全局目录写缓存；无须 IDE 隐藏配置。
        os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / "matplotlib-cache"))
        os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_DIR / "cache"))
    except OSError as exc:
        print(f"FAIL：无法准备输出目录：{exc}", flush=True)
        return 1

    dependencies = {}
    for name in ("numpy", "matplotlib", "trimesh", "cadquery"):
        print(f"检查依赖：正在导入 {name}…", flush=True)
        try:
            dependencies[name] = importlib.import_module(name)
            print(f"PASS：导入 {name} {version(name)}", flush=True)
        except Exception as exc:
            print(f"FAIL：导入 {name}：{type(exc).__name__}: {exc}", flush=True)

    if len(dependencies) != 4:
        print("FAIL：依赖未全部就绪，跳过 CAD 检查。", flush=True)
        return 1

    try:
        cq = dependencies["cadquery"]
        # CadQuery 本项目长度约定为 mm；本对象只用于环境验证。
        test_box = cq.Workplane("XY").box(*EXPECTED_MM)
        step_path = OUTPUT_DIR / "test_box.step"
        cq.exporters.export(test_box, str(step_path), exportType="STEP")
        if not step_path.is_file() or step_path.stat().st_size == 0:
            raise RuntimeError("STEP 导出未生成非空文件")
        print(f"PASS：导出测试 STEP：{step_path}", flush=True)

        imported = cq.importers.importStep(str(step_path))
        solids = imported.solids().vals()
        if len(solids) != 1:
            raise RuntimeError(f"重新读取后预期 1 个实体，实际为 {len(solids)}")
        if not imported.val().isValid() or not solids[0].isValid():
            raise RuntimeError("重新读取的几何无效")
        print("PASS：重新读取 STEP，包含且仅包含 1 个有效实体", flush=True)

        bounds = imported.val().BoundingBox()
        actual_mm = (bounds.xlen, bounds.ylen, bounds.zlen)
        if not all(
            math.isclose(actual, expected, rel_tol=1e-9, abs_tol=ABS_TOL_MM)
            for actual, expected in zip(actual_mm, EXPECTED_MM)
        ):
            raise RuntimeError(f"包围盒尺寸错误：{actual_mm} mm，预期 {EXPECTED_MM} mm")
        print(
            "PASS：包围盒 X × Y × Z = "
            f"{actual_mm[0]:.9f} × {actual_mm[1]:.9f} × {actual_mm[2]:.9f} mm"
            f"（绝对容差 {ABS_TOL_MM:g} mm）",
            flush=True,
        )
    except Exception as exc:
        print(f"FAIL：CAD 验证：{type(exc).__name__}: {exc}", flush=True)
        return 1

    print("PASS：全部环境检查通过；HFSS 导入尚未验证。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
