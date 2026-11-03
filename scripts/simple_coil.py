#!/usr/bin/env python3
"""学习用开口圆弧导线：不是 BISC、真实 Rx / Tx 或 RIID 结果。"""

import json
import math
import os
from pathlib import Path
import sys


# 练习参数：仅用于学习，长度单位统一为 mm，角度单位为度。
RADIUS_MM = 10.0
WIRE_DIAMETER_MM = 0.5
START_ANGLE_DEG = 15.0
END_ANGLE_DEG = 345.0
SAMPLE_COUNT = 201  # 中心线采样点数，包含起点和终点；不重复首点闭合。

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "simple_coil"


def point_at(angle_deg):
    """XY 平面的圆：x = R cos(theta)，y = R sin(theta)，z = 0。"""
    angle = math.radians(angle_deg)
    return (RADIUS_MM * math.cos(angle), RADIUS_MM * math.sin(angle), 0.0)


def save_preview(png_path, points=None):
    """points 为 CSV 读回坐标；不传时保留原有导线预览。STEP 不依赖采样。"""
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")  # 无须弹出窗口或依赖 IDE。
    import matplotlib.pyplot as plt

    if points is None:
        theta = np.linspace(
            math.radians(START_ANGLE_DEG), math.radians(END_ANGLE_DEG), 600
        )
        x, y = RADIUS_MM * np.cos(theta), RADIUS_MM * np.sin(theta)
        z = np.zeros_like(theta)
        start, end = point_at(START_ANGLE_DEG), point_at(END_ANGLE_DEG)
    else:
        # 新图的全部路径坐标和端点都来自实际保存后重新读取的 CSV。
        x, y, z = points.T
        start, end = points[0], points[-1]
    point_style = {} if points is None else {"marker": ".", "markersize": 4}

    fig = plt.figure(figsize=(12, 7.2))
    fig.subplots_adjust(left=0.07, right=0.95, bottom=0.12, top=0.80, wspace=0.24)
    fig.suptitle(
        "PRACTICE ONLY - open circular wire (not BISC)\n"
        f"Centerline radius: {RADIUS_MM:g} mm | Wire diameter: {WIRE_DIAMETER_MM:g} mm | "
        f"Angles: {START_ANGLE_DEG:g} to {END_ANGLE_DEG:g} deg"
        + ("" if points is None else f"\nCSV readback: {len(points)} points - centerline only"),
        fontsize=13,
    )
    top = fig.add_subplot(1, 2, 1)
    # plot 只连接相邻点，不追加首点，保留开口。
    top.plot(x, y, color="tab:blue", linewidth=2, label="Centerline (CCW)", **point_style)
    top.scatter(*start[:2], color="tab:green", s=55, zorder=3, label="Start")
    top.scatter(*end[:2], color="tab:red", marker="s", s=45, zorder=3, label="End")
    top.scatter(0, 0, marker="+", color="gray")
    top.annotate(
        f"Start {START_ANGLE_DEG:g} deg", xy=start[:2], xytext=(4, 6),
        arrowprops={"arrowstyle": "->", "color": "tab:green"}, color="tab:green",
    )
    top.annotate(
        f"End {END_ANGLE_DEG:g} deg", xy=end[:2], xytext=(4, -6),
        arrowprops={"arrowstyle": "->", "color": "tab:red"}, color="tab:red",
    )
    gap_deg = 360.0 - (END_ANGLE_DEG - START_ANGLE_DEG)
    top.text(RADIUS_MM * 0.6, 0, f"{gap_deg:g} deg gap", ha="center", fontsize=10)
    top.set(title="Top view (XY) - centerline", xlabel="X (mm)", ylabel="Y (mm)")
    limit = RADIUS_MM + WIRE_DIAMETER_MM + 1.5
    top.set_xlim(-limit, limit)
    top.set_ylim(-limit, limit)
    top.set_aspect("equal")
    top.grid(alpha=0.25)
    top.legend(loc="lower left")

    view = fig.add_subplot(1, 2, 2, projection="3d")
    if points is None:
        # 原有导线预览；CSV 点集图只画中心线，不添加导线表面。
        phi = np.linspace(0, 2 * np.pi, 24)
        tube_radius = RADIUS_MM + WIRE_DIAMETER_MM / 2 * np.cos(phi)
        view.plot_surface(
            np.cos(theta[:, None]) * tube_radius,
            np.sin(theta[:, None]) * tube_radius,
            np.broadcast_to(WIRE_DIAMETER_MM / 2 * np.sin(phi), (len(theta), len(phi))),
            color="lightskyblue", alpha=0.3, linewidth=0, shade=False,
        )
    view.plot(x, y, z, color="tab:blue", linewidth=2, label="Centerline", **point_style)
    view.scatter(*start, color="tab:green", s=55, depthshade=False)
    view.scatter(*end, color="tab:red", marker="s", s=45, depthshade=False)
    view.text(start[0], start[1], 1.2, "Start", color="tab:green")
    view.text(end[0], end[1], -1.8, "End", color="tab:red")
    view.set(title="3D view - centerline and wire preview", xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
    if points is not None:
        top.set_title("Top view (XY) - CSV points and adjacent segments")
        view.set_title("3D view - CSV points and adjacent segments")
    view.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
    view.set_box_aspect((1, 1, 1))  # 三轴使用相同尺度，不夸大导线厚度。
    view.view_init(elev=28, azim=35)
    fig.savefig(png_path, dpi=180, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def save_and_check_csv(csv_path):
    """复用圆弧公式采样，写入后从磁盘读回，逐项检查并返回读回数据。"""
    import numpy as np

    angles = np.linspace(START_ANGLE_DEG, END_ANGLE_DEG, SAMPLE_COUNT, endpoint=True)
    points = np.array([point_at(angle) for angle in angles], dtype=float)
    np.savetxt(
        csv_path, points, delimiter=",", header="x_mm,y_mm,z_mm",
        comments="", fmt="%.17g", encoding="utf-8",
    )
    with csv_path.open(encoding="utf-8") as csv_file:
        if csv_file.readline().rstrip("\r\n") != "x_mm,y_mm,z_mm":
            raise ValueError("CSV 表头必须为 x_mm,y_mm,z_mm")
    try:
        loaded = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    except (OSError, ValueError) as exc:
        raise ValueError(f"CSV 重新读取失败：{exc}") from exc

    tolerance_mm = 1e-9
    if loaded.shape != (SAMPLE_COUNT, 3):
        raise ValueError(f"CSV 形状错误：{loaded.shape}，预期 {(SAMPLE_COUNT, 3)}")
    if not np.isfinite(loaded).all():
        raise ValueError("CSV 包含非有限坐标（NaN 或无穷大）")
    if not np.allclose(loaded, points, rtol=0, atol=tolerance_mm):
        raise ValueError("CSV 读回坐标与保存前坐标不一致")
    if not np.allclose(loaded[0], point_at(START_ANGLE_DEG), rtol=0, atol=tolerance_mm):
        raise ValueError("CSV 起点不符合当前圆弧起始角和半径")
    if not np.allclose(loaded[-1], point_at(END_ANGLE_DEG), rtol=0, atol=tolerance_mm):
        raise ValueError("CSV 终点不符合当前圆弧终止角和半径")
    if not np.allclose(loaded[:, 2], 0, rtol=0, atol=tolerance_mm):
        raise ValueError("CSV 中存在不在 XY 平面上的点")
    radii = np.linalg.norm(loaded[:, :2], axis=1)
    if not np.allclose(radii, RADIUS_MM, rtol=0, atol=tolerance_mm):
        raise ValueError("CSV 中存在到原点的距离不等于中心线半径的点")
    if np.allclose(loaded[0], loaded[-1], rtol=0, atol=tolerance_mm):
        raise ValueError("CSV 起点与终点重复，路径被错误封闭")
    print(
        f"PASS：CSV 读回形状 {loaded.shape}，坐标有限且往返一致；"
        "端点、XY 平面、中心线半径及开口检查全部通过。",
        flush=True,
    )
    return loaded


def main():
    print("练习对象：开口圆弧导线，不是 BISC；复用现有虚拟环境。", flush=True)
    print(f"解释器：{sys.executable}", flush=True)
    try:
        if not (RADIUS_MM > WIRE_DIAMETER_MM / 2 > 0):
            raise ValueError("需要正的导线直径，且中心线半径大于导线半径")
        if not (0 <= START_ANGLE_DEG < END_ANGLE_DEG <= 360 and END_ANGLE_DEG - START_ANGLE_DEG < 360):
            raise ValueError("练习要求 0 <= 起始角 < 终止角 <= 360，且圆弧不闭合")
        if isinstance(SAMPLE_COUNT, bool) or not isinstance(SAMPLE_COUNT, int) or SAMPLE_COUNT < 2:
            raise ValueError("中心线采样点数 SAMPLE_COUNT 必须为大于等于 2 的整数")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / "matplotlib-cache"))
        os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_DIR / "cache"))
        print("正在加载 CadQuery…", flush=True)
        import cadquery as cq

        start = point_at(START_ANGLE_DEG)
        middle = point_at((START_ANGLE_DEG + END_ANGLE_DEG) / 2)
        end = point_at(END_ANGLE_DEG)
        # 三个圆上点定义精确圆弧；中点确保走过 330°，而不是跨过缺口的短弧。
        arc = cq.Edge.makeThreePointArc(cq.Vector(start), cq.Vector(middle), cq.Vector(end))
        path = cq.Wire.assembleEdges([arc])
        angle = math.radians(START_ANGLE_DEG)
        tangent = (-math.sin(angle), math.cos(angle), 0)
        radial = (math.cos(angle), math.sin(angle), 0)
        section_plane = cq.Plane(origin=start, xDir=radial, normal=tangent)
        wire = cq.Workplane(section_plane).circle(WIRE_DIAMETER_MM / 2).sweep(
            path, makeSolid=True, isFrenet=True
        )

        step_path = OUTPUT_DIR / "simple_coil.step"
        cq.exporters.export(wire, str(step_path), exportType="STEP")
        imported = cq.importers.importStep(str(step_path))
        solids = imported.solids().vals()
        if len(solids) != 1 or not imported.val().isValid() or not solids[0].isValid():
            raise RuntimeError("STEP 重新读取后不是单个有效实体")
        solid = solids[0]
        caps = [face for face in solid.Faces() if face.geomType() == "PLANE"]
        if len(caps) != 2 or not all(
            any((cap.Center() - cq.Vector(point)).Length < 1e-6 for cap in caps)
            for point in (start, end)
        ):
            raise RuntimeError("没有在预期起点和终点找到两个独立端面")
        expected_volume = math.pi * (WIRE_DIAMETER_MM / 2) ** 2 * RADIUS_MM * math.radians(END_ANGLE_DEG - START_ANGLE_DEG)
        if not math.isclose(solid.Volume(), expected_volume, rel_tol=1e-6):
            raise RuntimeError("导线体积与圆截面积 × 圆弧长度不符")
        print("PASS：STEP 重新读取为 1 个有效实体，具有位置正确的 2 个分离端面，体积符合预期。", flush=True)

        png_path = OUTPUT_DIR / "simple_coil.png"
        save_preview(png_path)
        csv_path = OUTPUT_DIR / "simple_coil.csv"
        loaded_points = save_and_check_csv(csv_path)
        points_png_path = OUTPUT_DIR / "simple_coil_points.png"
        save_preview(points_png_path, points=loaded_points)
        record = {
            "purpose": "practice_only_not_BISC_not_RIID",
            "source_script": "scripts/simple_coil.py",
            "centerline_definition": "(R*cos(theta), R*sin(theta), 0), counterclockwise from +X",
            "length_unit": "mm", "angle_unit": "deg",
            "radius_mm": RADIUS_MM, "wire_diameter_mm": WIRE_DIAMETER_MM,
            "start_angle_deg": START_ANGLE_DEG, "end_angle_deg": END_ANGLE_DEG,
            "start_mm": start, "end_mm": end,
            "cadquery_version": cq.__version__,
            "step_check": {"result": "PASS", "valid_solid_count": 1, "end_cap_count": 2, "volume_mm3": solid.Volume()},
            "sample_count": SAMPLE_COUNT,
            "sampling": "uniform_angle_including_endpoints_open_path",
            "csv_format": "provisional_internal_format: x_mm,y_mm,z_mm; centerline only",
            "csv_check": {"result": "PASS", "shape": list(loaded_points.shape), "absolute_tolerance_mm": 1e-9},
            "hfss_import_verified": False,
        }
        (OUTPUT_DIR / "parameters.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"PASS：STEP：{step_path}\nPNG：{png_path}\n参数与来源记录：{OUTPUT_DIR / 'parameters.json'}", flush=True)
        print(f"CSV：{csv_path}\nCSV 读回点集图：{points_png_path}", flush=True)
        print("CSV 为内部暂定格式，学长的 HFSS 读取流程尚未验证；STEP 的 HFSS 导入也尚未验证。", flush=True)
        print("几何成功不代表互感或效率提升。", flush=True)
        return 0
    except Exception as exc:
        print(f"FAIL：{type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
