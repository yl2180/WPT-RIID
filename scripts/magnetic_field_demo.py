#!/usr/bin/env python3
"""理想细导线完整圆环的 Biot–Savart 验证案例；不是 BISC 或高频全波仿真。"""

import json
import os
from pathlib import Path
import sys

import numpy as np


# 独立练习参数：不读取或闭合 simple_coil.py 的带缺口圆弧。
RADIUS_MM = 10.0
CURRENT_A = 1.0  # 从 +z 看向原点，正电流逆时针。
MU0_H_PER_M = 4 * np.pi * 1e-7  # 本例采用的自由空间磁导率近似值。
VALIDATION_SEGMENTS = (50, 100, 200, 400)
FIELD_SEGMENTS = 400
X_RANGE_MM = (-20.0, 20.0, 21)  # 起点、终点、点数（含两端）
Y_RANGE_MM = (-20.0, 20.0, 21)
Z_RANGE_MM = (5.0, 25.0, 21)
SLICE_Z_MM = 10.0
ARROW_STRIDE = 5  # 三个坐标方向每隔 5 个网格点取一次，共 125 支箭头。
ARROW_LENGTH_MM = 3.0  # 所有箭头等长，只表示方向；点颜色表示 |B|。

# 仅用于几何接触判定，不作为导线半径，也不加到物理公式的分母上。
ON_WIRE_TOL_M = 1e-12
TRANSVERSE_RATIO_LIMIT = 1e-12
RELATIVE_ERROR_LIMIT = 0.001  # 0.1%
CSV_HEADER = "x_mm,y_mm,z_mm,Bx_T,By_T,Bz_T"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "magnetic_field_demo"


def biot_savart(wire_points_mm, observation_points_mm, current_a, *, closed=False):
    """对任意折线做中点求积，返回 shape=(观察点数, 3) 的 [Bx, By, Bz]，单位 T。

    wire_points_mm: (N, 3)，按正电流方向排列，单位 mm。
    observation_points_mm: (3,) 或 (M, 3)，单位 mm。
    current_a: 有限实数，负值反转电流。
    closed=False: N 个点形成 N-1 段，不自动首尾相连。
    closed=True: N 个不重复首尾的顶点形成 N 段，最后一段明确连回首点。
    函数只使用输入折线，不知道它是否为圆环，也不调用圆环解析公式。
    """
    if not isinstance(closed, bool):
        raise ValueError("closed 必须为 True 或 False")
    if any(np.iscomplexobj(value) for value in (wire_points_mm, observation_points_mm, current_a)):
        raise ValueError("坐标和电流必须为实数，不能是复数")
    try:
        wire = np.asarray(wire_points_mm, dtype=float)
        observations = np.asarray(observation_points_mm, dtype=float)
        current = np.asarray(current_a, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("坐标和电流必须能够转换为实数") from exc
    if wire.ndim != 2 or wire.shape[1] != 3 or len(wire) < (3 if closed else 2):
        raise ValueError("导线坐标必须为 N×3；开口至少 2 点，闭合至少 3 点")
    if observations.shape == (3,):
        observations = observations.reshape(1, 3)
    if observations.ndim != 2 or observations.shape[1] != 3 or len(observations) == 0:
        raise ValueError("观察点必须为一个三维坐标或非空 M×3 坐标数组")
    if not np.isfinite(wire).all() or not np.isfinite(observations).all():
        raise ValueError("导线点和观察点必须全部有限，不能包含 NaN 或无穷大")
    if current.ndim != 0 or not np.isfinite(current):
        raise ValueError("电流必须是单个有限实数，单位 A")
    if closed and np.array_equal(wire[0], wire[-1]):
        raise ValueError("closed=True 时不要重复首点；函数会添加最后一段，否则产生零长度段")

    # 几何输入用 mm；进入 SI 物理公式前统一转为 m。
    wire_m = wire * 1e-3
    observations_m = observations * 1e-3
    starts = wire_m if closed else wire_m[:-1]
    ends = np.roll(wire_m, -1, axis=0) if closed else wire_m[1:]
    segments = ends - starts
    lengths_squared = np.sum(segments * segments, axis=1)
    if not np.isfinite(lengths_squared).all():
        raise ValueError("线段长度超出可计算数值范围")
    zero_segments = np.flatnonzero(lengths_squared == 0)
    if len(zero_segments):
        raise ValueError(f"第 {zero_segments[0]} 段为零长度线段（索引从 0 开始）")

    field_t = np.zeros((len(observations_m), 3))
    # 只对当前一段处理所有观察点，不创建 M×N×3 的大数组。
    for index, (start, end, dl, length_squared) in enumerate(
        zip(starts, ends, segments, lengths_squared)
    ):
        # 点到有限线段的最短距离：包含中点、其他内部位置和两端。
        fraction = np.sum((observations_m - start) * dl, axis=1) / length_squared
        closest = start + np.clip(fraction, 0.0, 1.0)[:, None] * dl
        distance_to_wire = np.linalg.norm(observations_m - closest, axis=1)
        touching = np.flatnonzero(distance_to_wire <= ON_WIRE_TOL_M)
        if len(touching):
            raise ValueError(
                f"观察点 {touching[0]} 位于理想导线第 {index} 段上或距其不超过 "
                f"{ON_WIRE_TOL_M:g} m；细导线模型在此不可用（索引从 0 开始）"
            )

        midpoint = (start + end) / 2
        r = observations_m - midpoint  # 从电流元中点指向观察点。
        distance = np.linalg.norm(r, axis=1)
        # ΔB = μ0 I/(4π) * (Δl × r)/|r|³；不往分母添加小常数。
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            field_t += np.cross(dl, r) / distance[:, None] ** 3

    field_t *= MU0_H_PER_M * float(current) / (4 * np.pi)
    if not np.isfinite(field_t).all():
        raise ValueError("磁场出现非有限结果，请检查坐标范围及观察点距离")
    return field_t


def circle_vertices_mm(segment_count):
    """完整圆环的 N 个独立顶点；不包含与 0° 重复的 360° 点。"""
    if not isinstance(segment_count, int) or segment_count < 3:
        raise ValueError("圆环线段数量必须是至少为 3 的整数")
    theta = np.linspace(0, 2 * np.pi, segment_count, endpoint=False)
    return np.column_stack((RADIUS_MM * np.cos(theta), RADIUS_MM * np.sin(theta), np.zeros(segment_count)))


def analytical_axis_field_t(z_mm, current_a):
    """完整圆环轴线解析解，仅用于验证；R 和 z 在公式中均为 m。"""
    radius_m = RADIUS_MM * 1e-3
    z_m = np.asarray(z_mm, dtype=float) * 1e-3
    result = np.zeros((z_m.size, 3))
    result[:, 2] = MU0_H_PER_M * current_a * radius_m**2 / (2 * (radius_m**2 + z_m**2)**1.5)
    return result


def validate_input_errors():
    """轻量检查几种重要错误入口，包含不在中点处的导线接触。"""
    line = [[0, 0, 0], [2, 0, 0]]
    triangle = [[0, 0, 0], [2, 0, 0], [0, 2, 0]]
    cases = [
        ("非法坐标形状", [[0, 0], [1, 0]], [0, 1, 0], 1, False),
        ("非有限观察点", line, [0, np.nan, 0], 1, False),
        ("非有限电流", line, [0, 1, 0], np.inf, False),
        ("零长度段", [[0, 0, 0], [0, 0, 0]], [0, 1, 0], 1, False),
        ("线段内部接触", line, [0.5, 0, 0], 1, False),
        ("线段端点接触", line, [2, 0, 0], 1, False),
        ("闭合最后一段接触", triangle, [0, 1, 0], 1, True),
        ("闭合重复首点", triangle + [triangle[0]], [0, 0, 1], 1, True),
    ]
    for name, wire, observation, current, closed in cases:
        try:
            biot_savart(wire, observation, current, closed=closed)
        except ValueError:
            continue
        raise RuntimeError(f"错误输入检查未通过：{name} 没有被拒绝")
    print(f"PASS：{len(cases)} 项非法输入/导线接触检查。", flush=True)


def validate_circle():
    """先完成轴线解析对照、收敛、方向和反向电流验证。"""
    observations = np.array([[0, 0, 0], [0, 0, RADIUS_MM]])
    theory = analytical_axis_field_t(observations[:, 2], CURRENT_A)
    theory_norm = np.linalg.norm(theory, axis=1)
    previous_error = None
    max_transverse_ratio = 0.0
    print("轴线验证：全部 B 分量单位 T；相对误差使用完整矢量范数。", flush=True)
    for count in VALIDATION_SEGMENTS:
        vertices = circle_vertices_mm(count)
        numerical = biot_savart(vertices, observations, CURRENT_A, closed=True)
        error = np.linalg.norm(numerical - theory, axis=1) / theory_norm
        transverse_ratio = np.linalg.norm(numerical[:, :2], axis=1) / theory_norm
        for i, label in enumerate(("圆心", "z=R")):
            numeric_text = ", ".join(f"{value:.10e}" for value in numerical[i])
            theory_text = ", ".join(f"{value:.10e}" for value in theory[i])
            print(
                f"{count:3d} 段 {label}: B数值=({numeric_text}); "
                f"B理论=({theory_text}); 矢量相对误差={error[i]:.8e} ({error[i]:.6%})",
                flush=True,
            )
        if not np.all(numerical[:, 2] > 0):
            raise RuntimeError("正向电流的轴线上磁场没有指向 +z")
        if not np.all(transverse_ratio < TRANSVERSE_RATIO_LIMIT):
            raise RuntimeError("轴线上横向磁场相对于理论场不够小")
        if previous_error is not None and not np.all(error < previous_error):
            raise RuntimeError("增加线段数量后，两个验证点的误差没有都减小")
        reversed_field = biot_savart(vertices, observations, -CURRENT_A, closed=True)
        if not np.allclose(reversed_field, -numerical, rtol=1e-12, atol=0):
            raise RuntimeError("反向电流没有产生逐分量反向的磁场")
        if not np.allclose(np.linalg.norm(reversed_field, axis=1), np.linalg.norm(numerical, axis=1), rtol=1e-12, atol=0):
            raise RuntimeError("反向电流改变了磁场大小")
        previous_error = error
        max_transverse_ratio = max(max_transverse_ratio, float(transverse_ratio.max()))

    if not np.all(error < RELATIVE_ERROR_LIMIT):
        raise RuntimeError("400 段时仍有验证点的矢量相对误差不小于 0.1%")
    print("PASS：正向/横向分量、误差递减、400 段误差 <0.1%、-1 A 反向等幅检查。", flush=True)
    return {
        "result": "PASS", "segments": list(VALIDATION_SEGMENTS),
        "observations_mm": observations.tolist(),
        "relative_errors_at_400_segments": error.tolist(),
        "max_transverse_to_theory_ratio": max_transverse_ratio,
        "negative_current_reverses_field": True,
    }


def save_and_read_field(csv_path, observations_mm, field_t):
    """每行是一个观察点位置及其磁场矢量；读回验证后供绘图使用。"""
    data = np.column_stack((observations_mm, field_t))
    np.savetxt(csv_path, data, delimiter=",", header=CSV_HEADER, comments="", fmt="%.17g", encoding="utf-8")
    with csv_path.open(encoding="utf-8") as stream:
        if stream.readline().rstrip("\r\n") != CSV_HEADER:
            raise RuntimeError("field.csv 表头错误")
        row_count = sum(1 for _ in stream)
    loaded = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    expected_count = X_RANGE_MM[2] * Y_RANGE_MM[2] * Z_RANGE_MM[2]
    if row_count != expected_count or loaded.shape != (expected_count, 6):
        raise RuntimeError(f"CSV 行数/形状错误：{row_count} 行、{loaded.shape}，预期 {expected_count}×6")
    if not np.isfinite(loaded).all():
        raise RuntimeError("CSV 存在 NaN 或无穷大")
    if not np.allclose(loaded[:, :3], data[:, :3], rtol=0, atol=1e-12):
        raise RuntimeError("CSV 坐标读回不一致（绝对容差 1e-12 mm）")
    if not np.allclose(loaded[:, 3:], data[:, 3:], rtol=1e-12, atol=1e-20):
        raise RuntimeError("CSV 磁场读回不一致（相对容差 1e-12、绝对容差 1e-20 T）")
    print(f"PASS：CSV 有 {row_count} 行数据，形状 {loaded.shape}，坐标和磁场有限且读回一致。", flush=True)
    return loaded


def plot_field(loaded, png_path):
    """只从 CSV 读回数据取磁场；圆环轮廓仅作位置参考。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_values, y_values, z_values = (np.unique(loaded[:, i]) for i in range(3))
    magnitudes_t = np.linalg.norm(loaded[:, 3:], axis=1)
    slice_mask = np.isclose(loaded[:, 2], SLICE_Z_MM, rtol=0, atol=1e-9)
    if slice_mask.sum() != len(x_values) * len(y_values):
        raise RuntimeError("CSV 中没有完整的 z=10 mm 热图切片")
    # CSV 按 x、y、z 嵌套顺序保存，z 最快变化；切片中 y 最快变化。
    heat = magnitudes_t[slice_mask].reshape(len(x_values), len(y_values)).T
    cube = loaded.reshape(len(x_values), len(y_values), len(z_values), 6)
    sparse = cube[::ARROW_STRIDE, ::ARROW_STRIDE, ::ARROW_STRIDE].reshape(-1, 6)
    sparse_magnitudes = np.linalg.norm(sparse[:, 3:], axis=1)

    fig = plt.figure(figsize=(13.5, 7.3))
    fig.subplots_adjust(left=0.07, right=0.94, bottom=0.18, top=0.80, wspace=0.27)
    fig.suptitle(
        "Biot-Savart demo - complete ideal thin-wire loop (not BISC)\n"
        f"R = {RADIUS_MM:g} mm | I = {CURRENT_A:g} A, CCW viewed from +z | {FIELD_SEGMENTS} segments",
        fontsize=13,
    )
    top = fig.add_subplot(1, 2, 1)
    heatmap = top.pcolormesh(x_values, y_values, heat, shading="nearest", cmap="viridis")
    top.set(title=f"Field magnitude at z = {SLICE_Z_MM:g} mm", xlabel="X (mm)", ylabel="Y (mm)")
    top.set_aspect("equal")
    heatbar = fig.colorbar(heatmap, ax=top, fraction=0.046, pad=0.04)
    heatbar.set_label("|B| (T)")
    heatbar.formatter.set_powerlimits((0, 0))
    heatbar.update_ticks()

    view = fig.add_subplot(1, 2, 2, projection="3d")
    nonzero = sparse_magnitudes > 0
    points = sparse[nonzero]
    view.quiver(
        *points[:, :3].T, *points[:, 3:].T,
        normalize=True, length=ARROW_LENGTH_MM, pivot="tail",
        color="#333333", linewidth=0.8, arrow_length_ratio=0.3,
    )
    dots = view.scatter(*sparse[:, :3].T, c=sparse_magnitudes, cmap="viridis", s=18, depthshade=False)
    loop = circle_vertices_mm(FIELD_SEGMENTS)
    loop = np.vstack((loop, loop[0]))  # 仅绘图闭合轮廓，不传回积分函数。
    view.plot(*loop.T, color="black", linewidth=1.4, label="Loop at z=0")
    view.set(title="Direction arrows; dot color = |B|", xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
    view.set(xlim=(-24, 24), ylim=(-24, 24), zlim=(0, 29))
    view.set_box_aspect((48, 48, 29))
    view.view_init(elev=24, azim=-55)
    view.legend(loc="upper left", fontsize=8)
    arrowbar = fig.colorbar(dots, ax=view, fraction=0.046, pad=0.10, shrink=0.82)
    arrowbar.set_label("|B| at arrow origin (T)")
    arrowbar.formatter.set_powerlimits((0, 0))
    arrowbar.update_ticks()
    fig.text(
        0.5, 0.05,
        f"CSV readback data | Coordinates: mm | Normalized arrows: {ARROW_LENGTH_MM:g} mm long, direction only\n"
        "Dot colors encode magnitude (T); each panel has its own color scale. Zero field: no arrow.",
        ha="center", fontsize=10,
    )
    fig.savefig(png_path, dpi=180, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main():
    print("独立完整圆环磁场验证：自由空间、理想细导线、稳恒电流。", flush=True)
    print(f"解释器：{sys.executable}", flush=True)
    try:
        if not np.isfinite(RADIUS_MM) or RADIUS_MM <= 0:
            raise ValueError("RADIUS_MM 必须为有限正数")
        if CURRENT_A != 1.0:
            raise ValueError("本验证案例要求 CURRENT_A=1.0；-1 A 在反向检查中单独计算")
        validate_input_errors()
        validation = validate_circle()
        # 只有以上验证全部通过，才计算观察区域和写入结果。
        x, y, z = (np.linspace(*axis) for axis in (X_RANGE_MM, Y_RANGE_MM, Z_RANGE_MM))
        observations = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
        print(f"正在计算 {len(observations)} 个观察点的三维磁场…", flush=True)
        field = biot_savart(circle_vertices_mm(FIELD_SEGMENTS), observations, CURRENT_A, closed=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / "matplotlib-cache"))
        csv_path = OUTPUT_DIR / "field.csv"
        loaded = save_and_read_field(csv_path, observations, field)
        plot_field(loaded, OUTPUT_DIR / "field.png")
        parameters = {
            "purpose": "algorithm_validation_only_not_BISC_not_full_wave_simulation",
            "source_script": "scripts/magnetic_field_demo.py",
            "model": "complete circular loop in XY, center=(0,0,0), ideal thin wire, steady current, uniform free space",
            "radius_mm": RADIUS_MM, "current_a": CURRENT_A,
            "positive_current_direction": "counterclockwise viewed from +z toward origin",
            "mu0_approx_H_per_m": MU0_H_PER_M,
            "mu0_approximation": "4*pi*1e-7 H/m (approximate, not an exact SI constant)",
            "unique_wire_vertices": FIELD_SEGMENTS, "segment_count": FIELD_SEGMENTS,
            "closure": "closed=True adds last-to-first segment exactly once; first vertex not repeated",
            "method": "polygonal chords with midpoint Biot-Savart quadrature, vector sum over segments",
            "observation_axes_start_stop_count_mm": {"x": X_RANGE_MM, "y": Y_RANGE_MM, "z": Z_RANGE_MM},
            "observation_count": len(observations),
            "units": {"input_coordinates": "mm", "formula_lengths": "m", "current": "A", "field": "T"},
            "on_wire_detection_tolerance_m": ON_WIRE_TOL_M,
            "singularity_handling": "reject contact with any input segment; no denominator regularization",
            "plot": {"slice_z_mm": SLICE_Z_MM, "arrow_stride": ARROW_STRIDE, "normalized_arrow_length_mm": ARROW_LENGTH_MM, "magnitude_encoding": "dot color in T"},
            "validation": validation,
            "field_csv_check": "PASS: row count, shape, finite values, roundtrip",
            "off_axis_accuracy_and_HFSS_validated": False,
        }
        (OUTPUT_DIR / "parameters.json").write_text(json.dumps(parameters, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"PASS：全部检查通过，输出位于 {OUTPUT_DIR}", flush=True)
        print("本例只验证稳恒细导线算法；未验证完整高频电磁行为、HFSS、互感或 RIID。", flush=True)
        return 0
    except Exception as exc:
        print(f"FAIL：{type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
