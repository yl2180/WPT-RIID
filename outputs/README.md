# 生成输出

存放脚本生成的几何文件及记录，默认不进入 Git。

`env_check/test_box.step` 是 `scripts/check_env.py` 生成的 **10 × 10 × 1 mm 环境测试长方体**，
不是 BISC 接收线圈，也不是任何实际线圈。可运行检查脚本重新生成。
本地导出和重新读取成功不代表通过 HFSS 导入验证。

`simple_coil/simple_coil.step` 是 `scripts/simple_coil.py` 生成的学习用开口圆弧导线，
中心线半径 10 mm，角度为 15° 至 345°，导线直径 0.5 mm。
它不是 BISC、真实 Rx / Tx 或 RIID 结果。同目录的 `simple_coil.png` 展示俯视图和三维图，
`parameters.json` 保存练习参数、单位、来源和检查结果。HFSS 导入尚未验证。

`simple_coil/simple_coil.csv` 默认保存 201 个中心线点（单位 mm，含首尾，不闭合），
表头为 `x_mm,y_mm,z_mm`，不含导线直径。
`simple_coil/simple_coil_points.png` 使用 CSV 读回数据展示采样点及相邻连接线，包含俯视图和三维图。
CSV 是项目内部暂定格式，尚未验证学长的 HFSS 读取流程。

`magnetic_field_demo/` 是独立完整圆环的理想细导线稳恒磁场验证输出：
`field.csv` 保存 9261 个观察点及 Bx、By、Bz（位置 mm、磁场 T），
`field.png` 从 CSV 读回数据生成 z=10 mm 热图与方向箭头图，
`parameters.json` 保存参数、单位、近似和验证摘要。
这些场点不是原开口线圈的中心线；本例不是 BISC 或完整高频电磁仿真。
