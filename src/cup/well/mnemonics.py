"""cup.well.mnemonics: 常用测井曲线简称常量表。

本模块集中维护项目内用于 LAS 曲线自动识别的 mnemonic 候选集，
供 Petrel / LAS 加载模块进行曲线匹配时复用。

边界说明
--------
- 本模块只保存常量，不包含任何 IO、数据转换或业务流程逻辑。
- 新增或调整简称时，应优先保证与项目现有井曲线命名习惯兼容。

核心公开对象
------------
1. `_VP_MNEMONICS`: 纵波速度或声波时差候选简称。
2. `_VS_MNEMONICS`: 横波速度或横波时差候选简称。
3. `_RHO_MNEMONICS`: 密度曲线候选简称。
4. `_GR_MNEMONICS` / `_POR_MNEMONICS` / `_PERM_MNEMONICS` / `_SW_MNEMONICS`:
   其他常见储层参数候选简称。
"""

_CALI_MNEMONICS = ("BS", "CALI", "HDAR", "HCAL")
_VP_MNEMONICS = ("DT", "AC", "DTC", "DTCO")
_VS_MNEMONICS = ("DTS", "DTSM", "DTSH")
_RHO_MNEMONICS = ("DEN", "RHOB", "RHOZ", "HDRA")
_GR_MNEMONICS = ("GR", "GR_CAL")
_POR_MNEMONICS = ("POR", "PHIE", "PHIT", "PHIE_HILT")
_PERM_MNEMONICS = ("PERM", "PERM_COATES_FFI")
_SW_MNEMONICS = ("SW", "SWE", "SWT", "SW_HILT")
