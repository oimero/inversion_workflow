"""cup.well.mnemonics: 常用测井曲线简称常量表。

本模块集中维护项目内用于 LAS 曲线自动识别的 mnemonic 候选集，
供 Petrel / LAS 加载模块进行曲线匹配时复用。

边界说明
--------
- 本模块只保存常量，不包含任何 IO、数据转换或业务流程逻辑。
- 新增或调整简称时，应优先保证与项目现有井曲线命名习惯兼容。

核心公开对象
------------
1. `CURVE_CATEGORY_MNEMONICS`: 第二步 LAS header 本地分类的公开候选集。
2. `CURVE_CATEGORY_PRIORITY`: 每个类别内选择 primary 的默认优先级。
3. `_VP_MNEMONICS` / `_RHO_MNEMONICS` 等旧常量：兼容现有加载函数。
"""

CURVE_CATEGORY_MNEMONICS = {
    "caliper": (
        "CAL",
        "CALI",
        "BS",
        "HCAL",
        "HDAR",
    ),
    "gamma_ray": (
        "GR",
        "GR1",
        "GR-NORM",
        "GR_CAL",
        "GAMMARAY",
        "GAMMARAY1",
        "GAMMARAY2",
        "GAMMARAY3",
        "GAMMARAY4",
        "GAMMARAY5",
        "GAMMARAY6",
        "GAMMARAY7",
        "GAMMARAY8",
        "GAMMARAY9",
        "GAMMARAY10",
        "GAMMARAY11",
    ),
    "p_sonic": (
        "DT",
        "DTC",
        "DTCO",
        "AC",
        "CALIBRATEDSONICLOG",
        "CALIBRATEDSONICLOG:1",
        "CALIBRATEDSONICLOG:2",
        "CALIBRATEDSONICLOG:3",
        "CALIBRATEDSONICLOG:4",
        "VP",
        "VP_MPS",
        "VPMS",
        "INPUTINTERVALVELOCITY",
        "OUTPUTINTERVALVELOCITY",
    ),
    "s_sonic": (
        "DTS",
        "DTSM",
        "DTSH",
        "DTSM_FAST",
        "DTSM_SLOW",
        "VS",
        "VS_MPS",
        "VSMS",
    ),
    "density": (
        "DEN",
        "RHOB",
        "RHOZ",
        "HDRA",
        "RHO",
        "RHO_GCC",
    ),
    "resistivity": (
        "RT",
        "LLD",
        "LLD1",
        "LLS",
        "MSFL",
        "ILD",
        "AT90",
        "RD",
        "RS",
        "RXO",
        "RLA1",
        "RLA2",
        "RLA3",
        "RLA4",
        "RLA5",
        "A40H",
        "P16H",
        "P28H",
        "P34H",
        "P40H",
    ),
    "sp": (
        "SP",
    ),
    "porosity": (
        "POR",
        "PHIE",
        "PHIT",
        "PHIE_HILT",
        "BFV",
        "CN",
    ),
    "permeability": (
        "PERM",
        "PERM_COATES_FFI",
    ),
    "water_saturation": (
        "SW",
        "SWE",
        "SWT",
        "SW_HILT",
    ),
}


CURVE_CATEGORY_PRIORITY = {
    "caliper": (
        "CAL",
        "CALI",
        "BS",
        "HCAL",
        "HDAR",
    ),
    "gamma_ray": (
        "GR",
        "GR1",
        "GR-NORM",
        "GR_CAL",
        "GAMMARAY",
    ),
    "p_sonic": (
        "DT",
        "DTC",
        "DTCO",
        "AC",
        "VP",
        "VP_MPS",
        "VPMS",
    ),
    "s_sonic": (
        "DTS",
        "DTSM",
        "DTSH",
        "VS",
        "VS_MPS",
        "VSMS",
    ),
    "density": (
        "DEN",
        "RHOB",
        "RHOZ",
        "HDRA",
        "RHO",
        "RHO_GCC",
    ),
    "resistivity": (
        "RT",
        "LLD",
        "LLD1",
        "LLS",
        "MSFL",
        "ILD",
        "AT90",
        "RD",
        "RS",
        "RXO",
    ),
    "sp": (
        "SP",
    ),
    "porosity": (
        "POR",
        "PHIE",
        "PHIT",
        "PHIE_HILT",
        "CN",
        "BFV",
    ),
    "permeability": (
        "PERM",
        "PERM_COATES_FFI",
    ),
    "water_saturation": (
        "SW",
        "SWE",
        "SWT",
        "SW_HILT",
    ),
}


DERIVED_OR_AUXILIARY_MNEMONICS = (
    "AI",
    "RC",
    "DRIFT",
    "RESAMPLEDAI",
    "RESIDUALDRIFTLOG",
    "TWTPICKED",
    "TWTPICKED2",
    "ONE-WAYTIME",
    "SESMIC",
    "SESMIC2",
    "INPEFA",
    "PEFA",
    "D-INPEFA_GR",
    "GRINPEFA",
    "FACIES",
    "FLUIDS",
    "LITH",
    "LITH_SHOW",
    "BOOL_POR",
    "VSH",
    "SAND??SHADIBI",
    "AMP",
)


_CALI_MNEMONICS = CURVE_CATEGORY_MNEMONICS["caliper"]
_VP_MNEMONICS = ("DT", "AC", "DTC", "DTCO", "VP", "VP_MPS", "VPMS")
_VS_MNEMONICS = ("DTS", "DTSM", "DTSH")
_RHO_MNEMONICS = CURVE_CATEGORY_MNEMONICS["density"]
_GR_MNEMONICS = CURVE_CATEGORY_MNEMONICS["gamma_ray"]
_POR_MNEMONICS = CURVE_CATEGORY_MNEMONICS["porosity"]
_PERM_MNEMONICS = CURVE_CATEGORY_MNEMONICS["permeability"]
_SW_MNEMONICS = CURVE_CATEGORY_MNEMONICS["water_saturation"]
