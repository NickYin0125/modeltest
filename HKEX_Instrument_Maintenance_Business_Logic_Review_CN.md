# HKEX Instrument Maintenance：业务逻辑与独立审查说明

版本：2026-10-10。对应实现：[`hkex_instrument_maintenance_full.rb`](./hkex_instrument_maintenance_full.rb)，SHA-256：`97ed0dfe878070b8f38665f8a331ba5507fe500c60ff65a02b912d9626e0dd68`。

## 1. 审查对象和结论边界

此文件描述**当前代码实际执行的逻辑**，供第三方逐项复核。设计依据是《HKEX Instrument Maintenance Engine v4》，但设计文档中的语句、注释、测试数据及本文件均不能替代港交所当日的 Instrument Definition／合约规格。港交所公开规格可以证明常规期限和部分行权价规则；具体某日是否挂牌、官方 `OrderBookID`、调整后合约和临时增开系列，仍须以权威 reference snapshot 验证。

此次交付仅把原 V2 业务引擎和 V3 文件夹适配器装入单个 Ruby 文件，移除 V3 对 V2 的运行时引用和 V2 单独运行时的示例入口。`VERSION = "4.1-naming-v2"` 与 `ADAPTER_VERSION = "3.0.0"` 均保留；规则与公开 API 未作业务变更。独立复核确认，V2 的可执行代码段及 V3 的可执行代码段分别与原文件逐字一致。

阅读下文时区分四种性质：**交易所规则**（港交所公开条文）、**算法解释**（代码对条文的确定性实现）、**系统政策**（清算测试环境的保留／合并／拒绝策略）、**官方数据必需**（不能凭算法推断的具体上市系列或 ID）。代码内 `source_reference`、`source_basis`、`algorithm_basis`、`rule_configuration` 和审计记录提供线索，但第三方仍应核对规则版本、生效日及输入来源。

## 2. 每日 SOD 的输入、输出和时间语义

业务前提：Trading 每日把 OMD 风格 reference 实体 CSV 发送到 Reference；业务日期逐日滚动。运行时输入是一整个当日文件夹，至少须包含 `Underlying` 和 `InstrumentClass`，`Instrument` 文件可为空，以支持从空 instrument 集合建立基线。其他来源可包括 `CombinationLeg`、行情、官方对账集、人工 override，以及 Market／Exchange 等查找表。文件名本身无业务含义，`FolderConfig.sources` 用 glob、字段映射、枚举翻译和 lookup 声明实体角色；尤其 `FCInstrument*.csv` 必须排除 `FCInstrumentClass*.csv` 的重叠匹配。入口见 `FolderAdapter`（约 2432 行）、`FolderRunner`（约 3224 行）。

核心时间字段：

| 字段 | 当前实现的含义 |
| --- | --- |
| `as_of_timestamp` | 输入快照／运行时间戳。 |
| `source_trading_date` | Trading 所发数据所属交易日，也是股票期权收盘价的取价日。 |
| `target_trading_date` | 要形成的下一交易日开盘 instrument 集合；未传时用交易日历求下一交易日。 |
| `target_open_time` | 判断 `datetime_first_trading`／`datetime_last_trading` 是否覆盖开盘时点。 |
| `last_trading_date` | 合约原始最后交易日；命名与经济身份使用它。 |
| `effective_last_trading_date` | 可选的有效最后交易日；默认生命周期判断优先使用它。生产可由批准的 `LifecyclePolicy.expiry_field` 显式选择字段。 |

**开盘有效性**是 `expiry >= target_trading_date`，因此在目标日到期的合约仍可存在于该日开盘集合；`expiry < target` 才属于一般到期剔除。运行状态必须是 `ACTIVE`，reference 状态不能是 `SUSPENDED`，首末交易时间必须覆盖目标开盘，`metadata.availability_date` 不能晚于目标日，所属 Class 必须可交易。见 `Instrument#active_at?`、`MaintenanceEngine#current_valid?`（约 1820 行）。这套截点选择是**系统政策**，应由清算团队确认，而不是声称所有港交所产品均采用同一时点。

`hkex_real` 模式要求来源可追溯、假期完整且覆盖 source／target 的交易日历；目标必须是 source 之后的下一交易日，并要求批准且生效中的 `LifecyclePolicy`。`test_synthetic` 仅供测试，不构成真实上市证明。官方月／周期满日期可覆盖推导值，但不一致时须提供覆盖许可和证据。港交所发布可用于比对的[衍生品交易日历](https://www.hkex.com.hk/Services/Trading/Derivatives/Overview/Trading-Calendar-and-Holiday-Schedule?sc_lang=en)。

## 3. 实体关系和标识

输入按 `Underlying.CommodityCode → InstrumentClass.CommodityCode → Instrument.InstrumentClassKey → CombinationLeg.OrderBookID` 关联；`InstrumentType` 从 `Market + InstrumentGroup` 推导。`InstrumentClassID` 是类的数值／源 ID，`InstrumentClassKey` 是引擎的类关联键；`Symbol` 是展示／业务代码，`OrderBookID` 是交易 reference ID，不允许互相替代。Class 属性中的 `market`、`instrument_group`、`modifier`、`contract_size`、报价因子和精度用于校验 Instrument。OMD raw 整数价格按相应 decimal 元数据**只解码一次**；已解码输入需显式标记 `decoded_decimal`，以防重复缩放。类、instrument 和官方 reference 来源必须明确配置编码方式，文件夹输入的 Class 必须明确配置 `generation_mode`。见 `ODPAdapter`、`InstrumentClass`、`Instrument`、`FolderAdapter`。

单腿经济身份为 `(InstrumentClassKey, LastTradingDate〔如缺失才用 EffectiveLastTradingDate〕, StrikePrice, PutOrCall, FinancialProduct, Modifier)`；组合单独使用 `(ClassKey, combination, OrderBookID 或 Symbol)`。因此同一个旧合约有效到期日被修订时，不应仅因此重命名；两个不同 `OrderBookID` 也可能被判为同一经济身份冲突。组合只维护现有 reference 记录和 legs，不由标准期限／行权价生成器自动新建。组合 parent／leg 的引用、数量、方向和比例会被校验。

关键 enum 接受 OMD 数值／代码或规范化名称：`UnderlyingType` 为 S/C/I/E/A/M；`InstrumentGroup` 包括 future、American／European call/put、组合、汇率与付款货币；`FinancialProduct` 包括 O/F/U/A/C/P/E/S；reference 状态为 1/2，runtime 状态为 1/2/4，`Tradable` 为 1/2。完整表及拒绝未知值的转换器在 `Enums`（约 76 行）。

## 4. 自动生成的产品范围

默认 `RuleCatalog` 仅匹配下表六种 profile。路由依据 `Market + InstrumentGroup + UnderlyingType/Code`，不是按文件名或 `Modifier` 猜测 weekly；`Modifier` 非普通值的调整系列拒绝标准生成。Class 还须 `Tradable=YES` 且 `generation_mode=STANDARD_NEW_SERIES`；`ca_created=Y` 的 Underlying 需官方 reference。其余类别可通过输入保留或 `reference_only`／批准规则配置处理，但默认生成器**不能覆盖全部 HKCC／SEOCH 产品矩阵**。见 `RuleCatalog`（约 647 行）和 `Generator`（约 1395 行）。

| Profile | 匹配的核心条件 | 常规期限基线 | 行权价／行情 |
| --- | --- | --- | --- |
| `HKCC_HSI_FUTURE` | HSI、Future、Market 34 | 现货月 + 3 个逐月 + 3 个季月；再 3 个 6/12 月与 3 个 12 月，共 13 个期限 | 无行权价 |
| `HKCC_HSI_MONTHLY_OPTION` | HSI、European Call/Put、Market 34 | 与 HSI futures 同一 13 期限结构 | HKCC 对应合约月 Closing Quotation；短／长期不同 grid |
| `HKCC_HSI_WEEKLY_OPTION` | HSI、European Call/Put、Market 39；profile 从 2026-07-27 生效 | 本周与下周；与现货月月权同日到期则不生成该周权 | 前交易日官方指数收盘值；weekly grid |
| `SEOCH_STOCK_OPTION_MONTHLY` | Stock、American Call/Put、Market 20/143 | 现货月 + 3 个逐月 + 3 个季月，共 7 期限 | source 当日标的股票收盘价；Group B |
| `SEOCH_STOCK_OPTION_WEEKLY` | Stock、American Call/Put、Market 18/145；profile 从 2026-07-27 生效 | 本周与下周；与现货月月权同日到期则不生成该周权 | source 当日标的股票收盘价；Group A |
| `HKCC_STOCK_FUTURE` | Stock、Future、Market 2/141 | 现货月 + 2 个逐月 + 2 个季月，共 5 期限 | 无行权价 |

期限结构依据港交所当前的[HSI futures](https://www.hkex.com.hk/Products/Listed-Derivatives/Equity-Index/Hang-Seng-Index-%28HSI%29/Hang-Seng-Index-Futures?sc_lang=en)、[HSI options](https://www.hkex.com.hk/products/listed-derivatives/equity-index/hang-seng-index-%28hsi%29/hang-seng-index-options?sc_lang=en)、[weekly HSI options](https://www.hkex.com.hk/Products/Listed-Derivatives/Equity-Index/Hang-Seng-Index-%28HSI%29/Weekly-Hang-Seng-Index-Options?sc_lang=en)、[stock options](https://www.hkex.com.hk/products/listed-derivatives/single-stock/stock-options?sc_lang=en) 与[stock futures 规格](https://www.hkex.com.hk/eng/prod/drprod/sf/Documents/HKEX_Stock_Futures_EN.pdf)。这些页面同时说明交易所可酌情增开更长期限或系列，因此表中数量是**规则最低基线**，不等于实际总挂牌数。

月权及上述 futures 的默认到期日为合约月倒数第二个交易日；周权为该周最后交易日。现货月在该月到期日**之后**才滚到下月；因此到期日当天开盘仍属于现货月。周权遇月权同日到期时在生成前被抑制。以上以港交所公开规格为依据，具体假期／特别安排仍以官方日历为准。

## 5. 行情、行权价和代码

`MarketReferenceData` 通过 `(underlying_code, quote_type, quote_date, contract_month)` 唯一定位报价；缺报价、同键多报价、真实模式误用 proxy 均会阻断。股票月权／周权用 source 日期的 `underlying_security_close`；HSI 周权用目标日前一交易日的 `official_index_close`；HSI 月权用目标日前一交易日、指定合约月的 `hkcc_closing_quotation`。HSI 报价合约月：目标日早于本月月权到期日取本月；到期日当天及之后取下月。这是代码的**算法解释**，应对照 HKCC 日终文件和月结日实例确认。见 `ReferencePriceRule`（约 1122 行）。

计算使用 `BigDecimal`。ATM 取最近有效行权价，中点取低档；跨价格带时按进入方向的档距逐级遍历。股票月权 Group B 最低为 ATM 上下各 2 档（共 5 档）；股票周权 Group A 为上下各 20 档（共 41 档），不足 20 个正价下档时要求人工复核。港交所[2026 年 EQD/06/26 通告](https://www.hkex.com.hk/-/media/HKEX-Market/Services/Circulars-and-Notices/Participant-and-Members-Circulars/SEHK/2026/EQD_06_26_e.pdf)明确 Group A／B、周权 20 档最低数量及交易所酌情增加的权利。

HSI 月权短期按 ATM 上下各 10% 的**向外覆盖**，长期按 20% 的**最近端点**解释；HSI 周权按 10% 向外覆盖。HSI 月权短／长期间距分别为 `<5,000: 50/100`、`5,000–<20,000: 100/200`、`≥20,000: 200/400`；周权为 `10/25/50`。这些间距可与[HSI 月权规格](https://www.hkex.com.hk/products/listed-derivatives/equity-index/hang-seng-index-%28hsi%29/hang-seng-index-options?sc_lang=en)及[周权规格](https://www.hkex.com.hk/Products/Listed-Derivatives/Equity-Index/Hang-Seng-Index-%28HSI%29/Weekly-Hang-Seng-Index-Options?sc_lang=en)逐档核对；端点舍入方式是算法解释。真实模式还要求有效且获批的 strike-grid 配置；输出记录 profile、报价、grid、覆盖政策和规则版本。见 `StrikeRule`（约 994 行）。

`SeriesNaming` 使用明确配置的 3–5 位 HKATS root、期货／Call／Put 各自月份字母、年份末位、期权行权价，以及周权 `W` 加两位到期日；月权／周权命名月的选择不同。命名格式参见港交所[HKATS User’s Guide](https://www.hkex.com.hk/Services/Trading/Derivatives/Infrastructure/HKATS/HKATS-User-Guide?sc_lang=en)。**格式正确不代表该系列正式存在**：真实模式仍需 `OfficialCodeResolver` 给出官方 `OrderBookID`、`Symbol` 和来源，或对已有官方 reference 行保留其来源；`NamedCodeResolver(synthetic: true)` 的确定性 ID 仅供测试。调整／Flexible／特殊组合、TMC 等需权威代码或序号，不能按常规 root 随意编造。见 `SeriesNaming`（约 1206 行）、`NamedCodeResolver`（约 1304 行）。

## 6. 维护、合并和对账的确定性顺序

`MaintenanceEngine#maintain`（约 1595 行）的顺序如下：

1. 验证业务日期、日历、生命周期政策；规范化并校验 Underlying、Class、Instrument、组合引用及 enum／精度。
2. 依 `target_underlying_ids` 确定本次 Class。`scope_mode=target_only` 只返回目标范围；`global_merge` 还原样并入未目标范围的当前 Instrument，并查跨范围代码冲突。FolderRunner 默认 `global_merge`。
3. 若 `mode=standard`，对允许生成的 Class 生成最低必要系列；`maintain_existing_only` 不生成；`reference_only` 以带来源的官方 reference 为候选。类级 `SUPPRESS_GENERATION` 可阻止生成。
4. 对生成候选执行目标日有效性过滤。当前已存在但状态暂停、除牌或尚未生效的同一经济身份不能被算法候选重新激活；逐合约 `SUPPRESS_GENERATION` 也在此阶段执行。
5. 从当前范围保留有效记录 `S`，剔除无效记录 `X`；按经济身份补入缺少的有效候选 `E`。普通情形为 `Final = S ∪ (E \ S)`，**不会因每日最低行权价窗口移动而删除仍有效的旧档位**。
6. 执行带来源、选择器和生效交易日的 `FORCE_ADD`、`FORCE_REMOVE`、`REPLACE_EXPIRY`；再检查最终唯一性、正价／期限／状态、Class 一致性和组合完整性。
7. 仅在提供 `official_reference` 时执行对账；按 `OrderBookID` 优先、`Symbol` 次之匹配，产生 `MATCHED`、`ATTRIBUTE_MISMATCH`（带字段差异）、`GENERATED_ONLY`、`OFFICIAL_ONLY`。`GENERATED_ONLY` 是**对账差异**，并非官方已挂牌证明。

每个 KEEP／ADD／REMOVE／SUPPRESS／override 决策记入 audit；生成受阻还列出 `blocked.reason_code`。`FolderRunner` 默认 `fail_on_blocked=true`，阻断时不发布输出；按需写出 `result.json`、`instruments.csv`、`audit.csv`、`reconciliation.csv`、`manifest.json`、`diagnostics.json`。输出使用暂存目录后原子发布；manifest 记录源文件 SHA-256、记录数、配置 SHA-256 和运行日期。`MaintenanceResult#projection` 只给简化的 instrument code＋expiration date，`full_model`／文件夹产物保留完整实体和审计，以便追查。

## 7. 第三方应逐项验证的业务问题

| 检查项 | 可复核证据／预期 |
| --- | --- |
| 业务日期及开盘截点 | 用生产假期表、source／target 日期和开盘时间，核对到期日当天保留、到期次日剔除、首末交易时间。 |
| 输入实体和比例尺 | 给出一份真实 OMD 文件夹、字段／enum 对照表、raw／decoded 标志；确认 `PriceQuotationFactor`、`ContractSize`、`StrikePrice` 和 `TickSize` 不重复缩放。 |
| 产品授权范围 | 对每个 Market／Class 标记“自动生成、只维护、官方 reference”；尤其 MHI、HSCEI、HSTECH、汇率、金属、特殊组合不能因为表中没有 profile 就认为不存在。 |
| 期限与行权价 | 以一日真实官方 series 对每个支持 profile 核对逐期限 expiry、ATM、跨价格带档距、月／周碰撞和酌情增开的额外档位。 |
| 命名与 ID | 按 Class 的真实 HKATS root／精度核对 Symbol；确认新增行有权威 `OrderBookID` 和来源，调整合约／Flex/TMC 不靠合成名称冒充官方记录。 |
| 维护和对账 | 对“空 Instrument 集合”和“已有大规模非到期集合”分别执行；复跑同一快照检验幂等；解释每个 `GENERATED_ONLY`／`OFFICIAL_ONLY`／属性差异。 |
| 规则治理 | 提供生效日期、批准者和来源版本；覆盖交易所临时通告、公司行动和系统开盘截点政策。 |

当前实现的重点边界：公开期限／行权价表只是最低要求，不能单独推导全市场的实际在册数量；已有有效的非标准档位会被保留。`reference_only` 是把官方记录作为候选补入，并不自动删除不在该官方列表中的有效当前记录；官方对账依赖外部权威快照，文件夹适配器不会自行下载或证明其完整性。`global_merge` 在指定部分 `target_underlying_ids` 时会原样保留非目标范围的记录，**不会替它们逐一重新判定到期状态**；若交付物须保证全局无已到期 Instrument，应覆盖所有 Class 或另设完整性闸门。空 Instrument 集合可以开始生成，但只有满足已支持 profile、行情、日历、批准配置和官方代码映射的 Class 才能在真实模式完成；否则会阻断。没有真实生产文件夹和同日官方 Instrument Definition，就不能宣称对某日全 HKCC／SEOCH 宇宙的数量已经验收。

## 8. 本次结构变更与测试证据

| 检查 | 结果 |
| --- | --- |
| 单文件隔离加载 | 把 full.rb 单独复制到临时目录，原 V2／V3 文件缺席时可 `require`；引擎和 FolderRunner 同时可用。 |
| 代码差异 | 除文件头注释、移除 V2 的单独执行 demo、移除 V3 `require_relative` 外，V2／V3 可执行代码段分别与原文件完全一致。 |
| 语法 | `ruby -c hkex_instrument_maintenance_full.rb`：`Syntax OK`。 |
| 当前 V2＋V3 测试直接针对 full.rb | 164 runs、827 assertions、0 failures、0 errors、0 skips；Luna 与本次独立验收均用临时改写的测试副本加载 full.rb，且临时副本已删除。 |
| 文件夹集成演示 | 使用合成 fixture 经 full.rb 的 `FolderRunner` 得到 49 条输出（KEEP 2、ADD 47），演示数据的官方对账为 MATCHED 2、GENERATED_ONLY 47。此数字仅证明示例路径可运行。 |
| 更早的两组顶层测试 | 它们单独加载更早的 `hkex_instrument_maintenance.rb` 时通过（31/102、43/123）；若强制它们加载 full.rb，会有 4 个旧预期不符。相同 4 个问题强制加载原 V2 时也存在：2 个真实模式缺少批准生命周期政策、1 个 adjusted modifier 的 block reason、1 个 ODP quotation factor 比例尺预期。因此不是本次合并的回归，也不应为了旧测试更改现行 V2 业务规则。 |

上述测试主要使用合成数据，证明可执行路径和结构等价，**不证明港交所当日上市全集正确**。独立审查的最终证据仍须来自同一业务日的真实 Trading 输入、批准规则配置、官方 Instrument Definition 和对账差异处置记录。
