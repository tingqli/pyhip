# QSA 优化日志

## 2026-09-27：32buffer / 128samples重新测试

- [本轮报告](../../../../mytest/mydata/qsa_buffers32_20260927_01/README.md)：10→32独立Q/K/V/indices/O，每实现128samples、每buffer4次、2warm，30例TP2/4/8×real3/47/low/high/short×direct/full，共7680raw、90门禁通过。永久benchmark默认32/128，CLI可覆盖，runtime不改。
- 加载上轮formal冻结phase/pipeline，不用后续ATT改过的同名源；mainAST仅formal(10,50)→(32,128)，每例采样前directELF/.text/资源匹配上轮，原选择/分母/路由/计时器不变。scratch仍每workspace复用。TP4/8是TP2激活headslice非真实多卡capture。
- Prepared真实L3TP2/4/8 base2503.593/2462.373/2452.253→phase2499.493/2461.313/2449.453µs，候选pad135.125/137.221/137.886T；L47base2498.914/2459.073/2450.092→2495.654/2459.353/2448.013，pad135.333/137.331/137.967T，仍未160。短M候选慢3.64/3.79/4.16%，full短慢2.56/2.34/2.30%，不采用。
- 真实TP4full本輪L3base3449.198/phase3446.178、L47base3428.397/phase3435.598us。第一32sample轮约2.45ms，第二约3.2ms，第三/四约3.6ms；两实现均逐轮变慢，路由完全相同，全部longtail保留。无kernel期间频率/热/阶段遥测，原因未确定，不将相对旧10buffer~40%增幅解释为buffer数量因果。
- 当前配置CPU检查通过默认/CLI覆盖/非法bounds/TP转发，正常44test函数AST不变，未再次跑44pytest；最初CPUmock继承mask失败独立记录，GPU矩阵全程unmasked无失败。1663证据文件hash，runtime/index未改，无硬件/新profile/SGLang/部署/commit/push。

## 2026-09-27：TP2/4/8永久覆盖、scratch无回读与160填充T未达

- [完整报告](../../../../mytest/mydata/qsa_direct_160t_20260927_01/README.md)：永久测试入口正常/perf/CLI默认TP2/4/8H12/H6/H3；真实TP2capture裁剪Qheads的TP4/8明确标derived，不称新分布式capture。新增scratch按CPUshape容量/稳定ptr/禁Tensorcpu/item/tolist/bool/热empty_like/graphindicesV更新。44正常passed213.79s，24perf deselected；runtime内核和1.7分流未改。
- PK+PV=1024*N*HK bytes，N12000HK1=12,288,000B(11.719MiB)，N32048=32,817,152B，只有实际N262144才256MiB。每layout/stream工作区复用，不从activeGPU结果定容量；全union仍保留buffer。普通LRU最多8，captured指针生命周期使总量可超8；不夸称整个QSA无临时tensor或无限制池。
- 160目标prepared含每次pack，真实Fpad337744756736→2110.9047296us门槛。65label/20探索收据316raw尝试跨tileQK-exp/PV-summary、PV分组、K流式、LDS半输出、多wave错相、globalLDS环、AGPR、cache/swizzle/alignment等；最接近phasepg8长输入约2500us，未到160。wideBN另声明非bitexact且按新padding；LDSring数值错误拒绝。
- 正式30例3000raw：2scope×5case×3TP，各10buf2warm50sample/实现，原timer，全gate通过。候选preparedL3TP2/4/8=2500.934/2458.673/2448.493us，135.047/137.369/137.940paddedT；L47=2498.173/2456.714/2447.354，135.197/137.478/138.004T。长形状与base≤约0.22%差异，M64慢3.55–4.16%，full短慢2.33–2.71%，不采用。完整TP4/8更多union，不以full171填充T替代direct160。
- 新ATT实际phasea78ee1bd2b58cdf73fbb67e3c95c84ee92c6db1008f78600962d0c8561dc4baf匹配formal，att_phasepg8/direct_3626_shader_engine_0_28.att。125wave全64/512000MFMA，原28fd49%附近→49.06%、VALU7.97→6.95、VMEMwait1.40→3.15，Vhi消费waitmedian136cyc抵消。针对性earlyVhi反事实3084–3121us更慢保留，不反复采旧候选。
- 全审计50性能收据3316raw，47成功141普通gate；3个0raw失败(wrapperprepare缩进/ArithValue类型/LDSring数值)保留。44test与runtime原hash/index验证；没有SGLang/硬件/stage/commit/push/部署。目标160明确未达，全部数据含失败/长尾保留。

## 2026-09-27：完整QSA的union/direct时间占比（源码只读）

- [完整报告](../../../../mytest/mydata/qsa_breakdown_20260927_01/README.md)：当前17/10分流、TP0H12，6例10buffer×10trace调用；rocprofv3 kernel/marker/copy，无ATT/PMC。时间占比按同60calls中各case阶段总时长/完整event总时长，不按query比或独立中位数推算。profile均值不替代普通50sample性能。
- L3M12000 union28.47%、direct53.04%、pack0.36%、dense6.06%；L47M12000 union31.61/direct50.09/pack0.34/dense5.85%；L3M11888 union59.41/direct21.19/pack0.31/dense6.21%；L47M11888 union41.85/direct39.15/pack0.35/dense6.09%。direct含pack分别53.40/50.43/21.50/39.51%。其它为恢复/校验、构表/排序和事件/发射间隙，所有长尾保留。
- 只算union+direct attention时，两者时间比例分别34.93/65.07%、38.69/61.31%、73.71/26.29%、51.67/48.33%。整个query batch的union/direct行比例分别32.74/50.17%、35.58/47.33%、66.53/16.22%、47.10/35.65%，另dense17.09%或17.25%；不是同分母。
- 高/低重合M2048H12路由全union/全direct仍有gated空kernel检查，非零启动时间。全部820dispatch（60原timer event前spin单列，760QSAkernel），CSV/JSON/correlation/纳秒/顺序无重叠核验；0copy/未分类/PMC，18普通门禁GPU2a4PTLEnabled；匹配当前正式ELF、实际trace输出/原参考/输入/planhash通过。初次CPU浮点微秒加法舍入导致零gap比较失败保留，改纳秒整数后通过，0GPU重试。

## 2026-09-27：当前packed与raw分支正式对照，生产源码只读

- [本轮报告](../../../../mytest/mydata/qsa_pack_vs_raw_20260927_01/README.md)：8例×5路径×50sample=2000正式raw，另20smoke；10buffer/2warm/GPU2a4/PTLEnabled，27门禁通过。prepared direct同query集合，packed每次pack在计时内；完整分qsa_raw_same_routes/current_packed/raw_auto，控制1.7路由与原生rho4差异。
- Prepared raw→packed：L3 sparse9949 3163.336→2513.993us(-20.527%，配对median0.828501即-17.15%)；L47 3051.676→2510.314(-17.740%)；lowH12 712.884→590.343(-17.189%)，lowH6 704.864→582.563(-17.351%)；highH12 692.643→574.123(-17.111%)，highH6 686.623→571.243(-16.804%)；M64H12 136.301→111.801(-17.975%)，H6 136.421→110.801(-18.780%)。当前真实packed99.665/99.811有效T，不到100，不用旧100T成绩替代或重采。
- 同路由完整raw→packed L3 3411.319→2909.096(-14.722%)，L47 3418.898→3001.036(-12.222%)；lowH12 810.925→689.783(-14.939%)，H6 800.124→674.904(-15.650%)；high全union相近；M64H12 168.221→145.320(-13.614%)，H6 166.661→143.921(-13.644%)。raw原生rho4真实全union3342.898/3251.997，非同路由；M64H12尾4行union导致526.982µs，不把72.42%全调用差距归功pack。
- raw计划生成4wave/CTA，AST只覆写当前prepare packed=False作独立逐字段/别名核验，timed运行当前rawGPU源码不变；不是清空一wave packedplan、改tensor形状或选中token。8例direct输出bitexact，五路径实际timed-output/reference/guard/input/planhash通过；本轮没重跑27pytest。
- 真rawELFe27713d7…248VGPR42SGPR32KiBLDS，packed28fd66d3…双kernel26/250VGPR、0/8KiBLDS，均0spill；gatedraw4b23c7ff…packed3ca79d04…。完整分支差异含几何/索引预取/归约/优先级，不是pack单因素归因。全部source/index未改，旧证据/硬件/SGLang/部署不动，无新ATT/PMC；新增报告与脚本只在mytest/mydata。

## 2026-09-26：重新标定union/direct，完整调用全真实输入改善

- [本轮报告](../../../../mytest/mydata/qsa_route_20260926_01/README.md)：入口为当前100T packed direct＋rho4。最终packed路径按实际M128/N64 union与每queryM16/N32 direct的填充工作比较，$10W_U≤17W_D$选union；GPU整数160*cdiv(U,16)≤17*sum(cdiv(tokens,32))。实际DirectPlan.packed_key决定启用；raw/ragged继续rho4，dense2051/原BQ/所有attention不变，没有新增kernel。
- 最终16例1600raw(10buf2warm50sample)全通过；8份真实完整QSA快1.50%–13.02%。TP0 L3/12000 3338.018→2903.916(-13.005%)，L47/12000 3247.318→2993.316(-7.822%)，L3/11888 2882.395→2839.096(-1.502%)，L47/11888 3135.257→2881.116(-8.106%)；TP1分别-13.020/-7.652/-1.583/-8.127%。H12共享75% 1081.586→680.903(-37.046%)；其余合成端点变化≤0.22%。所有longtail原样保留，不是每sample必胜。
- 真实M12000新路由dense2051＋L3 union3929/direct6020、L47union4269/direct5680；M11888L3 union7909/direct1928、L47union5599/direct4238。不是旧direct0，也不是此前forced-direct100T计时边界；TP0L3完整有效95.187/填充141.917T。
- rho1.75/pad1.4虽两主例探索约2903µs，但H12高重合528→596us，拒绝；BQ8使H6高重合318→535，拒绝；首版1.65正式1600raw中两M11888/L3约退化0.5%，保留后重标定1.7；GPUquota和recover/scatter融合因完整调用不受益拒绝。M11888参与反馈，不再称独立留出。
- 最终27正常测试59.07s全通过，新增GPU整数cutoff两侧/r1..32/HK2/raw/ragged/图内indices与V变化；原.02容差。attention整ELF/资源不变，direct源码c31466/packed88c6原样；recover/scatter/masks/order机器码.text不变（后三者debug位置导致整ELF不同）。仅compact planner.text变化，L3M12000ELF4ef15299…；qsa源5e2bdce5…union811dafc2…testb65ecbe6…。
- 全审计74run3916raw=716探索+1600首版+1600最终；73成功run219普通门禁全部GPU2/a4PTLEnabled；1模块导入失败0raw保留，CPU整ELF审计错误独立记录。8715证据文件hash，新独立插件包8文件源/内部导入通过，未部署，无SGLang源码/硬件写/stage/commit/push，13项用户index未变。

## 2026-09-26：DS交织与调用内KV预排，真实Direct有效100T

- [完整研究](../../../../mytest/mydata/qsa_direct_100t_20260926_01/README.md)：入口ed471dbc/ELFe27713d7（真正packed-FP32关闭）。先采新基线ATT；单纯D16 DS分组/lookahead、PV包装、cache预取/优先级最高约86.6有效T；宽LDS/跨tile流水等退化全部保留。
- 最终将KV在每次调用内按4token块预排，单编译host launcher提交pack+attention；计时包含两kernel，不缓存旧KV或移出准备。pack每CTA8块、V用4条DS交换，attention单query wave，无原每BN32的64条K数据bpermute/64条Vperm；索引预取、并行max/sum DS、整wavealpha==1跳rescale、PVsetprio2→0。原RNE/选择/加法树保留。
- 最初正式99.965207T/2506.453514µs未达标收据保留。后改宽pack与V向量读取+DS交换，候选Layer3/47达到100.131/100.271T；整理后四shapeELF/.text逐字相同，精确生产独立400raw：L3 3039.856→2502.334µs(-17.68%)、100.130有效/134.972填充T；L47 3035.136→2498.114(-17.69%)、100.299/135.200T；lowH12 712.364→588.963(-17.32%)、87.573/118.046T；lowH6 707.684→584.843(-17.36%)、44.095/118.878T。100有效门槛只真实两主例通过。
- 新私有模块_direct_packed.py，direct.py只在单请求/NK%4==0且offset余量安全时启用，非对齐/ragged保留raw。每次pack重建；graph修改K/V和union/direct active切换均验证；gated全union跳过KV搬运。输出未来NaN按BF16逐token屏蔽。没有改公共QSA/rho4/dense2051。
- 完整公开QSA另400raw：L3 3344.177→3340.178、L47 3247.398→3245.537（direct0，视为相近）；lowH12 812.344→686.323(-15.51%)、lowH6 803.004→675.123(-15.93%)。旧38/35ms、新真实72ms长尾全保留，不解释为时钟或筛样。
- 最终真实ELF28fd66d32c3c4fbf0324091a79a75bb5d808bdf98782780a3dfecc8bc126cfd4含pack(26VGPR/28SGPR/0LDS)与attention(250VGPR/42SGPR/8192LDS)，均0spill。最终ATT att_final/direct_18308_shader_engine_0_175.att，第三次attention前有pack，捕获ELF等于正式生产。
- ATT物理SIMD内部MFMA40.61→48.75%、DS指令19.26→1.02%、DSwait4.81→1.62%；非HBM/整卡利用率。入口124wave(93×65+31×64)，最终125wave(全64)，采样任务集合不同，不能直接比总MFMA声称工作减少；真实有效分母仍250558144512。
- 最终26QSA正常回归77.94s通过，新增1测试含HK2/NaN/guard/每次scratch刷新/graph改KV/动态路由。资源夹具严格检查两个kernel。插件build清单纳入私有模块，独立新target内部导入/hash通过，历史包与SGLang源码未改。
- 全审计69收据1980raw=1300正式(含首次100raw未达标)+680探索，5次编译失败0raw保留；全部普通门禁GPU2/a4 PTLEnabled通过，整个13暂存文件index不变；没有频率/功率/NUMA/PMC/部署/stage/commit/push。

## 2026-09-26：packed FP32真实禁用与类型化属性修复

- [完整报告](../../../../mytest/mydata/qsa_direct_packed_20260926_01/README.md)：原direct f40c7925已写通用passthrough，但安装版GPU→ROCDL转换丢失该属性。CPU最小IR复现和实际有/无属性ELF对照确认，两者均800d13cf、64条v_pk_mul_f32；未重复给同ELF计时。
- 先用独立compile-hint隔离的作用域ROCDL目标禁用验证（安装包未改，省略默认target避免append双对象误选），packed FP32为0。再验证类型化`llvm.target_features`可保留到LLVM；最小生产集成仅替换调用属性+解释注释，AST其它节点相同，无全局后端包装。
- 正式10buffer/2warmup/50sample四例400raw：Layer3 3119.916→3038.456µs(-2.61%)，Layer47 3116.857→3033.736(-2.67%)，低重合H12 733.884→713.244(-2.81%)，H6 729.464→707.904(-2.96%)。关闭后有效/填充T：82.462/111.157、82.591/111.330、72.314/97.477、36.430/98.212。另32探索raw保留，H12探索更慢也未删除；正式H6最大配对1.1945等慢样本保留。
- 原64packed乘法→128额外scalar乘法，静态v_mov_b32_e32 59→11，VGPR246→248；VMEM/MFMA/DS指令数相同，LDS32768/SGPR42/private及spill0。只说明实际codegen变化，不将静态数目作为精确时延归因；BF16概率/输出pack不是该开关的目标，原RNE未改。
- 当前生产源码ed471dbcf8aaa8c54c393b602a2e9495e68f5f2682d3d57a68760d2e6ae36412；真实两层ELFe27713d7e3613285e9c7e5f90bcec14032b3e84ec077f528a402a38ad45f1eac。精确生产四case ELF/.text/资源逐字等于正式测量目标版；实验及生产各原25QSA正常回归PASS，不是50个不同定义。整用户13文件index不变。
- 本轮没有新ATT/PMC或完整公开QSA性能；此前800d13cf的ATT/39.14%模型只对应旧版，不能重标为现在的packed关闭版。真实两层自然路由仍direct0；所有新增研究文件在PyHIP mytest/mydata，未改SGLang/硬件/安装编译器、未stage/commit/push。

## 2026-09-26：用户16×4布局尝试与提前V/消费者K等待

- [新轮报告](../../../../mytest/mydata/qsa_direct_16x4_pipeline_20260926_01/README.md)：入口direct1172dada…，隔离实现token=lane&15/channel=lane>>4、16B/lane、0/16/32/48覆盖64B，无K数据转置；同时测试V0/全部V在QK前及K等待移至consumer。参考union/dense内存提前原则，但不改变RNE/softmax/选择。
- Layer3四因素探索base3196.556/native-late6012.292/coalesced-early3188.017/native-early6041.771µs；全部V预取native6210.493也不胜出。正式4例600raw：native-stream Layer3/47≈6041µs、H12/6≈1360/1357µs，较入口慢79–89%，拒绝生产采用16×4。
- 最终只整合保留原合并K布局的earlyV0+consumer-Kwait，源码f40c7925d874a1af48c61509b4c2078ced86a2dfb26d53743c9ac9caa02bee1a。因精确ELF不同，单独正式400raw：L3 3185.997→3119.496(-2.09%)，L47 3187.277→3115.037(-2.27%)，H12 762.704→733.004(-3.89%)，H6 758.444→728.604(-3.93%)。有效/填充T分别80.320/108.269、80.435/108.424、70.364/94.849、35.395/95.422。
- 完整QSA另400raw：低重合H12 857.065→826.625(-3.55%)、H6 846.485→817.584(-3.41%)；真实L3/47约3341/3243µs前后相近（仍dense2051union9949direct0）。58.168ms旧H12和36.374ms新H6离群值保留，无删尾/动态时钟因果声明。
- 三版Layer3ATT各124完整wave/513856MFMA、KVx4load数相同；native16×4每轮少64数据bpermute，但VMEM指令72.36%、MFMA19.50%，K/V issue-stall中位56/60cyc。最终合并流水MFMA39.14%、VMwait1.79%；PC533vm8/PC712V0vm0/PC424V1vm16中位均4cyc；PV1前earlyzero已消失。所有占比局部模型，不代表HBM字节/全GPU利用率。
- 最终246VGPR/42SGPR/32KiBLDS/spill0，realELF800d13cf…/.textd9f18114…与正式+ATT相同。原未变数学helper/索引/GQA/尾部合同；原精度、跨实现所测输出bitexact。三实现各25普通测试PASS，首次候选符号夹具1error原样保留后改规范符号名，不改测试。
- 独立审计20收据1468raw=1400formal+68explore，全gatepass、输入/计划/实际timed-output/source/ELF/wholeuserindex核对；测试文件、union/dense/linear/qsa/common本轮未改。无SGLang/硬件/stage/commit/push，旧证据只读。

## 2026-09-26：补齐完整正式矩阵与重合区 / 三分支比较

- [独立新轮报告](../../../../mytest/mydata/qsa_formal_matrix_20260926_01/README.md)：生产源码只读，旧direct75270c…对照已交付1172dada…，16case/56组合/2800raw全部完成。每例新进程，10独立buffers/2warmup/50samples，轮换正反实现顺序，原cudaPerf；实质CPU输入/输出/逐bit membership-mask审计后立即门禁。48次门禁均use0%、VRAM≤3%、PTL Enabled，不sleep/轮询/重试；旧失败收据不改。
- 同QKV低重合H12/H6：direct1422.228/1418.728→759.644/755.444µs，约-46.6%；完整QSA1523.549/1512.828→856.445/843.364µs(-43.79/-44.25%)。强制exact union2125.952/1898.950µs，direct快2.799/2.514×。
- 同QKV shared高重合H12/H6：direct742.124/738.224，union405.602/214.561µs，union快1.830/3.441×。强制union仅研究RHO=inf；正常auto0/2048/0，完整新旧约相同，数值集合不变。
- 真实L3/L47整体稀疏9949行：direct6313.794/6311.913→3195.357/3190.357µs(-49.39/-49.45%)；union2787.815/2694.094µs仍快。有效/填充T：direct78.413/105.699与78.536/105.864；union89.876/213.370与93.003/210.471。正常dense2051+union9949，完整QSA3340.377/3245.437µs与原3338.857/3249.257相近。
- 按CPU原indices、原BQ10选择固定2000query窗口，不看时延：高[2060,4060)、低[10000,12000)，互不重叠。高rho1.4047/1.4005：direct710.024/712.223，union423.142/425.603µs；低rho2.7613/2.5219：direct736.044/740.023，union817.565/773.504µs，direct省9.97/4.33%。原rho4仍选union，下一步可校准gate但不能从窗口平均rho直接定全局阈值；窗口重新prepare的direct CTA不是原从2051开始的切分。
- Dense同集合前缀：H12/2048 dense161.240/161.841、union171.361/171.601；H12/2051 dense179.321/180.041、union175.581/175.721µs。H6/2048 dense102.760 vsunion105.801；2051 dense108.161 vsunion110.041µs。direct全部约392–398µs；dense仍需保留，长稀疏区dense=N/A（语义不等价）。
- 独立审计复算raw/工作量/三门禁/源与实际ELF，确认H12/H6direct仍上轮交付精确ELF、全部spill/private0。实际计时输出前后逐位、原QKV/indices/私有plan hashes不变；高低合成QKV hashes相同，只有indices不同。完整用户13文件stage index不变，无SGLang/硬件/生产分流/commit/push修改。无新ATT或模型profile。

## 2026-09-26：Direct K lane 合并读取与 PV 流水（新一轮）

- [本轮证据](../../../../mytest/mydata/qsa_direct_pipeline_20260926_01/README.md)以当前direct75270c…为入口，独立冻结全部用户index；不是上一轮最初direct基线。
- 主改动：K从token=lane&15/channel=lane>>4改为token=lane>>2/channel=lane&3，相邻4lane读连续64B；QK入口逆DS映射恢复原操作数。V继续复用K的block地址，BN32/M16/每链累加与RNE不变。
- 保留64block/wave寄存器索引缓存、非负索引移位mask、原生DS依赖、PV之间分批K预取。跨tile携带V、PV成对转置、逐MFMA的K请求等不一致/过小收益候选撤回。
- 正式10buffers/2warmup/50samples交错：H12 1427.627→763.204µs(-46.54%)，67.580有效/91.096填充T；H6 1424.308→758.704µs(-46.73%)，33.990有效/91.636填充T。两例门禁分别通过，共200raw。
- 随后real3 PRE gate use6%/VRAM3%失败，0raw；停止全部普通timing，不重试。正式收据保持complete=false，real47和formal auto未运行。完整QSA探索低重叠-39.3/-39.8%，真实两层direct0整体相近，不能称正式或端到端收益。
- 最终H12 ELF014efdda…、220VGPR/50SGPR/32KiBLDS/spill0，与新ATT相同。入口28wave/116032MFMA，新24wave/99456MFMA；每query64/65轮工作不变。完整waveblock读130/132→9，QKV向量读2072/2104不变。内部物理SIMD MFMA模型15.60→34.54%，VMEM指令46.26→12.26%，新增DS开销；不是HBM流量归因。
- 25QSA正常通过；增加所有64block缓存边界、HK2/NaN物理尾，另real3/47forced检查通过。两合成/两真实相对入口全部bitexact。18harness收据424raw=200已完成主例正式+224探索审计通过，完整矩阵仍失败停止状态。
- v01不透明DSasm+分离wait导致消费VALU越过等待，真实ISA/短N诊断确认；合并leaf后修复，最终使用原生DS，不放宽容差。编译类型错误、错误数值、路径错误和所有慢样本保留。
- qsa/dense/union/linear/共享helper本轮未改；用户13个staged文件完整index不变。无SGLang源码/部署/硬件/提交操作，旧证据只读。

## 2026-09-26：Direct授权优化与恢复排序复用

- 用户已确认继续direct性能分析。新[完整报告](../../../../mytest/mydata/qsa_linear_direct_20260926_01/README.md)记录低重合M2048/P30000/H12/H6和真实capture诊断；自然auto分别全direct与全union稀疏行，不混用路径。
- M有效比例H12=75%、H6=37.5%，N约98.91%。原direct有效32.75/16.44T，填充约44T；M浪费无法单独解释全部差距。4query选择复用低重合仅1.105×、真实两层2.44/2.54×；逻辑请求不是HBM流量。
- 保留K地址广播复用V、双QK独立链交织、下一K对与末PV重叠；d05探索H12/H6 1574.148/1573.528→1424.227/1420.488µs(-9.52/-9.73%)。d01fence、d03早V、d06两wave、d07集中PV操作数均不保留。
- 恢复重复检查已有sort，写入私有规范block序后由direct别名引用，删去第二个sort kernel/launch/M×512 buffer。公开indices只读，图每次刷新；原valid/duplicate/tail/padding校验不减弱。
- compact+mask融合正确但无一致收益，低重合退化，union整文件恢复原SHA。p01完整调用探索低重合约11%、真实约1.1%改善，但当时含后来撤回d07 PV，不重标成最终版本。
- 最终formal direct PRE gate use6%失败，0raw，formal auto未运行；停止性能不重试。linear B1causal正式200raw通过，8192为224.297有效/227.774填充T。最终direct仍缺有效正式时延。
- 原/最终ATT各28完整waves、116032MFMA；物理SIMD内部MFMA模型占比13.09→15.60%，VMEM wait58.90→33.42%，但VMEM指令区间增大，不能归因HBM。每BN32索引buffer读4→2、KV向量读32不变。
- 25QSA+28MHA=53功能通过，另加强错误检测1次复验；long noncausal Full整ELF保持d043ace…。17收据424raw审计通过，整个用户index未改，无SGLang/硬件/提交操作。

## 2026-09-26：选中块图与 Dense/MHA 性能对齐

- [新报告/交互图](../../../../mytest/mydata/qsa_mha_parity_20260926_01/README.md)：原两层M12000，X为4-token KV块，Y为query。蓝=indexer完整块，绿=因果尾，橙=union额外计算；N64虚槽单列，M128空行不冒充query。7200万标签像素与94无损分片已核验。
- dense保留因果长任务优先/蛇形worker配平；64对齐复用native DMA，非对齐保持完整VOFFSET；DS/DMA地址叶子融合另通过d02/d04正式50样本双层≥1%门槛。MFMA/数值表达式/waits不变。
- 同语义原生MHA/原dense/优化dense/union，Q2048和2051、两真实层、10buffers×50samples共800raw通过。2048原dense292.462/292.601→160.961/162.041µs；2051原236.662/236.761→179.201/179.641µs。原生同输入约293.4/225.3µs，union约171/175µs。不是长noncausal221T的同shape声明。
- union short-grid2/common-budget3/masked-u2均退化，已全部撤回，恢复入口SHA。dense静态common展开也不保留。没有direct优化或分析。
- 完整public QSA正式300raw：原5.249/5.055ms→优化dense5.156/4.945ms→no-dense5.082/4.880ms。本轮29–31/50样本>4ms，全部保留；不可用旧场次3.3ms代替、不无证据归因动态时钟。
- 短真实prefix正式轮完成M1的150raw后END门禁use13%失败，其余5形状未运行，不重试；全部150raw无效保留。仍保留dense分支、不硬编码新阈值。
- 24正常测试pass（新增2048/2051、dense任务prefix覆盖、HK2/ragged/NaN尾/guard graph），8perf deselect；审计1642raw=1300正式有效+192探索有效+150失败无效。完整用户staged index未变，未改SGLang/native MHA/direct、未commit/push。
- **TODO待确认：direct为何慢（重复访问、M方向浪费等仅假设），在用户确认1–2后继续。**

## 2026-09-26：Dense 是否仍需保留

- 纠正此前“当前2.6–2.8倍”：原始indices按冻结plan逐组unique复算，旧BQ8为Layer3/47 **2.774629/2.642250**；当前BQ10 **2.374045/2.263059**。分母为dense2051之后9949行，不能称全M12000。
- 分解为并集block扩张×N64舍入×M128填充。BQ8约2倍×1.007×1.333；BQ10约2.1–2.2倍×1.007×1.067。是work accounting，不是HBM流量或时延。
- 同一真实2051行因果前缀、prepared attention：dense236.841/236.762µs，union174.621/175.241µs；填充30.702G→28.790G FLOPs，填充吞吐129.633/129.676→164.870/164.286T。当前union已胜出，无需为追赶此dense路径再改内核。
- 完整真实12k public QSA（含恢复/校验/建表/排序/direct）：保留dense3434.498/3339.058µs，取消分流3300.858/3201.678µs，快3.89%/4.11%。GPU2/a4、10独立buffers/2warmups/50samples、AB/BA、400raw，全部门禁与输入/workspace哈希通过，长尾保留。
- 短请求M1/64/512探索2buffers/4samples仍dense更快；预定正式短矩阵在M1的**结束**门禁use10%失败，100raw无效保留，剩余case未执行，不重试。不能从探索硬编码全局阈值或无条件删除dense。
- 结论：已测M12000/P0长请求不需要额外dense前缀；全局dense实现仍保留，生产路由未改。无dense实验14项与生产21项通过。首轮两失败来自shared测试生成器在跨2051时priority未初始化；修复并在既有边界case加入回归，不改变计算内核或输入capture。
- [本轮完整报告](../../../../mytest/mydata/qsa_dense_branch_20260926_01/README.md)、[原始正式结果](../../../../mytest/mydata/qsa_dense_branch_20260926_01/formal_real/result.json)、[独立审计](../../../../mytest/mydata/qsa_dense_branch_20260926_01/audit.json)。未改SGLang/设备设置/四张staged CSV，没有新部署或全模型profile。

> 2026-09-26更新：真实TP0两层M12000，union填充213.148/210.036T通过210T，但有效89.783/92.810T未达到100T。
> 当前G12为BQ10/gridCU、GPU任务sort/snake；最终保留mask S0预取/S3 bit-select。详见文末及[本轮证据](../../../../mytest/mydata/qsa_union_210t_20260925_01/README.md#L1)。

> 当前API以[README](README.md)为准。2026-09-25整理后默认auto/rho4、dense2051、sorted BN32，
> BQ/grid固定为已验证值；仅保留auto与forced union。旧章节中的调参命令和模块名是历史记录。
> [本次代码/测试/机器码验证](../../../../mytest/mydata/qsa_cleanup_20260925_01/README.md)；未改写旧profile或旧性能结果。

## 2026-09-25：TP2 / TP4 三路精确分流

### 范围与预先声明的验收

- 主性能形状 M=12000；no_prefix P=0 / KV12000，chunk_prefill P=12000 / KV24000。
- TP2：Q12/KV1/D256；TP4：Q6/KV1/D256。TP8：Q3/KV1，仅正确性，不设性能门槛。
- BF16；512 个四-token 块和 0..3 个尾部，输出2051槽合同不变。
- 不改变逐-query Top-K、不扩大可见集合、不修改数值容差；有效 FLOPs 仍为有效token数×4×本地Qheads×256。
- shared 输入 selection_group 固定32，与 kernel BQ 解耦；同配置 seed17、索引哈希固定。
- 主目标仍约200有效TFLOPS；无法达到时明确失败，不用shared代替independent，不把无效并集MFMA计入分子。
- 正式原 cudaPerf、10独立buffers、2warmup、10samples；探索2buffers/6samples单独标记。
- 门禁 GPU use≤5%、VRAM≤20%、PTL Enabled/VECTOR,F8；不修改硬件配置、不等待或重采直到通过。

### 起点

- 上轮源码功能验收83项通过，TP1 independent约33.25/28.11T，shared181.97/187.75T。
- 原auto低重合仍调用Triton，原wave BF16单query/wave、BN16探索约26T，未采用。
- 本轮只修改本qsa目录；相邻MHA参考与冻结baseline保留。不存在旧qsa/opt.md，建立本日志后逐次追加。

### v01：实施顺序

1. 新增独立block-native FlyDSL direct，BN32/64候选，不依赖union建表；保留旧fallback和wave作对照。
2. dense_limit≤2051的前缀走真实dense causal：仅覆盖满足绝对可见长度条件的query。
3. union按实际计算tile对应query构造局部并集，先以BQ≤floor(128/G)确保每份并集只服务一个M128 tile。
4. 新DispatchPlan统一dense/direct/union，GPU gate保证互斥写入；direct模式跳过union。
5. 测试重点改TP2/4，TP8正确性；计时入口支持TP列表、固定selection_group、独立输出及完整raw/源码/ELF。

### v01-prep：静态准备

- 读取当前源码/技能/计时器；当前工作区无已跟踪修改，8张卡中GPU2保留约74%显存，其他卡初始空闲。
- dense采用原生linear参考；非BN64对齐上下文使用本地full-VOFFSET有界DMA，不以mask掩盖物理越界。
- dense静态检查及离线4形状编译通过；尚未据此声明GPU正确性或性能。

### v01-smoke：三路首版正确性

- direct新增BN32/64 block-native FlyDSL：每wave一个query和GQA heads，V占64KiB LDS，Q/K寄存器；BN32为4wave，BN64为2wave。
- auto不再调用Triton attention fallback；Triton仅用于动态union建表。direct模式无union分配、清零、scatter或compact。
- dense从每请求开头取 `min(M,max(0,2051-P))` 个query，保留原packed地址和绝对causal对齐。
- union effective BQ限制为 `min(requested_BQ, floor(128/G))`：TP2=10、TP4=21、TP8=32；每份局部并集仅供一个M128计算tile。
- GPU1入口PTL Enabled/VECTOR,F8，利用率0%；TP2/M37/P0或3000的direct/auto/union共6用例通过原FP32容差。
- [首版数值记录](results/tp24_v01_smoke.json)：最大绝对误差0.00749以内。首轮编译保留M0 warning；另一次仅从缓存执行并保存数值，不作为性能重采。
- 测试调整为TP2/4主性能与TP8功能，BN32/64、dense边界/guard和新DispatchPlan graph重放新增覆盖。

### v01-check：路径验证与首次测量门禁

- [路径测试](results/tp24_v01_routes.xml)：72 passed（147项未选），覆盖TP2/4/8 direct BN32/64、dense边界、graph和局部union。
- 首次direct-only BN32探索在TP2/no_prefix准备后被GPU1 use14%门禁拒绝，0个性能样本；[记录](results/tp24_v01_direct_bn32/no_prefix_tp2_independent_bq32/result.json)保留，不称性能成功。
- harness增加warmup之后对**实际使用的全部输入**重新验证causal/block/tail契约与索引哈希，补齐direct-only无union时的准备后审核；门禁阈值不变、无sleep。

### v02：block-native串行首版，未达到性能预期

- 两协议均2buffers/6samples、TP2/4、固定independent索引；GPU1门禁全部通过，原始结果分别在
	[BN32](results/tp24_v02_direct_bn32/no_prefix_tp2_independent_bq32/result.json)、
	[BN64](results/tp24_v02_direct_bn64/no_prefix_tp2_independent_bq32/result.json)同级四用例中。
- BN32 direct：TP2 no-prefix/chunk为22.12/21.86T，TP4为11.26/10.93T；同场baseline约29.2/28.5T和14.6/14.4T。
- BN64 direct：TP2为14.76/13.97T，TP4为7.35/7.00T；减少softmax迭代未抵消顺序load/wait、较少wave和LDS开销。
- 不采用BN64作为默认；未以较大的tile宣称优化成功。
- 下一候选v03：在16-token子片的QK MFMA前发射下一片K/V加载，V先写LDS，保留异步wait后的scheduler fence；检查寄存器/零spill与原数值门槛。

### v03：K/V预取无收益，继续保留失败证据

- [原始记录](results/tp24_v03_direct_prefetch/no_prefix_tp2_independent_bq32/result.json)同级4用例：TP2为21.48/20.57T，TP4为10.78/10.39T。
- 未优于v02，不能仅凭存在预取宣称隐藏延迟。数值和零spill通过，但额外活跃数据与LDS转置成本仍在。
- v04改为V按PV片段直接global向量加载，在寄存器进行2×2 BF16重排；不再将V写入LDS再读取。仅输出shuffle占32KiB LDS。
- 四个wave分别独立query，BN32/64都保持4wave；原始块/尾部VOFFSET边界不变，明确为不同内存路径的实验。

### v04：移除V LDS往返，采纳

- [BN32](results/tp24_v04_direct_globalv/no_prefix_tp2_independent_bq32/result.json)四用例：TP2 37.13/36.10T，TP4 18.62/18.16T，均优于同场baseline。
- [BN64](results/tp24_v04_direct_globalv_bn64/no_prefix_tp2_independent_bq32/result.json)：TP2 37.13/36.44T，TP4 18.62/18.27T，差别不大；BN32 VGPR214/SGPR55，BN64 VGPR234/SGPR71，scratch/spills均0。
- 保留BN32默认以留寄存器余量，BN64可选。以上2buffers/6samples仍是探索，非正式验收。
- v05：将下一16-token段V低半的加载放到当前PV高半前，检查能否进一步隐藏VMEM；不同variant均保留原始报告。

### v05/v06：完整分流与局部分组边界

- v05 PV局部预取与v04相近：TP2 37.15/36.22T，TP4 18.64/18.16T，不宣称微小差别为稳定加速。
- v06 auto/independent：TP2 41.23/35.80T，TP4 21.27/18.01T；dense前缀有效，但shared固定SG32时TP2/4仅40.46/38.57T与23.96/18.50T。
- 原因证据：[shared报告](results/tp24_v06_auto_shared/chunk_prefill_tp4_shared_bq32/result.json)表明TP4 BQ21有大量组跨越原SG32选择边界，仅215/572组走union；TP2 BQ10也跨界。未将改变selection_group视为修复。
- v07按硬件行容量取2幂BQ：TP2=8、TP4=16、TP8=32，且沿请求原始query边界对齐；dense切割后的首组允许短组，不把后续分组整体平移2051行。
- 输入和索引哈希仍固定SG32，本次只改执行分组；这是通用的tile边界策略，不更改模型选择集合。

- v07首跑数值通过但harness静态partition审核仍按旧2051起步等长分组，报 `Sparse-only metadata mismatch`，0性能样本；[失败记录](results/tp24_v07_aligned_shared/no_prefix_tp2_shared_bq32/result.json)保留。审核同步为请求原始边界对齐，未放宽验证。

### v07a：固定输入的对齐分组有效

- [shared测量](results/tp24_v07a_aligned_shared/no_prefix_tp2_shared_bq32/result.json)同级四用例：TP2 no-prefix/chunk 134.92/139.57T，TP4 122.52/128.71T；原SG32索引哈希逐字不变。
- 所有稀疏组都进入union，之前跨选择边界的退化消除。effective BQ8/16不是32，日志中明确记录。
- v08尝试每query排序512个块以改善低重合direct访问局部性；预分配排序buffer，Triton排序计入rebuild+run，原indices/block_indices不改写。

### v08/v09：排序与packet地址复用

- [v08 direct](results/tp24_v08_sorted_direct/no_prefix_tp2_independent_bq32/result.json)：TP2 37.62/37.70T、TP4 18.84/18.91T；计入排序后仍高于旧baseline，TP4/no-prefix收益较小但保留显式开关。
- v09将一个PV四-token packet内重复block读取/寻址合并；[完整auto](results/tp24_v09_packet_address/no_prefix_tp2_independent_bq32/result.json) TP2 43.16/37.04T、TP4 22.37/18.74T，含建表为41.35/35.54T和21.47/17.99T。
- v10减少无效预处理：先判断union膨胀gate，再只对active组计算前缀扫描/写compact表；block排序只处理实际走direct且非dense的行，graph重放时每次重新判断gate。

### v10：GPU建表按实际分流裁剪

- [探索报告](results/tp24_v10_gated_prepare/no_prefix_tp2_independent_bq32/result.json)四用例：TP2 run43.16/37.06T、plan+run41.72/35.80T；TP4 run22.38/18.73T、plan+run21.61/18.06T。
- 已优于当前同场baseline，但远未达到200；本轮不改变FLOPs定义，也不声称TP4只有6个heads就能维持TP2吞吐。
- 准备正式10buffer验收与完整正确性；测试中的旧BQ10/21和BN64两wave结构断言同步为采纳的8/16、统一4wave。
- audit默认审计冻结源码本身，`--check-current`才比较当前实现；历史报告不因后续优化而被篡改，TP8/参考baseline的target为空且不算达标失败。

### v11：完整测试通过后独立审查发现的边界修复

- [v10完整JUnit](results/tp24_v10_correctness.xml)：219 passed，包括TP2/4大形状及TP8功能；测试通过不等于不存在未覆盖边界。
- direct补齐wave查gate改为钳制到**当前direct tile的最后有效query**，不能读后续dense行对应的`query_tiles=-1`。
- 输出与Q/K/V重叠检查移到dense空调用早退之前，纯sparse模式也拒绝原地覆盖输入。
- sort_blocks=False时run读取当前inputs.block_indices而非prepare时保存的旧tensor；sort=True继续使用本轮重建的排序buffer。
- rebuild_plan显式切入inputs.q.device，避免当前设备不同导致Triton kernel和PyTorch zero在不同卡/stream执行。
- 增加auto gate翻转graph、无排序替换输入、纯sparse重叠拒绝等回归；上述修复不改变测试容差或选择语义。

- [v11回归](results/tp24_v11_boundary_fixes.xml)：19 passed，含auto gate正反翻转、sort=false替换tensor、跨当前GPU重建、输出alias拒绝。
- v12：direct以CTA-uniform gate检查四个实际query，全部由union处理时在进入计算/输出shuffle之前退出；禁止wave级提前退出破坏CTA barrier。

### v12：统一gate跳过无效direct CTA，最终功能验收

- [shared探索](results/tp24_v12_direct_ctagate/no_prefix_tp2_shared_bq32/result.json)四用例：TP2 139.34/144.78T、TP4 128.91/136.81T，计入建表后126.41/125.83T和110.26/109.97T；所有门禁通过，仍未达200。
- [完整JUnit](results/tp24_v12_final_correctness.xml)：238 passed、0 failed、0 skipped；包含TP2/TP4主矩阵、TP8功能、两种BN、dense阈值、guard、取消误差、graph gate变化、跨当前GPU和输出alias回归。
- Black88、Ruff F/I及Python语法通过；格式调整经AST对比不改变计算。开始冻结正式10buffers/2warmup/10samples的TP2/4结果；禁止用探索样本替换正式样本。

### 最终正式验收：TP2/TP4，10独立buffers × 10样本/scope

- GPU1 / PCI `0000:80:00.0` / MI308X gfx942；每场before/before_samples/after门禁均通过，PTL Enabled/VECTOR,F8，未写任何硬件策略。
- 固定M12000、seed17、SG32，effective BQ TP2=8 / TP4=16，dense_limit2051，direct BN32、sort=true。
- 三个scope分别使用独立output，所有10组输入完整校验、原indices哈希不变；240个正式计时样本全部保留。

| 正式报告 | candidate ms / T | plan+run ms / T | baseline ms | 完整调用加速 |
| --- | ---: | ---: | ---: | ---: |
| [TP2 independent/no-prefix](results/tp24_final_independent/no_prefix_tp2_independent_bq32/result.json) | 6.453 / 42.83 | 6.635 / 41.66 | 9.458 | 1.43x |
| [TP2 independent/chunk](results/tp24_final_independent/chunk_prefill_tp2_independent_bq32/result.json) | 8.142 / 37.12 | 8.448 / 35.77 | 10.568 | 1.25x |
| [TP4 independent/no-prefix](results/tp24_final_independent/no_prefix_tp4_independent_bq32/result.json) | 6.183 / 22.35 | 6.396 / 21.61 | 9.448 | 1.48x |
| [TP4 independent/chunk](results/tp24_final_independent/chunk_prefill_tp4_independent_bq32/result.json) | 8.110 / 18.63 | 8.392 / 18.01 | 10.573 | 1.26x |
| [TP2 shared/no-prefix](results/tp24_final_shared/no_prefix_tp2_shared_bq32/result.json) | 1.985 / 139.27 | 2.187 / 126.37 | 9.238 | 4.22x |
| [TP2 shared/chunk](results/tp24_final_shared/chunk_prefill_tp2_shared_bq32/result.json) | 2.089 / 144.66 | 2.402 / 125.82 | 10.223 | 4.26x |
| [TP4 shared/no-prefix](results/tp24_final_shared/no_prefix_tp4_shared_bq32/result.json) | 1.073 / 128.76 | 1.251 / 110.46 | 9.262 | 7.40x |
| [TP4 shared/chunk](results/tp24_final_shared/chunk_prefill_tp4_shared_bq32/result.json) | 1.106 / 136.63 | 1.372 / 110.12 | 10.187 | 7.42x |

- 主independent两用例均改善，**200有效TFLOPS目标仍未达到**；TP8只验正确性、target为空，未运行性能门槛。
- 不拿TP1历史181/188T替代本轮TP2/4结果；高重合shared不能代表实际模型Top-K。
- 最终union VGPR251/SGPR73/LDS64KiB，directBN32 VGPR214/SGPR67/LDS32KiB，dense bounded VGPR230/SGPR60；全部scratch/spills0。
- 八份报告经`audit --check-current`逐项复算raw中位、有效TFLOPS、240样本数、门禁、冻结当前源码和实际ELF SHA；全部通过。
- 正式结果之后只更新README/本日志，不改变被测计算实现。参考MHA、冻结baseline和历史结果未改写。

## 2026-09-25：迁移交接与清理状态

本节是清理后的继续工作入口；前面的逐版本日志保持历史含义。
本次**没有继续调性能、没有改动results目录**，只删除未使用路径、原样搬迁三个活跃helper、清理生成缓存并验证。

### 1. 清理决定与机械证明

- 删除当前源码中无调用的旧Triton fallback。
- 旧wave实验不能直接删除：direct仍使用它的`_load`、`_qk`、`_pv`。
	已将这三个函数逐字移入 [direct.py](direct.py)，移除旧模块import，将唯一helper绑定改成本地函数，再删除其余wave实验/launcher。
- 两个旧文件在 [正式报告冻结源码](results/tp24_final_independent/no_prefix_tp2_independent_bq32/source/direct.py) 同目录中保留；
	历史实现、IR/ELF、性能失败、原始样本均不改写、不重贴版本标签。
- 删除本目录生成的Python/pytest/Ruff缓存；不删除FlyDSL全局缓存、不清理其他实验目录。
- 其余文件均属于当前运行、测试、配置、来源或交接文档，不再为了文件数而合并。

迁移前后direct SHA-256：

| 对象 | SHA-256 |
| --- | --- |
| 清理前direct | `16f7de9611ffd829ad0832441d70f5794fa5a7025c343f8ac31b0272f6c35891` |
| 清理后direct | `81709d4cd738d0817b8e576156866b7c38c7e93e14fd79606e23b9f0efa18f94` |
| 搬迁的`_load`，含最终换行 | `e450995567f0b2ef0e7bdd5bce953f5f86d6b5c8c6624d14c93b7f14afd5563d` |
| 搬迁的`_qk`，含最终换行 | `42d4d3864b28ae4ba5c95b9274b1cbd6f24ce34d7f61dd8925bb27f5c46897ca` |
| 搬迁的`_pv`，含最终换行 | `2f53346097a59e40371af591857e38daf5ee9056dd55d3e7566407952b0f1dd9` |

已从冻结文件按以下三个唯一文本替换**完全复现清理后的direct字节**，不是只比较函数名或肉眼确认：
删除旧wave import；将三个helper的调用绑定去掉模块前缀；在DirectPlan前插入三个原函数。
函数体、dtype、偏移、MFMA顺序、wait/fence、接口和计时器不变。

清理验证分层：

| 层次 | 已验证事实 | 不等于 |
| --- | --- | --- |
| 源码 | 上述精确复现通过；无旧模块活跃import；其他计算文件未变 | 所有未来特化都机器码相同 |
| 静态 | Black88、Ruff F/I、Python语法、编辑器诊断通过 | GPU数值验证 |
| CPU | 64项契约测试通过 | 完整GPU矩阵重跑 |
| GPU | 18项direct/auto graph回归通过，覆盖TP2/4/8和BN32/64 | 新一轮性能测量 |
| 产物 | TP2/4的M12000/no_prefix/BN32 gated direct，与原正式ELF**整文件相同**，VGPR214/SGPR67、scratch/spill=0 | 未比较特化的逐位等价或跨机器时延不变 |

两个已比较的设备`.text` SHA-256：TP2 `d63a596657cd6b226095722c5131d398d379daba0ba3730665fd6af629e5553c`；
TP4 `e4fa92c7f10566ed09123a8d4be25b3c4d260b0cd4d65bcce01079baa9eff67b`。
未重新计时、未把本次结果写进历史results；18项回归的临时JUnit不是需要迁移的性能证据。

清理前results有2076个文件。按相对路径排序，将每项`relative_path + NUL + sha256 + newline`
串联再做SHA-256，聚合值为`74a9b2fe639a6d2ccf809d082ae45dc36189a29dfb10a3df476e5abe5ee25556`。
清理结束再次核验数量和聚合值，以证明历史证据没有被触碰。

### 2. 当前实现与约束（带到下一台机器）

- 主性能：TP2 Q12/KV1/D256、TP4 Q6/KV1/D256。TP8 Q3/KV1仅要求功能；不把TP1历史吞吐当成本轮结果。
- 两主用例M=12000表示**新增query数**。no_prefix P0/KV12000，chunk P12000/KV24000；BF16 Q/K/V，输出同Q。
- 四-token压缩、512个选中完整块、最多3个尾token；逻辑索引 `[M,2051]`，原块索引 `[M,512]`，有效前缀后-1。
- 每query保持独立选择，所有heads共享该query选择；最终attention读取原始K/V，不是压缩K。
- 默认seed17、benchmark SG32；SG与requested BQ分离，调BQ不改变输入哈希。independent/shared/recent都是合成负载，不是模型真实Top-K。
- `dense_limit=2051`：满足绝对可见长度≤2051的行走真实dense，原Q/O前t行、K/V前P+t行零复制；P12000没有dense行。
- `direct`：四wave/CTA，每wave一个query和GQA heads；BN32默认、64可选；V直接global→寄存器重排，只有输出shuffle占32KiB LDS。
- `union`：按原请求query边界对齐，effective BQ8/16/32；仅一个M128 tile使用该局部并集。共有完整块免mask，其余逐query mask；LDS64KiB。
- `auto`：dense、union active、direct inactive互斥写；不在CPU `.item()` 读gate。全union的direct CTA统一早退，不能wave提前退出CTA barrier。
- 默认attention计算全部FlyDSL；Triton负责union建表与可选block排序。direct模式不建union，但sort=true仍有GPU排序成本。
- `prepare(...)->DispatchPlan`持有dense/direct/union；`implementation.rebuild_plan`切入输入GPU重建gate和排序；`run(...,out=...)`不分配输出。
- 布局/长度/prefix/设备变化必须重新prepare；同布局选择变化同时更新indices和block_indices，再rebuild。
	graph需先warm，capture/replay地址固定；已验证gate正反翻转、排序刷新以及图外sort=false替换输入。
- 必须保留full-VOFFSET尾部边界、NaN guard、输出不重叠Q/K/V、padded wave只查有效query gate、异步wait后的scheduler fence。
- 精确指选择集合和因果语义不变；BF16浮点归约不要求跨实现逐位相同。保持原`rtol=atol=0.02`逐元素检查，relative L2仅补充。

### 3. 保留文件：运行与验收各有用途

除results外，当前保留19个文件；不是每个都参与热路径，但它们都用于继续开发、验证或交接。

| 分类 | 文件 | 保留理由 |
| --- | --- | --- |
| 核心入口 | [implementation.py](implementation.py)、[contract.py](contract.py) | 三路接口、输入/计划类型 |
| 计算与准备 | [dense.py](dense.py)、[direct.py](direct.py)、[kernel.py](kernel.py)、[plan.py](plan.py) | 当前执行的kernel与GPU建表；direct含三个搬迁helper |
| 输入与数学参考 | [inputs.py](inputs.py)、[reference.py](reference.py) | 固定seed用例、合法选择、独立FP32 oracle |
| 冻结对照 | [baseline.py](baseline.py)、[baseline_kernels.py](baseline_kernels.py) | 同场性能对照和数值交叉验证，不是无关fallback |
| 验收工具 | [test_qsa.py](test_qsa.py)、[bench.py](bench.py)、[audit.py](audit.py) | 正确性/边界/graph、原timer/raw/门禁、CPU证据审核 |
| 配置与来源 | [model_config.json](model_config.json)、[source_manifest.json](source_manifest.json) | Qwen固定revision字段、SGLang baseline哈希，不下载模型 |
| 包与缓存规则 | [__init__.py](__init__.py)、[.gitignore](.gitignore) | package-relative导入，忽略可重建缓存 |
| 文档 | [README.md](README.md)、[opt.md](opt.md) | 常用接口说明；本日志是当前状态、实验和下一步交接入口 |

### 4. 外部依赖与迁移边界

**推荐复制PyHIP checkout，而非只复制qsa目录，也不能只安装PyHIP wheel。** experiments/tests不包含在普通wheel里。
若做最小文件集合，必须保留相同相对目录和以下实际依赖：

- 三个MHA模块：[../mha/mha_pa_bf16_942.py](../mha/mha_pa_bf16_942.py)、
	[../mha/mha_pa_bf16_256_942.py](../mha/mha_pa_bf16_256_942.py)、
	[../mha/mha_pa_bf16_256_linear_942.py](../mha/mha_pa_bf16_256_linear_942.py)。它们是运行时helper/native dense依赖，不只是参考资料。
- 计时器：[../../../../src/pyhip/testing/misc.py](../../../../src/pyhip/testing/misc.py)，以及pyhip/testing和pyhip包初始化文件。
- 只读门禁：[../../../../tests/ops/gr_read/test_gr_read.py](../../../../tests/ops/gr_read/test_gr_read.py)，及tests/ops/gr_read的父包初始化文件；不调用GRRead模型或其kernel。
- package链：[../../../__init__.py](../../../__init__.py)、[../../__init__.py](../../__init__.py)、
	[../__init__.py](../__init__.py)、[../mha/__init__.py](../mha/__init__.py)。
- [../../../../pytest.ini](../../../../pytest.ini)、[../../../../conftest.py](../../../../conftest.py)、
	[../../../../pyproject.toml](../../../../pyproject.toml)、许可证/版权标记。
	SGLang快照的Apache-2.0来源不被PyHIP MIT覆盖，分发时一并携带
	[SGLang Apache-2.0 LICENSE](https://github.com/sgl-project/sglang/blob/540d564c19436f28f2644e2247350da56c124452/LICENSE)。

在目标机安装/选择已有兼容环境：Python≥3.10、ROCm PyTorch、配套Triton、FlyDSL、NumPy、msgspec、pytest。
内核仅支持gfx942；不假定其他AMD架构或NVIDIA可直接运行。ROCm下仍使用`torch.cuda`。
历史实测为Python3.10.12、Torch2.12.0+rocm7.2.4、Triton3.7.1；以新机实际import路径/版本为准，
不要用PyPI CUDA版Triton覆盖ROCm厂商发行版。

性能还需只读`rocm-smi`和支持PTL字段的AMD SMI；原机器临时bundle路径不属于迁移合同，
目标机可通过`--amd-smi`传入已有兼容工具。CPU审计不需要GPU工具；模块入口仍会加载NumPy/msgspec/Torch输入包。
纯审计也可直接运行 [audit.py](audit.py)，此脚本本身仅依赖标准库。

复制注意：

- 使用文件复制/rsync等保留实际工作树，确保新增或未跟踪文件也被带走；不要只依赖`git archive HEAD`遗漏本实验。
- 不带生成的Python、pytest、Ruff和本机JIT缓存；保留源码、config、许可证和需要的历史results。
- 当前results为证据而非运行依赖。可以不传全部探索产物，但建议至少保留tp24_final两组报告及其source/ELF/门禁、完整JUnit；
	部分复制后旧日志的其他链接可能不存在，不能据此补造或改写旧结果。
- 本次在原机器没有删减results。`bench`在新目录生成报告，绝不覆盖旧场次。

### 5. 新机器首先执行的检查

以下从PyHIP根目录执行；`python`应替换为选定的解释器。先运行CPU/功能，再进行计时。

```bash
PYTHONPATH=src python -c 'import torch, triton, flydsl; print(torch.__version__, triton.__version__, flydsl.__file__); print(torch.cuda.is_available())'
PYTHONPATH=src python -m pytest experiments/attention/flydsl/qsa/test_qsa.py -q -k cpu
PYTHONPATH=src python -m pytest experiments/attention/flydsl/qsa/test_qsa.py -q -k 'test_direct_matches_baseline_and_fp32 or test_auto_graph_gate_flips_rebuild_sorted_direct'
PYTHONPATH=src python -m pytest experiments/attention/flydsl/qsa/test_qsa.py -q -m 'perf or not perf'
python experiments/attention/flydsl/qsa/audit.py experiments/attention/flydsl/qsa/results/tp24_final_*/*/result.json
```

- 清理前完整矩阵为238项；本次64CPU+18GPU定向检查不能代替目标机完整验收。无GPU的skip不算功能通过。
- **对旧tp24_final报告不要使用`--check-current`期待通过**：helper搬迁让direct AST改变，严格检查应报告不一致。
	审计器未被放宽；旧结果只认证旧冻结源。下方复现证明可独立验证此次机械变化，目标机新测量生成新快照后再用`--check-current`。
- CPU复现清理（读取旧source快照，不写文件）：

```python
import ast
from pathlib import Path

root = Path("experiments/attention/flydsl/qsa")
saved = root / "results/tp24_final_independent/no_prefix_tp2_independent_bq32/source"
expected = (saved / "direct.py").read_text()
old_wave = (saved / "wave_kernel.py").read_text()
defs = {
		node.name: ast.get_source_segment(old_wave, node)
		for node in ast.parse(old_wave).body
		if isinstance(node, ast.FunctionDef)
}
helpers = "\n\n\n".join(defs[name] for name in ("_load", "_qk", "_pv"))
changes = (
		("from . import wave_kernel\n", ""),
		("    load, qk, pv = wave_kernel._load, wave_kernel._qk, wave_kernel._pv",
		 "    load, qk, pv = _load, _qk, _pv"),
		("class DirectPlan(", helpers + "\n\n\nclass DirectPlan("),
)
for old, new in changes:
		assert expected.count(old) == 1
		expected = expected.replace(old, new, 1)
assert expected == (root / "direct.py").read_text()
```

满足只读门禁之后，使用新的输出目录计时；物理GPU编号按目标机空闲设备设置，示例GPU0不是空闲保证：

```bash
env -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES PYTHONPATH=src python -m experiments.attention.flydsl.qsa.bench --gpu 0 --tp-list 2 4 --case all --algorithm auto --selection independent --selection-group 32 --output experiments/attention/flydsl/qsa/results/new_machine_independent
env -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES PYTHONPATH=src python -m experiments.attention.flydsl.qsa.bench --gpu 0 --tp-list 2 4 --case all --algorithm auto --selection shared --selection-group 32 --output experiments/attention/flydsl/qsa/results/new_machine_shared
```

保持原10buffers/2warmup/10samples、PTL Enabled/VECTOR,F8、GPU≤5%、VRAM≤20%。不修改PTL/功率/时钟，不循环等待门禁；
失败保留全部raw和错误。`candidate_run`与`plan_and_run`分开报告；初始化、JIT、cache gather、indexer和服务调度不在计时中。
有效FLOPs按原选择计数，不计并集padding或softmax额外算术。先核对SG32、seed17和indices哈希再比较优化前后。

### 6. 当前性能结论与下一轮任务

当前性能沿用本日志上一节八场正式报告，**没有新的清理后计时数据**：
TP2 independent run42.83/37.12T，TP4 independent22.35/18.63T；shared TP2 139.27/144.66T，TP4 128.76/136.63T。
完整调用较冻结baseline的主independent加速1.25–1.48x，但约200T目标仍未达到，不能用shared或TP1旧数据代替。

优先级与单变量实验：

1. **首段V低128维预取提前**：当前direct的首段V在softmax统计后发射，先只把低半提前，覆盖max/exp/sum窗口。
	 保持BN32、4wave、排序、gate不变；先测TP4/P12000，再测TP2。核对实际ISA、wait和214VGPR附近的live range，不盲目加双缓冲。
2. **四-token块V预排布**：转换为PV operand友好布局，跨query摊薄重复permutation/寻址。
	 转换与新V更新必须计入完整调用，不能只报预转换后的run；保留BF16位值、尾部和请求边界。
3. **TP特定cost model**：当前rho≤1.5未体现G、M/BN padding和common比例。direct固定M16，所以Q12→Q6有效FLOPs减半但指令不一定减半。
	 不可从TP2/TP4约8.1ms相同就断言HBM受限；区分逻辑字节与PMC实测流量。
4. **compact+mask构造融合**：active判定/选择性排序已经实现；进一步减少中间读写和launch，不能忽略跨CTA清零/atomic同步。
	 TP4 shared/chunk的run1.106ms、plan+run1.372ms差额约0.266ms只是两scope差，不是单独量到的plan耗时。
5. **次级探索**：M96/六计算wave或保持DMA职责时跳过无效32行；dense2048对齐主干+3行窄尾。
	 这些需重新验证barrier/布局，后者不改善P12000，不能同时引入多个变量掩盖归因。

已失败/已做的工作不要作为新方案重复：仅把BN变大、一般性K/V预取、V先写LDS再读、所有query共享无mask选择、
继续扩大并集、盲目增加四阶段展开或grid，都已有失败记录或语义风险。固定同一份输入和原容差，
每次在本日志追加假设、源码/ELF身份、正确性、raw、门禁、保留/拒绝理由。

### 7. 清理收尾检查

- 当前非results目录为19个文件、约260KiB；旧fallback/wave源码和本目录生成缓存已在磁盘上确认不存在。
- 2076个results文件数量与上述聚合SHA在清理前后完全一致；八份正式报告共240个raw样本按原口径审核通过。
- 文档本地链接和嵌入的机械迁移复现代码执行通过；严格`--check-current`对旧direct AST的拒绝行为保持不变。
- 未改参考MHA/计时器、未放宽数值/门禁/审计、未重跑性能；迁移后按本节步骤做新的完整验收，再继续优化。

## 2026-09-25：如何集成到 SGLang（方案，尚未实施）

本节基于本机SGLang revision `540d564c19436f28f2644e2247350da56c124452` 的实际调用链核对。
**本次只补充文档，没有修改SGLang生产代码、添加可用启动开关或完成服务端验收。**
当前238项实验测试、清理后回归和kernel性能不等于SGLang端到端集成已通过。

### 1. 接入位置与首版范围

推荐在现有 `QwenSparseAttnBackend.forward_extend()` 内增加默认关闭的实现分支，
不是替换全局attention backend，也不是把实验目录加入服务器的`sys.path`。
Qwen混合模型的full-attention侧会选择QSA backend；仅注册一个通用prefill backend名字，
不保证模型最终使用它。实际链路为：

```text
Qwen4ExpAttentionDecoderLayer.self_attention
		indexer: hidden_states -> logical topk_indices
		main attention: QKV projection -> Q/K norm + RoPE
		RadixAttention.forward
				hybrid full-attention backend
						QwenSparseAttnBackend.forward_extend
								cache write -> valid-row trim -> packed full-context K/V
								optional FlyDSL adapter -> [valid_rows, local_Q_heads, 256]
								_pad_extend_output -> flatten and restore DP padding
		existing sigmoid output gate -> O projection
```

首版只启用以下交集，其余在**发射新kernel之前**走原分支：

- 主模型的普通 `ForwardMode.EXTEND`、eager prefill，非draft/MTP runner。
	不仅判断`is_extend()`：该函数也包含MIXED、TARGET_VERIFY、SPLIT_PREFILL等模式；
	draft的某些prefill也可能使用普通EXTEND，因此还需检查runner角色。
- ROCm、输入设备架构恰为gfx942；`tensor.is_cuda`不能区分NVIDIA与ROCm。
- `qsa_profile.variant == QSA_VARIANT_COMPRESSED`，压缩比4、block_topk512、token预算2048。
	同一QSA backend也服务tokenwise DSA，不能对它应用四-token块恢复或dense阈值。
- 最终attention实际Q/K/V为BF16、D256、连续三维张量；首版以本地Q12/KV1、Q6/KV1为主，Q3/KV1做功能支持。
	从实际tensor与layer读取头数，不从全局`--tp`猜测；attention TP、DP和DCP可改变实际分片。
- Q/K/V同设备、16-byte对齐，当前实现要求各自byte span小于$2^{31}$；out连续且不与输入重叠。
  无法证明该ABI时不能通过强行reshape、忽略stride或修改descriptor extent来绕过检查。
- 首版不覆盖CP/DCP、交叉层共享导致K/V为空、量化/特殊KV布局、额外score修改、LSE返回等未验证合同。
	FP8权重模型仍可能产生BF16 Q/K/V；按实际输入检查，不按模型名或权重格式推断。
- 不接管decode、TARGET_VERIFY、DRAFT_EXTEND_V2、MTP索引复用、piecewise/breakable prefill graph。
	保留 `forward_extend()` 开头现有的speculative paged早退；空有效batch独立返回正确形状或保持原路径。

### 2. 需要查看或修改的生产边界

以下链接固定到已核对的SGLang版本；迁移到新revision后必须重新确认函数和合同。

| 生产位置 | 集成职责 |
| --- | --- |
| [QSA forward_extend](https://github.com/sgl-project/sglang/blob/540d564c19436f28f2644e2247350da56c124452/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py#L1370-L1472) | 推荐首版唯一attention分流位置；保留cache、gather、padding语义 |
| [prefill block选择](https://github.com/sgl-project/sglang/blob/540d564c19436f28f2644e2247350da56c124452/python/sglang/srt/layers/attention/qsa/qsa_indexer.py#L453-L507) | 当前生成block_indices后展开并删除；第二阶段在这里保留原块索引 |
| [QSA profile](https://github.com/sgl-project/sglang/blob/540d564c19436f28f2644e2247350da56c124452/python/sglang/srt/layers/attention/qsa/config.py) | 区分compressed/tokenwise，读取ratio、budget和index维度 |
| [模型indexer/attention衔接](https://github.com/sgl-project/sglang/blob/540d564c19436f28f2644e2247350da56c124452/python/sglang/srt/models/qwen4_exp.py#L1483-L1577) | 第二阶段将原block选择显式传给attention，保持现有MTP和stream同步 |
| [RadixAttention及custom-op参数](https://github.com/sgl-project/sglang/blob/540d564c19436f28f2644e2247350da56c124452/python/sglang/srt/layers/radix_attention.py#L150-L365) | eager kwargs和graph schema是不同边界；新增block字段需显式贯穿，不能假定自动传递 |
| [QSA metadata](https://github.com/sgl-project/sglang/blob/540d564c19436f28f2644e2247350da56c124452/python/sglang/srt/layers/attention/qsa/metadata.py) | 请求归属、序列长度、逻辑位置；压缩K仅供indexer，不是最终attention K/V |
| [主attention头分片](https://github.com/sgl-project/sglang/blob/540d564c19436f28f2644e2247350da56c124452/python/sglang/srt/models/qwen3_5.py#L999-L1082) | 读取本地Q/KV heads、D、scale，处理attention TP与KV复制差异 |
| [backend组合](https://github.com/sgl-project/sglang/blob/540d564c19436f28f2644e2247350da56c124452/python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py) | 确认最终full-attention侧为QSA；首版不改linear-attention和decode实现 |

### 3. 第一阶段：保持现有token索引接口，增加内部adapter

生产当前只把 `topk_indices[M,2051]` 传给attention；本实验需要额外的
`block_indices[M,512]`。首个可回滚集成可以在backend内部增加**device端恢复**，
不改模型返回值和RadixAttention签名。

在已验证的compressed/ordinary-extend合同下，令第i个query绝对逻辑位置为p：

$$c_i=\min\left(512,\left\lfloor\frac{p_i+1}{4}\right\rfloor\right),\qquad
B_{i,j}=\begin{cases}\lfloor I_{i,4j}/4\rfloor,&j<c_i\\-1,&j\ge c_i.\end{cases}$$

这里I是现有token索引。前`4*c_i`个有效项是完整块展开，之后才是0..3个尾token。
**不能不加有效完整块数判断就用 `topk_indices[:, :2048:4] // 4`**：短行的尾部紧跟有效块，
会被误认成完整块。恢复kernel还必须检查padding、每四项连续性、块起点对齐等前置合同；
测试对比恢复结果与生产indexer原始block输出，顺序也应保持。

这一方案仅用于当前固定四-token展开布局；上游若改变重排方式或变成tokenwise选择，应拒绝而非猜测。
把恢复成本计入adapter/服务端性能，不只测恢复后的run。它不改变Top-K计算，不省indexer本身。

从生产构造runtime输入的映射如下；不调用实验 `make_inputs()`、不读取实验JSON替代实际模型配置：

| 实验输入字段 | 生产来源/要求 |
| --- | --- |
| `q` | 主attention Q，裁剪到 `M=topk_indices.shape[0]`，reshape为`[M,layer.tp_q_head_num,D]` |
| `k`,`v` | 下一节两分支得到的原始packed K/V，**不是**compressed indexer K，也不是未转换的物理分页cache |
| `indices` | 当前layer/forward的logical `topk_indices`，int32连续，padding仍为-1 |
| `block_indices` | 第一阶段安全恢复，第二阶段由indexer直接传递；int32 `[M,512]` |
| `query_lens` | 当前forward每请求semantic extend长度，必须满足`sum(query_lens)==M` |
| `prefix_lens` | 当前总长度减extend长度，不是chunk编号或RoPE坐标 |
| `cu_q` | semantic extend长度的前缀和，显式输出连续int32 |
| `cu_k`,`kv_lens` | 各请求完整长度的前缀和与长度，显式连续int32 |
| `query_positions` | 每请求 `prefix+arange(extend_len)`，或已核对一致的indexer logical_positions |
| `query_sequence_ids` | packed query所属的请求行编号；不是 `req_pool_indices` 的全局槽号 |
| `max_seqlen_q/k` | host lengths求max，不在每层GPU tensor上`.max().item()` |
| `scale` | `layer.scaling`，当前D256为1/16；不能重复缩放 |
| 模型/profile字段 | 真实HF text config、QSAProfile与layer；不要硬编码TP2/4作为输入形状 |

实验的CaseSpec还携带seed/selection标签，生产adapter不需要它们；宜提取轻量runtime输入结构，
只传tensor、host lengths和已解析profile，不把整个ForwardBatch/ModelRunner交给kernel。
现有生产 `cumsum` 不保证输出int32，adapter必须显式转换，不能只把原tensor名称对接上。
CPU长度镜像缺失时应在本forward边界处理或保持原路径，不为此全局开启decode的CPU同步；
`needs_cpu_seq_lens=False` 不代表当前普通prefill完全不使用host lengths。

#### 无历史prefix分支

保留当前cache写入，直接传本次 `k[:M]`、`v[:M]` 的连续视图；
各请求Q与KV段长度相同，`cu_k=cu_q`，query位置从0开始。
FlyDSL dense分支仍只覆盖每请求前2051个可见token，其他query继续稀疏计算。

#### 有prefix / chunked prefill分支

保留现有layer pool getter和 `req_to_token[request,:seq_len]` 的gather逻辑；
先按照 `save_kv_cache` 约定写入本轮K/V，再收集prefix+当前chunk的完整K/V并拼接。
`cu_q`按新增长度，`cu_k`按完整长度；首个query的位置是P，不是0。
`save_kv_cache=False` 表示遵守调用方已有cache合同，不是可以遗漏新token的K/V。

首版沿用当前gather而不是同时改paged kernel，便于比较。将来直接访问paged cache应作为独立优化，
重新定义物理page表、对齐和buffer bounds，不能把logical token索引当成physical slot。

#### 后端执行伪代码（拟新增适配逻辑，不是已有API）

```text
forward_extend(..., topk_indices):
		preserve existing save_kv_cache handling
		trim semantic query rows; remember original padded row count
		preserve speculative paged and CPU bypasses
		compute real request lengths and packed Q/K/V as in current two branches

		if optional implementation is eligible before importing/launching it:
				recover original complete-block IDs with per-row complete-block counts
				build runtime inputs from this layer/forward, never from synthetic fixtures
				allocate or lease out[M, local_q_heads, D]
				prepare static layout when needed; rebuild current membership/sorted blocks
				execute FlyDSL dense/direct/local-union dispatch on current stream
		else:
				call the unchanged corresponding Triton prefill function

		return existing _pad_extend_output(out, original_query_row_count)
```

Q/K已经由模型做norm和RoPE，adapter不再做投影/归一化/旋转。
输出保持`[M,H,D]`、BF16，交给原 `_pad_extend_output()` 展平为`[padded_rows,H*D]`并补零。
已有sigmoid gate/O projection继续在模型层执行，不能在kernel中重复应用。

### 4. 第二阶段：从indexer传原始block，去掉恢复开销

正确性和回滚验证完成后，在 `select_prefill_tokens()` 的row-chunk循环内，
将每块 `block_indices` 写入本forward的 `[M,512]` 输出buffer，再执行原token展开。
无压缩K时写-1；不能只保留最后一块query的block结果。

推荐显式传递forward-local、layer-local的selection sidecar，例如
`token_indices + block_indices + logical_positions`，同时保留旧Tensor接口供其他backend使用。
需审查indexer、模型attention kwargs、RadixAttention及backend的所有调用者；
第一阶段仅eager，后续支持custom-op时必须修改其schema/fake输出/捕获路径。
当前piecewise边界显式包含token索引，**没有原始block参数**。

不要使用“模块全局last_blocks”或在共享metadata对象上放一个未分层的last-selection字段：
backend的 `get_indexer_metadata(layer_id,...)` 虽接收layer_id，当前可能返回同一forward metadata。
layer间选择会变化，overlap/多stream也会让无作用域的缓冲区被覆盖。
sidecar与输出在消费者完成前保持有效；如果indexer在alt stream上运行，
为新增block tensor保留与原token tensor同等的wait/record_stream生命周期约束。

第一阶段默认保留完整indexer计算和压缩cache更新，即使后续attention选dense。
进一步跳过短前缀评分/Top-K是第三个独立改动，必须保留pending K、RoPE位置和compressed cache状态，
并验证下一chunk及后续decode输出；不能为了dense fast path直接跳过整个indexer。

### 5. 可选依赖、启用开关和回退策略

- 把运行核心及三个MHA helper作为可安装、版本固定的模块或SGLang内vendor代码；
	不依赖运行机器上的PyHIP实验路径、开发容器绝对路径或仅含src的PyHIP wheel。
- 保留许可证与来源。只集成runtime核心、模型适配结构和编译依赖；
	不让生产import基准、pytest、AMD SMI、硬件空闲门禁、实验input生成器或结果审计。
- `implementation`顶层会导入FlyDSL，因此**eligibility/开关判断要在lazy import之前**；
	未启用、CPU/NVIDIA/其他ROCm架构或tokenwise模型不应因缺少FlyDSL而启动失败。
- 建议先加默认关闭的专家级选择开关；如命名 `SGLANG_USE_FLYDSL_QSA_PREFILL`，
	**这只是建议名，当前尚未实现，不能直接用它启动新路径**。
	新SGLang变量必须在Envs注册typed descriptor并通过`.get()`读取，测试用`.override()`。
	若做公开CLI，则在ServerArgs注册namespace metadata并从resolved config读取，不在业务代码直读raw seed。
- 实验中的`mode=auto/direct/union`是新实现内部算法选择，不能与“是否启用可选实现”的部署开关混为一谈。
- 默认关闭：旧路径完全不变。可选启用时，不支持的硬件/profile/ABI或明确缺少可选包可在launch前回退并记录原因。
	若用户要求强制使用新实现，应在初始化/预检时明确报错，不能悄悄使用baseline。
- 不允许`except Exception: old_attention(...)`包住编译和GPU执行：非法metadata、数值失败、device fault必须暴露，
	不能在状态已写入或设备异常后自动重算以掩盖错误。
- idle/PTL检查只属于benchmark验收。生产GPU本来就繁忙，服务运行时不得调用它、修改时钟功率或等待GPU空闲。

### 6. Serving计划与scratch生命周期

首版可每forward重建计划以保证正确，但不要把实验 `prepare()` 原样放到每层热路径后就称为低开销集成：
它包含host metadata构造、buffer分配，dense准备还核对device CU与host长度，可能发生同步。

生产建议拆分：

1. 每forward一次，从已知host lengths生成静态request布局、dense区间、local union/direct tile元数据。
2. 依据device、dtype、真实head数、请求长度/prefix布局、BQ/BN和容量获取独占scratch lease。
	 仅总M相同不足以复用布局，两个batch可有完全不同的请求边界和prefix。
3. 每层拿到本层block选择后，重建动态membership、gate、masks和direct排序，随后运行。
4. 顺序层可在消费者完成后复用scratch；并发forward/stream不得写同一组计划buffer。
	 不永久缓存按shape得到的active位图或排序结果，不把显存地址复用误认为内容没变。
5. 代码/JIT特化缓存与数据计划分开；只缓存验证过的代码身份，不缓存上一次输出。

遵循SGLang的runtime-context资源/stream/buffer生命周期，不修改只读ScheduleBatch来传临时值，
也不把请求级计划塞入永久model配置。每次launch使用输入设备当前stream，host metadata重建与device值更新明确分离。

### 7. CUDA graph / torch.compile 放到独立阶段

当前实验图测试只证明固定layout、固定地址下rebuild+run可重放，
**不证明SGLang的piecewise/breakable prefill已经能接入**。
已核对的SGLang版本特意未把Qwen4-Exp加入breakable prefill支持列表，理由就是host侧QSA metadata。
首版不要移除该限制；decode现有graph路径保持不变。

未来启用时必须同时完成：可捕获的metadata更新、预分配scratch、所有特化warmup、
block sidecar跨custom-op传递、稳定地址与生命周期、每次replay重新构造当前选择，
并测试shape bucket padding、graph fallback以及主/draft各自语义。
不能在capture中编译新特化、做`.tolist()`/`.item()`或动态host分配。

### 8. 验证清单和分阶段合入

| 阶段 | 必须证明的内容 |
| --- | --- |
| 包装与静态能力检查 | 未安装FlyDSL/非gfx942/不同QSA profile时旧路径正常；显式启用状态可观测；CPU import不拉起GPU依赖 |
| 原始选择桥接 | 从生产indexer获取真实token/block，短行恢复不吞尾、不引入重复，跨row chunk保留全部M行，未来token永不出现 |
| 后端数值 | 同一批实际Q/K/V与相同真实选择，对比旧prefill、FP32 oracle；TP2/4主验收，TP8功能；不放宽现有容差 |
| KV与布局 | P0/非0、分多chunk、radix prefix命中、不同请求长度、空query请求、非4/64对齐、DP padding、cache write顺序及save=false合同 |
| 生命周期 | 多层不同选择、连续batch相同M不同prefix、排序buffer刷新、输出alias、非当前GPU/stream、sidecar不串层、不串请求 |
| 未接管路径 | decode、TARGET_VERIFY、DRAFT_EXTEND_V2、draft普通EXTEND、tokenwise DSA均保持原路径和输出；首版禁用的graph/CP/DCP不被误入 |
| 服务精度与性能 | 单机kernel过关后再跑主模型请求、logits/生成质量和长上下文回归；测完整TTFT、throughput、后续decode行为，而非只报kernel TFLOPS |

测试按SGLang [测试指南](https://github.com/sgl-project/sglang/blob/540d564c19436f28f2644e2247350da56c124452/test/README.md)放置和注册。
建议先添加device桥接与backend单测，再做TP2/4服务测试；不要把实验目录的自定义argparse入口直接当CI测试。
新增或修改KL一致性测试时使用项目已有方法校准，不能借集成理由放宽精度阈值。

性能需拆开报告：indexer评分/Top-K、块恢复或sidecar、KV gather、动态plan/sort、attention、
全部forward以及服务端TTFT/吞吐。当前实验不含KV gather和indexer，不能把1.25–1.48x kernel链加速直接外推成服务加速。
加入分支命中/回退原因、dense/direct/union行数和临时显存统计，确认真实负载选择重合度；
不把SG32 synthetic shared成绩替代生产采样。

推荐合入顺序：**可选包与eligibility → 保持token接口的eager adapter → 真实请求/缓存验证 →
原block sidecar和scratch复用 → 服务端性能验收 → 最后考虑graph、paged直读和indexer短前缀跳算**。
每阶段默认关闭并能恢复旧路径；任何尚未完成的阶段在日志明确标记，不用实验图/合成测试代替生产验收。

## 2026-09-25：TP2 第一阶段插件接入与真实模型 profile

本节是上述方案的**本机可回滚实验接入**，不是修改SGLang生产源码或全量上线认证。
按用户约束，新增代码、测试和报告均在PyHIP的mytest；SGLang和原启动/测试脚本只读。
当前SGLang为c9ef753，不是方案链接中的540d564；已重新核对并固定backend/indexer/expand的源码SHA。

- 使用当前SGLang原生general-plugin entry point＋HookRegistry，而非修改模型/ServerArgs或把experiments目录加入服务器sys.path。
	[插件说明](../../../../mytest/sglang_flydsl_qsa_plugin/README.md)；默认关闭，`PYHIP_QSA_PREFILL=1`启用。
- 6个QSA runtime文件＋3个MHA helper在wheel构建时逐字打包，许可证/哈希随包保存；不导入合成输入、benchmark、pytest或AMD SMI。
- 原`forward_extend`继续处理KV写入、prefix gather、裁剪和padding；只在主runner ordinary eager EXTEND的compressed BF16/gfx942合同内替换sparse prefill调用。decode/spec/graph/tokenwise/CP/DCP保持旧路。
- 第一阶段GPU恢复原block，按真实完整块数检查四项连续、对齐、tail、padding、因果和重复块，错误显式暴露；无异常后baseline重算。
- 独占scratch按backend/device/stream/完整length+prefix布局/heads/scale缓存，逐layer重建gate与sort；首次布局仍有实验prepare的小metadata回读，不称零同步。输出独立分配。

### 验证

- 目标机原完整矩阵：[238 passed](../../../../mytest/qsa_original_full_tests.xml)，无skip。
- 最终插件：[27 passed](../../../../mytest/qsa_plugin_final_tests.xml)，含生产选择恢复、非法token、TP2/4/8、prefix gather/空请求、padding/save=false、动态选择、预热隔离和单卡双stream。
- TP2实际模型各rank校验48个layer/layout案例，逐元素检查共289284行，另做FP32边界/采样oracle；保持rtol=atol=0.02。并非模型准确率或超长上下文验收。
- 补充同进程跨GPU测试的夹具ambient-device错误与修正后的FlyDSL特化跨卡限制均保留失败记录。最终0.1.1显式限制每进程单GPU；SGLang本次TP2为每rank独立进程，服务采集没有此错误。

### Profile结果（不重采/覆盖旧base）

完整材料：[TP2 QSA profile报告](../../../../mytest/sglang_tp2_qsa_20260925_01/README.md)。
负载仍为nominal12000→5、4请求、并发1，4/4成功；真实3×12000＋1×11888 prefill、20 decode graph，与旧base一致。
每rank新QSA命中48次；真实query行分流dense17.13%、direct70.66%、union12.20%，不能用synthetic shared吞吐替代。

| 单rank累计GPU kernel时间，48次调用 | TP0 | TP1 |
| --- | ---: | ---: |
| 原sparse attention | 467.187ms | 466.883ms |
| 新attention本体 | 315.330ms | 315.345ms |
| 恢复＋校验 | 10.253ms | 10.242ms |
| plan＋sort | 8.732ms | 8.746ms |
| 新三段合计 | 334.316ms | 334.332ms |
| 三段GPU工作比值 | 1.397x | 1.396x |

TP0本体dense11.381ms、union25.123ms、direct278.826ms，按source scope＋correlation区分同名kernel。
indexer相关GPU工作209.734→210.084ms，未跳过；本负载无prefix gather。
上述不包含host调度、输出分配、未改上游操作或indexer，不是完整TTFT加速。
Median TTFT1083.10→1074.18ms；两场未交错、地址/server seed不同，candidate额外预热11888特化，mean改善不能全部归因于QSA。
profiler导出等待计入benchmark duration；不将其吞吐当无profiler基准。

采集用0.1.0固定包；之后0.1.1仅加单GPU进程防误用检查，9个runtime文件逐字相同并已机械核对，未重贴旧trace版本。
两ranktrace完整，PTL均Enabled/VECTOR,F8，服务/GPU资源已清理。保留ROCtracer重复flow警告与未完成的长上下文/模型指标/sidecar/graph等边界。

## 2026-09-25：GPU kernel 角色命名（0.1.2）

- 按用户要求，为FlyDSL添加显式profiler符号：`direct_qsa_bf16_d256`、`union_qsa_bf16_d256`、
	`dense_qsa_bf16_d256_bounded`；dense复用的共享MHA实现命名为`dense_mha_bf16_d256`。
	MHA只有这一装饰器名称变化，计算与调用接口不变。
- Triton准备kernel改为`direct_qsa_sort_blocks`与`union_qsa_scatter_membership`、
	`union_qsa_compact_membership`、`union_qsa_score_masks`，避免混淆排序与建表成本。
- [33项定向GPU测试](../../../../mytest/qsa_kernel_names_tests.xml)通过，覆盖TP2/4/8、direct两种BN、
	union局部分组、dense对齐/尾部、graph gate；fixture从实际编译IR断言四类FlyDSL符号与零spill。
- [源码重现证明](../../../../mytest/qsa_kernel_names_proof.json)逐字核对9个runtime文件：
	只有4个FlyDSL显示名和4个Triton函数名/调用变化，adapter未变。
- 新[0.1.2包](../../../../mytest/qsa_plugin_dist/pyhip_sglang_qsa_plugin-0.1.2-py3-none-any.whl)与target_v3已生成，原生插件加载预检通过。
	target_v1/v2、旧wheel及两轮trace不改写；没有重跑模型profile或性能计时，不能将命名后的产物称为旧场次实测。

## 2026-09-25：两层真实输入重放，调整分流达到半base

按用户新要求，新增数据/profile/日志统一在mytest/mydata大容量卷；不移动旧数据，不改SGLang。
[完整报告](../../../../mytest/mydata/qsa_real_study_20260925/README.md)、
[预声明协议](../../../../mytest/mydata/qsa_real_study_20260925/PROTOCOL.md)、
[实际命令](../../../../mytest/mydata/qsa_real_study_20260925/COMMANDS.md)。

- 用原TP2 profile负载捕获layer3/47、rank0/1、真实M12000/11888，8份BF16 Q/K/V及全部选择/长度/位置/scale/输出。
	只捕获profile窗口原始调用，同stream独立GPU克隆；导出trace后CPU序列化。克隆扰动该trace，性能用离线重放，不称dump零开销。
- 逐张量/文件SHA验证；原baseline两条GPU函数AST与当前SGLang一致。真实数据正确性8passed，容差仍rtol=atol=.02。
- 有效探索比较15种配置，再细化union grid1/2/4和dense2048；forcedunion或auto rho4最好。
	direct-only、BN64、BQ4、nosort等不解决目标；union_only/dense2048也未同时改善两层。
- 原rho1.5仅13.3%–21.9%非dense行走union。去重后唯一block/逐query选择block总和为21.44%–25.81%，
	聚合复用3.87–4.66×；rho中位1.74–2.14、最大3.021。rho4使本批全部非dense行走union，逐query选择和causality不变。
- 保留dense2051、effectiveBQ8、grid_multiplier2、BN32、sort=true；无计算内核修改。

### 正式10buffer/10sample验收（非探索样本）

全部8份数据在物理GPU0/PCI0000:0a:00.0重放，原cudaPerf、每buffer/scope2warmup、相邻sample逆序，
每case7scope×10sample，合计560raw。base输出预分配；full为recover+async assert+rebuild+dispatch，
不含indexer、KV gather、分配、JIT或服务调度。

| 8case范围 | full ms | 原base耗时比例 | full有效TFLOPS |
| --- | ---: | ---: | ---: |
| 原base | 9.7085–9.8237 | 100% | 28.04–28.18 |
| 原auto/rho1.5 | 6.9168–7.0800 | 70.46%–72.91% | 38.64–39.96 |
| dense+union | 3.7906–4.5679 | 39.04%–46.51% | 60.51–72.18 |
| auto/rho4 | 3.8223–4.6012 | 39.37%–46.85% | 60.07–71.58 |

两个候选在**每一份**输入上都显式断言full<=0.5base，
[性能单测8passed](../../../../mytest/mydata/qsa_real_study_20260925/formal_performance.xml)。
[最终审计/CSV](../../../../mytest/mydata/qsa_real_study_20260925/audit_final/timings.csv)从raw重算；
24份门禁均PTL Enabled/VECTOR,F8、利用率0%、VRAM<=5%，源码/实际ELF零scratch/spill核对通过。
原sweep_v1结束利用率20%失败，186raw保留但不作为性能结论；后续按原harness校验/源码冻结/end gate顺序执行，无sleep或轮询。

### 可用插件0.1.4与限制

新增`PYHIP_QSA_MODE`和`PYHIP_QSA_UNION_INFLATION`，默认仍auto/1.5；推荐本批显式auto/4以保留direct fallback。
[配置收据](../../../../mytest/mydata/qsa_real_study_20260925/selected_config.json)、
[0.1.4包](../../../../mytest/mydata/qsa_real_study_20260925/packages/pyhip_sglang_qsa_plugin-0.1.4-py3-none-any.whl)。
31项插件回归+8份真实输入经打包Registry首次prepare/后续rebuild，共
[39passed](../../../../mytest/mydata/qsa_real_study_20260925/plugin_tuned_tests.xml)；原生5hook/lazy import预检通过。
9个runtime文件与bridge均与0.1.3采集包逐字相同，历史包/trace不换标签。

不宣称无profiler端到端2×，也不宣称相对auto1.5再减半（实际下降33.7%–46.0%）。
调优后全模型服务profile、其他层/真实长prefix/并发shape及TP4/TP8性能尚未验证。

## 2026-09-25：最新源码 auto4 TP2 全模型 profile

按用户要求，从当前源码重新构建0.1.4到新的独立target，再运行原模型TP2 launcher、warmup与profile负载；
[报告/双rank trace入口](../../../../mytest/mydata/sglang_tp2_qsa_latest_20260925_01/README.md)。
没有输入克隆、没有改SGLang、没有覆盖旧profile；当前9个runtime及插件源哈希与包一致。

- 4/4请求成功，实际3×M12000+1×M11888，20decode steps；每rank48次新QSA。
- 每rank48个真实layer/layout通过原rtol=atol=.02全行对照和FP32采样；profile外验证，不作为模型准确率评测。
- 最新GPU恢复+plan+attention累计TP0/1=216.738/216.491ms，原base=467.187/466.883ms，
	新/旧46.39%/46.37%，约2.16×；旧auto1.5三段334.316/334.332ms。
- 最新attention本体198.870/198.638ms；TP0 dense11.391ms、union186.609ms、direct gate0.870ms。
- 全部12个QSA层：dense17.13%、union82.87%、direct工作行0；auto仍有direct gate/skip成本。
- Median TTFT1083.10→1013.54ms为分场profile观测，server seed/地址及预热不同，不称端到端2.16×；duration含导出等待。
- 双trace完整，QSA相关ROCm correlation无缺失/歧义；原始duplicate-flow警告保留。服务已结束，GPU0/1均回到空闲。
- [TP0单trace三表](../../../../mytest/mydata/sglang_tp2_qsa_latest_20260925_01/triage_tp0.md)显示通信/GEMM/MoE仍是主要工作；
	单trace不据此证明overlap/fusion收益。真实长prefix、其他并发shape、TP4/TP8调优性能仍未验证。

## 2026-09-25～26：Union本体、填充210T / 有效100T

### 范围和结果

- 原输入固定TP0 Layer3/47、M12000/P0、H12/HK1/D256 BF16、scale1/16。前2051行dense不计union分子/时间，9949行union全部有效选择不变。
- 最初主目标有效210T；用户在v22后明确改为**填充≥210T且有效≥100T**。各报告保留当时目标，不以修改分子重贴标签。
- 物理GPU2/0000:a4:00.0、MI308X80CU、PTL Enabled/VECTOR,F8；原cudaPerf，10buffer、2warmup、每版50sample、AB/BA，全部raw保留。
- union-only正式：Layer3原版4085.762024µs→**2790.714979µs，89.782778有效/213.148314填充T**；
	Layer47原版3843.799949µs→**2699.674606µs，92.810498有效/210.035667填充T**。
	中位时延−31.70%/−29.77%，配对speedup中位1.458785/1.421289；**有效100T未达**。
- 两版有效分子同为250558144512 FLOPs。BQ8→10、有效M96→120，填充分子L3 695205888000→594836193280，L47 662037331968→567027957760；不是用旧分子计算新时延。
- 第一次完整QSA对照包含recover/check/plan/sort/dense/union/direct，另200raw：L3 4705.845118→3432.297945µs，L47 4371.783018→3336.638093µs。
	完整有效80.533831/82.842698T、时延下降27.06%/23.68%；不是整模型TTFT或旧SGLang基线对照。

### 保留实现与ATT证据

- Mask在S0发射有界buffer读，已有S2 vmcnt(0)退休后S3消费，去掉原S5的串行mask load/wait。
	保留signed bit-select产生原FP32 bits或−inf，LLVM生成bfe/bfi；probability pack仍原half-up，原QK/PV/lazy-rescale及所有必要wait/barrier不变。
- Block表32-bit buffer寻址、固定lane mask offset、S3预算2；G12 BQ10/grid≤CU，其它G维持原BQ16/32和grid≤2CU。
- 每次rebuild增加按N64成本sort/snake任务顺序；`-1`pad一致跳过。最终每排序CTA最多4096任务，超大batch分chunk，无atomic counter或CPU回读。
- 最新[ATT manifest](../../../../mytest/mydata/qsa_union_210t_20260925_01/att_ordered/ui_output_agent_62957_dispatch_311/filenames.json#L1)：GPU2/CU1/SE0/allSIMD、dispatch311、8完整wave、worker68每wave12任务。
	891904动态MFMA/224512 DMA4，逐PC hitcount、stitch/endpgm和捕获ELF核验通过。
- 原masked S5 vmcnt等待median238/mean292.6cycles，最终不再有该VMEM等待；S5 active1156→612，S3/S4同时增加，不能仅报局部缩短。
	最终common/masked physical-SIMD MFMA union79.45%/70.08%，masked仍有10.48% VMEM completion-wait优先分类及两个约22万cycles长等待。
	不同采集worker/任务不同，不把局部busy差当整卡因果收益，不称无气泡。S0的18条buffer读实际为16DMA＋2寄存器读。

### 未采用尝试与验证

- 保留v01～v47全部候选源/IR/ELF/ISA/结果，56份报告合计1096raw（包括两个正式scope400raw）。本轮没有失败性能门禁。
- v22动态N32数值失败；v31/32六wave有VGPR spill9；v41向量四wavespill24，修复生命周期后仍慢。
	M96跳无效wave、BN32双CTA、SMEM表、lookahead、展开、wave优先级、wait后移均未胜出。
	v34贪心分配本体稍快但建表重；V预转置计时含转换，修正bank布局后收益很小，均未保留。
- 最终[88项JUnit](../../../../mytest/mydata/qsa_union_210t_20260925_01/release_functional.xml#L1)0失败/错误/跳过，1053.289s：67 BF16 MHA＋21 QSA，含8真实输入和新任务顺序边界。
	独立排序测试1passed；18deselected为未改FP8/SWA10＋性能8，原O `.02`、LSE `.002`及重复bitexact不变。
- 原source633302ef…→最终source`d71332aafa8bc92ae18045c9154eb1a5c05fce0dd881f8885b6a9125f69c2a00`。
	正式/ATT整ELF均`091fcf13c25a253615307ba29cbceaf7592365d1b95287319e43aab9d15679d5`，VGPR246/SGPR81/LDS65536/private0/spill0。
- [独立审计](../../../../mytest/mydata/qsa_union_210t_20260925_01/audit.json#L1)复算全部正式样本、两种分子、地址/源/ELF、门禁和JUnit；`complete=true`而`targets_both_met=false`。
	详细表格、原始证据和限制见[完整报告](../../../../mytest/mydata/qsa_union_210t_20260925_01/README.md#L1)。
- 所有改动在PyHIP；前轮MHA工作和四份staged CSV未动，无本轮stage/commit/push。没有重建/部署独立插件或重跑整模型profile。

## 2026-09-27：记录统一约定与10buffer三分支/完整QSA矩阵（本次开始）

- 按用户最新要求，**今后所有优化记录及性能表只追加到本文件，不再新增Markdown报告**。既有报告/源码快照/失败/原始trace保留，不回写；遗漏研究在本节后补齐。
- 独立Q/K/V/indices/output buffer恢复并保持**10个**；每实现128samples、2warmup。32buffer数据只作历史，不再是默认。
- 本轮矩阵固定TP2/4/8 local H12/H6/H3：原TP0 Layer3/47×M12000/11888；两层M12000截取前2048/2051行dense等价前缀；M2048/P30000低/高重合与M64/P30000低重合。共11输入×3TP=33case。
- dense等价前缀比较prepared dense/direct/union及完整qsa；稀疏区比较prepared direct（含每次KV pack）/强制exact union及完整qsa。稀疏区dense不可等价，表中写N/A，不能用无mask全attention冒充。
- 完整qsa逐case记录dense/union/direct行数及百分比；独立kernel+marker+copy trace记录每个kernel时长/完整event占比、pack、validation/plan与间隙。普通128样本与profile10call统计分开，不相加独立中位数。
- 所有GPU测量GPU2/a4、PTL Enabled/VECTOR,F8，原cudaPerf，entry/pre/post门禁，实际计时输出/原容差/FP32/输入/plan/源码身份核验；失败保留即停，无硬件设置写、host等待轮询或重采择快。
- TP4/8保留原TP2 KV与indices，仅取对应Q heads/参考输出，是派生local-head重放，**非真实多卡TP4/8采集**。运行时内核和1.7路由不改。原始新数据集中于mytest/mydata/qsa_unified_20260927_01，使用Python/JSON/CSV/XML而不新建Markdown。

## 2026-09-27补录：此前遗漏的整理、布局、计数器与输入研究

以下是既有证据的补录，**不是本轮重新优化或重采**。文中旧源码、旧路由、旧“最快”仅指当时版本；不覆盖上文失败记录，也不把旧ATT重贴为当前packed实现。

### 1. 2026-09-25两轮精简与单一公共接口

- 第一轮胜出路径清理：QSA固定当时auto/rho4、dense2051、sorted BN32，删除公开direct-only/BN64/nosort及调参矩阵；不是删除auto内部direct。D256分页固定已验v73，未动其它D128/192/FP8/gfx950/SWA活跃路径。
- 同一快照统计Python行数14514→11786（−2728），插件runtime185227→116799bytes。32个helper逐字移动、52个函数不变；13个实际特化的**整ELF、.text及资源相同**，两份真实输出bitexact。语义裁剪/测试合并不冒称机械搬迁。
- QSA/插件/真实输入66passed，BF16 MHA52passed；旧8capture、560raw、16个half-base断言、24门禁重新审计通过。没有新性能或模型profile。夹具的dtype/rescale/空输入/alias错误保留，未放宽原O `.02` / LSE `.002`。原baseline在特殊rescale例自身不满足独立oracle，不把该缺陷转嫁给重构。
- 第二轮收敛为唯一`qsa(q,k,v,indices,*,query_lens,prefix_lens,softmax_scale,out)`；contract与显式prepare/rebuild公共接口删除，union代码/plan合并，D256 helper并入MHA common。linear/page1/page4与SHUFFLE-5D page32/64/128具有不同ABI，保留两类内核，不误删为重复。
- 正常测试、FP32参考、8份capture回放、性能收敛到[test_qsa.py](test_qsa.py)；临时接入收敛到[sglang/plugin.py](sglang/plugin.py)、[sglang/baseline.py](sglang/baseline.py)、[sglang/profile_model.py](sglang/profile_model.py)。旧一次性工具只归档，不成为第二套永久入口。
- 第二轮133个定义逐字核对，13/13特化整ELF相同；72passed（QSA20＋BF16 MHA52），原容差和重复bitexact。0.2.0独立包原生5hook、disabled不导入Torch/FlyDSL、enabled不导入experiments；两真实输出bitexact。
- 第二轮性能在GPU0采样前use20%门禁失败，**0raw即停**；未重跑整模型，不能把旧0.1.4模型trace称为0.2.0。50→24个Python文件、14031→11943行是另一快照统计范围，不能与第一轮减行量直接累加。
- 历史证据：[第一轮报告](../../../../mytest/mydata/qsa_cleanup_20260925_01/README.md)、[第二轮报告](../../../../mytest/mydata/qsa_consolidation_20260925_01/README.md)、[133定义证明](../../../../mytest/mydata/qsa_consolidation_20260925_01/source_final/source_proof.json)、[13特化等价](../../../../mytest/mydata/qsa_consolidation_20260925_01/codegen/equivalence.json)。当前额外私有packed模块是后续性能工作，不改变单一公共API。

### 2. D256 linear 220T参考的真实范围与失败尝试

- B1连续Q10240/KV2583/H24/HK2/D256、BF16、noncausal/noLSE、grid80/512线程。原cudaPerf热`run()`，不含中央API校验/JIT/分配；10buffer、2warmup、50sample/版、AB/BA共100raw。
- 同场基线3000.074983µs / 216.672329有效T → 保留版**2932.175040µs / 221.689778有效T**，时延−2.2633%。全部慢尾保留；只此连续Full通过220T，不代表QSA短因果、稀疏direct、分页或TTFT。
- 保留四项：非尾DMA地址producer与DS/DMA合并leaf；API已验证B1后静态绑定长度；满足`0<K%64<=32`时裁剪最后N32与drain；编译器可见、数值不变的概率half-up pack仅限静态连续分支。
- 最后实际执行40×N64＋N32，故填充分子**652298158080**，实际填充**222.462T**；旧41×N64矩形会虚报225.209T，不采用。有效分子650033233920不含softmax/转换。
- 最终VGPR235→228、SGPR62→52、LDS64KiB、scratch/spill0；正式/最终/ATT整ELF`d043ace4aab22dbcc2374fdce86f46468ec76880624269944d8cde7afd2981c7`，.text`11b85dcdadde5d2a541c6c9e3722ad1b583b3884eb9055b621a371387cecd420`。
- ATT中V转置16条perm从S5分散到S4，S5缩短而S4增长；局部稳态MFMA模型82.4905%→79.1204%，**没有提高到无气泡**。整体少12288条MFMA/1536条DMA4来自裁剪尾部，不能用局部利用率替代完整计时。
- 全局启用新pack曾使page1 3048.876→3253.397µs、page4 3038.156→3140.497µs，拒绝；最终分页ELF恢复原版，隔离后未重测分页，不宣称分页220T。
- 39份收据688raw含失败：强制转置/预算/phase展开/早max/operand streaming/次序等未胜；v08/v11/v17及v19分页门禁失败；formal v19虽220.154T但结束use20%，**无效**。v21有效219.276T未达，实质改进v22才221.690T。87功能passed（MHA67＋QSA20），新增15个尾长覆盖，容差未改。
- [完整历史](../../../../mytest/mydata/mha_linear_220t_20260925_01/README.md)、[原始100样本](../../../../mytest/mydata/mha_linear_220t_20260925_01/formal_v22_linear/linear/raw.jsonl)、[独立审计](../../../../mytest/mydata/mha_linear_220t_20260925_01/audit.json)。后续causal8192的224.297有效T属于上文另一研究，不与本noncausal分母混用。

### 3. 原始K的4lane布局与当时最快raw Direct

- CPU布局研究只解释原K `[N,HK,256]` 的读取责任，**不是预先转置K**。每lane每load16B，相邻4lane合计64B；8个packet覆盖同token的512B。对lane $\ell$、packet $j$，地址为 $K_{base}+(\tau[\lfloor\ell/4\rfloor]HK+hkv)512+64j+16(\ell\bmod4)$。
- 一个N16×D256片段共8192B逻辑读取；BN32两个片段共16KiB/wave。全wave单次load分散到16个token行，不能说成连续1024B，更不能由此推出实际HBM事务/物理L1行大小。MFMA前仍需逆DS重排。
- [地址图与CPU覆盖验证](../../../../mytest/mydata/qsa_k_global_layout_20260926_01/README.md)、[verification.json](../../../../mytest/mydata/qsa_k_global_layout_20260926_01/verification.json)保留。示意块ID不是capture选块，本研究无GPU重采。
- 当时raw源码`f40c7925…`、ELF`800d13cf…`：4query waves/CTA；加载Q和64个block IDs，初始K0/K1；每轮先发V0，QK消费者`vmcnt(8)`；原FP32 max/exp/sum与RNE；发V1，PV之间发next K；`vmcnt(16)`消费V1，next K留给下一QK消费者；最后输出LDS转置。BN32每wave64条MFMA，索引每8轮更新。
- 历史L3 prepared9949行3121.816993µs，80.260356有效/108.188519填充T；当时“最快”不等于当前KV-packed，也非最优证明。匹配ATT124完整waves，513856MFMA：局部MFMA39.1397%、VMEM指令18.3120%、VMEMwait1.7920%、DS指令19.7687%、DSwait4.9666%、VALU14.5246%、其它1.4963%。
- K/V消费者等待中位均4cycle，但64-ID缓存刷新中位540/P90 700cycle；不能说所有等待已消失。仅整理已有[raw伪代码与ATT](../../../../mytest/mydata/qsa_direct_fastest_20260926_01/README.md)，原[ATT](../../../../mytest/mydata/qsa_direct_16x4_pipeline_20260926_01/att_production/direct_62860_shader_engine_0_172.att)未移动。随后typed packed-FP32关闭及KV预排已改变ELF，旧trace不能重贴当前版本。

### 4. PMC否定“免K转置使HBM流量增加4倍”

这是旧raw4lane与被拒16×4方案的对照，不是当前packed。真实TP0 L3 M12000、prepared query[2051,12000)，10buffer；read/write/cache三个独立pass共360次direct派发，90统计样本、3600原始counter行。

| 全kernel指标，中位数 | raw4lane | 16×4免转置 | 后者/前者 |
|---|---:|---:|---:|
| VMEM读指令 | 20942696 | 20942696 | 1.000000× |
| TCP tag/cache查找，含hit/miss | 332376336 | 824732880 | **2.481322×** |
| TCP读tag冲突累计cycle | 2865312 | 494744352 | **172.666834×** |
| L1→L2读请求 | 153453982.5 | 150936890.5 | 0.983597× |
| 本地HBM目标读请求 | 10124502.5 | 8871733.5 | 0.876264× |
| 本地HBM目标读字节 | 1295702592 | 1135349152 | **0.876242×** |
| 写字节 | 61126656 | 61126656 | 1.000000× |

- 写字节恰为9949×12×256×2。L2命中率92.9319%→93.8288%；没有HBM读放大4×。16lane/64B地址桶解释模型的K查找4倍加其它不变，合计恰好吻合2.481322×；这是地址模型，不是独立测得K-only HBM或物理cache-line大小。
- 本机gfx942读字节公式为 $128R_{128}+64(R-R_{128}-R_{32})+32R_{32}$，$R_{128}=\mathrm{TCC\_BUBBLE}$。全部EA读目的地分类为本地DRAM；不能用旧仅32/64B公式低估。
- 计数边界是TCC→EA/Fabric、本地HBM**目标**，不是UMC物理HBM总线；MALL未测。TCP event60为查找次数，event10为跨实例累计冲突cycle，172.67×不等于时延倍数；`TCP_GATE_EN1`才是interface-clock分母。
- CSV已聚合TCC64/TCP96/SQ16实例，不能再乘4XCC。PMC进程内PTL自动Disabled、前后Enabled，未写硬件；与普通Enabled时延是不同条件，不据PMC给普通时延作定量归因。
- [历史报告](../../../../mytest/mydata/qsa_hbm_counters_20260926_01/README.md)、[逐指标summary](../../../../mytest/mydata/qsa_hbm_counters_20260926_01/analysis/summary.json)、[配对ratio](../../../../mytest/mydata/qsa_hbm_counters_20260926_01/analysis/ratios.json)、[原始cache计数](../../../../mytest/mydata/qsa_hbm_counters_20260926_01/cache/pmc/pass_1/pmc_counter_collection.csv)。三版精确ELF匹配，选择/原输出/输入/plan/index审计通过。

### 5. 8lane连续128B仍未减少L1查找，拒绝采用

- 8相邻lane各16B读128B，N16×D256完整覆盖；保留原V地址，以native DPP或DS恢复QK操作数。CPU HK1/HK2×有效token0..16共34种覆盖与逆映射通过；不是仅凭“更连续”断言更快。
- 4lane、8lane+DPP、8lane+DS的TCP查找均**332376336**，tag冲突均**2865312**；无预期减半。DPP的HBM目标读字节1.043703×、LDS类指令1.056197×，未减少流量。分母`TCP_GATE_EN1`变大使百分比变小，不能称冲突工作减少。
- L3正式10buffer/2warmup/50sample：3121.816993→**3219.757080µs，慢3.1373%**；配对ratio中位1.031331894，全部50对都慢。DPP比DS快才进入正式，未达到比基线快1%的预定条件，不扩大正式或集成。
- 4case探索48raw＋正式100raw；原25QSA测试两候选各跑一次，另6短尾/HK2/NaN/guard边界，不冒称50个不同测试定义。VGPR246/248/250，均无spill；DPP额外64DPP/BN32及更多hazard wait只是指令事实，无ATT不能逐条定量归因全部3.14%。
- PMC另30统计样本，进程内PTL Disabled，普通时延则Enabled；别名`coalesced_stream/native_stream`在本轮指8lane DPP/DS，不是再次测16×4。[报告](../../../../mytest/mydata/qsa_direct_8lane_20260926_01/README.md)、[普通时延复算](../../../../mytest/mydata/qsa_direct_8lane_20260926_01/analysis/timing.json)、[PMC](../../../../mytest/mydata/qsa_direct_8lane_20260926_01/analysis/pmc.json)保留，生产未改。

### 6. 选择条纹、重复内容与RoPE的证据边界

- CPU确定性重建：原SGLang `random` dataset从ShareGPT抽prompt token IDs，重复/截断再decode；seed42四prompt原长749/9/107/59，重编码12000/12000/11888/12000。**不是独立均匀随机UTF-8/token**。原采集没存input_ids，不声称原prompt逐字确认。
- 第一请求lag749 token一致率100%。capture主attention K非旋转维64:256在lag748/749/750/1498的余弦：L3 **.5594/.9985/.5595/.9984**，L47 **.5709/.8911/.5723/.8866**；V lag749分别.9988/.9589。蓝色列频率在187/374块有峰，749token=187.25块，不能硬凑748token周期。
- 最近完整块选中率L3 100%、L47 99.34%；最近16块95.28%/76.27%；block0两层100%，但不是硬编码sink证明。蓝色选中率≥80%列只有7/17个，union蓝+橙计算率≥80%列为249/209，union扩张使条纹加粗。
- 实际indexer是Q4/KV1/D128，rotary_dim64、theta10000000；Q位置i、压缩4-token K位置4b，旋转项是相对位置 $R_i^TR_{4b}$，一般不抵消。主D256 Q/K并非indexer D128输入，不能用它替代消融。
- 现有证据支持**重复内容＋因果/近邻偏好＋union扩张**共同作用；没有pre-RoPE indexer Q/K/logits与固定内容的位置消融，不能量化RoPE贡献，也不说全是/完全无关RoPE。没有按749硬编码策略。
- [完整CPU证据](../../../../mytest/mydata/qsa_pattern_explanation_20260926_01/analysis.json)、[频率/自相关图](../../../../mytest/mydata/qsa_pattern_explanation_20260926_01/pattern_evidence.png)、[历史说明](../../../../mytest/mydata/qsa_pattern_explanation_20260926_01/README.md)。其中rho4/“direct待授权”和no-dense性能是当时状态，当前以1.7路由为准。

### 7. Layer3的M11888不是M12000截断

- 两份实际capture的Q/K/V从row0即不同，没有完整相同行；共同9837个稀疏query位置没有一行具有完全相同512-block集合。跨输入选块交集/512均值44.894%、中位38.867%，不是相邻query复用率。
- 当前BQ10逐组U均值1133.058/939.622、中位1180/957；填充work ratio中位1.829985/1.483771，聚合块复用4.5249×/5.4553×。完整10query组的1.7规则等价U≤1088，因此实际11888更多组走union。
- M12000：U3929/D6020；假设只截掉末112行：U3929/D5908；**真实M11888：U7909/D1928**。同样完整组中4060行D→U、80行反向，净3980，不能由少112行解释。
- 同一次profile4个名义12000请求中实际3×12000＋1×11888；重编码解释与确定性重建相符，但原prompt/tokenIDs缺失，不能定位原始112个token变化位置。没有新GPU测量或SGLang改动。
- [完整身份/路由说明](../../../../mytest/mydata/qsa_layer3_lengths_20260927_01/README.md)、[capture identity](../../../../mytest/mydata/qsa_layer3_lengths_20260927_01/analysis/capture_identity.json)、[CPU routes](../../../../mytest/mydata/qsa_layer3_lengths_20260927_01/analysis/routes.json)。本轮矩阵保留两份实际输入，不用12000截断冒充11888。

## 2026-09-27：当前源码10buffer统一TP2/4/8实测

### 1. 范围、计时与分母

- [预声明协议](../../../../mytest/mydata/qsa_unified_20260927_01/protocol.json)的33例全部完成：4真实输入＋4dense前缀＋low/high/short，分别TP2/4/8；111个实现/输入组合，**14208条普通raw**。每实现10个独立Q/K/V/indices/O、校验调用后2warmup、128samples；buffer=sample%10，所以0–7各13次、8–9各12次。实现按预定轮转/反向顺序交错，所有样本等权参与中位数，无筛尾。
- 10个输入buffer在同case的各实现间共享相同地址，输出各自独立且使用allocation起点；guard view只用于正确性。prepared direct与公开QSA各有自己的workspace scratch，均在10buffer间复用，**不是10份PK/PV，也不保证冷cache**。
- prepared dense/union不含恢复或构表；prepared direct含**每次KV pack＋attention**，不含cold prepare。完整`qsa()`包含recover/check/rebuild/sort/dense/union/pack/direct、预分配O，不含JIT、indexer、KV gather、服务调度或TP通信。各scope不能相加独立中位数。
- 所有real/dense数据来自原TP2 rank0 L3/47；TP4/8保留K/V/indices，Q与captured O取前6/3heads连续重放。TP2/H12是真实capture，TP4/H6、TP8/H3是**派生单卡本地形状**，不是分布式服务或新多卡capture。合成三TP各自seed17生成；同TP low/high QKV相同，仅选择不同；跨TP不宣称所有QKV相同。
- 稀疏prepared真实区间为query[2051,M)，M12000/11888分别9949/9837行；完整QSA仍覆盖全M。dense前缀取原M12000的Q/K/V/indices前2048或2051行，三分支严格等价。其它稀疏区dense=N/A，不测无mask的非等价dense。
- 全行原capture或baseline `.02` 对照、选定行FP32、guard、重复bitexact、CPU逐bit plan/mask与最后一次**实际采样输出**检查通过；采样后不重跑覆盖O再验。输入/计划/源码/index前后不变；用CPU从原indices独立重建全部路由和work。
- [计时源码](../../../../mytest/mydata/qsa_unified_20260927_01/measure.py)、[trace源码](../../../../mytest/mydata/qsa_unified_20260927_01/trace_driver.py)、[CPU独立审计](../../../../mytest/mydata/qsa_unified_20260927_01/analyze.py)、[结果JSON](../../../../mytest/mydata/qsa_unified_20260927_01/analysis/results.json)、[111组合CSV](../../../../mytest/mydata/qsa_unified_20260927_01/analysis/timing.csv)。每case目录保留raw、addresses、完整ELF/ISA/IR、planner及三阶段门禁。

有效分子为 $F_{eff}=4DH\sum_i n_i$，$D=256$、$n_i$为query真实有效token数。填充分子仅计实际QK/PV MFMA：

$$
F_{direct}=4D\,HK\cdot16\sum_i32\lceil n_i/32\rceil,
\qquad F_{union}=4D\,HK\sum_g128\cdot64\lceil U_g/16\rceil.
$$

direct的16是**每query的MFMA M16头槽位**，不是读取token数；无效wave不进入MFMA循环。union的U是组内4-token块并集，按整个M128/N64 tile计算，不能改成逐query direct工作量，否则掩盖union扩张。dense按每head M128 query tile的因果可见宽度向N64取整；本dense路径没有long linear的末N32裁剪。完整填充F只加实际启用组和dense，empty gate、pack、softmax/校验不增加MFMA分子。吞吐统一 $F/(t_{\mu s}10^6)$。

### 2. Dense等价前缀：三分支与完整QSA

每格为 **中位µs / 有效T / 填充T**；TP列全篇表示local H12/H6/H3。M2048 direct使用packed，M2051物理N不4对齐，使用当前raw fallback，**不是强制给packed改形状**。

| 输入 | TP | dense | direct（pack含在内，如适用） | union | 完整QSA |
|---|---:|---:|---:|---:|---:|
| dense3_2048 | 2 | 160.661 / 160.477 / 170.424 | 298.361 / 86.413 / 116.961 | 171.881 / 150.001 / 165.839 | 212.321 / 121.431 / 128.958 |
| dense3_2048 | 4 | 102.281 / 126.037 / 133.849 | 293.981 / 43.850 / 118.704 | 106.321 / 121.248 / 166.635 | 154.141 / 83.632 / 88.816 |
| dense3_2048 | 8 | 100.281 / 64.276 / 68.260 | 292.961 / 22.002 / 119.117 | 103.641 / 62.192 / 85.472 | 152.020 / 42.400 / 45.028 |
| dense3_2051 | 2 | 178.821 / 144.602 / 171.693 | 386.282 / 66.941 / 90.604 | 175.181 / 147.607 / 164.343 | 230.961 / 111.958 / 132.933 |
| dense3_2051 | 4 | 107.201 / 120.605 / 143.200 | 383.362 / 33.725 / 91.295 | 109.961 / 117.578 / 163.637 | 159.441 / 81.089 / 96.281 |
| dense3_2051 | 8 | 106.160 / 60.894 / 72.302 | 382.562 / 16.898 / 91.485 | 106.920 / 60.461 / 85.439 | 158.201 / 40.863 / 48.518 |
| dense47_2048 | 2 | 161.400 / 159.742 / 169.643 | 298.422 / 86.396 / 116.937 | 172.161 / 149.757 / 165.569 | 213.101 / 120.987 / 128.486 |
| dense47_2048 | 4 | 102.780 / 125.425 / 133.199 | 294.521 / 43.770 / 118.486 | 106.440 / 121.112 / 166.448 | 154.941 / 83.201 / 88.358 |
| dense47_2048 | 8 | 100.560 / 64.097 / 68.070 | 292.802 / 22.013 / 119.182 | 104.001 / 61.976 / 85.176 | 152.181 / 42.355 / 44.980 |
| dense47_2051 | 2 | 179.481 / 144.071 / 171.062 | 386.702 / 66.868 / 90.506 | 175.601 / 147.254 / 163.950 | 231.141 / 111.871 / 132.829 |
| dense47_2051 | 4 | 107.321 / 120.471 / 143.040 | 383.702 / 33.695 / 91.214 | 109.961 / 117.578 / 163.636 | 159.220 / 81.202 / 96.414 |
| dense47_2051 | 8 | 106.160 / 60.894 / 72.302 | 383.242 / 16.868 / 91.323 | 106.961 / 60.438 / 85.407 | 158.001 / 40.914 / 48.579 |

dense通常胜出；TP2/M2051的union本体略快，但额外构表成本不在prepared表内，不能据此自动更改完整QSA路由。

### 3. 真实稀疏后缀与完整QSA

每格仍为 **中位µs / 有效T / 填充T**。dense对以下prepared稀疏集合均为**N/A（不等价）**；direct均每次pack。完整QSA包含另2051行dense及构表，分子/行数与prepared不同。

| 输入 | TP | direct | 强制exact union | 完整QSA |
|---|---:|---:|---:|---:|
| real3_12000 | 2 | 2611.194 / 95.955 / 129.345 | 2921.875 / 85.753 / 203.580 | 2941.955 / 93.957 / 140.082 |
| real3_12000 | 4 | 2471.532 / 50.689 / 136.654 | 1962.910 / 63.823 / 215.251 | 2467.533 / 56.011 / 170.882 |
| real3_12000 | 8 | 2451.833 / 25.548 / 137.752 | 1149.506 / 54.493 / 213.016 | 1627.468 / 42.461 / 155.173 |
| real47_12000 | 2 | 2505.133 / 100.018 / 134.821 | 2697.274 / 92.893 / 210.223 | 3009.656 / 91.843 / 139.157 |
| real47_12000 | 4 | 2501.453 / 50.083 / 135.019 | 1943.050 / 64.475 / 207.901 | 2478.113 / 55.771 / 169.047 |
| real47_12000 | 8 | 2454.193 / 25.523 / 137.619 | 1127.666 / 55.548 / 209.889 | 1599.189 / 43.212 / 152.803 |
| real3_11888 | 2 | 2747.974 / 90.153 / 121.523 | 2676.613 / 92.556 / 182.511 | 2996.476 / 91.306 / 155.208 |
| real3_11888 | 4 | 2770.795 / 44.705 / 120.522 | 2004.670 / 61.790 / 175.721 | 2440.952 / 56.043 / 150.603 |
| real3_11888 | 8 | 2402.252 / 25.782 / 139.012 | 1043.965 / 59.326 / 209.112 | 1500.567 / 45.582 / 150.597 |
| real47_11888 | 2 | 2806.195 / 88.282 / 119.002 | 2699.474 / 91.773 / 200.570 | 3203.916 / 85.394 / 136.511 |
| real47_11888 | 4 | 2510.953 / 49.331 / 132.994 | 1883.390 / 65.769 / 208.536 | 2431.113 / 56.270 / 165.430 |
| real47_11888 | 8 | 2410.213 / 25.697 / 138.553 | 1137.846 / 54.431 / 210.510 | 1608.968 / 42.511 / 153.641 |

### 4. 高低重合与短M

low/high：M2048、P30000、N32048；short：M64、P30000、N30064、低重合。dense均**N/A（不等价）**。每格 **中位µs / 有效T / 填充T**。

| 输入 | TP | direct（含pack） | 强制exact union | 完整QSA |
|---|---:|---:|---:|---:|
| low | 2 | 589.523 / 87.490 / 117.934 | 2127.872 / 24.239 / 194.172 | 689.423 / 74.812 / 100.845 |
| low | 4 | 584.583 / 44.115 / 118.930 | 1900.250 / 13.571 / 182.468 | 677.564 / 38.061 / 102.610 |
| low | 8 | 582.983 / 22.118 / 119.257 | 1291.707 / 9.982 / 179.078 | 671.683 / 19.197 / 103.508 |
| high | 2 | 574.863 / 89.721 / 120.941 | 405.642 / 127.150 / 171.126 | 527.922 / 97.699 / 131.488 |
| high | 4 | 570.343 / 45.216 / 121.900 | 214.301 / 120.339 / 165.344 | 321.061 / 80.323 / 110.364 |
| high | 8 | 568.163 / 22.695 / 122.368 | 109.841 / 117.391 / 161.294 | 211.241 / 61.041 / 83.870 |
| short | 2 | 111.940 / 14.399 / 19.409 | 695.364 / 2.318 / 18.614 | 145.140 / 11.105 / 14.969 |
| short | 4 | 111.041 / 7.258 / 19.566 | 917.625 / 0.878 / 11.619 | 143.801 / 5.604 / 15.109 |
| short | 8 | 109.761 / 3.671 / 19.794 | 1197.846 / 0.336 / 5.876 | 141.161 / 2.855 / 15.391 |

本轮低重合direct比union快约2.22–3.61×，高重合union比direct快约1.42–5.17×；这是本体scope，不是整模型倍数。M64 union有效并行任务少且组并集扩大，本体更慢；不能因填充吞吐接近或更高就选union。

### 5. 完整QSA的query数与占比

真实输入每格 **行数（占全部M的百分比）**，不是时间比例；所有数字从GPU plan与原indices CPU复核。最后一列单独使用稀疏行分母。

| 输入 | TP | dense | union | direct | 稀疏query内union/direct % |
|---|---:|---:|---:|---:|---:|
| real3_12000 | 2 | 2051（17.092%） | 3929（32.742%） | 6020（50.167%） | 39.491 / 60.509 |
| real3_12000 | 4 | 2051（17.092%） | 9341（77.842%） | 608（5.067%） | 93.889 / 6.111 |
| real3_12000 | 8 | 2051（17.092%） | 9949（82.908%） | 0（0%） | 100 / 0 |
| real47_12000 | 2 | 2051（17.092%） | 4269（35.575%） | 5680（47.333%） | 42.909 / 57.091 |
| real47_12000 | 4 | 2051（17.092%） | 9933（82.775%） | 16（0.133%） | 99.839 / 0.161 |
| real47_12000 | 8 | 2051（17.092%） | 9949（82.908%） | 0（0%） | 100 / 0 |
| real3_11888 | 2 | 2051（17.253%） | 7909（66.529%） | 1928（16.218%） | 80.401 / 19.599 |
| real3_11888 | 4 | 2051（17.253%） | 9837（82.747%） | 0（0%） | 100 / 0 |
| real3_11888 | 8 | 2051（17.253%） | 9837（82.747%） | 0（0%） | 100 / 0 |
| real47_11888 | 2 | 2051（17.253%） | 5599（47.098%） | 4238（35.649%） | 56.918 / 43.082 |
| real47_11888 | 4 | 2051（17.253%） | 9613（80.863%） | 224（1.884%） | 97.723 / 2.277 |
| real47_11888 | 8 | 2051（17.253%） | 9837（82.747%） | 0（0%） | 100 / 0 |

以下分流在TP2/4/8一致；dense3与dense47分别保留独立计时，但路由相同。

| 输入组 | TP | dense行/% | union行/% | direct行/% |
|---|---|---:|---:|---:|
| dense3_2048、dense47_2048 | 2、4、8 | 2048 / 100% | 0 / 0% | 0 / 0% |
| dense3_2051、dense47_2051 | 2、4、8 | 2051 / 100% | 0 / 0% | 0 / 0% |
| low | 2、4、8 | 0 / 0% | 0 / 0% | 2048 / 100% |
| high | 2、4、8 | 0 / 0% | 2048 / 100% | 0 / 0% |
| short | 2、4、8 | 0 / 0% | 0 / 0% | 64 / 100% |

[全33case三分支query CSV](../../../../mytest/mydata/qsa_unified_20260927_01/analysis/queries.csv)。当前TP8真实后缀全union，但公开调用仍提交gated direct/pack检查，时间不严格为0。

### 6. 完整QSA逐kernel占比：口径与名称

- 另开33个rocprofv3进程，kernel＋marker＋copy selected regions，每case仍10独立buffer、2warmup后10call，**330call全部保留**；无ATT/PMC。不是上方普通128样本的阶段拆分，不与它们拼成一次测量。
- 下面每格均为 **每call平均µs / 占完整GPU event的%**。分子是该kernel在同10call中的duration总和，分母是同10call的`cudaPerf`完整event总和；不是kernel各自median相加，也不是query百分比。
- CSV与JSON逐dispatch时间戳/correlation一致，按ROCTx external correlation归属每call；整数ns求和，并确认同call kernel无重叠。3570次dispatch中只排除原timer已在event前的**330个spin**，其它3240个QSA kernel全部统计；无copy、未知kernel。ROCTx区间含spin，不能拿其CPU区间作分母。
- `residual`＝完整event−全部QSA kernel时长，包含发射间隙、事件边界及profile相关停顿；不是另一个kernel或可直接消除的开销。首call及长尾不删，短M时residual占比很大，尤其不能据此倒推普通运行阶段时长。
- FlyDSL产物从实际warmed callable完整导出，每case trace ELF均匹配其普通测量；code-object注册的符号/内存尺寸对应，未把另一shape或旧ATT的二进制换标签。Triton产物也保存；Torch reduce的trace `Scratch_Size=12`属于Torch验证kernel，不把“FlyDSL attention零scratch/spill”扩展成所有kernel均零scratch。
- 下表使用一对一短名避免重复超长C++模板；**没有在profiler重命名kernel**，完整字符串在[symbols.json](../../../../mytest/mydata/qsa_unified_20260927_01/analysis/symbols.json)。每项是单一kernel，不把多个kernel暗中合并。

| 表内短名 | 实际符号/唯一模板识别 | 职责 |
|---|---|---|
| `recover` | `_qsa_recover_blocks` | 恢复、排序、校验4-token块 |
| `compare_eq` | `vectorized_elementwise_kernel`，`CompareEqFunctor<int>` | errors==0 |
| `boolean_all` | `reduce_kernel<512,1,...ReduceOp<bool,...>>` | 错误位全量归约 |
| `assert_async` | `_assert_async_cuda_kernel<bool>` | GPU异步断言 |
| `membership_zero` | `vectorized_elementwise_kernel`，`FillFunctor<int>` | 清零membership |
| `scatter` | `union_qsa_scatter_membership` | 逐query散射成员 |
| `compact` | `union_qsa_compact_membership` | 并集压缩及GPU gate |
| `score_masks` | `union_qsa_score_masks` | 精确逐query mask |
| `order_tasks` | `union_qsa_order_tasks` | 任务排序/蛇形 |
| `dense` | `dense_qsa_bf16_d256_bounded` | dense前缀；N64对齐时内部用native DMA |
| `union` | `union_qsa_bf16_d256` | union attention或空gate |
| `pack` | `direct_pack_kv_bf16_d256` | 每次KV预排或空gate |
| `direct` | `direct_qsa_bf16_d256` | packed direct attention或空gate |

原始数据：[逐kernel CSV](../../../../mytest/mydata/qsa_unified_20260927_01/analysis/kernels.csv)、[330call纳秒分解](../../../../mytest/mydata/qsa_unified_20260927_01/analysis/calls.json)、[首例原始kernel CSV](../../../../mytest/mydata/qsa_unified_20260927_01/real3_12000_tp2/trace/qsa_kernel_trace.csv)。同结构覆盖所有33case。

#### real3_12000

| kernel / 统计项 | TP2 µs / % | TP4 µs / % | TP8 µs / % |
|---|---:|---:|---:|
| recover | 199.201 / 6.687% | 198.909 / 7.802% | 198.717 / 11.649% |
| compare_eq | 5.240 / 0.176% | 5.512 / 0.216% | 5.508 / 0.323% |
| boolean_all | 7.604 / 0.255% | 7.364 / 0.289% | 7.076 / 0.415% |
| assert_async | 4.472 / 0.150% | 4.476 / 0.176% | 4.472 / 0.262% |
| membership_zero | 6.568 / 0.220% | 6.088 / 0.239% | 5.388 / 0.316% |
| scatter | 40.536 / 1.361% | 40.048 / 1.571% | 44.556 / 2.612% |
| compact | 16.508 / 0.554% | 18.436 / 0.723% | 11.096 / 0.650% |
| score_masks | 18.932 / 0.636% | 55.496 / 2.177% | 85.108 / 4.989% |
| order_tasks | 19.664 / 0.660% | 8.096 / 0.318% | 5.164 / 0.303% |
| dense | 180.369 / 6.055% | 107.329 / 4.210% | 105.793 / 6.202% |
| union | 845.144 / 28.372% | 1856.354 / 72.813% | 1159.990 / 67.999% |
| pack | 10.628 / 0.357% | 8.328 / 0.327% | 4.756 / 0.279% |
| direct | 1575.480 / 52.890% | 176.265 / 6.914% | 11.076 / 0.649% |
| residual | 48.436 / 1.626% | 56.768 / 2.227% | 57.184 / 3.352% |
| full mean | 2978.784 / 100% | 2549.469 / 100% | 1705.885 / 100% |
| full median（µs） | 2908.275 | 2470.093 | 1636.888 |

#### real47_12000

| kernel / 统计项 | TP2 µs / % | TP4 µs / % | TP8 µs / % |
|---|---:|---:|---:|
| recover | 199.053 / 6.477% | 198.645 / 7.960% | 198.781 / 12.044% |
| compare_eq | 5.240 / 0.171% | 5.284 / 0.212% | 5.496 / 0.333% |
| boolean_all | 7.548 / 0.246% | 7.368 / 0.295% | 7.228 / 0.438% |
| assert_async | 4.520 / 0.147% | 4.604 / 0.184% | 4.656 / 0.282% |
| membership_zero | 6.532 / 0.213% | 6.452 / 0.259% | 5.368 / 0.325% |
| scatter | 39.428 / 1.283% | 38.576 / 1.546% | 43.788 / 2.653% |
| compact | 17.668 / 0.575% | 18.452 / 0.739% | 11.184 / 0.678% |
| score_masks | 23.144 / 0.753% | 57.588 / 2.308% | 79.841 / 4.838% |
| order_tasks | 19.808 / 0.645% | 8.108 / 0.325% | 5.220 / 0.316% |
| dense | 180.161 / 5.862% | 107.577 / 4.311% | 105.976 / 6.421% |
| union | 974.333 / 31.704% | 1897.798 / 76.046% | 1134.998 / 68.770% |
| pack | 10.496 / 0.342% | 8.408 / 0.337% | 4.588 / 0.278% |
| direct | 1542.620 / 50.196% | 103.084 / 4.131% | 11.004 / 0.667% |
| residual | 42.620 / 1.387% | 33.632 / 1.348% | 32.292 / 1.957% |
| full mean | 3073.172 / 100% | 2495.577 / 100% | 1650.421 / 100% |
| full median（µs） | 3005.236 | 2449.693 | 1608.568 |

#### real3_11888

| kernel / 统计项 | TP2 µs / % | TP4 µs / % | TP8 µs / % |
|---|---:|---:|---:|
| recover | 197.069 / 6.790% | 197.201 / 9.068% | 197.105 / 12.742% |
| compare_eq | 5.472 / 0.189% | 5.532 / 0.254% | 5.268 / 0.341% |
| boolean_all | 7.648 / 0.264% | 7.172 / 0.330% | 7.332 / 0.474% |
| assert_async | 4.696 / 0.162% | 4.520 / 0.208% | 4.544 / 0.294% |
| membership_zero | 6.504 / 0.224% | 6.072 / 0.279% | 5.596 / 0.362% |
| scatter | 37.528 / 1.293% | 37.180 / 1.710% | 40.108 / 2.593% |
| compact | 24.804 / 0.855% | 18.316 / 0.842% | 10.936 / 0.707% |
| score_masks | 30.196 / 1.040% | 47.188 / 2.170% | 69.557 / 4.497% |
| order_tasks | 15.132 / 0.521% | 9.112 / 0.419% | 4.920 / 0.318% |
| dense | 180.205 / 6.209% | 107.573 / 4.946% | 105.996 / 6.852% |
| union | 1737.657 / 59.875% | 1661.789 / 76.414% | 1055.050 / 68.205% |
| pack | 8.568 / 0.295% | 4.732 / 0.218% | 4.404 / 0.285% |
| direct | 603.115 / 20.782% | 11.428 / 0.525% | 23.912 / 1.546% |
| residual | 43.544 / 1.500% | 56.908 / 2.617% | 12.160 / 0.786% |
| full mean | 2902.139 / 100% | 2174.724 / 100% | 1546.888 / 100% |
| full median（µs） | 2850.575 | 2106.711 | 1508.868 |

#### real47_11888

| kernel / 统计项 | TP2 µs / % | TP4 µs / % | TP8 µs / % |
|---|---:|---:|---:|
| recover | 197.281 / 6.673% | 196.925 / 8.127% | 197.313 / 11.723% |
| compare_eq | 5.416 / 0.183% | 5.308 / 0.219% | 5.524 / 0.328% |
| boolean_all | 7.696 / 0.260% | 7.296 / 0.301% | 7.240 / 0.430% |
| assert_async | 4.552 / 0.154% | 4.496 / 0.186% | 4.568 / 0.271% |
| membership_zero | 6.536 / 0.221% | 6.376 / 0.263% | 5.332 / 0.317% |
| scatter | 38.948 / 1.317% | 38.372 / 1.584% | 40.184 / 2.388% |
| compact | 20.588 / 0.696% | 31.348 / 1.294% | 11.080 / 0.658% |
| score_masks | 24.628 / 0.833% | 54.580 / 2.252% | 84.756 / 5.036% |
| order_tasks | 15.288 / 0.517% | 9.156 / 0.378% | 5.096 / 0.303% |
| dense | 180.593 / 6.109% | 107.769 / 4.447% | 105.917 / 6.293% |
| union | 1229.194 / 41.578% | 1792.313 / 73.964% | 1144.970 / 68.028% |
| pack | 10.232 / 0.346% | 8.632 / 0.356% | 4.624 / 0.275% |
| direct | 1178.226 / 39.854% | 134.609 / 5.555% | 10.780 / 0.640% |
| residual | 37.212 / 1.259% | 26.032 / 1.074% | 55.708 / 3.310% |
| full mean | 2956.391 / 100% | 2423.213 / 100% | 1683.093 / 100% |
| full median（µs） | 2887.775 | 2382.132 | 1616.689 |

#### dense3_2048

下面四张dense前缀表均只启动recover/三个Torch验证kernel/dense；五个plan kernel、union、pack、direct均**未启动，0µs/0%**，不是略去空gate。每格仍为平均µs / 完整event百分比。

| kernel / 统计项 | TP2 µs / % | TP4 µs / % | TP8 µs / % |
|---|---:|---:|---:|
| recover | 40.984 / 15.309% | 41.168 / 22.505% | 41.088 / 19.869% |
| compare_eq | 5.260 / 1.965% | 5.172 / 2.827% | 4.888 / 2.364% |
| boolean_all | 4.948 / 1.848% | 5.136 / 2.808% | 5.020 / 2.428% |
| assert_async | 4.472 / 1.670% | 4.460 / 2.438% | 4.200 / 2.031% |
| dense | 185.225 / 69.190% | 100.992 / 55.207% | 111.961 / 54.140% |
| residual | 26.816 / 10.017% | 26.004 / 14.215% | 39.640 / 19.169% |
| full mean | 267.705 / 100% | 182.933 / 100% | 206.797 / 100% |
| full median（µs） | 220.261 | 160.481 | 159.041 |

#### dense3_2051

| kernel / 统计项 | TP2 µs / % | TP4 µs / % | TP8 µs / % |
|---|---:|---:|---:|
| recover | 50.380 / 15.420% | 41.200 / 19.274% | 41.052 / 20.068% |
| compare_eq | 5.728 / 1.753% | 4.936 / 2.309% | 5.188 / 2.536% |
| boolean_all | 6.324 / 1.936% | 5.260 / 2.461% | 5.332 / 2.607% |
| assert_async | 4.952 / 1.516% | 4.228 / 1.978% | 4.468 / 2.184% |
| dense | 230.665 / 70.603% | 105.889 / 49.536% | 117.865 / 57.617% |
| residual | 28.660 / 8.772% | 52.248 / 24.442% | 30.660 / 14.988% |
| full mean | 326.710 / 100% | 213.761 / 100% | 204.565 / 100% |
| full median（µs） | 307.101 | 165.781 | 164.380 |

#### dense47_2048

| kernel / 统计项 | TP2 µs / % | TP4 µs / % | TP8 µs / % |
|---|---:|---:|---:|
| recover | 41.152 / 15.356% | 44.616 / 22.174% | 41.088 / 18.555% |
| compare_eq | 5.056 / 1.887% | 4.912 / 2.441% | 5.176 / 2.337% |
| boolean_all | 5.132 / 1.915% | 4.948 / 2.459% | 5.184 / 2.341% |
| assert_async | 4.292 / 1.602% | 4.184 / 2.079% | 4.380 / 1.978% |
| dense | 161.693 / 60.335% | 101.376 / 50.384% | 113.005 / 51.031% |
| residual | 50.668 / 18.907% | 41.172 / 20.462% | 52.612 / 23.759% |
| full mean | 267.993 / 100% | 201.209 / 100% | 221.445 / 100% |
| full median（µs） | 220.281 | 160.901 | 159.181 |

#### dense47_2051

| kernel / 统计项 | TP2 µs / % | TP4 µs / % | TP8 µs / % |
|---|---:|---:|---:|
| recover | 44.900 / 15.144% | 41.240 / 20.702% | 41.220 / 21.385% |
| compare_eq | 5.084 / 1.715% | 5.008 / 2.514% | 5.136 / 2.665% |
| boolean_all | 5.276 / 1.779% | 5.212 / 2.616% | 5.392 / 2.797% |
| assert_async | 4.468 / 1.507% | 4.184 / 2.100% | 4.456 / 2.312% |
| dense | 178.397 / 60.169% | 105.865 / 53.143% | 104.445 / 54.186% |
| residual | 58.368 / 19.686% | 37.700 / 18.925% | 32.104 / 16.656% |
| full mean | 296.494 / 100% | 199.209 / 100% | 192.753 / 100% |
| full median（µs） | 238.921 | 165.301 | 164.020 |

#### low

以下low/high/short都没有dense query，dense未启动为**0µs/0%**；其它实际启动项逐一保留。

| kernel / 统计项 | TP2 µs / % | TP4 µs / % | TP8 µs / % |
|---|---:|---:|---:|
| recover | 54.184 / 6.787% | 41.032 / 5.023% | 41.020 / 5.227% |
| compare_eq | 5.064 / 0.634% | 5.080 / 0.622% | 5.232 / 0.667% |
| boolean_all | 5.076 / 0.636% | 5.252 / 0.643% | 5.304 / 0.676% |
| assert_async | 4.328 / 0.542% | 4.460 / 0.546% | 4.496 / 0.573% |
| membership_zero | 6.028 / 0.755% | 5.464 / 0.669% | 5.028 / 0.641% |
| scatter | 25.696 / 3.219% | 38.860 / 4.758% | 25.216 / 3.213% |
| compact | 4.820 / 0.604% | 4.764 / 0.583% | 4.744 / 0.605% |
| score_masks | 4.372 / 0.548% | 4.528 / 0.554% | 4.456 / 0.568% |
| order_tasks | 4.616 / 0.578% | 4.580 / 0.561% | 4.456 / 0.568% |
| union | 4.536 / 0.568% | 4.724 / 0.578% | 4.764 / 0.607% |
| pack | 24.236 / 3.036% | 24.268 / 2.971% | 24.168 / 3.080% |
| direct | 584.835 / 73.253% | 576.575 / 70.588% | 572.975 / 73.018% |
| residual | 70.580 / 8.841% | 97.228 / 11.903% | 82.844 / 10.557% |
| full mean | 798.372 / 100% | 816.816 / 100% | 784.704 / 100% |
| full median（µs） | 710.684 | 703.124 | 695.764 |

#### high

| kernel / 统计项 | TP2 µs / % | TP4 µs / % | TP8 µs / % |
|---|---:|---:|---:|
| recover | 41.580 / 7.106% | 41.384 / 10.424% | 54.264 / 17.034% |
| compare_eq | 5.376 / 0.919% | 5.052 / 1.272% | 5.160 / 1.620% |
| boolean_all | 5.172 / 0.884% | 5.264 / 1.326% | 5.088 / 1.597% |
| assert_async | 4.480 / 0.766% | 4.340 / 1.093% | 4.500 / 1.413% |
| membership_zero | 6.000 / 1.025% | 5.384 / 1.356% | 5.052 / 1.586% |
| scatter | 27.832 / 4.757% | 26.484 / 6.671% | 40.148 / 12.603% |
| compact | 17.496 / 2.990% | 13.760 / 3.466% | 11.220 / 3.522% |
| score_masks | 13.712 / 2.343% | 4.772 / 1.202% | 4.872 / 1.529% |
| order_tasks | 4.580 / 0.783% | 4.448 / 1.120% | 4.380 / 1.375% |
| union | 405.330 / 69.273% | 213.941 / 53.886% | 108.420 / 34.034% |
| pack | 4.732 / 0.809% | 4.720 / 1.189% | 4.912 / 1.542% |
| direct | 3.884 / 0.664% | 3.920 / 0.987% | 4.044 / 1.269% |
| residual | 44.948 / 7.682% | 63.556 / 16.008% | 66.504 / 20.876% |
| full mean | 585.123 / 100% | 397.026 / 100% | 318.566 / 100% |
| full median（µs） | 540.942 | 335.141 | 228.961 |

#### short

| kernel / 统计项 | TP2 µs / % | TP4 µs / % | TP8 µs / % |
|---|---:|---:|---:|
| recover | 6.824 / 2.857% | 6.848 / 2.446% | 7.152 / 2.447% |
| compare_eq | 5.228 / 2.189% | 5.296 / 1.891% | 5.788 / 1.980% |
| boolean_all | 5.028 / 2.105% | 5.124 / 1.830% | 5.100 / 1.745% |
| assert_async | 4.576 / 1.916% | 4.524 / 1.616% | 4.432 / 1.516% |
| membership_zero | 4.820 / 2.018% | 4.684 / 1.673% | 4.752 / 1.626% |
| scatter | 4.660 / 1.951% | 4.648 / 1.660% | 4.544 / 1.554% |
| compact | 4.648 / 1.946% | 4.628 / 1.653% | 4.656 / 1.593% |
| score_masks | 4.492 / 1.881% | 4.468 / 1.596% | 4.452 / 1.523% |
| order_tasks | 4.288 / 1.795% | 4.408 / 1.574% | 4.424 / 1.513% |
| union | 4.512 / 1.889% | 4.560 / 1.629% | 4.472 / 1.530% |
| pack | 24.156 / 10.115% | 24.020 / 8.578% | 37.228 / 12.736% |
| direct | 110.073 / 46.090% | 95.109 / 33.966% | 93.104 / 31.851% |
| residual | 55.516 / 23.246% | 111.693 / 39.889% | 112.208 / 38.386% |
| full mean | 238.821 / 100% | 280.009 / 100% | 292.313 / 100% |
| full median（µs） | 175.061 | 170.441 | 169.620 |

### 7. 慢尾、条件差异与本轮结论

- **10buffer不保证消除时间漂移。** 本轮real3/M12000/TP2的四个32sample区间direct中位为2487.65/3457.54/3285.98/2501.97µs，union为2788.09/3783.30/3589.76/2786.98µs，QSA为2903.30/3859.74/3762.48/2905.82µs；多路径同时波动，最终表仍用全部128样本。
- real3/M11888/TP4完整QSA四段中位2098.05/2814.31/2834.19/2098.35µs，全128中位2440.952µs；不能拿两段快值或另一场profile中位2106.711µs替换。32sample是描述性切片，不是32独立buffer；实际始终10个buffer轮转。
- 普通完整QSA的极端长尾：real3/M12000/TP4 **100826.088µs**、TP8 **96823.311µs**，real47/M12000/TP2 **69417.839µs**；dense3/M2051/TP2也有11986.503µs。所有原始样本及每scope min/max、32sample切片、10buffer轮次统计保留在[结果JSON](../../../../mytest/mydata/qsa_unified_20260927_01/analysis/results.json)。未测运行期间频率/热/外部负载，原因未确定，不能归因buffer/cache/clock或“必然JIT”。
- profile另场10call中位/均值有不同扰动：例如dense3/M2051/TP2 profile中位307.101µs，而普通中位230.961µs；short/TP4 residual39.889%、TP8 38.386%。本表是这次trace的实际份额，不推断为无profiler稳态或端到端服务成本。
- 当前低重合优先direct、高重合优先union；TP2真实两个M12000后缀prepared direct较union快，但TP4/8的union更有利。公开路由还包括局部组选择、构表和dense，不能仅按整段prepared中位决定每个query组。
- real47/M12000/TP2 direct本场100.018有效T，real3仅95.955T；**不宣称两层同时达100T**。所有当前direct填充吞吐仍低于160T；不拿union215T或完整QSA170.882T替代direct160目标。没有采用前轮被拒phasepg8或修改生产路由。

### 8. 本轮交付校验

- [审计](../../../../mytest/mydata/qsa_unified_20260927_01/analysis/audit.json)：33普通case、111组合、14208raw；33trace、330call、3570dispatch，330预event spin剔除、3240QSA dispatch全部归属。普通＋trace共**198门禁通过**，GPU2/a4、PTL Enabled/VECTOR,F8、use最大0%、VRAM最大2%；不代表运行时固定时钟。
- 实际输出、原`.02`＋FP32、guard、输入hash、完整plan/mask、source、全git index均通过；全部FlyDSL attention/pack产物零private/spill。trace CSV/JSON一致、同call无kernel重叠；0未分类/0copy/0PMC/0ATT。采集时源与匹配普通产物的哈希保存到每case，不以AST检查代替GPU数值。
- 永久[test_qsa.py](test_qsa.py)已恢复`BENCHMARK_BUFFERS=10`，`BENCHMARK_SAMPLES=128`保留，CLI help同步。[CPU配置回归](../../../../mytest/mydata/qsa_unified_20260927_01/config_check.json)验证默认、10/50覆盖、非法bounds及TP2/4/8转发；对32buffer版仅常量/help两个AST节点改变，正常测试与benchmark函数体不变。此默认修改没有重跑全部44pytest，不把33case测量冒称44项回归。
- runtime源码仍为qsa`5e2bdce5…`、union`811dafc2…`、dense`db71b1c3…`、direct`c31466f4…`、packed`88c6ec5b…`；当前测试SHA`724054a511ce6b6fb76305ac283b0b1b3e033db60ff7158354408f5e142f0c73`。未改SGLang、硬件、历史包或旧结果，未stage/commit/push，绘图服务保持运行。
- 新研究只创建Python/JSON/CSV/二进制等证据，**没有新增Markdown文件**；已有README仅纠正当前默认与入口，所有研究补录和表格追加本日志。只读结构探针曾误从`strings`取`kernel_symbols`而KeyError，修正取顶层后分析通过，没有GPU失败或重采。

## 2026-09-27：Direct pack内存上限与完整调用/逐kernel伪代码

### 1. PK+PV上限：64MiB/工作区，超出回退raw

- [direct.py](direct.py)新增内部常量`MAX_PACKED_KV_BYTES = 64 * 1024 * 1024`，限制**两份额外packed scratch合计**，不是每份64MiB，也不包含调用者原始K/V。
- `prepare()`用`K.numel()*K.element_size() + V.numel()*V.element_size()`计算；只有不超过上限、单请求、物理N四token对齐且原uint32偏移安全条件成立才分配PK/PV。**等于上限允许，严格超过不分配任何PK/PV**，直接生成原4-wave raw DirectPlan，且不调用pack kernel。
- BF16/D256下所需bytes为`1024*N*HK`，故HK1/2/4分别最多N65536/32768/16384。与TP本地Q heads多少无关；若HK改变则按实际HK计，不按模型max context或direct活跃行数估算。
- N12000/HK1仍12,288,000B；N32048仍32,817,152B；N65536恰64MiB可pack，N65540超过；N262144/HK1需256MiB，现回退raw。上文旧256MiB描述是原无预算时的容量公式，不再表示当前会分配。
- `_Workspace`原本就令`union.packed_direct = (direct.packed_key is not None)`；超限后自然为False，因此分流使用raw已有的`U*r <= 4*sum(selected_blocks_including_tail)`，**不是继续拿packed的1.7交叉点配raw**。dense范围和逐query选择集合不变；fallback不等于全query强制direct，union仍可承担适合的组。
- 决策只在cold plan创建时读取host tensor元数据，无`.cpu()`/`.item()`/`.tolist()`、无GPU active回读或`mem_get_info()`探测，不捕获OOM再补救。64MiB是静态资源策略，不是性能测出的最佳阈值，也不保证低于阈值绝不会OOM。
- 限额是**每DirectPlan/工作区**的PK+PV，不限制indices、union plan、输出或Torch临时tensor；多个stream/layout、pinned graph会累计，进程总量可以大于64MiB。原普通LRU/graph pinning不改；不是全局内存池或动态配置。现有进程的旧缓存/旧graph不会因修改源码自动转换，应使用新进程加载新版。
- 保持10独立I/O buffer、128sample默认；本轮只改host预算、增回归及文档，**不重采性能**。上轮33case的KV均低于64MiB，路径不变，但旧性能收据仍属于当时冻结源码，不重贴新版本。

验证结果：

- 新CPU测试在旧代码上按预期失败：N65540/HK1在超限后仍调用`empty_like`，[失败JUnit](../../../../mytest/mydata/qsa_pack_limit_20260927_01/host_before.xml)保留。
- CPU meta tensor覆盖HK1/2/4、上限下4行/恰上限/超4行共9种；禁止GPU回读、free-memory查询及同步，检查超限0次packed分配。
- GPU TP2/4/8×HK1/HK2×恰上限/超4行共12种边界，原`.02`全68行FP32、guard、重复bitexact、shared/independent图内切换和V更新通过；超限时禁止`empty_like`与packed `run()`，确认实际raw路径/分流。独立边界4项passed168.15s，见[边界JUnit](../../../../mytest/mydata/qsa_pack_limit_20260927_01/boundary_after.xml)。
- 完整唯一入口正常回归**48passed、24性能项未选**，112.80s；14条上游Torch JIT弃用warning保留。包括原8份capture×TP2/4/8、SGLang backend流程及新预算边界，见[完整JUnit](../../../../mytest/mydata/qsa_pack_limit_20260927_01/normal_all.xml)。
- [源码核验](../../../../mytest/mydata/qsa_pack_limit_20260927_01/source_check.json)：所有原正常测试AST不变，direct除host `prepare()`外的函数/类逐字不变，其它运行时源未改。真实M12000/TP2的dense/union/gated与ungated packed **4个完整ELF及.text匹配上一轮**，零private/spill，[产物核验](../../../../mytest/mydata/qsa_pack_limit_20260927_01/codegen_check.json)。这不是超限路径性能结论。
- 伪代码下列PK/PV置换公式另经CPU逐元素1024项双射及两步DS bit交换核验，[置换验证](../../../../mytest/mydata/qsa_pack_limit_20260927_01/pack_permutation.json)。Pylance语法检查通过；原环境`reportMissingImports`未靠代码shim掩盖。

### 2. 记号与完整QSA伪代码（CPU调度）

以下为**当前实现的逻辑伪代码**，不是可运行API，不承诺逐指令/逐cycle等同ISA。保留真实分流、舍入、边界和流水关系；`parallel`、`reduce`等概括lane协作，不能据此重排FP32加法树或删wait/barrier。`softmax_scale`默认1/16；exp实际为AMD FP32 `exp2`近似指令。

| 记号 | 含义 |
|---|---|
| `Q[M,H,256]`、`K/V[N,HK,256]` | 连续BF16；`G=H/HK<=16` |
| `I[M,2051]` | 原只读逐query token IDs；请求内相对编号 |
| `pos[i]`、`seq[i]`、`kv_len[s]` | host长度生成的device元数据 |
| `RB[M,512]`、`errors[M]` | 私有排序后的完整块IDs、ABI错误位 |
| `meta[g]` | `(first_query,rows,request_K_base,kv_len,first_position)` |
| `DM[g,b]` | uint32 membership位图，bit q表示组内query q选中块b |
| `UB/UM[g,j]`、`counts[g]` | compact block/member表；`(U,common_N64_tiles)` |
| `active[g]`、`query_tile_id[i]` | 1=union、0=direct；dense行tile ID为−1 |
| `Masks[g,nt,q,quarter]` | 32-bit word的低16位对应本lane四个4-token块 |
| `PK/PV` | 额外packed K/V，元素数与原K/V相同，但逻辑布局不同 |

实现入口：[qsa.py](qsa.py)；plan选择：[direct.py](direct.py)、[union.py](union.py)；dense视图：[dense.py](dense.py)。

```text
function qsa(Q, K, V, I, host_query_lens=None, host_prefix_lens=None,
			 softmax_scale=None, out=None):
	validate host shape/dtype/device/byte spans/contiguity/alignment
	require BF16 D256, H%HK==0, H/HK<=16, gfx942, inference-only
	default queries=(M,); default prefix=(N-M,) for one request, else all zero
	validate host lengths exactly describe packed Q and KV; scale finite and >0
	validate I is contiguous int32[M,2051] on the same GPU
	O = out if supplied else empty_like(Q)
	validate O; forbid O overlap with Q/K/V/I
	if M==0: return O

	with process lock and Q.device:
		require this process uses one GPU
		stream = current_stream(Q.device)                  # 每次取当前stream
		key = (device, stream, query_lens, prefix_lens, H, HK, scale)
		W = workspace_cache.get(key)
		if W is absent:
			if graph_capture: error("warm this layout/stream first")
			W = make_workspace_from_host_shapes(...)       # cold allocation
			insert W; evict noncaptured old layouts while cache length>8
			# captured项保留，可能使总数超过8
		mark W recently used; W.captured |= graph_capture
		bind W metadata to this call's Q/K/V/I              # 不缓存旧输入指针

		launch _qsa_recover_blocks(I, pos, kv_len, seq, RB, errors)
		ok_elements = launch CompareEq(errors, 0)
		ok = launch BooleanAll(ok_elements)
		launch _assert_async_cuda_kernel(ok)               # 无CPU回读

		if W.union exists:
			launch membership_zero(DM)
			launch union_qsa_scatter_membership(RB, pos, meta, DM)
			launch union_qsa_compact_membership(DM, UB, UM, counts, active,
												PACKED_DIRECT=W.union.packed_direct)
			launch union_qsa_score_masks(UB, UM, counts, meta, active, Masks)
			launch union_qsa_order_tasks(counts, active, Order)

		for request in W.dense.calls:
			launch dense_qsa_bf16_d256_bounded(Q_view, K_view, V_view, O_view)
		if W.union exists:
			launch union_qsa_bf16_d256(..., Masks, active, Order, O)
			if W.direct.PK exists:
				launch direct_pack_kv_bf16_d256(K,V,PK,PV,active)
				launch direct_qsa_bf16_d256(Q,PK,PV,...,O)   # 1 query wave/CTA
			else:
				launch direct_qsa_bf16_d256(Q,K,V,...,O)     # raw 4 waves/CTA
	return O                                               # hot返回不等待GPU完成
```

同一stream内launch按序依赖；公开路径即使全部union/全部direct，也照常launch另一分支的GPU gate检查。没有稀疏后缀时不建union/direct，不launch这两类kernel。benchmark的event前`spin_kernel`不是QSA的一部分。

```text
function make_workspace_from_host_shapes(...):
	create CU offsets, kv_len, pos, seq, RB[M,512], errors[M]
	for request s:
		dense_count[s] = min(query_len[s], max(0,2051-prefix[s]))
		prepare dense Q/O views for first dense_count[s] rows
		prepare K/V views of prefix[s]+dense_count[s] rows
		# 一次cold prepare核对CU元数据；hot run不回读
	if sum(dense_count)==M: return dense-only workspace

	BQ = min(32, floor(128/G)) if G==12
		 else min(32, largest_power_of_two_not_above(128/G))
	# 当前TP2/4/8的BQ为10/16/32；沿原请求query边界对齐
	group each request from dense_count[s] to next BQ boundary, then by BQ
	allocate union metadata/DM/UB/UM/Masks/counts/active/Order/query_tile_id

	packed_bytes = K.numel*K.element_size + V.numel*V.element_size
	packed = one_request and N%4==0 and packed_bytes<=64*1024*1024
			 and K.numel*2 + HK*1536 + 512 < 2^32
	waves = 1 if packed else 4
	build direct metadata in groups of waves after each dense prefix
	if packed and direct has tasks: allocate PK like K and PV like V
	else: PK=PV=None                                      # 不先分配再丢弃
	union.packed_direct = (PK is not None)
	direct.source_blocks aliases RB                       # 不重复排序
	direct.active/query_tile_id alias union's device plan
	return workspace
```

### 3. 恢复与验证：4个GPU kernel

#### 3.1 `_qsa_recover_blocks`

每个program处理一行query；完整块最多512个，slots向4096取整仅为并行检查，实际I仍2051列。直接输出原duplicate检查已经排序的RB，不再额外启动direct排序。

```text
kernel _qsa_recover_blocks(row i):
	p = pos[i]; length = kv_len[seq[i]]; visible = p+1
	complete = min(visible//4,512); n = 4*complete + visible%4
	for j in parallel 0..511:
		(a,b,c,d) = I[i,4*j : 4*j+4]
		full[j] = j<complete
		valid[j] = a>=0 and a%4==0 and (b,c,d)==(a+1,a+2,a+3)
				   and d<visible and d<length
		block[j] = a//4 if full[j] and valid[j] else -1
	ordered = sort_ascending(block[j] if full[j] else INT32_MAX)
	duplicate = any(j>0 and j<complete and ordered[j]==ordered[j-1])

	bad = p<0 or visible>length or any(full and not valid) or duplicate
	for slot in parallel 0..4095:
		token = bounded_load(I[i,slot], slot<2051, default=-1)
		if slot<2051:
			if slot>=n: bad |= token!=-1
			else:
				bad |= token<0 or token>=visible or token>=length
				if slot>=4*complete:
					bad |= token != 4*(visible//4) + slot-4*complete
	RB[i,j] = ordered[j] if j<complete else -1
	errors[i] = int(bad)
```

#### 3.2 Torch `CompareEqFunctor<int>` elementwise kernel

```text
kernel compare_eq(errors, ok_elements):
	for i in parallel valid indices:
		ok_elements[i] = (errors[i]==0)
```

#### 3.3 Torch `reduce_kernel<...,ReduceOp<bool,...>>`

```text
kernel boolean_all(ok_elements, ok):
	each participating thread reduces its bool elements with logical AND
	combine thread/warp/block partials with AND, identity=True
	write ok = AND(ok_elements[0:M])
```

线程/向量宽度由Torch模板与输入shape选择，不是固定独立QSA kernel实现；上轮实测符号为512线程版本。

#### 3.4 Torch `_assert_async_cuda_kernel<bool>`

```text
kernel assert_async(ok):
	if not ok[0]: trigger device assertion("Invalid compressed QSA token/block/tail ABI")
```

错误异步传播，不调用`ok.item()`；无效I不是可以继续使用的attention输入。上面三个Torch kernel可以分配小bool临时tensor，不因PK/PV有界就宣称整个hot path零分配。

### 4. Union plan：5个GPU kernel

#### 4.1 Torch `FillFunctor<int>`（membership_zero）

```text
kernel membership_zero(DM):
	for element in parallel DM: element = 0
```

这是`DM.zero_()`的实际GPU工作，每次rebuild执行；未启用组的旧compact/mask存储不要求全部清零，消费者必须受active/common gate保护。

#### 4.2 `union_qsa_scatter_membership`

```text
kernel union_qsa_scatter_membership(group g, local_query q):
	(first,rows,_,_,_) = meta[g]
	if q>=rows: return
	i=first+q; bit=uint32(1)<<q
	for block in parallel RB[i,0:512]:
		if block>=0: atomic_OR(DM[g,block], bit, relaxed)
	visible=pos[i]+1
	if visible%4!=0:
		atomic_OR(DM[g,visible//4], bit, relaxed)            # 含不完整尾块
```

只合并执行读取的并集，不把其余query没选的token当作可见；精确屏蔽由score_masks负责。

#### 4.3 `union_qsa_compact_membership`

```text
kernel union_qsa_compact_membership(group g, PACKED_DIRECT):
	bits[b] = DM[g,b] for valid physical blocks, else 0
	present[b] = bits[b]!=0
	U=sum(present); r=meta[g].rows; p0=meta[g].first_position
	for q in parallel 0..31:
		visible[q]=p0+q+1
		s[q]=min(visible[q]//4,512) + int(visible[q]%4!=0)
		n[q]=4*min(visible[q]//4,512) + visible[q]%4
	if PACKED_DIRECT:
		steps=sum(ceil(n[q]/32) for q<r)
		enabled = 160*ceil(U/16) <= 17*steps
	else:
		enabled = U*r <= 4*sum(s[q] for q<r)                # budget/raw回退
		# prepared forced-union研究可用RHO=inf；公开路径固定4
	active[g]=enabled; counts[g]=(U,0)
	if not enabled: return

	all_queries = (uint32(0xffffffff) >> (32-r))
	common[b] = present[b] and bits[b]==all_queries and 4*b+3<=p0
	other[b] = present[b] and not common[b]
	C=sum(common)
	destination[b] = inclusive_scan(common)[b]-1 if common[b]
					 else C+inclusive_scan(other)[b]-1
	for b where present[b]:
		UB[g,destination[b]]=b
		UM[g,destination[b]]=bits[b]
	counts[g].common_N64_tiles = C//16
```

common块按原block ID递增置于前部，其余块也递增；只对完整16块common组免mask。末尾不足16个common块与其它块一起进入masked tile。

#### 4.4 `union_qsa_score_masks`

```text
kernel union_qsa_score_masks(group g):
	if active[g]==0: return
	U, common_tiles = counts[g]
	r=meta[g].rows; p0=meta[g].first_position; length=meta[g].kv_len
	for nt in common_tiles .. ceil(U/16)-1:
		for (q,quarter) in parallel [0:BQ) x [0:4):
			word=uint32(0)
			for group in 0..3:
				slot=16*nt + 2*quarter + 8*(group//2) + group%2
				safe=(slot<U and q<r)
				member=bounded_load(UM[g,slot], safe, 0)
				block=bounded_load(UB[g,slot], safe, 0)
				selected=safe and ((member>>q)&1)!=0
				for offset in 0..3:
					token=4*block+offset
					keep=selected and token<=p0+q and token<length
					word |= uint32(keep) << (4*group+offset)
			Masks[g,nt,q,quarter]=word
```

未选中成员、未来token、物理尾部、填充slot均为0。每word低16位直接对应attention内每lane的16个score；不能简单按全并集做无mask softmax。

#### 4.5 `union_qsa_order_tasks`

```text
kernel union_qsa_order_tasks(sort_chunk):
	# SLICES=ceil(BQ*G/128)，当前G<=16且BQ受限，实际为1
	# TASKS=num_groups*HK*SLICES; SIZE<=4096; GRID为固定worker数
	rank = chunk*SIZE + parallel_range(0,SIZE)
	valid = rank<TASKS
	group = (rank % (TASKS//HK)) // SLICES
	cost = ceil(counts[group].U/16) if valid and active[group] else 0
	key = (int64(cost)<<SHIFT) + (TASKS-1-rank) if valid else -1
	sorted_key = sort_descending(key)
	task = TASKS-1-(sorted_key & ((1<<SHIFT)-1)) if sorted_key>=0 else -1
	row,column = divmod(rank,GRID)
	destination = row*GRID + (column if row even else GRID-1-column)
	bounded_store(Order[destination],task, destination<ceil(TASKS/GRID)*GRID)
```

同cost按原task ID递增，inactive成本0、padding任务−1。每chunk独立排序，不是跨所有任务的全局排序；长任务优先加蛇形分摊，没有原子任务队列/CPU active计数。

### 5. Dense/union共有的数值与八阶段流水（内部逻辑，非额外kernel）

两者使用BM128/BN64/D256、512线程/8wave、64KiB LDS和4+4wave错相。dense按单head的128query组织M行；union按query×GQA head组织M行。**其概率BF16转换为half-up，而direct为RNE；不可统一成同一舍入说明。** 三类输出O最终均为软件BF16 RNE。

```text
s = softmax_scale * log2(e)
half_up_bf16(x) = high16(uint32_bits(x)+0x8000)
RNE_bf16(x) = high16(uint32_bits(x)+0x7fff+((uint32_bits(x)>>16)&1))

function dense_or_union_tile_pipeline(NT, load_selected_KV, mask_scores):
	DMA first K tile to LDS; wait VMEM/LDS; CTA barriers
	S0 = mask_scores(QK_FP32_MFMA(Q,K0), tile=0)
	m = max(row_max(S0)*s, -1e30) + 1
	z = FMA(S0,s,-m); l=0; A_low=A_high=0
	prefetch bounded next K
	for t=1 .. NT-1:
		# 逻辑依赖如下；真实发射按下表S0..S7交织
		S = mask_scores(QK_FP32_MFMA(Q,K[t]), tile=t)
		P = exp2_FP32(z)                                   # 上一个tile
		l += row_sum_FP32(P)                               # 用未量化概率求分母
		Pb = half_up_bf16(P)
		A += PV_FP32_MFMA(Pb,V[t-1])
		candidate = row_max(S)*s
		changed = candidate > m+7                         # 原lazy-rescale规则
		new_m = candidate+1 if changed else m
		next_z = FMA(S,s,-new_m)
		if any_lane_in_wave(changed):
			alpha=exp2_FP32(m-new_m)
			A*=alpha; l*=alpha                             # 历史+上一tile一起变基准
		m=new_m; z=next_z
		keep next K prefetch and required publish/overwrite waits
	load final V; P=exp2_FP32(z)
	l+=row_sum_FP32(P); A+=PV_FP32_MFMA(half_up_bf16(P),V[last])
	retire VMEM/LDS; CTA barriers
	O=RNE_bf16(A*(1/l if l>0 else 0))
	LDS output transpose; bounded stores only to valid query/head rows
```

这里的`row_sum/row_max`指当前helper的局部pair-tree与跨lane归约，不授权更换结合顺序。lazy阈值7与偏置1均为原FP32/log2域行为，不改成每tile标准max后声称bitexact；实际S0..S7还把summary/center交织进PV。

| 阶段 | 核心工作与交织 |
|---|---|
| S0 Memory | V(t−1) global→LDS DMA与K(t)低半LDS读；union masked路径在此预取mask |
| S1 Compute | QK(t)低半MFMA与上一tile的exp2交织 |
| S2 Memory | K(t)高半LDS读，V发布与必要VMEM/LDS等待、CTA rendezvous；union mask在此退休 |
| S3 Compute | QK(t)高半；上一tile概率sum/BF16 half-up pack；union应用mask bitselect，准备后续K地址 |
| S4 Memory | K(t+1) DMA与V(t−1)低半LDS读；跨lane sum及部分V operand重排 |
| S5 Compute | PV低D128与累积row_sum/当前score max交织；dense在此处理当前因果mask |
| S6 Memory | V(t−1)高半LDS读，跨lane max与必要K发布等待 |
| S7 Compute | PV高D128与当前score center交织，lazy rescale A/l，完成barrier |

这是两kernel内部阶段，不是8次launch。prologue/drain另有等待；有交织不等于MFMA100%忙。证据与确切helper分别在[union.py](union.py)、[dense.py](dense.py)、[MHA common](../mha/_common.py)、[linear PV helper](../mha/mha_pa_bf16_256_linear_942.py)。

### 6. `dense_qsa_bf16_d256_bounded`伪代码

每个非空eligible请求独立launch。host view长度`Qn=t`、`Kn=P+t`，bottom-right因果位置为`P+q`，不打包Q/K/V；满足N64对齐时仍是此符号，内部切native DMA。最后N64不会套用long noncausal linear的N32裁剪。

```text
kernel dense_qsa_bf16_d256_bounded(Qview,Kview,Vview,Oview,Qn,Kn):
	allocate 64KiB shared LDS
	blocks=ceil(Qn/128); tasks=H*blocks
	for work=blockIdx.x; work<ceil(tasks/CUS)*CUS; work+=CUS:
		row,column=divmod(work,CUS)
		rank=row*CUS + (column if row even else CUS-1-column)
		if rank>=tasks: continue                            # 全CTA一致跳过
		head=rank%H; qb=blocks-1-rank//H                   # 最长因果tile优先
		q0=128*qb; valid=min(128,Qn-q0); hkv=head//G
		NT=min(ceil(Kn/64),ceil((q0+valid+Kn-Qn)/64))
		each wave loads its 16 Q rows for this head
		load_KV(t)=original K/V[hkv] tokens [64*t:64*t+64)
		mask_score(q,k)=keep iff k<Kn and k<=Kn-Qn+q
		use dense_or_union_tile_pipeline(NT,load_KV,mask_score)
		# 4+4wave错相，必须保留所有consumer/producer waits和CTA barriers
		store only q<Qn to this request's original O view
```

非对齐物理尾使用完整byte VOFFSET让descriptor检查真正全偏移，不能只在score处mask却越界DMA。其它请求/dense外行不写。

### 7. `union_qsa_bf16_d256`伪代码

```text
kernel union_qsa_bf16_d256(..., Masks, active, Order):
	allocate 64KiB shared LDS
	for work=blockIdx.x; work<ceil(TASKS/GRID)*GRID; work+=GRID:
		ordered=Order[work]
		if ordered<0: continue
		mapped=(ordered%(TASKS//HK))*HK + ordered//(TASKS//HK)
		g=mapped//(HK*SLICES); hkv=mapped%HK
		row_offset=((mapped//HK)%SLICES)*128
		if active[g]==0: continue                           # 全CTA一致，未读旧mask
		first,r,k0,length,p0=meta[g]
		U,common_tiles=counts[g]; NT=ceil(U/16)
		for row_slot in this CTA's M128:
			q,head=divmod(row_offset+row_slot,G)
			valid=q<r and q<BQ and head<G
			Qfragment=bounded_load(Q[first+q,hkv*G+head,:],valid,zero)
		load_KV(t):
			for compact token position j=0..63:
				slot=16*t+j//4
				block=UB[g,min(slot,U-1)]                   # 有界预取可重复最后块
				token=4*block+j%4
				read original K/V[k0+token,hkv,:] into LDS  # 无KV pack；物理尾有界
		mask_scores(S,t):
			if t<common_tiles: return S
			bits=Masks[g,t,min(q,BQ-1),lane//16]
			return original_score_bits where bits[i]==1, else -inf
		run dense_or_union_tile_pipeline(NT,load_KV,mask_scores)
		# common前缀分组展开免mask；后续S0预取mask、S3消费
		write valid query/head results through LDS output transpose
```

传入kernel名为`MEMBERS`的参数实际是`plan.score_masks`，不是`UM`原始membership表。精确成员/因果限制已包含在mask里。物理KV不是4对齐时切full-VOFFSET DMA；compact padding/最后V drain仍受边界保护，不能误称直接读完所有物理尾都是安全的。

### 8. `direct_pack_kv_bf16_d256`伪代码与元素布局

只有host已允许PK/PV时才launch；超64MiB完全不启动该kernel。公开路径为gated版本，每个wave在GPU检查所有union组是否存在`active==0`，无需CPU返回计数。全union仍有检查launch成本，但不读写K/V/PK/PV。

```text
kernel direct_pack_kv_bf16_d256(K,V,PK,PV,active,GATED):
	if GATED:
		lane=threadIdx.x%64; needed=False
		for start=0 .. active_count-1 step64:
			i=start+lane
			value=active[min(i,active_count-1)]
			needed |= i<active_count and value==0
		if ballot(needed)==0: return                       # 此wave无需做pack

	# 256线程/4wave；两wave对应同block/head的D低/高128
	# 一个CTA分4轮处理8个(block,KVhead)单位
	half=(threadIdx.x//64)%2
	base_pair=blockIdx.x*8 + threadIdx.x//128
	for u=0..3:
		pair=base_pair+2*u; block,head=divmod(pair,HK)
		if pair>=N//4*HK: bounded no-op
		for (token_in_block,dim) assigned to this wave/half:
			PK[location(block,head,Jk(token_in_block,dim))] = K[4*block+token_in_block,head,dim]
			PV[location(block,head,Jv(token_in_block,dim))] = V[4*block+token_in_block,head,dim]
```

对`t∈[0,4)`、`d∈[0,256)`，以下为**BF16元素偏移**（raw字节地址还要×2）：

```text
location(b,h,j) = ((4*b + j//256)*HK + h)*256 + j%256
Jk(t,d) = 256*(d//64) + 128*((d%64)//32) + 32*((d%32)//8) + 8*t + d%8
Jv(t,d) = 512*(d//128) + 128*((d%8)//2) + 8*((d%128)//8) + 4*(d%2) + t
```

同4-token/head单位中各1024个元素都写一次，无重复/遗漏。实际K每lane读取16B后直接写映射地址；V交换两个token bit与两个register-index bit（两轮native DS），再组合16-bit半字。**只搬BF16 bits，不量化、不做attention、不改公开K/V**。有direct工作时pack当前整个物理KV容量，非只pack本轮活跃query选中块；每次调用/graph replay刷新，不能跨不同KV复用旧内容。

### 9. `direct_qsa_bf16_d256`：packed单wave版本

此符号也用于raw版本，必须结合plan/ELF区分。packed每CTA一个query、一个KV head，BF16 MFMA的M16容纳该query的G个Q heads；H12/6/3并未把M tile变成12/6/3。

```text
kernel direct_qsa_bf16_d256_packed(Q,PK,PV,O,RB,meta,active,query_tile_id):
	task=blockIdx.x; tile,hkv=divmod(task,HK)
	first,rows,k0,length,p0=meta[tile]                       # rows=1
	if GATED and active[query_tile_id[first]]==1: return    # 全CTA一致
	i=first; visible=p0+1
	complete=min(visible//4,512); n=4*complete+visible%4
	NT=ceil(n/32)
	s=FP32(softmax_scale*log2(e))
	load Q[i,hkv*G : hkv*G+G,:], pad unused heads to M16
	cache=RB[i,0:64]                                      # 每lane一个block ID
	selected_token(j):
		if j<4*complete: return 4*cached_block(j//4)+j%4
		if j<n: return 4*(visible//4)+j-4*complete
		return descriptor_out_of_bounds
	prefetch first K0/K1 (two N16 fragments) from PK
	m=-1e30; l=0; A_low=A_high=0
	for t=0 .. NT-1:
		derive PV addresses from current packed K block bases using DS
		issue V0_low/V0_high (first N16, two D128 halves)
		if (t+1)%8==0:
			issue next 64 IDs at chunk=min((t+1)//8,7)
		wait until previous K ready (vmcnt(8), compiler dependencies retained)
		S0,S1=paired_QK_FP32_MFMA(Q,K0,K1)                  # 无K数据逆DS转置
		scores=FP32(concatenate(S0,S1)*s); mask j>=n to -inf
		m_new=max(m,packed_row_max(scores))
		alpha=exp2_FP32(m-m_new)
		P=exp2_FP32(scores-m_new)
		l=l*alpha+packed_row_sum(P)                        # 分母用FP32 P
		if any_lane(alpha!=1): A*=alpha
		wait V0 ready; wait/load next IDs at cache boundary
		P0,P1=BF16_RNE(P)                                 # 与union half-up不同
		issue V1_low/V1_high (second N16)
		A_low += PV_MFMA(P0,V0_low); issue bounded next K0
		A_high+= PV_MFMA(P0,V0_high); issue bounded next K1
		wait V1 ready (vmcnt(16), next K可继续在途)
		A_low += PV_MFMA(P1,V1_low)
		A_high+= PV_MFMA(P1,V1_high)
		m=m_new
	wait final bounded K prefetch; barrier
	O=BF16_RNE(A*(1/l if l>0 else 0))
	transpose through 8KiB LDS; guarded store valid G heads
```

- V从packed物理4-token块读取，最后query可能只见其中1–3个token；`_mask_value_tail`在PV前**把未选future token对应BF16 bits清0**，避免`0*NaN`污染，不能只依赖score/概率mask。
- `packed_row_sum/max`保留lane16/32/48并行交换及现有归约树；QK仍两条独立N16累加链，PV八组D16通道。PV周围`s_setprio(2)→0`是wave仲裁，不改GPU频率。
- next K偏移clamp到最后合法N16片段，最后迭代仍有有界重复预取；伪代码中的prefetch不是允许删除尾wait。两kernel由同一编译host launcher按stream先pack后attention提交。

### 10. `direct_qsa_bf16_d256`：raw fallback四wave版本

超64MiB、ragged或物理N非4对齐等情况使用[direct.py](direct.py)中的同名raw实现。公开API、选择集合、FP32累加与BF16 RNE不变；不是把packed plan的buffer简单清空，而是cold prepare重新生成4query metadata。

```text
kernel direct_qsa_bf16_d256_raw(Q,K,V,O,RB,meta,active,query_tile_id):
	task=blockIdx.x; tile,hkv=divmod(task,HK)
	first,rows,k0,length,p0=meta[tile]                       # 1<=rows<=4
	if GATED:
		needed = any(active[query_tile_id[first+min(w,rows-1)]]==0 for w=0..3)
		if not needed: return                             # 必须整个CTA一起退出
	for wave w in parallel 0..3:
		safe_row=first+min(w,rows-1)
		valid = w<rows and (not GATED or active[query_tile_id[safe_row]]==0)
		i=first+w; visible=p0+w+1
		n=4*min(visible//4,512)+visible%4
		NT=ceil(n/32) if valid else 0
		load Q and first64 sorted IDs; invalid waves用安全/OOB地址
		selected token = 4*RB[i,j//4]+j%4 for complete-block slots,
		                 causal tail token for final n%4 slots
		raw byte address = ((k0+token)*HK+hkv)*512 + channel_bytes
		prefetch raw K0/K1:
			adjacent four lanes read one token's contiguous64B
			eight16B/lane packets cover each N16×D256 fragment
		m=-1e30; l=0; A=0
		for t=0..NT-1:
			DS broadcast current K token-base addresses for V
			issue V0 low/high; wait K at consumer (vmcnt(8))
			inverse-lane DS transpose K0/K1 into native QK operands
			S0,S1=paired_QK_FP32_MFMA(Q,K0,K1)
			scores=FP32(concatenate(S0,S1)*FP32(softmax_scale*log2(e)))
			mask slots>=n to -inf
			new_m=max(m,raw_row_max(scores)); alpha=exp2_FP32(m-new_m)
			P=exp2_FP32(scores-new_m)
			l=l*alpha+raw_row_sum(P); A*=alpha; m=new_m     # 原raw归约树，每轮rescale
			wait V0; every8iterations load next64 IDs and wait
			RNE-pack P0/P1; issue V1 low/high
			byte-permute raw V operands immediately before each PV
			PV(P0,V0_low), next K0 prefetch
			PV(P0,V0_high), next K1 prefetch
			wait V1 (vmcnt(16)); PV(P1,V1_low/high)
		wait outstanding VMEM
	all four waves rendezvous even if invalid/union-routed
	normalize and RNE pack; output transpose through32KiB LDS
	guarded stores only for valid direct query/head lanes
```

raw query组可以跨union的BQ边界，所以CTA先判断“是否任何wave需direct”，内部再逐wave gate；padding wave必须查夹紧到本组有效行的tile ID，不能读dense行的−1。无效wave可跳MFMA loop，但不能绕过共享输出barrier提前退出。

### 11. 调用清单与文档范围检查

- 恢复/校验4个＋plan5个＋dense＋union＋pack＋direct，完整packed单请求通常共13个kernel；raw取消pack为12个；纯dense为5个。多请求dense按eligible请求数分别launch；没有所有场景固定13次的保证。
- `dense_or_union_tile_pipeline`、lane重排、输出shuffle、原生DMA是内联/内部逻辑，不是新增GPU kernel。旧`direct_qsa_sort_blocks`已被恢复排序复用删除；QSA不在此重新做indexer/Top-K或KV gather。计时器spin仅属测试框架。
- 此节对应64MiB host上限后的当前源码；所有旧优化/性能/失败/ATT记录保留原身份。本轮只在PyHIP改代码与追加现有文档，无SGLang修改、硬件写、stage/commit/push或新Markdown。

## 2026-09-27：为什么real3_12000完整QSA慢于最快prepared分支

本节只复算已有[统一结果](../../../../mytest/mydata/qsa_unified_20260927_01/analysis/results.json)、[逐call纳秒分解](../../../../mytest/mydata/qsa_unified_20260927_01/analysis/calls.json)及原始128sample，不重跑GPU或改分流。**prepared分支与完整QSA不是同一计时范围；当前混合分流也不保证取整批最快分支。**

### 1. 普通计时的差距存在，但比较范围不同

| real3_12000，本地TP | prepared direct µs | prepared union µs | 完整QSA µs | 完整QSA相对最快prepared |
|---|---:|---:|---:|---:|
| TP2 / H12 | 2611.194 | 2921.875 | 2941.955 | +330.761µs / +12.667% |
| TP4 / H6 | 2471.532 | 1962.910 | 2467.533 | +504.623µs / +25.708% |
| TP8 / H3 | 2451.833 | 1149.506 | 1627.468 | +477.962µs / +41.580% |

以上是各自全部128样本的中位数，以及中位数之差/比；不是逐sample差值或ratio的中位数。TP4/8仍为原TP2 capture的派生local-head重放。

- [实际测量边界](../../../../mytest/mydata/qsa_unified_20260927_01/measure.py)：real case的`begin=2051`。prepared direct和union都只计算**query[2051,12000)的9949行**，前2051行保持sentinel；恢复、校验、建表已在计时前完成。direct包含每次pack，union只运行已构好的精确并集attention。
- 完整`qsa()`计算**全部12000行**：还要算前2051行dense，并且每次运行恢复/排序/错误检查、membership清零/scatter/compact/mask/task sort；按真实输入重建，不能直接复用上一调用的选择结果。
- 本轮是预分配输出、预热后的hot call，不把差距解释为首次JIT、I/O buffer克隆或cold PK/PV allocation。dense/union/direct写入query集合互斥，不是把同一query重复计算后相加。

### 2. 同一完整调用的trace把额外工作直接量出

下表使用另场**同10个profile call的平均µs**；逐kernel整数ns求和，分解项来自同一批调用，因此可以相加。不是上表普通128sample中位数的精确差值分解，不能跨两场直接相减推断分流收益。

| 完整QSA内部工作 | TP2 | TP4 | TP8 |
|---|---:|---:|---:|
| 恢复/排序＋3个Torch验证kernel | 216.517 | 216.261 | 215.773 |
| 5个union构表/排序kernel | 102.208 | 128.165 | 151.313 |
| 前2051行dense | 180.369 | 107.329 | 105.793 |
| union attention | 845.144 | 1856.354 | 1159.990 |
| direct pack | 10.628 | 8.328 | 4.756 |
| direct attention | 1575.480 | 176.265 | 11.076 |
| event/发射间隙等residual | 48.436 | 56.768 | 57.184 |
| **稀疏部分：union＋pack＋direct** | **2431.253** | **2040.947** | **1175.822** |
| **其余：恢复＋构表＋dense＋residual** | **547.531** | **508.523** | **530.063** |
| **完整event均值** | **2978.784** | **2549.469** | **1705.885** |

TP2和TP4分别约**548µs/509µs**的工作与间隙不属于“只计算后9949行”的稀疏attention本体，占本场完整event的18.381%/19.946%。TP8已经无direct工作行，仍有约530µs的其它成本及空gate launch，说明不能简单把完整QSA预计为prepared union的1.15ms。

`residual`不是独立kernel，也不全是可删除开销。所有launch在同一stream按序执行；混合分支耗时相加，**不是并行执行后取max，更不是取min(direct,union)**。不能按query比例直接缩放两条prepared时延，因为各组选择、填充、task数量、gate扫描及实际运行条件不同。

### 3. 为什么还会选到整批较慢的分支

| 本地TP | dense query | union query | direct query | 稀疏query内U/D |
|---|---:|---:|---:|---:|
| TP2 | 2051 | 3929 | 6020 | 39.491% / 60.509% |
| TP4 | 2051 | 9341 | 608 | 93.889% / 6.111% |
| TP8 | 2051 | 9949 | 0 | 100% / 0% |

- 分流在每个query组上按`10*union_padded_work <= 17*direct_padded_work`决定，**不是先对整批两个kernel测速取最小值**。整段direct较快不表示每一个局部组都更适合direct；整段union较快也不能证明每一个局部组都更快。
- 但反过来也不能宣称当前1.7就是完整调用最优：它是已测交叉点的静态近似，没有显式计入“为少数query额外启动direct＋pack”的整批成本、不同TP的task并行度及两分支都启动时的成本。
- TP4只有608行走direct，仍付出本场direct176.265µs＋pack8.328µs。是否值得把这些行从union移出，需要同场对照其在union中的实际增量，**现有prepared中位数与另场profile不能证明**。TP2的3929行union同理。
- 因此本表足以解释“完整调用为什么比本体慢”，**不足以证明混合QSA优于完整的全direct或全union方案**。这不是已经证实的数值错误，也不是已经证明的最优分流。

正确的同scope后续对照应是：①恢复/校验＋dense＋全稀疏direct（含pack，不建union计划）；②恢复/校验＋dense＋强制全稀疏union（每次构表）；③当前完整auto。三者均输出全部12000行、相同验证/分配边界，10独立buffer交错测量；不能拿现有prepared数字替代这两个尚未测量的完整方案，也不直接相加独立中位数。

此case的PK+PV仅12,288,000B，TP2/4/8的HK均1，未触发新64MiB上限；此次差距不是预算fallback造成。现有普通计时中的多段变慢/长尾仍全部保留，原因未确定，本节不归因频率或cache。

## 2026-09-27：TODO总表（新增5D layout与准备kernel优化）

本轮按用户“记录”要求登记，**没有实现5D支持、没有改准备kernel或启动测试/性能采集**。本表汇总QSA、相关SGLang接入及D256 MHA参考中仍未闭环的事项；旧章节保留当时状态，本表区分待办、可选后续、已完成和已否决，避免把历史TODO重复当作新任务。

### 1. 用户新增的两项主要待办

- [ ] **QSA-T01：为QSA增加5D layout支持。** 当前[qsa.py](qsa.py)明确要求Q/K/V均为3D；相邻MHA的5D能力不等于QSA已支持。先核对并对齐既有[MHA SHUFFLE-5D合同](../mha/mha_pa_bf16_942.py#L801-L856)：Q仍为`[M,H,D]`，K为`[Npage,HK,D/8,S,8]`，V为`[Npage,HK,S/8,D,8]`，参考page size为32/64/128。这是现有可复用ABI的起点，实施前明确页表/stride/逻辑长度，不能把任意五维reshape都叫支持。
	- 保留现有3D入口和单一公共`qsa()`，明确logical selected token/block到physical page/slot的映射、每请求prefix与最后一页有效长度；不能把logical indices直接当physical地址。
	- 评估dense、union、direct的原生5D KV读取及有界fallback，目标包括避免不必要的全KV gather/转换；若首版需要转换，必须单列并纳入完整调用成本，不能藏在计时外。5D缓存布局也不等于当前4-token PK/PV私有布局，不能直接混用descriptor。
	- 验收覆盖TP2/4/8本地形状、GQA/HK、不同页大小/非连续页表、prefix/ragged/页尾、NaN与输出guard、stream/graph更新；与等价3D结果及FP32参考按原容差比较，保留原选择/causality。PK+PV合计64MiB预算、超限fallback与无GPU回读约束继续成立。

- [ ] **QSA-T02：简化并优化准备kernel的开销。** 当前有稀疏后缀的完整调用在attention前有4个恢复/验证kernel和5个union准备kernel。已有real3_12000 trace中，恢复/验证加构表的TP2/4/8均值约**318.726/344.426/367.086µs**；不包括dense、pack或residual，不代表这些时间可全部省掉。
	- 分别核对`_qsa_recover_blocks`中的展开读取/排序/重复检查、三个Torch错误验证kernel的launch与bool临时tensor，以及membership清零/scatter/compact/score_masks/task sort的读取、写入和launch成本。
	- 优先消除可证明冗余的中间结果与重复遍历，评估有依据的融合/更紧凑布局，并区分cold host metadata与每层必须刷新的device选择。已有的恢复排序复用、common免mask、S0 mask预取/S3 bitselect已经完成，不再当作待实现收益。
	- 不删有效性/duplicate/tail/padding检查，不放宽数值容差，不跨层/跨调用缓存旧active或mask，不引入CPU device-scalar回读或不安全跨CTA等待。之前recover+scatter、compact+mask融合候选并无一致收益，必须有新的机制假设，不能只因“少一个kernel”原样重试。
	- 验收同时报告各准备kernel及**完整QSA**，覆盖真实两层两长度、TP2/4/8、dense-only、低/高重合、短M、raw/超预算fallback和后续5D路径。普通时延与profile占比分开，不能只报准备kernel更快却忽略attention/全调用回退。

### 2. 其余主要TODO（QSA-T03～T12）

| ID | 待办 | 当前状态与完成标准 |
|---|---|---|
| QSA-T03 | **同scope的完整全direct / 全union / auto对照** | 尚未完成。三者均恢复/校验并输出全部query，保留dense前缀；全direct不建无用union计划、pack在内，全union每次构表。先覆盖real3/47与TP2/4/8，再按固定协议扩展；prepared9949行数字不能替代完整12000行对照。 |
| QSA-T04 | **基于完整成本完善分流** | 当前1.7局部填充work规则已实现，但不保证整批最优。依赖T03，评估少数direct/union行的额外launch、pack、构表和task并行度，分别考虑TP、layout及raw预算fallback；收益须覆盖完整调用，不能硬编码capture或在CPU读active。 |
| QSA-T05 | **查明时间漂移与长尾来源** | 10/32buffer历史轮均有多路径共同变慢，普通QSA出现约70～100ms长尾；因果尚未定位。以有界诊断采集运行期遥测、逐阶段/host提交时间关联，保留全部raw与未知区间，不凭空闲频率、buffer数或局部ATT猜结论，不原样重采择快。 |
| QSA-T06 | **Direct填充160T目标及效率改进** | 明确未达，`phasepg8`未采用。需新假设围绕VMEM发射、MFMA/VALU/DS交织、索引与尾部成本，并计入每次pack；TP2/4/8、短M、raw/超64MiB路径不得被忽略。历史两主例有效100T曾通过，但后来样本不足100，不宣称稳定余量或拿full/union填充T代替direct目标。 |
| QSA-T07 | **Union有效100T与工作膨胀控制** | 历史两主例填充210T已通过，约89.8/92.8有效T未过100。研究精确membership下更细分组/混合M tile、M/N填充和输出/流水成本；新增plan成本计入full，按实际MFMA工作计分子，不能扩大可见集合或沿用旧较大填充分子。 |
| QSA-T08 | **Dense小M/尾部效率与长请求是否拆dense** | M2048/2051本体对照已完成；M2051最后3行、较小G与启动/收尾仍有优化空间。“长请求不拆dense”仅历史实验，短真实prefix正式矩阵曾门禁失败；需在当前源码/代表性负载上完成全调用验收，不能全局删除dense或只凭prepared本体设置阈值。 |
| QSA-T09 | **扩大真实工作负载与缓存路径验证** | 现有8份capture及合成边界不能替代真实长prefix、多请求、多个prefill chunk、radix cache命中、不同M/P/HK、DP padding与save=false的服务验收。补非重复自然长文本、其它层/模型及超64MiB raw路径性能；新采集保存prompt/token IDs与配置身份，旧输入不替换。 |
| QSA-T10 | **最新源码部署、实际TP4/8采集与全模型profile** | 首阶段插件和旧TP2模型profile已完成；当前packed/1.7/64MiB版尚无完整的新服务验收，TP4/8现有结果仍为派生local-head。构建新冻结target并按真实TP2/4/8采集，检查每rank实际替换与源码身份；不得把旧0.1.4 trace重贴为当前版本。 |
| QSA-T11 | **模型精度、长上下文及无profiler端到端性能** | warmup/逐层O对照不等于模型指标。仍需用户指定长上下文测试、logits/生成质量、后续decode不变、并发/TTFT/吞吐验证；profile导出耗时不能算正常服务吞吐，kernel加速不能直接外推端到端。 |
| QSA-T12 | **SGLang graph / torch.compile独立接入** | 核心QSA固定layout/stream的graph回归已通过，但插件仍绕过graph/compile。未来需可捕获metadata、所有特化warmup、稳定scratch、custom-op/schema/fake输出、bucket padding及选择每次刷新；不得直接解除现有piecewise/breakable限制。涉及SGLang源码须另获明确修改授权。 |

T03/T04的计时范围与证据见[完整调用差异说明](opt.md#L1866)；T05～T08的失败和当前性能范围见[统一33case结果与限制](opt.md#L968)。T09～T12继承[接入验证边界](sglang/README.md)，不是声称这些场景当前已支持或已失败。

### 3. 历史后续/可选研究（仍未完成，不自动扩大当前任务）

| ID | 事项 | 边界 |
|---|---|---|
| QSA-O01 | **由indexer显式传递原block选择，避免展开后再恢复** | 历史第二阶段尚未实施，可作为T02的另一条路线。需forward/layer-local sidecar贯穿调用链、row-chunk完整覆盖、stream生命周期和验证，不使用全局last_blocks。涉及SGLang修改需明确授权，不能单纯删恢复而丢失原ABI验证。 |
| QSA-O02 | **短dense前缀跳过不必要的indexer评分/Top-K** | 独立于dense attention优化，尚未实施。必须保留compressed cache、pending K、RoPE位置及下一chunk/decode语义；不能因为attention全dense就跳过整个indexer/cache更新。 |
| QSA-O03 | **进程级scratch预算/容量池、graph释放与更广pack布局** | 每工作区PK+PV≤64MiB及raw fallback已完成；全进程多个layout/stream/pinned graph累计仍无总预算。只有需要时才设计graph安全回收/独占lease及ragged多请求pack，不能并发共用缓冲或引入active回读；不是当前64MiB需求未完成。 |
| QSA-O04 | **量化RoPE对选块条纹的因果贡献** | 周期749/近邻/union扩张及两份capture差异已分析；RoPE贡献未做消融。需要indexer旋转前Q、压缩K和logits，固定内容比较位置/旋转；主attention D256 Q/K不能冒充D128 indexer输入，也不硬编码周期。 |

decode、draft/verify、tokenwise DSA、CP/DCP、量化KV、额外score修改与LSE输出等仍属当前插件/接口之外，**不是本次用户已经要求实现的功能清单**；继续显式拒绝或在launch前沿用原实现，若要支持需独立立项与验收。

### 4. 相关D256 MHA历史未闭环验收（与QSA目标分开）

| ID | 事项 | 当前判定 |
|---|---|---|
| MHA-R01 | SHUFFLE-5D D256的原230T目标 | 仍未通过原正式验收，默认保留v73。continuous noncausal linear的221.690有效T不等于5D分页230T，不能因T01复用布局而视为目标已达。 |
| MHA-R02 | linear page1/page4对旧v98的严格1.10倍时延边界 | 历史page4约1.1033倍未过，尚无当前新版同scope再验闭环；与当前默认v73对照及QSA单接口性能是不同问题，不能拿不同baseline或吞吐比例替代原时延门槛。 |

范围依据：[MHA接口/历史性能说明](../mha/README.md)、[MHA原优化记录](../mha/opt.md)。这里只保留关联未闭环状态，不启动MHA调优、不把全部旧实验候选逐项重新列为活动TODO。

### 5. 已完成、已替代或已否决的旧TODO

| 旧条目 | 最新状态 |
|---|---|
| “direct为何慢，等待确认后再分析” | 已获授权并完成多轮K合并读取、V/K消费者流水、typed packed-FP32禁用、每次KV预排及ATT/PMC分析；不再标为未开始。剩余性能目标归T06。 |
| “重新标定原rho4” | packed路径1.7规则已完成；raw/ragged/预算fallback保留rho4。整批成本进一步优化归T04，不把现有1.7再次报成待实现。 |
| 单API/测试入口、kernel角色名、记录集中 | 已完成；所有新优化记录追加本文件，不新建Markdown。 |
| TP2/4/8本地三分支/full对比、query与逐kernel占比 | 33case/111组合/14208raw及330trace call已经完成；真实多卡TP4/8服务仍归T10。 |
| pack内存上限、无GPU回读与graph刷新 | 已完成64MiB/工作区合计上限，48项正常回归通过；不等于全进程内存池已完成。 |
| QSA完整流程、各GPU kernel及两种direct伪代码 | 已追加完整说明，不再待写。 |
| M12000与M11888是否截断、K布局图、HBM是否4倍 | 已证实为不同输入；地址布局已核验；旧16×4的HBM目标读字节未放大4倍，不能把L1 lookup或累计stall当HBM/墙钟。RoPE定量因果另归O04。 |
| 原生linear B1 causal任务配平、连续Full220T | 已完成各自限定范围；不自动覆盖其它布局、长度、分页或5D230T。 |
| 第一阶段SGLang插件/旧TP2 profile“尚未实施” | 已由后续五hook插件、原始capture与两轮TP2 profile推进完成；当前新源码部署与完整服务指标仍归T10/T11。 |
| 16×4免K转置、8lane DPP/DS、phasepg8及无收益融合 | 已测而未采用；错误数值、慢例、门禁失败与ATT全部保留。不是待合并功能，不无变化重试旧候选。 |

最早的“QSA约200有效T”以及union曾经的“210有效T”保留为历史目标/未达记录；后续明确口径分别为direct160填充T、union210填充且100有效T。不能静默改写旧验收，也不把已被后续要求替代的目标重复算一条新增TODO。direct历史两主例有效100T通过属于特定场次，不代表所有未来case/样本稳过100T。

### 6. 待办执行时继续遵守的统一约束

- 本次登记合计**12项主要TODO、4项历史后续/可选研究、2项关联MHA未闭环验收**；完成项与否决候选另列，不混数。用户新增T01/T02优先，T03提供T04和准备优化的公平基线，T05并行作为性能可信度问题处理；其余按依赖单独排期，不代表本轮已经启动。
- 代码、临时脚本与数据仍只在PyHIP；数据放mytest/mydata，所有新叙述只追加现有[opt.md](opt.md)。涉及SGLang源码的sidecar/graph等需明确授权。
- 性能保持**10独立I/O buffer**，默认每实现128sample、2warmup、原cudaPerf及门禁；具体新scope采样协议预先写清，不覆写旧数据、不挑快段、不用profile均值代替普通中位数。TP4/8派生与真实多卡数据必须分开。
- 不改选择/causality/原精度门槛，不用CPU同步换取表面简化；检查实际计时输出、input/plan/source/ELF身份与资源。没有新验收结果前，只能标“待实现/待验证”，不能把减少kernel数、5D形状可reshape或局部ATT改善当作已交付收益。

## 2026-09-27：T01原生SHUFFLE-5D、SGLang接入与pack边界

本节推进上方T01，不重写登记时的历史状态。用户明确允许此次修改SGLang，并在初版资源失败后选择“继续优化原生5D，保持零spill验收”。因此dense、union、direct均保留原生5D读取，不用隐藏gather/转换替代零spill要求。T02准备融合、T03同scope全direct/全union、T04完整成本路由和T10/T12模型部署/graph仍是独立任务。

本轮工作区与全部原始失败、JUnit、冻结源码、IR/ELF/ISA及性能数据在[study协议](../../../../mytest/mydata/qsa_5d_20260927_01/protocol.json)所在目录；没有新增Markdown报告，没有修改PTL/时钟/功率/NUMA，没有stage/commit或改动历史包。下面分别回答支持范围、page64意义、能否省pack，性能不能由减少kernel数推断。

### 1. SGLang此前已有5D缓存，不等于QSA此前已端到端支持

- [MHA缓存布局与writer](../../../../../../sglang/python/sglang/srt/mem_cache/memory_pool.py)：ROCm且AITER启用时，`SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d`选择SHUFFLE布局；默认仍NHD，HND优先。BF16的向量宽度是`X=16/dtype_bytes=8`，不是2。分配和`launch_reshape_and_cache_shuffle_5d`写入在T01之前已存在。
- [AITER既有5D attention](../../../../../../sglang/python/sglang/srt/layers/attention/aiter_utils.py)：decode有`pa_decode_gluon`；prefill在无prefix时使用新生成3D K/V，有prefix时从5D gather成linear，再走LINEAR prefill。其注释明确说明当前page64/BF16或FP8/SHUFFLE预填充缺少对应编译配置。因此“支持5D缓存”不应被写成“所有prefill原生5D且无转换”。
- T01之前[QSA backend](../../../../../../sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py)的无prefix路径用fresh 3D K/V；prefix路径按token对cache第0维index_select，而5D第0维是page；decode选中KV提取也按3D拆shape。仅打开5D缓存flag不能让这些路径自动正确。
- 本轮授权内的SGLang改动：新增[页表及5D消费边界](../../../../../../sglang/python/sglang/srt/layers/attention/qsa/paged.py)，从device `req_to_token`/request IDs构造页表并检查每page内部slot连续、page起点对齐及物理范围；backend把原始5D K/V和页表交给该边界，保留有效query裁剪、输出padding和原KV写入职责。未启用插件时边界保留正确的5D→linear fallback，启用native插件时跳过gather。
- [decode选中KV提取](../../../../../../sglang/python/sglang/srt/layers/attention/qsa/sparse_attn.py)增加正确的5D K/V字节寻址，槽地址在page乘法前升64位；backend scratch维度使用layer的D256，而不是误取5D的`shape[2]=32`。这里仍是原decode消费者需要的**selected KV compact**，不是PyHIP direct每attention全KV的PK/PV pack。
- 3D无prefix仍直接使用fresh K/V，不为“看shape”而调用会等待layer transfer的cache getter。`is_shuffle_5d_cache()`仅通过真实MHA或Hybrid内部full pool的静态layout字段判断，不假设所有KVPool都有同名字段、不用getattr默默兜底。

### 2. 5D下page-size=64到底表示什么

令$S$为page size、$x$为请求内逻辑token，`page_table[request,x//S]`给物理page ID $p$，页内偏移$o=x\bmod S$。BF16 D256布局为：

$$
K:[N_{page},H_K,32,S,8],\qquad
V:[N_{page},H_K,S/8,256,8].
$$

每个head的地址（单位：BF16元素）分别为

$$
K(p,h,d,o)=((((pH_K+h)32+\lfloor d/8\rfloor)S+o)8+d\bmod8),
$$

$$
V(p,h,o,d)=((((pH_K+h)(S/8)+\lfloor o/8\rfloor)256+d)8+o\bmod8).
$$

- **page64=每物理KV页64个原始token。** HK1/D256/BF16时一页K为32KiB、一页V为32KiB，总64KiB；不是64个选中block、不是topk64，也不是必须以BN64做direct。
- [QSA压缩池](../../../../../../sglang/python/sglang/srt/mem_cache/qsa_kv_pool.py)的compress ratio4使一个full-KV page对应16个压缩indexer block，`compressed_slot=full_slot//4`。压缩indexer K是独立缓存，不是主attention K/V的5D维度。
- direct仍按BN32选中token处理，即最多8个完整四token块，可能来自不同物理页；四token块因为S是4的倍数且page对齐，不会跨页。union使用BN64 compact选中token，不要求这些token在同一物理页。
- [参数覆盖](../../../../../../sglang/python/sglang/srt/arg_groups/overrides.py)中，通用5D flag在HIP上将未指定page size默认到64；compressed Qwen4-Exp模型另外无条件固定page64。PyHIP kernel合同支持32/64/128不表示该模型服务配置会接受其它page作为最终解析值。本轮没有修改这些默认值/覆盖逻辑。

### 3. 可以省direct pack，但必须实现新的寻址/操作数布局

结论是**可以，而且原生5D路径已经不分配PK/PV、不启动`direct_pack_kv_bf16_d256`，dense/union也不做全KV转换**。这不是把旧packed kernel的descriptor直接指向5D：旧PK/PV是四token内按QK/PV MFMA操作数预排的私有布局，SHUFFLE5D的channel/page分块不同，需要本轮的页翻译和K/V访存适配。

调用仍只有[qsa.py](qsa.py)的`qsa(...,page_table=...)`：

- Q仍BF16连续`[M,H,256]`；5D K/V均连续、16B对齐，每buffer字节跨度小于2GiB；S只允许32/64/128，H/HK≤16。不是任意五维reshape。
- `page_table`为同device连续int32 `[requests,max_pages]`，有效列覆盖各`query+prefix`逻辑长度，存物理page ID而非slot。每调用GPU验证当前有效page；未用padding列可为−1，不缓存旧表指针。相同shape换新页表pointer也正确。
- 5D省略prefix默认0，绝不把物理cache容量减M当成prefix；空请求/零query段仍保留请求编号。page0和末页padding在测试中用NaN填充，不能靠softmax概率0掩盖`0*NaN`。
- 每调用仍恢复/验证原2051 token-index ABI，重建union选择/mask/order，再按互斥query范围运行dense、union、direct。5D新增page-ID校验，不删除duplicate/tail/padding检查。
- 3D单请求4对齐且PK+PV≤64MiB继续按原pack路径运行；3D超过预算或ragged继续raw4wave。**5D不受这份额外PK/PV预算限制，因为它不创建这两份scratch**，但原2GiB单buffer寻址上限仍在；大型SGLang物理池超过native范围时插件会回原fallback，不能宣称任意模型缓存容量都走native。
- 省略的是每attention的全缓存重排及额外全局K/V副本，不是新token cache writer、页表构造、错误校验、选中tile到LDS搬运或MFMA布局交换。SGLang页表构造成本也不是零，T02可另行优化。

### 4. 最终原生内核机制与失败历程

[_paged.py](_paged.py)提供三个明确GPU符号：`dense_qsa_bf16_d256_5d`、`union_qsa_bf16_d256_5d`、`direct_qsa_bf16_d256_5d`。

- **Direct：**复用[_direct_packed.py](_direct_packed.py)的单query-wave数学/成对QK/RNE/PV/输出，替换为5D地址。K按8BF16 packet读；V协作读取后仅在PV消费点做两个channel bit的lane/register交换，避免在预取点提前等待。typed `llvm.target_features -packed-fp32-ops`保留；>256的报告VGPR数不能单独作为spill结论。
- **Dense：**保留因果最长tile优先/蛇形调度，5D只保留1个（S64/128）或2个（S32）page start，避免16个地址跨phase同时存活；S128的第二个BN64tile使用`token&63`而不是重复加page内64偏移。原3D展开/调度保留。
- **Union：**[_paged_union.py](_paged_union.py)使用BM128/BN64、8wave/64KiB LDS。每个排序task一个CTA而不是持久worker跨task循环；这是消除剩余SGPR spill的关键。compact K采用适合5D的协作lane分配及已有MHA式swizzled LDS，V直接进入PV操作数LDS；V DMA覆盖QK，下一tile K DMA覆盖PV。不是3D的深4+4wave错峰展开，不能预设原吞吐保持。
- [_paged_common.py](_paged_common.py)把物理四token起点与末块有效token数编码在同一整数，load前/OOB及PV前逐BF16屏蔽最后部分块，覆盖物理页尾NaN。
- [CPU地址证明](../../../../mytest/mydata/qsa_5d_20260927_01/layout_proof_v9.json)枚举8192个K DWORD destination与消费者读取映射，并验证64lane V操作数的两bit交换。它证明地址/值排列，不证明HBM流量、L1line大小或预估几倍加速。

失败全部保留：

| 版本/阶段 | 实测结果与处理 |
|---|---|
| 初始smoke | Uint16→Uint32转换形式不被FlyDSL支持，改为显式LLVM zext；原失败与修复后数值保留。 |
| v1 | 数值/graph正确但dense SGPR spill，union同时private/VGPR/SGPR spill，未通过资源验收。 |
| v2最初 | 更改helper却复用v1 ELF；属于stale cache诊断，不是新代码资源结果。独立cache后才看到实际变化。 |
| v2实际～v5 | dense地址live-state优化逐步实现零spill；union深流水仍private/VGPR或SGPR spill，未采用这些资源失败版本。 |
| v6～v7 | 浅union没有private/VGPR spill，但仍13～17个SGPR spill；只用DS broadcast不够。 |
| v8 | ISA定位到持久worker跨task保留基指针；改每task一个CTA后通过三项零spill，TP2/4/8×page32/64/128与后续真实capture功能通过。首个性能主例却明显退化。 |
| v9 | 改协作K和V channel交换。一次JIT静态helper的`continue`被AST rewrite误处理导致导入失败，去掉不必要JIT装饰后通过；不是GPU数值失败。direct主例改善但仍慢于3D。 |
| v10 | V/QK、下一K/PV轻量重叠，复用页翻译；保持零spill，union有所改善，仍未达到3D性能。采用当前原生功能实现，但不宣称性能目标通过。 |

FlyDSL当前依赖收集不会递归追踪任意`module.attribute(...)`调用的helper实现；当前launcher/body用显式callable import以参与依赖hash。每轮独立cache保留，实际IR/ELF验证，未修改安装包或清空共享cache。普通Python AST检查与GPU JIT检查分开，不能仅凭编辑器missing-import提示或源码外观断言编译等价。

### 5. 第一主例的性能事实（后续完整矩阵另附）

真实Layer3/M12000、TP2 local H12/HK1/D256，原cudaPerf、10独立buffer、2warmup/buffer、每实现128sample；所有raw/门禁/实际timed O均保留。prepared direct/union只算query[2051,12000)的9949行，full算全部12000行，**不能跨列混scope**。

| 原生实现阶段 | 同场3D direct含pack µs | 5D direct µs | 同场3D union µs | 5D union µs | 同场3D full µs | 5D full µs |
|---|---:|---:|---:|---:|---:|---:|
| v8 | 2509.713 | 10522.413 | 2788.374 | 10325.251 | 2904.934 | 11351.177 |
| v9 | 2510.354 | 6739.316 | 2787.835 | 9964.294 | 2905.995 | 10989.440 |
| v10 | 2511.193 | 6742.396 | 2787.195 | 9263.609 | 2904.795 | 10288.455 |

原始结果：[v8](../../../../mytest/mydata/qsa_5d_20260927_01/performance/real3_12000_tp2/result.json)、[v9](../../../../mytest/mydata/qsa_5d_20260927_01/performance_v9/real3_12000_tp2/result.json)、[v10](../../../../mytest/mydata/qsa_5d_20260927_01/performance_v10/real3_12000_tp2/result.json)。v10保留3D full的10.19ms和5D full的17.18ms等慢样本，不滤除。

v10 5D direct/union/full相对同场3D仍约2.68×/3.32×/3.54×耗时。3D packed用pad1.7，而5D沿用rho4，真实稀疏行全union；因此full也含路由差异，不能把全部退化或改善归因单一访存变化。未测PMC/ATT反事实，不声称HBM放大若干倍或某个指令独自解释时延。

同场[实际kernel trace证明](../../../../mytest/mydata/qsa_5d_20260927_01/performance_v10/real3_12000_tp2/trace_proof.json)：10组buffer分别运行native direct/union/full，共30call、150kernel，**无direct pack、无全KV gather**。这只证明省掉pack机制，并不推翻上表性能退化。

所以本轮给配置选择的建议是：**可以用5D省PK/PV，但目前不要仅为性能启用5D；默认继续保留3D/NHD。** T01的功能、无pack和零spill与“5D比3D更快”是不同验收项，后者未通过，也不能用MHA连续linear的221.69T或旧direct100T成绩替代。

### 6. v10完整15例性能矩阵（冻结测量版本，非最终v11性能）

完整[raw/资源/门禁审计](../../../../mytest/mydata/qsa_5d_20260927_01/analysis/results.json)覆盖5类×TP2/4/8，共**96个实现组合、12288个普通raw样本**，另保留v8/v9主例1536raw，总13824raw。原cudaPerf、10独立buffer、2warmup/buffer、128sample/实现；普通时延与trace共53次门禁均通过，固定GPU2/0000:a4:00.0、PTL Enabled/VECTOR,F8。源、input/page table、选择/plan、actual timed O和guard逐项核验。

real3/47用TP0原M12000 capture；TP4/8只裁剪Q heads到6/3，**不是实际分布式TP4/8采集**。low/high为M2048/P30000、独立选块/每32query共享优先顺序、seed17；dense3_2051是原Layer3前2051行的派生causal前缀。转换到5D及构造物理页表均在准备阶段，比较的是resident cache，排除了cache writer和SGLang req_to_token→page_table过程，不是服务器端到端数据。

下表各单元格为**3D / 5D µs**，最后列是full中位数比。real prepared从2051行开始，其它prepared从0开始；full始终全部query。强制union仅用于同选择语义的分支对照，不改变公开auto策略。

| 用例 | TP | prepared direct 3D/5D µs | prepared union 3D/5D µs | full QSA 3D/5D µs | 5D/3D full |
|---|---:|---:|---:|---:|---:|
| real3_12000 | 2 | 2511.193 / 6742.396 | 2787.195 / 9263.609 | 2904.795 / 10288.455 | 3.542× |
| real3_12000 | 4 | 2483.294 / 6690.237 | 1960.431 / 6659.197 | 2464.993 / 7459.741 | 3.026× |
| real3_12000 | 8 | 2456.393 / 6658.316 | 1147.626 / 4211.823 | 1626.269 / 5378.409 | 3.307× |
| real47_12000 | 2 | 2506.654 / 6712.877 | 2693.155 / 8840.548 | 2998.196 / 9861.794 | 3.289× |
| real47_12000 | 4 | 2479.513 / 6678.716 | 1891.910 / 6369.635 | 2445.634 / 7172.439 | 2.933× |
| real47_12000 | 8 | 2451.334 / 6654.896 | 1126.347 / 4059.822 | 1596.089 / 5040.247 | 3.158× |
| low | 2 | 589.223 / 1511.528 | 2127.352 / 7347.139 | 689.403 / 1621.929 | 2.353× |
| low | 4 | 583.984 / 1499.788 | 1900.770 / 6513.996 | 675.763 / 1597.169 | 2.364× |
| low | 8 | 582.983 / 1491.069 | 1291.287 / 4517.345 | 673.283 / 1584.949 | 2.354× |
| high | 2 | 575.103 / 1509.149 | 405.783 / 1304.667 | 528.322 / 1419.868 | 2.688× |
| high | 4 | 570.723 / 1499.808 | 214.501 / 662.843 | 318.502 / 768.284 | 2.412× |
| high | 8 | 567.563 / 1494.048 | 110.000 / 335.801 | 211.301 / 437.862 | 2.072× |
| dense3_2051 | 2 | 386.362 / 794.524 | 175.221 / 523.203 | 230.581 / 725.084 | 3.145× |
| dense3_2051 | 4 | 383.482 / 788.704 | 110.521 / 331.641 | 159.200 / 486.342 | 3.055× |
| dense3_2051 | 8 | 382.182 / 786.885 | 107.401 / 331.101 | 157.761 / 481.362 | 3.051× |

3D dense3_2051的KV长度不能4对齐，因此表内direct3D是原raw4wave、不是pack；其它四类direct3D均含每次全KV pack。dense本体另外测得：TP2 179.281/671.423µs，TP4 108.081/433.262µs，TP8 106.760/428.003µs，同样3D/5D。

以下给出5D**有效T / 填充T**，不可互换。有效FLOPs=`4*D*QH*sum(actual selected token counts)`；direct填充按M16/N32，union按M128/N64实际并集tile，dense按实际因果M128/N64 tile，full逐互斥分支累加填充工作；F不含softmax/构表/校验额外算术。

| 用例 | TP | direct有效/填充T | union有效/填充T | full有效/填充T |
|---|---:|---:|---:|---:|
| real3_12000 | 2 | 37.162 / 50.093 | 27.048 / 64.212 | 26.867 / 60.800 |
| real3_12000 | 4 | 18.726 / 50.483 | 18.813 / 63.449 | 18.527 / 58.698 |
| real3_12000 | 8 | 9.408 / 50.725 | 14.872 / 58.137 | 12.848 / 46.497 |
| real47_12000 | 2 | 37.325 / 50.313 | 28.342 / 64.139 | 28.029 / 60.611 |
| real47_12000 | 4 | 18.758 / 50.570 | 19.668 / 63.420 | 19.269 / 58.462 |
| real47_12000 | 8 | 9.413 / 50.751 | 15.429 / 58.299 | 13.710 / 48.410 |
| low | 2 | 34.123 / 45.996 | 7.020 / 56.236 | 31.800 / 42.866 |
| low | 4 | 17.195 / 46.356 | 3.959 / 53.229 | 16.146 / 43.530 |
| low | 8 | 8.648 / 46.627 | 2.854 / 51.206 | 8.135 / 43.866 |
| high | 2 | 34.176 / 46.069 | 39.533 / 53.206 | 36.325 / 48.889 |
| high | 4 | 17.195 / 46.356 | 38.906 / 53.457 | 33.567 / 46.120 |
| high | 8 | 8.630 / 46.535 | 38.399 / 52.760 | 29.448 / 40.462 |
| dense3_2051 | 2 | 32.545 / 44.050 | 49.422 / 55.026 | 35.662 / 42.343 |
| dense3_2051 | 4 | 16.393 / 44.375 | 38.985 / 54.256 | 26.584 / 31.564 |
| dense3_2051 | 8 | 8.215 / 44.478 | 19.524 / 27.590 | 13.430 / 15.946 |

全样本中最大长尾为real3/TP8/5D full第7个sample **240992.798µs**，留在raw并参与中位数，未因异常慢而删除。普通中位数、配对ratio中位数、min/max和工作量均在JSON单独保存，不用profile平均数替换它们。

### 7. 分流比例与逐kernel占比

真实case的dense都为2051/12000=17.092%；下表U/D是稀疏9949行内的数量（不是按时延缩放）。

| 用例/TP | 3D U/D query | 5D U/D query | 5D稀疏区U/D比例 |
|---|---:|---:|---:|
| L3 TP2 | 3929 / 6020 | 9949 / 0 | 100% / 0% |
| L3 TP4 | 9341 / 608 | 9949 / 0 | 100% / 0% |
| L3 TP8 | 9949 / 0 | 8605 / 1344 | 86.491% / 13.509% |
| L47 TP2 | 4269 / 5680 | 9949 / 0 | 100% / 0% |
| L47 TP4 | 9933 / 16 | 9949 / 0 | 100% / 0% |
| L47 TP8 | 9949 / 0 | 9501 / 448 | 95.497% / 4.503% |

low三TP在3D/5D均2048行全部direct；high均2048行全部union；dense3_2051均2051行全部dense。v10并未为5D重新校准rho4，所以TP8可出现与3D不同的分流，不能把完整call对比视为只删pack的单因素实验。

在[预先补充的TP4/8 trace协议](../../../../mytest/mydata/qsa_5d_20260927_01/trace_protocol.json)下，共保留real3三TP和low TP2四场trace，每场10buffer×native direct/union/full=30call，总590kernel（real3各150，low没有dense所以140）。所有kernel均通过host correlation归入唯一user annotation（不重复计同名GPU投影），整数ns求和，无漏归属、无pack/gather，实际trace O比同路径参考逐bit一致。

以下real3表中每格为**单次full kernel平均µs（占该full全部kernel总时间%）**。分母不是普通cudaPerf event，不含host/launch间隙，故不直接相加来解释普通中位数。每TP有10个full call。

| GPU kernel/工作 | TP2 | TP4 | TP8 |
|---|---:|---:|---:|
| recover/sort/check block ABI | 199.356 (1.953%) | 199.084 (2.692%) | 199.064 (3.727%) |
| validate physical pages | 4.639 (0.045%) | 4.519 (0.061%) | 4.655 (0.087%) |
| Torch compare errors==0 | 5.295 (0.052%) | 5.139 (0.069%) | 5.211 (0.098%) |
| Torch all reduction | 7.799 (0.076%) | 7.487 (0.101%) | 8.007 (0.150%) |
| Torch async assert | 4.667 (0.046%) | 4.475 (0.061%) | 4.567 (0.086%) |
| membership fill | 6.539 (0.064%) | 6.271 (0.085%) | 5.587 (0.105%) |
| scatter membership | 38.623 (0.378%) | 39.427 (0.533%) | 41.355 (0.774%) |
| compact/gate | 27.339 (0.268%) | 18.135 (0.245%) | 10.659 (0.200%) |
| score masks | 49.027 (0.480%) | 59.323 (0.802%) | 74.495 (1.395%) |
| task sort/order | 19.647 (0.192%) | 8.283 (0.112%) | 5.347 (0.100%) |
| native dense5D | 668.923 (6.552%) | 431.485 (5.834%) | 425.185 (7.961%) |
| native union5D | 9149.443 (89.623%) | 6588.039 (89.083%) | 3492.386 (65.393%) |
| native direct5D | 27.483 (0.269%) | 23.755 (0.321%) | 1064.061 (19.924%) |
| **全部kernel总计** | **10208.781** | **7395.424** | **5340.580** |

TP2/4 full的direct query为0，但仍有gate kernel launch时间；无pack launch。low/TP2的full kernel均值总1633.169µs，direct1521.779µs/93.180%，union空gate4.567µs/0.280%，其余为恢复/校验/构表。所有详细名称/原始duration保存在上述审计JSON。

### 8. 资源、回归、独立插件及v11安全补丁

v10[完整正常回归](../../../../mytest/mydata/qsa_5d_20260927_01/validation_v10/result.json)为**64 passed、24 perf deselected、0skip、1099.352s**，测试覆盖3D原有48项加5D page32/64/128×TP2/4/8、HK2/ragged/空段、NaN物理尾页、超过64MiB物理cache、table pointer替换、graph改KV/table/indices、无pack分配，以及6个SGLang prefix0/3000×TP插件调用。8份原始capture（两rank、两层、两长度）各跑TP2/4/8并分别校验3D/5D，24份5D是布局派生回放，不是新服务器capture。

v10全部144个native5D实际编译特化：direct55、union55、dense34。资源范围如下，**每个特化private/VGPRspill/SGPRspill三项均0**；VGPR计数不等同spill数量。

| native5D | VGPR | SGPR | LDS bytes |
|---|---:|---:|---:|
| direct | 276 | 56～58 | 8192 |
| union | 220～222 | 92～97 | 65536 |
| dense | 250～252 | 83～89 | 65536 |

15例测量中51个3D编译签名与T01前统一研究对应签名的**整ELF、.text和资源全部一致**，包括dense、union、raw及有/无gate packed direct；不是只按源码AST相似宣称不变。[主例独立匹配收据](../../../../mytest/mydata/qsa_5d_20260927_01/representative_3d_equivalence.json)。所有native5D ELF/ISA在各case artifacts中，历史code object没有改名重贴。

SGLang本轮[原55项加新增5D 5项注册回归](../../../../mytest/mydata/qsa_5d_20260927_01/sglang_all_v3.xml)：**60 passed、0skip、14.820s**。覆盖实际SHUFFLE writer、页表对齐/非法request、prefix读取、decode selected gather/graph更新、真实backend D256 decode结果及无prefix5D分派；3D no-prefix原测试不需要fake getter即可通过。未宣称整模型精度、TTFT或服务graph已验收。

v10独立0.3.0包的[禁用惰性](../../../../mytest/mydata/qsa_5d_20260927_01/package_disabled.json)和[启用六hook](../../../../mytest/mydata/qsa_5d_20260927_01/package_enabled.json)均通过：真正SGLang entry-point发现/注册/应用、五上游文件ABI失败保护、禁止gather的六个5D backend调用，未从experiments导入runtime。5D捕获含page_table和物理shape hash；克隆整个物理池可能很大，默认不采集输入，不应隐藏此成本。

另外尝试永久CLI的5D benchmark时，TP2的10组独立页表/缓存与预热输出检查完成，**采样前GPU use7%>5%**，按门禁停止、0raw、不重试、不继续TP4/8计时。该[失败收据](../../../../mytest/mydata/qsa_5d_20260927_01/cli_5d_benchmark/tp0_layer3_m33_4df92c88dcb0_tp2/result.json)保留；不能称此CLI三TP性能验收完成，也不能把前面的15例独立研究计时换名为CLI成功。

最终静态LDS账本检查发现v10最后一次投机K DMA在退出循环后没有显式`vmcnt(0)`，而输出transpose马上复用该LDS。现有数值测试未观察到失败，但`_stage_end()`只是CTA rendezvous，不是DMA counter drain；因此**v11追加显式vmcnt/lgkmcnt清零＋CTA屏障再写O**，防止异步写覆盖。这是正确性安全补丁，不是性能优化。本节v10样本和code object继续只代表v10，**不能当成v11的正式性能**；在前述性能门禁失败后不重跑性能，仅重新进行最终代码的功能/资源与独立包验证。

[v10→v11语义差异证明](../../../../mytest/mydata/qsa_5d_20260927_01/epilogue_source_delta.json)确认device body只新增该drain＋barrier（并清理两个unused import）。最终审计另发现dense/union文件EOF空白行；编辑器多次未能移除，停止重复改写并保留两条`git diff --check`告警，不宣称空白检查通过。[最终字节/AST收据](../../../../mytest/mydata/qsa_5d_20260927_01/final_source_identity_v2.json)确认相对运行中冻结源码仅union EOF换行数不同，**含行位置的AST完全一致**；新最终包复制当前字节，旧v10/v11包只读保留。编辑器仍有Torch/Triton的`reportMissingImports`（severity2），系统`/bin/python3`实际imports与GPU测试正常；不安装替代包掩盖诊断。

### 9. T01最终功能闭环（v11）

- [x] **T01原生5D接口/SGLang接入/省PK+PV/零spill功能验收完成**，限定于前述BF16 D256、page32/64/128、连续对齐5D、每buffer<2GiB和固定indices ABI，不表示性能优化达标或所有服务模式已验收。上方原TODO勾选保持其登记时历史状态，以本节为最新状态。
- 最终v11 [JUnit](../../../../mytest/mydata/qsa_5d_20260927_01/validation_v11/tests.xml)：**64 passed、24 perf deselected、0失败/0错误/0skip，1163.088s**。144个native5D特化仍为direct55/union55/dense34，三项spill均0；VGPR/SGPR/LDS范围与v10功能矩阵相同。原[外层收据](../../../../mytest/mydata/qsa_5d_20260927_01/validation_v11/result.json)保留`complete=false`/外层exit1，因为运行期间union EOF空白变化触发文件hash检查，不篡改成“全绿”；pytest本身exit0。最终源码通过上述带行号AST证明衔接，并重新验证最终包/真实机器码。
- [三TP最终实际ELF与末尾drain](../../../../mytest/mydata/qsa_5d_20260927_01/epilogue_v11/result.json)导出9个dense/union/direct specialization，三份real3 union重复输出对原capture通过，实际ISA确认输出首个DS写前有`vmcnt(0) lgkmcnt(0)`及CTA barrier，其后无未完成K→LDS load。TP2/4/8 union ELF SHA前缀分别`6498d87a…`/`96e72aa4…`/`6667a5f7…`，这些不是v10计时ELF。
- 最终新target的[禁用惰性](../../../../mytest/mydata/qsa_5d_20260927_01/package_final_disabled.json)与[启用六hook](../../../../mytest/mydata/qsa_5d_20260927_01/package_final_enabled.json)均通过，版本0.3.0；11个runtime文件逐字节等于当前源码、五源ABI有效且拒绝错误hash、6个原生5D backend调用无gather、无experiments源树依赖。仅构建/验证包，**未启动整模型服务**，旧target全部保留。
- 最终永久CLI [check-only](../../../../mytest/mydata/qsa_5d_20260927_01/cli_5d_check_final/checks.json)对实际5D插件capture按TP2/4/8完成3例功能回放；这不重试前面的性能门禁失败，没有新增v11时延。
- 保留的15例/13824raw与失败版本身份另见[测量完整性审计](../../../../mytest/mydata/qsa_5d_20260927_01/retained_measurement_audit.json)、[失败收据索引](../../../../mytest/mydata/qsa_5d_20260927_01/failed_attempts_index.json)和[cache writer未修改证明](../../../../mytest/mydata/qsa_5d_20260927_01/source_scope_audit.json)。原opt前180814字节保持SHA `8701b171…`，既有历史/用户index/AITER CSV和9078绘图服务保留。

**性能结论不变：**冻结v10矩阵5D full为3D的2.07～3.54倍耗时，最终v11仅补安全同步、未重测性能，不能宣称更快。T02/T03/T04、5D访存/流水继续优化、真正多卡服务与模型指标仍待后续独立验收；不默认切换5D、不自动部署到服务、不将功能完成包装成性能目标通过。

## 2026-09-27：MHA 5D/linear同场比较，以及QSA三个5D分支为何退化

用户指出MHA是QSA基础，要求实际对比[mha_pa_bf16_256_paged_942.py](../mha/mha_pa_bf16_256_paged_942.py)与[mha_pa_bf16_256_linear_942.py](../mha/mha_pa_bf16_256_linear_942.py)，并确认QSA的5D慢是否只发生在union。本轮只做测试/原因定位，**没有修改MHA/QSA/SGLang生产代码或切换默认实现**；诊断候选只在mytest/mydata，没有合入production。

先给结论：**不是只有union。当前QSA5D的dense、union、direct都明显慢。这个退化主要说明T01的5D适配没有保留原MHA5D的数据通路和流水优势，不能概括成“5D布局本身比linear差”。** 原MHA5D与最新连续linear在本场只差约3%～4%，而QSA三分支相对各自3D可差2～4倍；且只换union V-LDS布局即可使同语义、同数学输出的5D耗时下降约三分之一。

### 1. 比较协议、源码与语义范围

[正式协议](../../../../mytest/mydata/qsa_mha_layout_20260927_01/protocol.json)、[全部重算结果](../../../../mytest/mydata/qsa_mha_layout_20260927_01/analysis/results.json)：固定GPU2/0000:a4:00.0，gfx942/MI308X80CU，PTL Enabled/VECTOR,F8。原cudaPerf、**10独立I/O buffers、2warmup/buffer、128sample/实现**，label循环移位并交替反序，sample%10轮换buffer。

- 主矩阵16case：非因果Full的H24/HK2锚点及TP2/4/8 H12/6/3-HK1；causal Q=KV2048/2051×TP2/4/8；real Layer3/47稀疏后9949行×TP2/4/8。共96个实现组合、**12288raw**。
- MHA5D用page64、相同逻辑K/V映射到随机物理page、末页NaN；连续linear用相同逻辑顺序K/V。非因果Full另测linear page1/page4及两版的nonpersistent grid，不把不同页粒度说成相同物理布局。
- causal2048/2051选中全部可见token，MHA和QSA各分支语义真正相同；可同场直接对比。real M12000只对QSA union/direct的query[2051,12000)进行同selected对照，**没有拿无稀疏mask MHA输出替代真实QSA**。
- 所有阶段都是预分配/预热后的attention热调用；QSA direct3D若符合条件，计入每次pack。layout转换、页表/indices生成、QSA recovery/plan、JIT、参考和输出allocation都在计时外；这是本体比较，不是完整QSA或SGLang服务。
- MHA全O对完整FP32参考按原`.02/.02`检查；真实QSA对原capture加FP32抽样行；10buffer重复输出逐bit一致、guard、所有实际timed O/input/page/plan/source均验证。每个实际ELF/.text/ISA/资源导出；三项spill均0。
- 主矩阵48个门禁通过，另6场诊断18个门禁通过，总66次。所有慢阶段/长尾保留；TP4/8仍只是local-head形状或TP2 capture派生，不是新分布式TP运行。

### 2. MHA本身：最新连续linear略快，5D约等于mapped linear

Q10240/KV2583、BF16 D256、noncausal/noLSE、persistent，以下均为**全部128sample中位µs**。H24/HK2是原MHA benchmark锚点，不给它附加未经验证的服务TP编号。

| 本地QH/HK | 连续linear µs | SHUFFLE5D/p64 µs | linear/page1 µs | linear/page4 µs | 连续linear / 5D有效T | 5D/连续linear耗时 |
|---|---:|---:|---:|---:|---:|---:|
| 24/2锚点 | 4441.724 | 4607.105 | 4599.345 | 4587.184 | 146.347 / 141.094 | 1.0372× |
| 12/1（TP2形状） | 2193.732 | 2278.192 | 2270.133 | 2269.493 | 148.157 / 142.664 | 1.0385× |
| 6/1（TP4形状） | 744.204 | 771.124 | 772.765 | 765.545 | 218.365 / 210.742 | 1.0362× |
| 3/1（TP8形状） | 374.362 | 385.882 | 389.222 | 385.682 | 217.047 / 210.567 | 1.0308× |

因此对**当前两份源码**，本场没有复现“5D比连续linear略快”；5D比连续linear慢3.08%～3.85%，与mapped page1/page4则约在−0.86%～+0.73%之间。nonpersistent两版同样比较，5D/linear耗时比约1.0293～1.0417，不能把结果只归因persistent开关。这里保留各自现行task顺序，不把它写成完全同scheduler的单因素布局实验。

重要历史边界：旧记录里page1/page4对**v98实验**较慢，而当前paged文件是保留的**v73默认**；最新连续linear后来获得静态B1、N32尾裁剪与DMA leaf改进。旧candidate/旧mapped-mode对比不等于当前默认paged vs当前连续linear。本轮锚点linear整ELF仍是`d043ace4…`，与历史221.69T的真实ELF一致，不是代码偷偷退回旧版。

本轮长形状出现明显共同快慢阶段：[分段披露](../../../../mytest/mydata/qsa_mha_layout_20260927_01/temporal_disclosure.json)。锚点每16sample的linear中位约2939.8/4486.0/2941.1/4480.4/2940.9/4475.3/3822.6/4462.6µs；5D同步约3047.2/4620.3/3044.7/4628.4/3048.9/4620.6/3894.0/4613.1µs。上表**仍使用全样本4441.724/4607.105**，不挑快段复报220T。锚点配对ratio中位1.03766，与中位数比1.03723单列；没有运行期时钟/温度因果数据，不给共同变慢指定原因。

### 3. 同一causal问题中，QSA三个5D分支全部退化

下表为µs，各项均输出同一causal注意力结果（并非同一个分支算法的所有细节相同）；direct/union是强制分支本体，正常公开qsa在此长度仍走dense。

| Q=KV / TP | MHA linear | MHA5D | QSA dense3D / 5D | QSA union3D / 5D | QSA direct3D / 5D |
|---|---:|---:|---:|---:|---:|
| 2048 / 2 | 161.321 | 193.622 | 161.201 / 618.603 | 171.661 / 522.423 | 299.082 / 790.784 |
| 2048 / 4 | 102.440 | 128.601 | 102.760 / 425.382 | 105.921 / 321.841 | 294.022 / 782.224 |
| 2048 / 8 | 100.881 | 102.760 | 100.780 / 424.083 | 103.801 / 321.162 | 291.881 / 777.264 |
| 2051 / 2 | 172.321 | 206.561 | 179.361 / 674.924 | 174.701 / 524.323 | 386.923 / 802.384 |
| 2051 / 4 | 103.641 | 136.001 | 108.320 / 433.783 | 110.481 / 332.741 | 384.782 / 795.724 |
| 2051 / 8 | 101.961 | 104.881 | 106.841 / 433.022 | 108.320 / 331.002 | 381.662 / 788.824 |

- dense5D/dense3D为**3.76～4.21倍**；dense5D/原MHA5D也为**3.19～4.13倍**。这在没有稀疏并集/page碎片语义的同一个dense问题上已经发生，直接否定“只有union稀疏性导致”的解释。
- union5D/union3D约**3.00～3.09倍**；direct5D/direct3D约**2.07～2.66倍**。2048的3D direct包含pack；2051不能四token对齐，3D direct是raw4wave。因此2051不能叫“只省pack”的比较。
- QSA3D dense与MHA linear的2048结果基本相同；2051 QSA bounded地址比原MHA还有小额代价。MHA5D causal在TP2/4慢约20%～31%，但远小于QSA5D自己的3～4倍退化；linear的单请求causal最长tile优先snake与原paged升序task顺序不同，也是本体比较的真实差异，未做单独调度因果归因。

实际稀疏问题同样如此，当前**v11**已重新测量，不再借用上轮v10时延。prepared query[2051,12000)、direct3D含每次pack：

| real layer / TP | union3D / 5D µs | direct3D / 5D µs | union5D/3D | direct5D/3D |
|---|---:|---:|---:|---:|
| L3 / 2 | 2787.495 / 9254.411 | 2506.073 / 6734.057 | 3.320× | 2.687× |
| L3 / 4 | 1958.990 / 6640.836 | 2475.493 / 6694.577 | 3.390× | 2.704× |
| L3 / 8 | 1147.286 / 4162.602 | 2450.833 / 6658.437 | 3.628× | 2.717× |
| L47 / 2 | 2692.515 / 8838.048 | 2502.913 / 6711.277 | 3.282× | 2.681× |
| L47 / 4 | 1889.890 / 6349.015 | 2470.754 / 6680.436 | 3.359× | 2.704× |
| L47 / 8 | 1125.926 / 4011.182 | 2448.833 / 6653.496 | 3.563× | 2.717× |

本轮没有重新测full QSA，不能把这些本体相加或取最小当作完整call；有效/填充FLOPs与T均在每case JSON单列。

### 4. 为什么原MHA5D优势没有传到QSA5D

“基于MHA改出”只说明共享数学/helper，**不意味着保留了同一访存/流水实现**。当前真实调用路径是：

| 分支 | 3D路径 | 5D路径与关键差异 |
|---|---|---|
| dense | [dense._bounded_body](dense.py)主要复用linear body；连续DWORD DMA、linear LDS | 仍是该linear-shaped body套[_paged_common.dma](./_paged_common.py)，不是调用原paged MHA body；K按channel离散，V两次ushort load＋寄存器合并＋DS store来适配linear LDS。 |
| union | [union._body](union.py)保留linear式八阶段、4+4wave错相、mask预取/common循环展开、持久worker | [_paged_union.body](_paged_union.py)为压低live state重写成短流水：每排序task一个CTA，所有wave同相，当前tile softmax→PV依赖没有原跨tile exp/QK重叠；V-LDS stride另外产生冲突。 |
| direct | [_direct_packed](_direct_packed.py)单query wave读专门PK/PV布局，V四条128b load/半片，无V操作数重排 | 5D为另一种物理布局，V八条64b load/半片，再在PV消费点增加lane/register exchange；仍要block→page二次地址翻译，不是原MHA5D的整page连续DMA。 |

**Dense的访存是直接可见的失配。** 原MHA5D的V一条wave DMA使64lane各读4B，合起来256B连续区域；16个packet协作覆盖整N64/D256 tile。当前QSA dense5D保留linear consumer的channel2/lane分配，但SHUFFLE V相邻channel隔16B，因此一lane要读两条16-bit指令，另合并/DS写。K同样沿错的channel方向跨page stripe，而不是原MHA5D的token/channel-packet协作方式。

从实际源码提取公式的[静态地址分组模型](../../../../mytest/mydata/qsa_mha_layout_20260927_01/address_model_v2.json)，HK1/page64完整N64/D256、8waves，每组请求按16lane与64B地址bucket计数：

| 通路 | 8wave tile的VMEM指令数 | 有效请求字节 | 地址bucket访问数 |
|---|---:|---:|---:|
| 原MHA5D K | 128 | 32768 | 512 |
| QSA dense5D K | 128 | 32768 | 2048 |
| 原MHA5D V | 128 | 32768 | 512 |
| QSA dense5D V | 256 | 32768 | 8192 |

这证明同样有效字节被更分散的指令地址承载，**不是实测HBM流量变4倍/16倍，也不声明物理cache line=64B**。本轮实际ISA的单N64阶段，原MHA5D有32个K/V DWORD DMA；QSA dense5D有16个K DMA＋32个V ushort load并DS写，另有页地址读取。数值相同不代表这个适配是高性能实现。

**Union既有稀疏寻址成本，也有可以避免的实现退化。** 原MHA5D一个N64 tile大致对应一页（S32时两页），按页预取并保持规则地址；union的N64是16个独立四token块，可以落在16个物理页，需要block→page翻译和选中/尾部mask。该成本不能消失，但不足以解释全部3倍差距，因为dense也退化，而且下面的LDS对照可以显著改善。

当前union5D每个V读取地址为`32768+(lane>>4)*4096+(lane&15)*64+offset`，原MHA5D为`32768+(lane>>4)*4096+(lane&15)*16+offset`。对于[仓库记录的gfx942 ds_read_b128服务组](../../../../docs/swizzle.ipynb)与32个DWORD bank，当前stride64映射在每组内四个不同地址落同bank，模型为**4路冲突**；原MHA stride16模型为1。当前还保留每packet处理完整/部分四token块的分支，ISA中的这些动态分支不能当成无条件每次执行的load/wait；不重复上轮“静态VMEM数就是HBM流量”的错误。

**Direct增加了内循环指令与操作数交换。** BF16 V四token只占8token向量的一半，当前5D每半D128要8条64-bit load，而专用PV布局只需4条128-bit load。四个半片合计每BN32从16条V load变32条，新增两次页翻译；实际对应稳态回边静态VMEM从33到51（含可选index-prefetch分支），DS bpermute从8到72，MFMA仍64条。VGPR250→276、SGPR42→58，private/spill仍0。这个工作量差和地址/等待差是可见机制，但没有把每个指令或高VGPR独立量化为某个百分比；零spill不是同吞吐证明。

### 5. Union V-LDS诊断对照：约三分之一时延可由布局改动消除

为检验而不是只猜，预先登记[诊断协议](../../../../mytest/mydata/qsa_mha_layout_20260927_01/diagnostic_protocol.json)，创建仅研究目录可用的[vswizzle候选](../../../../mytest/mydata/qsa_mha_layout_20260927_01/vswizzle.py)。生产代码不改，candidate保留原body AST/数学/所有wait/调度/selection，将V LDS地址bit4/5与bit7/8做XOR，并按逆映射交换DMA source lane。

[CPU逐元素证明](../../../../mytest/mydata/qsa_mha_layout_20260927_01/bank_model.json)验证16384个BF16 V payload不变、每条global指令每16lane的地址多重集不变，`ds_read_b128`模型冲突由4→1。6例实际GPU结果与当前5D逐bit一致，原FP32/capture容差不变，guard/actual timed O/inputs/plans/sources通过；仍零private/VGPRspill/SGPRspill。

| 诊断用例 | 原union5D µs | 仅V-LDS XOR候选 µs | 耗时下降 | 同场union3D µs |
|---|---:|---:|---:|---:|
| causal2048 TP2 | 522.843 | 353.642 | 32.362% | 171.361 |
| causal2048 TP4 | 321.401 | 216.841 | 32.532% | 105.561 |
| causal2048 TP8 | 320.542 | 212.781 | 33.618% | 104.160 |
| real3 TP2 | 9161.650 | 6196.074 | 32.369% | 2788.475 |
| real3 TP4 | 6588.296 | 4431.264 | 32.740% | 1959.811 |
| real3 TP8 | 4152.782 | 2813.255 | 32.256% | 1147.386 |

18实现组合、2304raw、18次门禁全部通过；每例同场原版/候选/3D、10buffer/128sample。第一次诊断仅因mydata symlink解析后的freeze路径不在lexical root而失败、0raw；修正脚本路径后在新目录执行，失败收据保留。

这个干预实验强烈支持**V-LDS映射是重要瓶颈**，不是5D物理布局必然慢；但它不等于“原时延准确32%全是bank conflict”。代码生成也变化：real3 TP2候选VGPR222→252、SGPR95不变、零spill；单N64回边MFMA128、barrier8、LDS-read64及load/分支数不变，地址ALU和reg分配变化。未采PMC bank计数，所以只报告模型与干预效果。候选仍比3D慢约2.0～2.5倍，原8阶段错相/访存路径/页翻译等剩余问题仍需分别优化验证。

### 6. 下一步应修什么，不应下什么结论

1. **dense优先真正复用原MHA5D的协作K/V和LDS布局**，而不是继续用ushort gather模拟linear consumer。因其同语义causal问题已测出3～4倍额外代价，这比先怀疑5D缓存本身更直接。
2. **union先修V-LDS映射，再保留零spill重建原4+4wave错相与跨tile流水**；把block/page地址准备、部分块处理移出正常完整tile热路。不能直接删除page/tail/NaN检查，也不能为了流水复活此前SGPR/private spill版本。
3. **direct分别评估5D V协作读取、操作数布局及页索引开销**；省一次全KV pack后增加每query每Ntile工作，未必划算。不能把“no pack/no scratch”当成高性能目标本身。
4. 保留本轮原MHA/linear性能、不同页粒度、causal调度和共同变慢的边界。**没有证据说5D先天劣于linear，也没有证据说最新默认MHA5D必然更快。** 当前QSA三分支的适配效率需要改进，这是实现问题，不能由“MHA是基础”推出性能自动继承。

本轮主矩阵＋诊断合计**22场、14592raw、66次门禁**，全源/实际ELF/资源与原始样本保存在新研究目录。MHA/QSA生产文件逐字节未改、两仓库HEAD/index未改、旧报告与已启动的9078绘图服务保留；只把本节追加现有opt，没有新增Markdown，也没有部署诊断候选或运行整模型。

## 2026-09-27：补正——5D V可以128-bit读取，64-bit是当前direct的选择

用户指出SHUFFLE-5D V本来就为向量读取设计。该指正成立：前文“5D V需要8条64-bit load”把当前实现选择说成布局限制，不准确。准确说法是：**当前direct的`value_load()`每lane每次调用（N16×D128片段）发出8条64-bit load；专用packed V的对应调用发出4条128-bit load。这不是5D只能64-bit，也不是5D的最优实现证明。**

BF16的V形状为`[P,HK,S/8,D,8]`，最后一维是同channel的8个连续token。固定page/head/group8/channel后，8个BF16正好连续16B、起点16B对齐，可以直接用`buffer_load_dwordx4`读取；S32/64/128都满足。当前MHA的global→LDS实现具体选择协作DWORD DMA、随后128-bit LDS读，这也是实现策略，不影响布局支持global128的事实。

当前direct保留四token block的MFMA操作数分工：每lane负责4个选中token×8个channel，共64B有效payload。对于每个channel，这4个token占16B物理向量的下半8B或上半8B，因此当前每channel读64-bit，总8次。专用PV预排把同一四token块的两个channel放在连续16B中，4次128-bit即可得到相同64B有效payload。

不能做的只是**保持所有地址/分工不变，机械把8个64-bit load改成4个128-bit load**：

- 从8token group起点读128bit得到同channel的8token，不是两个channel各4token；若仅选中一个四token半块，其中一半没有被选择。
- 若从上半块起点（16B向量内偏移8B）直接扩宽，连续数据是当前channel的token4～7加下一个channel的token0～3，不是两个channel都取token4～7，也不能声称这个地址16B对齐。
- **正确的128bit方案完全可行：**向下对齐到8token group，读完整16B，再按块奇偶选低/高64bit；当前lane分工下仍为每channel一次，8次128bit读128B、只用64B，不会仅靠改位宽自动降到4条。这里的字节数是指令请求量，不是HBM实测流量。
- 若同一个8token group的两个四token块都被选择，可重新分配loader/lane并共享一次128bit读取的两半；也可允许孤立半块overfetch后丢弃。需要联合设计载入分工、寄存器/LDS重排、复用与MFMA输入布局，而不是以现有lane分工否定128-bit方案。
- 多读的未选中、future或padding元素必须丢弃/在PV前清零，不改变精确选中集合，避免`0*NaN`；物理page/descriptor边界仍要有效。

[CPU地址与payload核验](../../../../mytest/mydata/qsa_mha_layout_20260927_01/v128_contract_check.json)覆盖page32/64/128的28672个16B物理向量和57344个四token半块，证明128-bit读取/选半片的可行性及直接扩宽的错配；这只是地址/语义证明，**本次没有新GPU计时、没有实施128-bit候选、没有修改生产kernel**。后续应比较真正的128-bit协作方案，不能把上一轮现有64-bit路径的性能拿来代表5D原生向量设计的上限。

## 2026-09-27～28：修正三个原生5D分支——dense达标，union/direct仍未全部达到linear

### 任务、约束和验收结论

用户要求“修正三个5D分支的实现，使得性能和linear相当”。本轮实际重写并验证了三个分支，而非仅列TODO；开始前将“相当”明确为**同场、同scope、全部样本中位数的5D/3D比值≤1.10**。这一数值是本轮预先宣布的操作性标准，不是用户原话中的数值，不因结果失败而放宽。

**最终并未完成全部性能目标。** 保留实现为clean2；dense六个TP/长度组合全部达标，union仅1/15个scope按上述口径通过，direct为0/15，完整`qsa()`为8/21。不得把“功能与零spill通过”“比旧5D快很多”写成“三分支与linear相当”。最新正式结果就是本节clean2，不是T01 v10/v11或早期探索值。

固定GPU2／PCI0000:a4:00.0／gfx942 MI308X80CU；系统/bin/python3、Torch2.12.0 ROCm7.14、FlyDSL0.3.2。PTL Enabled/VECTOR,F8、650W未写；原`cudaPerf`不变。每版10个独立I/O buffer、每buffer2warmup、正式128samples、探索20samples，AB/BA交替，保留所有样本。新的输入副本、cache、日志、ELF/ISA、trace、独立包均在mytest/mydata，本轮没有新Markdown。

- [预先固定协议](../../../../mytest/mydata/qsa_5d_parity_20260927_01/protocol.json)
- [最终结论与主表](../../../../mytest/mydata/qsa_5d_parity_20260927_01/analysis_final/summary.json)
- [完整57场统计、配对比值、16sample时间分块及长尾](../../../../mytest/mydata/qsa_5d_parity_20260927_01/analysis_final/formal_audit.json)
- [全部探索、失败和日志清单](../../../../mytest/mydata/qsa_5d_parity_20260927_01/analysis_final/campaign_inventory.json)

### 保留实现：没有隐藏KV转换或路由规避

1. **dense**：调用真正的[MHA paged `_body`](../mha/mha_pa_bf16_256_paged_942.py)，而不是把linear数据通路换成5D地址。保留QSA最长causal tile优先/蛇形persistent任务分配；K/V原生DMA、4＋4wave错相。每plan一个FP32 `ones`标量供原MHA descaling参数使用，不是KV scratch。原3D bounded body已去掉失效的5D分支。
2. **union**：BM128/BN64、8wave、64KiB LDS和4＋4wave八阶段流水。K/V LDS改为**四token块优先**，每wave负责两个block；K/PV均直接匹配MFMA载荷，V用128bit LDS读、输出128bit LDS/store。common前缀四次展开、masked段保持精确membership。每次先构造`[tiles,CAP,2]` int32 K/V byteoffset表，准备kernel计入branch/full时延，KV本体不复制。长请求保留排序persistent worker；`TASKS<=GRID`用相同body的单任务特化，消除短请求无收益循环状态。
3. **direct**：新[_paged_direct.py](_paged_direct.py)，N32、每query/HK一个wave，LDS=0。每次GPU将已选择且相邻的两个四token块稳定放到地址表前部；`[M,513]` int32只存地址与paired数。完整成对N32区每个N16×D128片段真正用**4次128-bit V读**，两lane组共享8token载荷并在PV前交换一位；剩余单块用8次64-bit并在PV前清理NaN尾。两个静态循环避免动态load分支合流产生早期VM等待。没有扩选token，没有PK/PV，也不运行全KV pack/gather。成对重排可能改变浮点归约顺序，保证原容差与同版repeat，不声明跨3D/5D bitexact。

原3D packed direct仍包含**每次pack**，64MiB PK+PV预算不变；原3D raw fallback与公共`qsa()`契约不变。路由也未为了掩盖慢分支而改变：3D packed为pad1.7，raw/5D仍rho4。新增metadata和pair成本均包含在5D计时里；它们不是T02准备开销优化已经完成的证据。

### 同步审查与资源失败均保留

调优中曾将union高半片LDS等待推迟至barrier后；虽然目标数值测试通过，源码生命周期审查发现慢/STAGGER组在S2/S6 rendezvous后可能遇到快组覆盖旧K/V。因此最终版在**STAGGER组的S2/S6 barrier前强制lgkmcnt(0)**，其余消费者前等待及最终vmcnt/lgkmcnt drain保留。早期U33约3112.817µs是旧等待版本，不能作为最终安全版时延；安全clean1同场为3131.876µs，正式clean2另列下表。

首次完整clean1为65个数值测试通过，但两个short union特化（M33/P3000/H12或H3/page64）各有4个SGPR spill，整轮仍失败。没有将private=0误写成零spill；shortfix去掉`TASKS<=GRID` persistent循环后，原签名VGPR/SGPR从238/106变为230/96，三个spill字段都归零，再执行完整clean2。

- [失败完整clean1](../../../../mytest/mydata/qsa_5d_parity_20260927_01/full_clean1_validate/result.json)
- [shortfix六个adapter回归](../../../../mytest/mydata/qsa_5d_parity_20260927_01/shortfix1_validate/result.json)
- [最终clean2：65 passed、24 perf deselected、0 skipped](../../../../mytest/mydata/qsa_5d_parity_20260927_01/full_clean2_validate/pytest.xml)
- [最终327对象／364kernel实例及资源](../../../../mytest/mydata/qsa_5d_parity_20260927_01/full_clean2_validate/result.json)

最终145个native特化为dense34、union55、direct56；全部private/VGPRspill/SGPRspill=0。dense VGPR223～224、SGPR82～90、LDS65536；union230～237／96～106／65536；direct252～254／42～50／0。另核验50个Triton地址准备对象，三字段同样为0。正常测试覆盖page32/64/128×TP2/4/8、HK2、ragged/空请求、物理NaN padding、guard、repeat、页面/K/V/indices图更新、旧8份真实capture×TP、本地SGLang adapter；新增address-scratch投毒/稳定pair集合/图重放刷新回归。

NaN验证针对本任务原有物理padding和direct causal尾；不额外承诺“任意逻辑有效但被某query排除的NaN/Inf都不传播”。union/dense的共享PV会出现IEEE `0*NaN`，这是与原MHA/3D相同的非有限输入边界，不应把已过padding测试扩大为任意非有限输入支持。

### 正式结果：dense六例全部与linear相当

scope为同一全部query前缀，单位µs；完整QSA含recovery/校验/调度，不是单kernel相加。

| 长度 | TP | dense 3D→5D | 5D/3D | 完整qsa 3D→5D |
|---|---:|---:|---:|---:|
| 2048 | 2 | 160.660→160.761 | 1.0006 | 211.361→213.762 |
| 2048 | 4 | 101.840→102.901 | 1.0104 | 152.841→156.121 |
| 2048 | 8 | 99.880→102.020 | 1.0214 | 150.681→154.921 |
| 2051 | 2 | 178.960→173.881 | 0.9716 | 229.881→227.061 |
| 2051 | 4 | 107.680→105.041 | 0.9755 | 158.821→158.521 |
| 2051 | 8 | 106.340→103.840 | 0.9765 | 157.060→156.681 |

这里的TP4/8是原TP2 capture的local head H6/H3派生回放，不是实际分布式TP4/8新采集。dense合成输入同样不进行TP通信。

### 正式结果：真实两层union/direct/full

真实输入为TP0、M12000/P0。独立branch仅测query[2051,12000)、9949行；full测全部12000行，不能混用scope。下表单位ms，R=全样本中位数之比，P=128个配对ratio的中位数。

| scope | TP | L3 3D→5D | R / P | L47 3D→5D | R / P |
|---|---:|---:|---:|---:|---:|
| union | 2 | 4.2211→4.7565 | 1.1268 / 1.1269 | 4.0614→4.5777 | 1.1271 / 1.1280 |
| union | 4 | 2.4670→2.5807 | 1.0461 / 1.1170 | 2.7196→3.0760 | 1.1310 / 1.1243 |
| union | 8 | 1.1479→1.2785 | 1.1138 / 1.1133 | 1.1252→1.2619 | 1.1215 / 1.1210 |
| direct | 2 | 2.5038→3.9343 | 1.5713 / 1.5757 | 2.4985→3.9033 | 1.5623 / 1.5656 |
| direct | 4 | 2.4618→3.8684 | 1.5714 / 1.5720 | 2.4573→3.8427 | 1.5638 / 1.5645 |
| direct | 8 | 2.4475→3.8388 | 1.5684 / 1.5682 | 2.4471→3.8127 | 1.5580 / 1.5603 |
| qsa | 2 | 3.5884→4.2422 | 1.1822 / 1.2600 | 3.2171→3.9292 | 1.2214 / 1.1859 |
| qsa | 4 | 3.3542→3.6637 | 1.0923 / 1.0791 | 3.2945→3.5159 | 1.0672 / 1.0528 |
| qsa | 8 | 1.6246→2.1926 | 1.3496 / 1.3490 | 1.5940→1.9435 | 1.2193 / 1.2191 |

L3 TP4 union按预定R口径通过，但P为1.1170，不能据此声称典型配对开销<10%。所有正式独立union的R范围1.0461～1.2328；独立direct范围1.1035～2.1452；full范围0.9877～4.1780。

TP2 union双方明显同步变慢：L3的0～31及112～127样本约2.789→3.13ms，中间32～111约4.23→4.78ms；L47对应约2.698→3.03ms和4.07→4.60ms。每16样本分块的R仍约1.122～1.131。完整中位4.22/4.06ms不是误写，不能只保留快段替换正式分母。没有kernel期间频率遥测，不归因为时钟、温度、NUMA或PTL。

最大正式长尾是L3 TP4 union5D sample56/buffer6 **54.645489ms**，同对3D2.977896ms；全部保留。表中的漂移与长尾不是新的分布式服务观测。

### 正式结果：高低重合与短请求

以下同列为3D→5D µs；low/high为M2048/P30000，high沿原generator每32query共享priority，short为M64/P30000。

| case | TP | union | direct | full qsa |
|---|---:|---:|---:|---:|
| low | 2 | 3023.796→3350.997 | 592.543→1269.506 | 689.243→1368.067 |
| low | 4 | 1905.330→2242.712 | 588.343→1256.166 | 678.764→1354.327 |
| low | 8 | 1291.327→1478.928 | 586.444→1258.047 | 670.823→1349.427 |
| high | 2 | 406.442→469.342 | 578.103→1140.546 | 528.523→593.803 |
| high | 4 | 214.521→252.901 | 573.203→1127.286 | 319.022→360.602 |
| high | 8 | 110.120→135.761 | 572.463→1121.506 | 211.041→239.881 |
| short | 2 | 696.244→811.284 | 116.861→128.961 | 149.641→625.203 |
| short | 4 | 918.365→1076.845 | 115.480→127.761 | 148.341→191.481 |
| short | 8 | 1197.967→1422.488 | 114.201→126.721 | 145.641→196.201 |

short TP2 full4.178倍明显高于独立direct1.104倍，不能用独立direct替代full结论；当前rho4与pad1.7路由差异和构表/排队成本都仍存在。此轮不通过改路由“解决”用户要求的三个分支性能。

工作量沿用$F_{effective}=N_{selected} H\cdot4\cdot256$。direct填充为$4\cdot16\cdot32\cdot256\sum_q\lceil n_q/32\rceil$；union为$4\cdot128\cdot64\cdot256\sum_t\lceil U_t/16\rceil$。TP2 L3独立direct有效F250558144512、填充F337744756736，正式有效100.071→63.686T、填充134.893→85.846T，不能称5D达到100T或160填充T。[CPU逐tile选集/FLOPs/query比例模型](../../../../mytest/mydata/qsa_5d_parity_20260927_01/analysis_final/work_model.json)独立复现selection并逐项匹配21个full实际routing；不是实测HBM流量。full的路由和每branch工作量均单独记录，不能用相加的独立中位数推算full。

### 主要失败候选及可复用教训

全部原始版本/源码/ELF/测试/日志见清单，以下仅列机制与代表值；这些是20sample探索，不能替代上方128sample正式验收。

| 路径 | 机制/结果 | 决策 |
|---|---|---|
| dense D1 | 真实MHA paged数据通路＋QSA causal配平，TP2/2048约160.881µs | 保留并完成六例正式 |
| union U1 | 新native PV输出channel顺序错误；常量channel V可复现 | 修复epilogue，失败保留 |
| union U2/U3/U4 | 原MHA流水、提前byteoffset、common展开：4.312→3.947→3.571ms | 后续继续改 |
| union U5/U6 | 显式PV和融合leaf约3.630/3.737ms | 未因指令减少认定胜出 |
| union U7～U10 | K128/V64 vector→LDS，数值可过但先spill；短生存期解spill后5.008ms | 拒绝 |
| union U11/U13 | persistent导致VGPR spill；显式uniform＋延后future后U14零spill3.426ms | 仅保留通过组合 |
| union U16～U19 | 四token块优先LDS、step交错、uniform SOFFSET约3.220→3.205→3.176ms | 主体保留 |
| union U20/U21 | 每wave只读两个地址对；SGPR spill后pin消除，但3.839ms | 拒绝 |
| union U24～U33 | QK/exp、late-V、实际关闭packed-FP32、分块地址准备；最好旧等待约3.113ms | 先补安全wait再重新测 |
| union U26/U28/U31/U34 | M0/SOFFSET强制leaf、inline页表、masked展开等造成spill | 全部拒绝 |
| union U35/U36/U37 | inline global页表约3.654/3.555ms，MFMA优先级3.300ms | 拒绝 |
| direct D1～D7 | native V ownership、地址表、延后high V、移除输出LDS：约4.35→4.17→4.21ms | 零spill并不保证parity |
| direct D3/D18/D19 | 机械128bit选半约7.963ms；延后选半D18 spill，N16容纳后4.465ms | 拒绝 |
| direct D9 | opaque VMEM inline asm后repeat出现巨大误差 | 恢复编译器可见native load |
| direct D10/D17 | N16少VGPR仍4.371ms；D128输出拆分重复QK6.210ms | 拒绝 |
| direct D12 | V1288token载荷映射另一PV segment，4.486ms | 拒绝 |
| direct D13/D14 | 4lane V packet＋PV前lane置换，先spill，解spill后4.590ms | 拒绝 |
| direct D22b/D29/D31 | 真4x128合作、两个静态循环、对齐替代地址DS：4.048→3.955→3.934ms | 保留机制，仍未parity |
| direct D23/D24 | 统一物理8token含掩码padding4.845ms；V包相位导致spill | 拒绝 |
| direct D27 | 8路task转置4.597ms | 拒绝；不声称task编号等于XCD |
| direct D32～D36 | Kstream4.004ms、四query wave3.973ms、双V预取3.978ms、CTA split4.514ms、延迟rescale3.963ms | 没有达到目标，恢复较简洁D31 |
| study Triton | compiler-native多wave direct17.045ms | 不入生产 |
| study two-pass/split | score/P物化8.099ms；全局FP32 split4归约4.740ms | 不入生产，不保留大scratch |

缓存实验纠错必须保留：当前FlyDSL `raw_ptr_buffer_load` 的`aux`接受Python int/IntegerAttr，传IR Value会被wrapper静默丢弃。D8所谓coherent实验实际ISA没变，不能说明cache策略无效。gfx942本机LLVM定义是SC0=bit0(1)、NT=bit1(2)、SC1=bit4(16)；D16曾误称SC1，实际NT且7.690ms；D15真正SC0约4.363ms、D25真正SC1约4.048ms，均无改善，最终恢复0。不能沿用其它gfx架构GLC/SLC名称猜本机编码。

### ATT：只回答局部发射/等待，不冒充HBM或端到端归因

两次GPU2/SE0/CU1/四SIMD采样均核对实际ELF、完整wave和共同稳态窗口，排除每wave首末四轮；每次10buffer实际输出校验。原64-bit V路径D11的124wave均64轮，MFMA模型占比27.34%、VMEM指令区间66.01%、VMEM完成等待1.19%。动态成对D28为28.90%、36.85%、12.21%，另DS等待5.47%；具体branch merge处出现median284/512cycles的vmcnt0，推动最终拆成静态循环。

这些区间按实际physical SIMD合并，不把wave等待简单相加；不是HBM字节、整GPU利用率、clock因果或普通timer时延。D28不等于最终clean2，最终未再采ATT，不能拿其PC/比例给clean2贴标签。

- [D11完整ATT分析](../../../../mytest/mydata/qsa_5d_parity_20260927_01/direct_dir11_att_analysis.json)
- [D28成对ATT分析](../../../../mytest/mydata/qsa_5d_parity_20260927_01/direct_dir28_att_analysis.json)

### 交付完整性、包与剩余工作

正式57场/114label共14592raw，加60场探索2400raw＝**16992普通计时样本**；正式171、探索180次门禁均通过，另ATT6快照，总357。84次正确性/collection尝试中27次失败全部保留（不将重复测试相加称独立覆盖）；203份日志保留。最终完整功能65passed用时1560.55s；第一次full资源失败的1612.71s也保留。

清理了旧5D bounded/packed分支和所有失败候选死代码；只保留新native消费者。源码AST特化核验之外，[实际3D产物对照](../../../../mytest/mydata/qsa_5d_parity_20260927_01/three_d_artifact_audit.json)匹配原T01 v11的132个GPU2签名（dense29/union39/raw direct27/packed37），整ELF、`.text`、全部资源均一致；代表124个不同ELF、169个kernel函数。该身份结论不包括不匹配签名，也不代表5D数值与3D bitexact。

独立[plugin_clean2源码manifest](../../../../mytest/mydata/qsa_5d_parity_20260927_01/plugin_clean2/pyhip_qsa_runtime/source_manifest.json)包含9个QSA模块＋3个MHA依赖＋plugin共13源码。已验证[禁用惰性加载](../../../../mytest/mydata/qsa_5d_parity_20260927_01/plugin_clean2_disabled.json)、[启用六hook/五源ABI拒绝/6个原生5D adapter执行](../../../../mytest/mydata/qsa_5d_parity_20260927_01/plugin_clean2_enabled.json)，没有experiments导入或隐藏gather/PK/PV。本轮没有SGLang源码变更、未部署整模型、未采新的多卡TP4/8或TTFT/throughput。

计时输出核验边界：每layout10个buffer先按原参考和FP32抽样行`.02/.02`校验，正式前poison，正式后直接bitwise验证**实际计时保留输出**，没有先重跑覆盖；但不是每个sample单独保存输出，也未把所有plan数组落盘。full workspace深度身份和完整第三方运行环境并非全部冻结，不能扩大审计证明范围。

本轮保留的是**正确、零spill、显著修正旧5D低效实现，但未达到全部parity**的版本。T01功能继续成立；当前真正完成的性能子项只有dense六例。union的稳态约12%差距和short/high差距、direct真实约56%～57%及合成约2倍差距仍待解决；T02/T03/T04/T05/T06/T07/T10/T11/T12等既有TODO不能因此标完成。后续需要更有效的native数据消费/请求组织或完整成本模型，而不是放宽容差、允许spill、免费KV转换、删慢样本或通过路由绕过慢分支。

## 2026-09-28：对照3D/5D pipeline，分离page-size与V的四／八token物理布局

用户追问：“5d union比3d节省了转置，理论上更快，对照3d和5d的pipeline找出差距；direct理论上计算是一致的，对比3d pipeline找差距；还是说pagesize=64会有额外成本，如果改成pagesize=4会不会好一些？”

**本轮结论：算量相同不代表访存流水等价。union的主要可验证改进不是缩页，而是Vvec8与四token选块之间的物理载荷不匹配；direct还明显受K的page内channel跨度、地址指令及V协作重排影响。隔离Vvec4＋S4原型在两层TP2 direct已接近3D，但它是新缓存ABI，不是现有SHUFFLE5D只改page-size。** 生产Python、默认路由、SGLang和之前13源码插件完全不变；下面数据不能给生产clean2重贴性能标签。

研究：[预先固定协议](../../../../mytest/mydata/qsa_pipeline_pages_20260928_01/protocol.json)、[对照入口](../../../../mytest/mydata/qsa_pipeline_pages_20260928_01/study.py)、[纯CPU汇总](../../../../mytest/mydata/qsa_pipeline_pages_20260928_01/summary.json)、[实际ISA循环账本](../../../../mytest/mydata/qsa_pipeline_pages_20260928_01/isa_pipeline.json)、[地址窗口模型](../../../../mytest/mydata/qsa_pipeline_pages_20260928_01/address_model.json)。新增文件仅在mytest/mydata，无新Markdown。

### 1. union：同样八阶段，但省掉的转置不等于整个Memory阶段更便宜

准确基线是[union.py](union.py)，不是拿MHA dense和QSA sparse互比；5D为[_paged_union.py](_paged_union.py)。两者每个wave、一个N64稳态phase均为：

| 阶段 | 3D与5D共同工作 | 关键差别 |
|---|---|---|
| S0 Memory | 读K(t)低N32的16条LDS128，同时16条DWORD DMA装V(t−1) | 3D的DMA leaf把M0、LDS读、DMA固定交错；5D native lowering另生成SOFFSET/M0和hazard间距 |
| S1 Compute | 32条QK MFMA，与上一tile的exp交错 | 数量相同；5D显式双N16交错，3D原schedule仍保留 |
| S2 Memory | 16条LDS128读K(t)高N32，退休V DMA | 3D慢组按特定读顺序用lgkm8提前退休将被覆盖部分；5D慢组安全lgkm0，不能直接删 |
| S3 Compute | 32条QK MFMA、mask、P打包与local sum | 相同精确membership、相同common/unroll覆盖 |
| S4 Memory | 16条LDS128读V(t−1)低D128，同时16条DWORD DMA装K(t+1) | 3D逐步准备V转置；5D载荷已转置，但仍需等读者与DMA |
| S5 Compute | 32条PV MFMA、sum/max | 3D后续V转置穿插PV，部分成本被覆盖；5D无该转置 |
| S6 Memory | 16条LDS128读V(t−1)高D128，退休K DMA | 3D渐进消费；5D慢组覆写前退休仍是必要安全条件 |
| S7 Compute | 32条PV MFMA、center/rescale | 同样4＋4错相，下一phase沿用同一LDS槽 |

真实ELF的完整单phase回边（未触发rescale支路）核对：**两者均128 MFMA、64 LDS128、32条K/V DMA、8个s_barrier**。3D含136条v_perm_b32，其中128用于V转置、8用于P打包；5D仅余8条P打包。3D回边为12条s_waitcnt，5D两个wave-group版本为8或7条，**不是5D有更多barrier／更多MFMA／8倍K重读**。把Python的sched_barrier当CTA barrier也不成立。

V的物理请求才是重要差异：

- 3D一条V DMA每wave取256B连续数据；当前5D Vvec8每channel有8token，但一个QSA块只要其中4token。[_dma_v](_paged_union.py#L65-L74)的地址是`base + (lane>>1)*16 + (lane&1)*4`，即**每16B取8B**；64lane仍只请求256B，却分布在512B跨度，触及8个对齐64B地址窗口，而连续载荷为4个。这是地址窗口模型，不是声称HBM流量必然翻倍。
- 此式不依赖S=64；S32/S16/S8仍有相同8token interleave。因此仅缩page并不消除V的空洞。
- 隔离[Vvec4 DMA](../../../../mytest/mydata/qsa_pipeline_pages_20260928_01/variants.py)保留同一4token块优先LDS、同一QK/PV body、同一安全barrier，仅将V缓存改为四token内层和连续lane×4的DMA。两层union降低约8.4%～8.6%；S64已接近3D，S4反而略慢。VGPR237/SGPR106/LDS64KiB与原5D相同，三spill字段0。这比“少转置理论应更快”提供了可测的原因约束，但没有将全部改善归为未测的HBM或cache counter。

### 2. direct：MFMA相同；V打包粒度、跨lane重排和K地址跨度不同

基线[_direct_packed.py](_direct_packed.py)每次call先pack，计时包含该pack。一个query-wave、N32循环，3D和5D都是32条QK＋32条PV＝**64条MFMA**，不是5D多算。5D成对分组会改变选块遍历顺序，因此只宣称相同选集／算量／原误差门限，不宣称和3D逐位相同。

| 每N32循环，单wave | 3D packed | 当前5D成对区间 | 当前5D未成对区间 |
|---|---:|---:|---:|
| K128加载 | 16 | 16 | 16 |
| V128加载 | 16 | 16 | 0 |
| V64加载 | 0 | 0 | 32 |
| 额外V跨lane交换 | 0 | **32条ds_swizzle**，伴随选择及编译器wait | 0 |
| 原有地址／softmax交换 | 保留 | 同类操作仍存在 | 同类操作仍存在 |

所以“真正4×128读取”只说明一个V半片的请求宽度，**并不代表它已经是MFMA原生operand**。3D pack一次就把两channel×四token放在16B，所有query直接复用；Vvec8成对路径读的是一channel×八token，仍需交换token-half与channel的归属。实际成对回边41条wait（含compiler插入的swizzle依赖退休），不是源码两次lgkm0就等于两次实际等待；不能将wait条数换算成毫秒损耗。

预取顺序也不一样，必须以ISA而非只看Python判断：3D把第一个N16 segment的V低／高D128都发出，再vmcnt8释放K进入QK。5D源码延后V高半；**实际成对ISA把高半挪到vmcnt4之后、第一条QK之前**，仍不能利用K退休之前的那段延迟；未成对高半与QK交错。第二个N16 segment的高半则在第一组PV后才发。机械把两段高V都提前的Vvec4实验虽然数值通过，却产生1／2个VGPR spill和8／12B private，已拒绝计时，没有降低零spill要求。

K还有真实page-size成本：[_paged_common.key_load](_paged_common.py#L29-L34)的part间距为`64*S`字节。S64是4096B，除首part外的7个offset不能放进12bit buffer立即数字段，实际每个K半片还需对应向量地址加法；S8是512B，0…3584可作为立即数；S4是256B，且四token的全部D256集中在2KiB微块。不能把“跨度大”直接说成64倍不coalesced：本轮分离块地址模型中每条K128指令的64B窗口数相同，变化的是跨part的4KiB地址footprint和地址生成。无PMC，不宣称L1/L2命中率或TLB miss数已测。

### 3. page-size=4不是现有SHUFFLE5D参数，而是新布局实验

当前公开契约：[K(P,HK,32,S,8)、V(P,HK,S/8,256,8)](qsa.py#L122-L127)，入口只接受S32/64/128。SGLang BF16 vectorized pool按16B向量取X=8并断言S能被X整除；S4使V的S/8轴无法表示非零token。S8/16在物理公式上可行，但本轮仅prepared sparse branch研究，未扩公开支持列表。

实验Vvec4使用 **K(P,HK,32,S,8)，V(P,HK,S/4,256,4)**，S=64或4；每次attention直接读这个resident cache，不做KV pack/gather。V读取16B可以包含两channel×四token，不是强迫只用64-bit。direct沿用每call地址sidecar的稳定选集顺序，但Vvec4没有使用“8token pair必在同一页”的读取假设，所以S4页间不相邻仍正确。新的cache writer／allocator／其它消费者兼容性未实施。

缩小page会扩大page-table占用：固定logical length下S4表长约S64的16倍；但是当前[_direct_page_offsets／_union_page_offsets](_paged.py#L18-L62)已经按每个选中四token块查表，**查表操作数并不会自动变成16倍**。变的是不同表项数量、复用和物理地址公式。S4让完整选块的块内token偏移为0，但不能取消查表、scatter、launch或每call刷新。

### 4. 同场实测：两层TP2、10buffers、每label128samples

M12000/P0、H12/HK1、仅query[2051,12000)。3D每call pack、5D每call地址准备均计时；layout初始化、allocation、JIT、block recovery和union计划构建不计，与此前prepared branch边界一致。每场labels轮换起点并反转次序，所有raw保留。**这不是full qsa或整模型性能。**

| resident布局 | Layer3 union ms | Layer47 union ms | Layer3 direct ms | Layer47 direct ms |
|---|---:|---:|---:|---:|
| 3D原版 | 4.191 | 4.026 | 2.474 | 2.472 |
| 原5D Vvec8，S64 | 4.678 | 4.500 | 3.923 | 3.889 |
| 原Vvec8，S32 | 4.681 | 4.501 | 4.031 | 3.992 |
| 原Vvec8，S128 | 4.693 | 4.508 | 3.947 | 3.908 |
| 原Vvec8，S16（仅研究） | 4.681 | 4.505 | 3.729 | 3.689 |
| 原Vvec8，S8（仅研究） | 4.676 | 4.492 | 3.548 | 3.501 |
| **新Vvec4，S64** | **4.284** | **4.111** | **3.226** | **3.250** |
| **新Vvec4，S4** | **4.327** | **4.159** | **2.585** | **2.585** |

- union Vvec4/S64相对3D R=1.0223/1.0213，配对中位P=1.0321/1.0354；Vvec4/S4 R=1.0325/1.0331、P=1.0363/1.0383。原Vvec8/S64仍R=1.1163/1.1178、P=1.1221/1.1235。缩S8没有有效改变union瓶颈。
- direct原Vvec8/S8比S64改善9.5%／10.0%，仍慢41.6%～43.4%；Vvec4/S64仍慢30.4%～31.5%；**Vvec4/S4 R=1.0450/1.0456、P=1.0451/1.0459**，比旧5D S64快约34.1%／33.5%。不能把Vvec4收益全部归因于page4。
- 有效F两层同为250558144512。direct3D有效101.270/101.352T，Vvec8/S64为63.876/64.430T，Vvec4/S4为96.911/96.930T。union Vvec4/S64为58.486/60.945T；全部label的有效T与原始min/max已在汇总中保存。未用“更高填充F”更换分母。
- union仍有双方共同快／慢阶段，完整中位未筛；Layer3 direct3D最大3.668ms等长尾全部保留。不是新的分布式TP4/8或运行时频率证据。

四场共**4096普通raw**，12次普通采样门禁通过。入表104个保存编译对象、106个kernel metadata实例三个字段均0；两层Vvec4/S4 direct VGPR244、SGPR50、LDS0。所有版本对capture和FP32抽样oracle保持`.02/.02`，10buffer实际计时保留输出逐位对本版已验证输出、输入／plan哈希不变、越界guard不变；并有M33/P3000跨尾页合成检查。尚未完成新布局HK2/ragged/graph/TP4/8/dense/decode/plugin全矩阵，因此不把原型加入公开运行时。

### 5. 本轮ATT门禁失败如实保留；不据其时长宣布归因完成

为当前3D与S64各采union/direct两个kernel，共4个trace。两场入口和采样前合格，但结束GPU利用率分别9%／10%，超过5%门限；保留原始结果、未重采、未改PTL或时钟。普通4096raw是此前独立且通过所有门禁的场次，不受此失败重新标记为profile时间。

CPU已确认4个trace ELF与相应普通场次整ELF一致，union各8个wave、direct各125个wave，全部instruction stitching完整、无pc_index指向注释错误。ISA与已执行指令可作结构核对；[union ATT受限分析](../../../../mytest/mydata/qsa_pipeline_pages_20260928_01/att_union_analysis.json)、[direct ATT受限分析](../../../../mytest/mydata/qsa_pipeline_pages_20260928_01/att_direct_analysis.json)明确写`performance_gate_passed=false`。这些诊断按指令／wave统计，不是共同resident窗口的物理SIMD墙钟分解；不将其累计stall求和当整卡时间，不报告HBM、clock或稳态占比的因果百分比。

另外保留：首次重装FlyDSL隔离wrapper遗漏原注解，GATED变为dynamic而编译失败；修正仅在[研究绑定](../../../../mytest/mydata/qsa_pipeline_pages_20260928_01/variants.py)恢复constexpr注解。后续提前V高半的资源失败同样保留，未进行性能采样。没有把任何失败候选写入生产。

下一步设计方向应把**64token分配页**与**4token计算微块**分开：可研究在S64宏页内将K/V都按4token微块存放，争取保留小块连续性而不使页表变长16倍；这需要新物理ABI／cache writer，不是本轮已实现或测过的生产方案。若必须保持现有Vvec8 ABI，则应改进union的成对DMA组织以及direct的operand交换/地址预计算/预取生存期，而不是断言省转置必然快，或盲目调page-size即可解决。

补充[最终CPU审计](../../../../mytest/mydata/qsa_pipeline_pages_20260928_01/delivery.json)：HK1时S4的K微块内部地址与原3D direct packed K逐元素相同（19456个BF16元素），区别仍有物理页顺序及页表翻译；这解释了为什么S4能接近packed消费者，而不是仅“页表更小”——实际上页表更大。Vvec4/原Vvec8两种DMA进入同一union LDS的4096个字节标签检查一致，两层各标签整ELF相同。此前正文追加前的opt前缀、生产clean2所有Python哈希、两个仓库HEAD/index以及原9078服务均通过完整性检查；新布局完整生产验收仍未完成。

## 2026-09-28：按用户要求撤销QSA 5D支持，后续默认PyHIP .venv

**最新状态取代上方T01／clean2的“当前支持”描述：QSA恢复为仅3D，停止维护本轮原生5D接入。** 用户明确要求“后续缺省都使用pyhip下的.venv python环境；因为5d v最少是8个token连续，和现在compact ratio=4冲突，撤销本次5d支持的修改，包括对sglang的修改；将当前性能、限制记入文档”。本节仅追加，前面的成功、失败、测试、性能和结论均作为当时历史保留，不改写旧收据。

### 撤销原因与保留结论

- 现有BF16 SHUFFLE V的物理内层是8token，而本QSA `compress_ratio=4`按四token完整块选择；这是**物理读取粒度与稀疏选块粒度不匹配**，不是数学上不能计算或5D不能128-bit读取。旧实现已能正确、零spill读取，但要承受半向量载荷、跨lane operand重排或额外地址／预取成本。
- union省掉V转置并未自动超过3D：3D可把部分转置与PV交错，旧5D的V DMA每16B只取其中8B。原Vvec8从S64缩到S8几乎不改善union；direct虽改善约10%，仍慢约42%～43%。S4不满足原V(P,HK,S/8,256,8)合同，不能只改一个page-size参数。
- Vvec4/S4研究原型给出不同物理ABI的性能线索，不是现有5D支持已达标；未完成HK2/ragged/graph、TP4/8、dense/decode、cache writer及SGLang整模型接入，不保留为生产fallback，也不在本次撤销中继续推进新布局。

### 撤销前最后实测性能（历史数据，不是本次.venv复测）

MI308X/gfx942、物理GPU2、两层TP2本地H12/HK1、M12000/P0、query[2051,12000)。每实现10个独立buffer、2warmup、128samples；3D direct包含每次KV pack，5D包含每次地址准备。**均为prepared分支，非full qsa、TTFT或吞吐**；普通4096raw及12次通过门禁的结果完整保留在[页长／布局研究汇总](../../../../mytest/mydata/qsa_pipeline_pages_20260928_01/summary.json)。

| 版本（L3 / L47） | union ms | direct ms | 状态 |
|---|---:|---:|---|
| 3D | 4.191 / 4.026 | 2.474 / 2.472 | 恢复保留的实现；本表仍是撤销前测量 |
| 原5D Vvec8/S64 | 4.678 / 4.500 | 3.923 / 3.889 | 本次撤销；相对同场3D约1.116～1.118×／1.573～1.585× |
| 原Vvec8/S8 | 4.676 / 4.492 | 3.548 / 3.501 | 仅研究，不提供公开支持 |
| 新Vvec4/S64 | 4.284 / 4.111 | 3.226 / 3.250 | 新ABI原型，仅历史 |
| 新Vvec4/S4 | 4.327 / 4.159 | 2.585 / 2.585 | 新ABI原型；direct约1.045×，不是原5D达标 |

同场3D direct有效101.270/101.352T，原5D S64为63.876/64.430T，新Vvec4/S4为96.911/96.930T。union有共同快慢阶段，完整中位未筛；上述Vvec4/S64 union的配对ratio中位为1.032/1.035，与中位数之比1.022/1.021不同。最后两场ATT结束门禁9%／10%失败，保留且未重采，不据其时长作因果百分比。

更广的[clean2正式矩阵](../../../../mytest/mydata/qsa_5d_parity_20260927_01/analysis_final/summary.json)为57个scope、14592条正式raw：dense比值0.972～1.021且6/6通过；union仅1/15、direct0/15、full8/21满足预定≤1.10门槛。65项旧环境功能测试、327个编译对象三个spill字段全0是撤销前验证，不能重标为本次回退后的新环境测试。Vvec4原型及更早T01数据也不覆盖缺失的全栈验收。

### 精确回退范围

以[T01之前的逐文件SHA清单](../../../../mytest/mydata/qsa_5d_20260927_01/sources_before.json)及其冻结before源码为依据，不按Git HEAD整体重置，不影响更早3D优化和用户暂存修改。

- PyHIP恢复[qsa.py](qsa.py)、[dense.py](dense.py)、[direct.py](direct.py)、[_direct_packed.py](_direct_packed.py)、[union.py](union.py)及[test_qsa.py](test_qsa.py)的pre5D正文。删除page_table公开参数、5D校验/工作区/地址scratch、三分支5D分派、5D回放与专属测试。
- 删除四个5D专属活动模块；撤销前完整版本仍可查阅[dispatcher归档](../../../../mytest/mydata/qsa_5d_revert_20260928_01/before/pyhip/experiments/attention/flydsl/qsa/_paged.py)、[公共载荷归档](../../../../mytest/mydata/qsa_5d_revert_20260928_01/before/pyhip/experiments/attention/flydsl/qsa/_paged_common.py)、[direct归档](../../../../mytest/mydata/qsa_5d_revert_20260928_01/before/pyhip/experiments/attention/flydsl/qsa/_paged_direct.py)、[union归档](../../../../mytest/mydata/qsa_5d_revert_20260928_01/before/pyhip/experiments/attention/flydsl/qsa/_paged_union.py)。上方历史章节指向这些旧活动文件的相对链接现已失效，应使用本归档或各场冻结source；不为兼容旧研究脚本重新引入活动5D shim。
- SGLang恢复QSA backend的3D prefill及decode scratch维度，恢复3D compact K/V读取；删除本轮新增的页表／5D prefill边界和注册5D测试。分别保留[旧边界归档](../../../../mytest/mydata/qsa_5d_revert_20260928_01/before/sglang/python/sglang/srt/layers/attention/qsa/paged.py)和[旧测试归档](../../../../mytest/mydata/qsa_5d_revert_20260928_01/before/sglang/test/registered/kernels/test_qsa_5d.py)。**未撤销SGLang原有MHA 5D、memory pool、QSA压缩池、cache writer或环境变量。** 当前QSA不应仅打开vectorized_5d来运行prefix/decode。
- [插件](sglang/plugin.py)恢复3D五hook、三个上游ABI SHA和0.2.0／八源码打包，不再引用已删除SGLang边界。唯一保留的非快照文案修正是日志／报告准确写`dense2051;packed=pad1.7;raw=rho4`，不重新声称已过时的auto4策略。旧0.3.0／clean2冻结包只保留作证据，与回退后的backend ABI不兼容，不能继续部署。
- **3D packed direct不是5D新文件，必须保留。** 其每call pack、64MiB预算、raw fallback、dense2051、pad1.7路由、TP2/4/8 local-head回放、原`.02/.02`门槛均未撤销。恢复的原测试正文在8份capture齐全时对应48项正常case和24项perf，不把删除5D测试说成3D覆盖减少。

### 新默认环境与本次验证限制

后续Python默认使用PyHIP的.venv解释器；本次所有Python命令均通过该环境执行，PyHIP工作区的编辑器选择也已核对为该环境。对SGLang工作区的选择操作虽返回成功，复查仍报告其自身.venv，因此不声称两个编辑器选择都已持久生效；后续命令继续显式指定用户要求的PyHIP解释器。现有[环境配置](../../../../.venv/pyvenv.cfg)为`include-system-site-packages=false`，实查缺少torch、flydsl、triton、pytest、msgspec及sglang。未修改此隔离设置、未静默借系统site-packages、未切回旧系统Python，也未擅自安装另一套ROCm依赖。

已恢复文件按pre5D正文及AST核验，保留可能的EOF换行差异；插件仅有前述策略日志的显式差异。可用标准库完成语法、归档、上游ABI、惰性导入和八源码包构建检查；实际pytest尝试见[阻塞日志](../../../../mytest/mydata/qsa_5d_revert_20260928_01/pytest_attempt.log)，为`No module named pytest`。**本次未执行GPU数值／资源／性能回归，不能宣称48项重跑通过或新环境时延不变。** 后续若运行测试，需先在.venv补齐兼容ROCm依赖。

撤销前状态与27文件快照见[回退清单](../../../../mytest/mydata/qsa_5d_revert_20260928_01/before.json)。新报告、JSON与源码归档全部位于mytest/mydata；文档旧正文逐字保留，快照用非Markdown后缀，没有新增Markdown报告。原始样本、失败版本、ELF/ISA、trace及历史插件均不删除；没有stage/commit/reset、硬件写入或模型部署，原9078绘图服务保留。

最终[回退审计](../../../../mytest/mydata/qsa_5d_revert_20260928_01/result_final.json)通过：9文件恢复、6文件删除归档，SGLang两源码逐字等于pre5D，相关未暂存差异已清空，两个仓库HEAD/index及其它原有dirty文件未改；当前[八源码3D独立包manifest](../../../../mytest/mydata/qsa_5d_revert_20260928_01/plugin_3d_final/pyhip_qsa_runtime/source_manifest.json)与最终源码相同，三个ABI来源哈希匹配，禁用entry point实际执行且未导入Torch/FlyDSL/experiments。dense/union仍有历史EOF空白告警；一次清理未成功后停止，不把正文／AST一致说成所有文件字节完全一致。首个包保留为快照，以带final后缀的包对应最终字节。本次没有新增GPU性能样本。

## 2026-09-28：独立.venv全面复测3D，207项同场时延无回退，保留全链资源缺口

### 1. 结论与验收边界

针对用户“全面测试性能，确认回到最好水平”，本轮实际完成**54组QSA＋12组相关MHA，207个配对scope，59,904条普通计时raw**；另有76项正常正确性测试与6组独立full-QSA profile。不是只比源码，也不是使用撤销前的旧计时冒充新环境结果。

- **当前／冻结最佳3D同场无回退：207/207项满足预先规定的中位数比≤1.03。** 比值范围0.991776～1.011268，配对ratio中位范围0.995352～1.011268；阈值未在结果出来后放宽。这里3%是本轮预先采用的操作性门槛，不是用户另行规定的数字。
- **QSA主要绝对成绩回到历史最好水平附近。** TP0真实M12000的L3/L47 full为2.899278/2.990318ms，对应原routing最终正式版2.903916/2.993316ms；M11888为2.834735/2.874315ms，对应2.839096/2.881116ms。不是所有历史候选、单样本的绝对最低纪录。
- **不能宣称“全部场景、全部kernel验收通过”。** MHA causal8192/H12的两版共同从约1.84ms进入约2.82ms慢阶段，完整128样本中位2.817175ms，未复现旧1.838490ms的完整中位。全部样本保留，没有仅取前48个快样本。
- **attention零spill，但全链零spill不通过。** 扩展到Triton规划器后发现冻结最佳版也存在的compact寄存器spill；超64MiB边界甚至有private scratch。原严格矩阵在首个low/TP2、0raw处停止；剩余48组按明确独立的“legacy diagnostic”协议完成测量，不能拿诊断完成覆盖严格资源失败。

总入口：[原始协议](../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/protocol.json)、[资源失败后的诊断协议](../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/diagnostic_protocol.json)、[完整raw/ELF/门禁/路由审计](../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/analysis.json)、[最终环境/资源/历史/profile证据](../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/final_evidence.json)、[机器生成统一表格](../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/tables.txt)。本节之后“当前结果”指这些新收据；上一节缺依赖、未跑GPU是回退当时的真实状态，历史正文未改。

### 2. 独立.venv环境、冻结版本及正确性

全部新Python工作继续使用PyHIP的.venv，Python3.10.12；Torch2.12.0+rocm7.14.0.git6bbd260、Triton3.8.0+rocm7.14.0.git4cff872、FlyDSL0.3.2、pytest9.0.3、msgspec0.21.1、NumPy2.2.6、amdsmi26.5.0+2b22ab0195。实际Torch/Triton/FlyDSL导入均来自.venv，`include-system-site-packages=false`，系统dist-packages不在`sys.path`；SGLang/AITER保留显式安装的源码editable映射。`pip check`无损坏依赖。

镜像记录中的原Torch/Triton wheel档案已不存在，故从镜像已安装distribution的RECORD及载荷构造本地wheel，再安装到.venv；**这些是重建wheel，不是找回的原始wheel文件**。19个core＋41个adapter＋85个runtime wheel的39,890个安装载荷文件在最终复核中逐个哈希一致，版本固定；另固定six1.16.0、distro1.7.0。原provenance、RECORD差异及原wheel哈希都保留在[core清单](../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/env_core/manifest.json)、[adapter清单](../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/env_adapter2/manifest.json)、[runtime清单](../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/env_runtime2/manifest.json)与[最终安装验证](../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/environment_verified.json)。未静默启用系统site-packages或切解释器。

环境准备失败也保留：最初误选py2 wheel tag；压缩manylinux tag需按`parse_tag`展开；已满足base但带extras仍须遍历依赖；只用find-links让pip选入20个较新间接依赖，文件审计发现后用明确版本及no-index纠正。[初始冻结收据](../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/identity.json)中的初始环境不匹配没有擦除，以后生成的environment_verified及final_evidence为最终安装结论。

硬件仍是物理GPU2/PCI0000:a4:00.0、MI308X/gfx942、80CU、HIP7.14.60850，PTL Enabled/VECTOR,F8、650W；没有修改功率、时钟、PTL、NUMA或CU mask。所有QSA/MHA运行时Python源码、原timer、SGLang源码、两个仓库HEAD/index及原有dirty文件在追加本节前逐一核对未改。本轮没有部署模型；原9078绘图服务保留。

冻结比较对象是T01之前的3D正文，复制成独立模块namespace，当前与冻结各自拥有工作区/编译缓存；不是把同一个Python对象换标签。没有恢复任何QSA5D shim。正常测试结果：

| suite | passed | deselected | skipped / failures / errors | 用时 |
|---|---:|---:|---:|---:|
| QSA原正常回归，含8份capture×TP2/4/8及3D SGLang backend | 48 | 24 perf | 0 / 0 / 0 | 825.109s |
| 相关MHA `bf16_linear_d256` | 28 | 52其它 | 0 / 0 / 0 | 478.004s |

[QSA JUnit](../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/validation_qsa/tests.xml)、[MHA JUnit](../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/validation_mha/tests.xml)。QSA有14条deprecation警告；MHA一条测试内empty CUDA Graph警告，均未藏成skip。QSA测试一度停留在41个进度点，实际CPU栈在FlyDSL/LLVM union编译，不是GPU死锁；[当时栈](../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/validation_qsa_stack.txt)保留。QSA48项导出的131个编译记录与历史3D同family/signature的**整ELF及资源131/131一致**，见[校正family映射后的历史核验](../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/validation_qsa/historical_comparison_v2.json)；首份仅匹配104项的记录也保留。

### 3. 普通计时方法及覆盖

采用未修改的`cudaPerf`事件计时，原device-side spin在start event之前；未加host sleep。每scope每实现128个正式样本、10个独立Q/K/V/indices和输出buffer、每buffer2次warmup；所有实现共享同一组输入，各有独立输出和scratch，不宣称同输出/scratch地址。prepared计划按实现复用，不是10套plan/PK/PV，也不强制flush cache。current/frozen为AB/BA，全调用三label采用起点轮换＋反转，包含原SGLang sparse baseline。

- dense只计真实eligible前缀；forced union/direct在**完全相同的剩余稀疏query行**上各自计算，不能改成只测auto分给该分支的行。dense-only额外强制两稀疏分支计算全行。
- prepared direct包含每次pack；union不含构表；prepared调用中包含`bind()`。full是真正公共`qsa()`，含恢复、校验、构表、gate、pack与attention；不含首次JIT、引用计算、indexer或外部KV gather。不能将三个独立分支的中位相加冒充full。
- 原`.02/.02`不变；capture/原baseline全输出对比和抽样FP32 oracle，MHA为全部query的分块FP32 oracle。独立CPU审计selected-set、membership、common-prefix、score-mask、active gate、任务排列。guard测试使用独立storage；自然输出在正式采样前poison，**结束时先核对实际留下的10个输出再做其它launch**。current/frozen逐位相同，输入、计划、源码哈希不变。
- 66场普通测量共198次entry/pre-sample/post门禁均通过：use≤5%、VRAM≤20%、PTL固定。没因长尾删样本或重试取快；6场profile另有18次门禁，全部通过。
- 8份实际capture覆盖rank0/rank1、L3/L47、M12000/M11888。TP4/8只从TP2 Q及输出裁成H6/H3，HK1，**不是实际分布式TP4/8采集或服务**。M11888是另一份原始capture，不是M12000截断。

研究入口为[bench](../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/bench.py)、[固定顺序矩阵](../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/matrix.py)、[MHA入口](../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/mha_bench.py)、[独立profile入口](../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/trace_driver.py)。目录排他创建；首个严格low资源失败、一次启动器缩进解析失败、一次最终审计历史ELF子目录查找失败都保留，后两者没有GPU计时。另保留untimed32warp资源探测的2048线程超过1024上限错误；不是候选性能结果。

### 4. 全部QSA普通中位时延

单位**µs**，每格依次为**dense / forced union / forced direct / full qsa**；“—”表示没有dense行。54场共195scope、56,832raw（含每场full原SGLanglabel）。除六个dense-only场外均为legacy diagnostic，资源缺口见第7节；这里给的是实际时延，不代表全链资源通过。

合成输入：dense2048/2051为P0；low/high为M2048/P30000随机独立／每32行共享优先选块；short为M64/P30000；share50/share75固定共同选块比例、seed117；raw为M33/P30000、NK30033非4倍数；ragged_hk2为Q(9,0,33)/P(30000,17,5)、HK2、H为常规H的两倍；overbudget为M68/NK65540，PK+PV略超64MiB，走raw而不pack。一般合成seed17。

| 输入 | TP2 | TP4 | TP8 |
|---|---:|---:|---:|
| dense2048 | 161.100 / 171.281 / 298.842 / 211.801 | 102.640 / 105.761 / 294.161 / 153.581 | 100.601 / 103.441 / 291.522 / 151.640 |
| dense2051 | 179.341 / 174.801 / 386.682 / 230.441 | 108.001 / 110.361 / 383.263 / 159.161 | 106.641 / 107.000 / 381.842 / 157.920 |
| low | — / 2127.573 / 588.124 / 688.464 | — / 1901.612 / 582.504 / 677.704 | — / 1291.968 / 581.843 / 672.504 |
| high | — / 406.503 / 573.283 / 527.943 | — / 214.761 / 569.164 / 319.042 | — / 110.241 / 567.423 / 210.441 |
| short | — / 696.244 / 111.920 / 145.921 | — / 919.805 / 110.840 / 144.761 | — / 1199.207 / 109.641 / 142.321 |
| share50 | — / 1409.508 / 582.143 / 680.764 | — / 1280.888 / 579.064 / 669.904 | — / 993.345 / 575.523 / 665.404 |
| share75 | — / 951.566 / 577.623 / 678.724 | — / 822.445 / 571.564 / 666.425 | — / 669.284 / 571.723 / 1312.988 |
| raw | — / 721.224 / 133.721 / 471.042 | — / 950.046 / 133.301 / 285.222 | — / 1252.608 / 132.601 / 285.841 |
| ragged_hk2 | 14.181 / 663.044 / 133.361 / 179.342 | 14.240 / 658.744 / 133.241 / 179.701 | 14.360 / 658.164 / 133.001 / 177.341 |
| overbudget | — / 805.185 / 135.641 / 171.761 | — / 1181.167 / 135.561 / 697.105 | — / 1879.412 / 135.501 / 725.005 |
| tp0 L3 M11888 | 180.541 / 2349.993 / 2453.713 / 2834.735 | 108.641 / 1647.249 / 2433.053 / 2095.551 | 107.121 / 1043.486 / 2413.853 / 1501.388 |
| tp0 L3 M12000 | 180.621 / 2783.617 / 2503.996 / 2899.278 | 108.941 / 1958.672 / 2487.255 / 2460.735 | 107.041 / 1150.187 / 2467.655 / 1628.830 |
| tp0 L47 M11888 | 181.080 / 2588.413 / 2454.973 / 2874.315 | 108.960 / 1828.950 / 2435.133 / 2372.512 | 107.201 / 1137.706 / 2421.133 / 1609.289 |
| tp0 L47 M12000 | 180.861 / 2694.137 / 2501.456 / 2990.318 | 108.921 / 1886.891 / 2481.315 / 2440.255 | 107.160 / 1125.926 / 2463.394 / 1596.329 |
| tp1 L3 M11888 | 180.262 / 2346.103 / 2451.024 / 2827.068 | 108.561 / 1642.576 / 2431.364 / 2091.060 | 107.201 / 1043.690 / 2413.484 / 1500.475 |
| tp1 L3 M12000 | 180.481 / 2778.095 / 2501.814 / 2895.795 | 108.841 / 1954.310 / 2480.513 / 2457.093 | 107.021 / 1148.646 / 2463.993 / 1625.809 |
| tp1 L47 M11888 | 180.442 / 2583.601 / 2452.860 / 2868.423 | 108.760 / 1823.230 / 2434.973 / 2363.713 | 107.041 / 1133.966 / 2418.913 / 1605.169 |
| tp1 L47 M12000 | 180.301 / 2685.356 / 2497.755 / 2983.358 | 108.641 / 1882.138 / 2478.004 / 2434.844 | 107.041 / 1124.031 / 2463.044 / 1594.015 |

这也保留了现有路由的局限：share75/TP8 full约1.313ms，不如强制direct约0.572ms；raw、超预算小M的rho4尾组也可让full明显高于direct。当前与冻结的路由、输出和时延均相当，所以不是本次撤销引入的回退，也未改阈值隐藏这些场景。短M强制union≈0.696/0.920/1.199ms，而full≈0.146/0.145/0.142ms；不得混淆二者。

真实TP0 M12000 full相对同场原SGLang sparse：L3 TP2/4/8为**3.377×/3.853×/5.684×**，L47为**3.272×/3.878×/5.786×**。原baseline时延分别9.791/9.480/9.258ms、9.784/9.463/9.236ms；这是同一局部attention调用范围，不是模型TTFT或服务吞吐。

以下是真实M12000的**有效T / 填充T**；D256的有效F为$4\cdot256\cdot H\sum_i n_i$，各scope严格使用自己实际计算的query行，union填充按M128/N64，direct按每query的M16/N32。full填充按实际分流相加，非attention准备时间只进入时延，不伪造F。

| 输入 | TP | dense | union | direct | full |
|---|---:|---:|---:|---:|---:|
| L3 | 2 | 143.161 / 169.982 | 90.012 / 213.692 | 100.063 / 134.882 | 95.340 / 142.144 |
| L3 | 4 | 118.679 / 140.913 | 63.961 / 215.716 | 50.368 / 135.790 | 56.165 / 171.354 |
| L3 | 8 | 60.393 / 71.707 | 54.460 / 212.890 | 25.384 / 136.869 | 42.426 / 155.043 |
| L47 | 2 | 142.971 / 169.756 | 93.001 / 210.467 | 100.165 / 135.019 | 92.437 / 140.056 |
| L47 | 4 | 118.701 / 140.938 | 66.394 / 214.089 | 50.489 / 136.115 | 56.637 / 171.669 |
| L47 | 8 | 60.326 / 71.627 | 55.634 / 210.213 | 25.428 / 137.105 | 43.289 / 153.076 |

本轮TP2 prepared direct的100有效T重现，但裕量很窄；不将此推广为所有场次稳定≥100T。最新前轮direct2.474/2.472ms比本轮2.504/2.501ms略快；输入和ELF相同，但旧harness预绑定输入，本轮`bind()`在计时内，且跨场/交错集合不同，不能声称逐微秒完全复现。union本轮2.784/2.694ms已回到约2.79/2.70ms快区，而不是上一轮4.19/4.03ms共同慢区；160填充T direct仍未实现，union100有效T仍未实现。

### 5. 相关MHA：同场无回退，不隐藏8192共同慢阶段

本表是native linear `.run()`热调用，不是公共API校验全开销。Q/KV均BF16、HK1、D256、B1、persistent、noLSE；causal项Q=KV，full项Q10240/K2583、noncausal。每行当前／冻结共256raw，12行共3072raw。

| 输入 | TP | 当前µs | 冻结µs | 当前/冻结 |
|---|---:|---:|---:|---:|
| causal2048 | 2 | 161.081 | 161.361 | 0.998265 |
| causal2048 | 4 | 102.060 | 102.040 | 1.000191 |
| causal2048 | 8 | 100.121 | 100.120 | 1.000005 |
| causal2051 | 2 | 172.041 | 171.841 | 1.001164 |
| causal2051 | 4 | 102.760 | 102.800 | 0.999616 |
| causal2051 | 8 | 101.161 | 101.320 | 0.998431 |
| causal8192 | 2 | 2817.175 | 2817.215 | 0.999986 |
| causal8192 | 4 | 952.285 | 952.166 | 1.000125 |
| causal8192 | 8 | 511.023 | 511.303 | 0.999452 |
| full | 2 | 2214.672 | 2223.711 | 0.995935 |
| full | 4 | 745.044 | 740.584 | 1.006022 |
| full | 8 | 373.262 | 373.342 | 0.999786 |

8192/TP2与[旧formal_linear](../../../../mytest/mydata/qsa_linear_direct_20260926_01/formal_linear/result.json)的causal8192 current实际ELF逐字相同（SHA以eb708b3d开头），同shape/flags/timer，但旧10buffer/50sample中位1838.490µs，本轮10/128中位2817.175µs，高53.233%。本轮16样本时间块的current/frozen中位依次为：1840.030/1839.490、1838.230/1838.410、1837.030/1837.430、2823.496/2819.755、2824.675/2820.755、2821.615/2824.515、2823.055/2819.835、2822.175/2822.775µs。**不是current独有变慢**，也不能因前48个快样本存在就把128样本整体写成1.84ms。原因未定位，不能无证据归因clock、温度或功率；新旧random seed及地址也不同。

full历史同scope H12/H6/H3为2193.732/744.204/374.362µs，本轮2214.672/745.044/373.262µs，约+0.95%/+0.11%/−0.29%；新旧均128样本、10buffers，ELF相同，但seed及交错候选集合不同。H24/HK2的旧220T场景不是本表，不宣称所有MHA形状达到220T。

### 6. 完整QSA query比例与每kernel时间占比

六份[独立profile结果](../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/final_evidence.json)各10次热full调用；host annotation经runtime correlation关联到全部780个kernel，未丢失调用。每call恰13个kernel，attention及planner的实际ELF与该shape普通场次一致，输出/输入/源码重新核验。没有ATT/PMC，也没有用profile时间代替上表cudaPerf中位。

| 输入 | TP | dense行 | union行 | direct行 | profile每call GPU kernel时长和µs |
|---|---:|---:|---:|---:|---:|
| L3 M12000 | 2 | 2051 (17.092%) | 3929 (32.742%) | 6020 (50.167%) | 2954.691 |
| L3 M12000 | 4 | 2051 (17.092%) | 9341 (77.842%) | 608 (5.067%) | 2502.980 |
| L3 M12000 | 8 | 2051 (17.092%) | 9949 (82.908%) | 0 | 1680.908 |
| L47 M12000 | 2 | 2051 (17.092%) | 4269 (35.575%) | 5680 (47.333%) | 3056.431 |
| L47 M12000 | 4 | 2051 (17.092%) | 9933 (82.775%) | 16 (0.133%) | 2485.464 |
| L47 M12000 | 8 | 2051 (17.092%) | 9949 (82.908%) | 0 | 1644.968 |

下表分母只取相应profile内10call全部GPU kernel的duration之和，单位**%**，非端到端墙钟占比：

| kernel | L3 TP2 | L3 TP4 | L3 TP8 | L47 TP2 | L47 TP4 | L47 TP8 |
|---|---:|---:|---:|---:|---:|---:|
| dense attention | 6.089 | 4.293 | 6.496 | 6.326 | 4.331 | 6.693 |
| union attention | 29.465 | 74.175 | 69.954 | 32.339 | 76.856 | 69.059 |
| direct attention | 52.921 | 7.090 | 0.704 | 50.074 | 4.184 | 0.723 |
| recover blocks | 6.968 | 8.189 | 12.246 | 6.711 | 8.269 | 12.489 |
| scatter membership | 1.423 | 1.595 | 2.779 | 1.365 | 1.609 | 2.829 |
| compact membership | 0.577 | 0.758 | 0.676 | 0.588 | 0.764 | 0.687 |
| score masks | 0.666 | 2.271 | 5.237 | 0.761 | 2.355 | 5.510 |
| order tasks | 0.687 | 0.332 | 0.310 | 0.667 | 0.336 | 0.314 |
| pack KV | 0.353 | 0.339 | 0.256 | 0.335 | 0.347 | 0.282 |
| Torch bool reduce | 0.272 | 0.306 | 0.443 | 0.266 | 0.307 | 0.465 |
| Torch fill | 0.237 | 0.260 | 0.326 | 0.228 | 0.258 | 0.342 |
| Torch compare | 0.194 | 0.213 | 0.314 | 0.189 | 0.212 | 0.338 |
| Torch async assert | 0.150 | 0.179 | 0.258 | 0.151 | 0.172 | 0.270 |

TP8 direct行数虽为0，gated direct和pack的空工作launch仍存在，约0.7%/0.3%是固定launch/门控成本，不能报成没有kernel；TP4 L47只16个direct行却约4.18%，同样暴露固定成本。没有修改这一分流/launch策略来美化复测。

### 7. 资源、长尾和未解决限制

普通场次201个current/frozen编译记录对，**整ELF、`.text`、函数bytes及resources全部一致**；318对Triton planner特化也整ELF/资源一致。加上正常测试及profile，595个attention/pack编译对象记录、800个kernel metadata实例、233个去重ELF的private/VGPRspill/SGPRspill三字段全部为0。数量包含重复shape/场次记录，不能称595个唯一kernel。

但是`union_qsa_compact_membership`未满足全链零spill。以下为实际readelf字段，当前/冻结完全相同；每场通常有auto与forced两个特化，不把它们误计为不同case：

| 场景 | 组数 | private bytes | VGPR spill count | SGPR spill count |
|---|---:|---:|---:|---:|
| 8份真实capture×TP2/4/8 | 24 | 0 | 0 | 8 |
| low/high/short/share50/share75×TP2/4/8 | 15 | 0 | 0 | 80 |
| raw/ragged_hk2×TP2/4/8 | 6 | 0 | 0 | 120 |
| overbudget×TP2/4/8 | 3 | 2064 | 515 | 194 |

因此不是只有“SGPR spill但无scratch”一种情形，**overbudget有真实private scratch**。六个dense-only场的当前FlyDSL/Triton导出无这些非零字段；本轮没有审计所有Torch原生及原SGLang baseline内核来建立额外“全库零scratch”声明。

原[low/TP2严格失败收据](../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/low_tp2/result.json)在输出正确之后、计时之前因SGPRspill80停止，raw为空；原协议没有被改写。后续diagnostic目录只是为了完成性能问题的诊断，全部非零资源显式记录`resource_acceptance=false`，不通过严格验收。untimed探测确认同一个compact函数在真实C4096由4warp改8warp可零spill，C8192改16warp可零spill，精确计划不变；[真实探测](../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/planner_probe_tp0_layer3_m12000_tp2/result.json)、[低重合探测](../../../../mytest/mydata/qsa_3d_revalidate_20260928_01/planner_probe_low_tp2/result.json)。**这些特化未计时、未覆盖更大C、未采用到生产或冻结基线**；不能据此宣布已修复。若下一轮要求全链零spill，应独立处理compact扫描资源规模并保留当前性能对照。

长尾全部保留，门禁通过不等于每sample都无抖动。例如TP0 L47 M12000/TP8 full中位1.596ms、最大73.154ms；TP1 L47 M11888/TP4 full中位2.364ms、最大73.696ms；还有TP0 L3 M11888/TP4 dense最大1.476ms（中位108.641µs）。本轮按预定中位门槛通过，**没有证明p99/最大值SLA已达标**，也未将这些长尾归因于硬件或Python原因。16样本段及全部raw可从analysis追溯，未重跑筛快。

最终限制仍包括：只验证本地GPU2；TP4/8为派生local-head而非多卡服务；没有模型部署/TTFT/吞吐/长上下文指标；没有重新宣称所有旧5D结果适用于3D；QSA公共API仍仅3D，64MiB pack预算、raw/ragged rho4及pad1.7均不变。原9078服务、旧包/失败/trace/源快照保留，无stage/commit/reset或硬件写入。严格结论是**QSA主要性能恢复、当前相对冻结版无回退；全链资源和MHA8192全程绝对最好水平尚未达标**，而不是“全面无条件通过”。

## 2026-09-28：修复全链零spill，包含Torch验证kernel；53组有效性能无回退

本节响应“解决：全链零spill未通过”及后续“继续”。**前节的资源缺口已在当前3D代码修复**：54组完整QSA实际dispatch逐一匹配其ELF，570次调用的private/VGPRspill/SGPRspill三字段全部为0；新独立包的6例、72次dispatch也全部为0。不是只检查attention或把旧非零字段忽略。正常回归最终63通过，原精度和路由不变。性能完成53/54组，唯一失败场次因门禁而无计时，不声称54组全部性能验收完成。

权威收据：[最终独立审计](../../../../mytest/mydata/qsa_zero_spill_20260928_01/final_analysis.json)、[最终全部数值表](../../../../mytest/mydata/qsa_zero_spill_20260928_01/final_tables.txt)、[预先固定协议](../../../../mytest/mydata/qsa_zero_spill_20260928_01/protocol.json)、[最终源码身份](../../../../mytest/mydata/qsa_zero_spill_20260928_01/final_identity.json)。旧36场阶段审计、旧资源失败及前节所有历史数据保留，不改写成这轮的成功结果。

### 1. 两个根因与最小运行时改动

**根因A：compact的向量/扫描宽度随最大上下文增长。** 原实现取`C=next_power_of_2(max_blocks)`，两组cumsum及相关掩码同时存活，30k到超预算域导致SGPR spill甚至private scratch；仅升warp不能覆盖任意长度，也受1024线程上限约束。

[union_qsa_compact_membership](union.py)改为固定`C≤1024`、4warp、两遍分块扫描：第一遍累计total/common数，再使用原整数gate；第二遍分别带`common_base`和`other_base`前缀压缩。块间按升序前进，块内仍cumsum，故得到与原版逐元素一致的“完整common块升序＋其它块升序”，不是只保持集合相同。`common_count//16`、尾块排除、partial query group及packed pad1.7/raw rho4均不变；禁用组仍不写无效输出区域。无新增planner scratch、无CPU回读，不把全局临时张量当作spill规避手段。

**根因B：Torch原生布尔归约也有private scratch。** 本轮新增SDK的code-object注册与kernel enqueue回调，在真实full `qsa()`范围记录每个dispatch、符号、runtime private和ELF。冻结版13个kernel中，`reduce_kernel<...,ReduceOp<bool,...>>`实际private为**12B**，虽然其VGPR/SGPR spill count为0；这同样违反三字段全零，不能用“不是寄存器spill”排除。原[完整链路失败证据](../../../../mytest/mydata/qsa_zero_spill_20260928_01/chain_before_real/resource_audit.json)同时记录compact的SGPRspill8。

[qsa.py](qsa.py)新增`_qsa_check_errors`：固定1024元素一段，逐段OR错误标志，写入workspace一次分配的单个bool `valid`。原`_qsa_recover_blocks`仍每行写完整errors，原`torch._assert_async`和错误语义保留；移除`(errors == 0).all()`两个Torch launch及热调用临时bool张量。`valid`每次调用必写、按stream工作区复用并随graph固定生命周期，不在host读取。完整真实调用由13个kernel变为**12个**；dense-only由5个变为4个。这是全链资源修复，不是关掉校验或把非法输入放过。

运行时仅修改上述两个文件；dense、direct、packed direct及MHA attention计算源码不动，SGLang源码不动、QSA5D不恢复。原API、`.02/.02`精度、64MiB pack预算、分流阈值和query选集保持不变。

### 2. 正确性、graph与持久回归门禁

- 最终[正常测试JUnit](../../../../mytest/mydata/qsa_zero_spill_20260928_01/validation_final/tests.xml)：**63 passed、24 perf deselected、0 skipped/failures/errors**，110.115s；首次完整轮63项780.05s、定向15项8.67s收据也保留，时间差含已编译缓存，不能作为性能加速。
- 原48项保留，含8份真实capture×TP2/4/8、SGLang 3D adapter、raw/HK2/ragged、预算边界、output guard、选集变化与graph。新增6项errors归约边界/graph测试，长度1、1023、1024、1025、12000、261632；逐个置错/恢复首行、段边界和末行，验证结果不会沿用上一轮标志。
- 新增9项compact域测试：511、1024、1025、3000、4097、8012、16385、65536、1048575 blocks，每项含rows1/3/10/16/32、空/高重合/低重合位集、packed/raw/forced三种模式；CPU验证gate、count、common段、输出顺序及未写区域guard。另有[36个生产kernel域边界特化](../../../../mytest/mydata/qsa_zero_spill_20260928_01/production_domain_boundaries/result.json)，覆盖1023、2048/2049、4096、8192/8193、16384、32768/32769、65535/65537、1048576 blocks。大域是独立planner资源/数值验证，不冒称分配了对应完整4M-token KV并测完整attention。
- [非法公共输入隔离测试](../../../../mytest/mydata/qsa_zero_spill_20260928_01/invalid_public_verified.json)：1025行先正常执行，再破坏第1024号行（跨1024段边界），实际由`at::native::_assert_async_cuda_kernel<bool>`报HSA异常并退出，拒绝语义保留。初始日志分类器只认CUDA报错短语而误判的收据保留，后续仅CPU按明确kernel名核实，未重跑GPU到成功。
- [test_qsa.py](test_qsa.py)的常规资源fixture现在检查全部已编译Triton恢复/验证/规划器实际ELF三字段，而非仅FlyDSL IR。热路径/graph回归还禁止Tensor equality/all、host读取及`empty`/`empty_like`，并检查新`valid`与PK/PV地址稳定，防止旧12B归约被重新加回。
- 所有timed输出在任何替代launch前逐位回验；current/before输出、精确compact顺序、有效membership/score mask与route均相同。输入、已初始化plan范围及scratch哈希不变。最终增强测试在计时之后完成，只有测试文件变化，运行时字节与所有测量一致。

### 3. 全链实际资源验收，不遗漏Torch kernel

[采集器](../../../../mytest/mydata/qsa_zero_spill_20260928_01/chain_capture.cpp)仅记录code-object load、符号注册和指定完整QSA范围内的enqueue；无PMC/ATT、无PTL/clock写入。每个dispatch按kernel_id找到code_object_id，读取实际ELF `.name` 对应metadata，核对runtime/symbol private与ELF private一致，并要求三项字段存在且为0；不会用另一函数的最后一条metadata代替本kernel。FlyDSL及Triton产物与普通测量逐字匹配；Torch assert/fill也在同一实际dispatch审计中。

| 原失败来源 | 修复前三字段 private / VG spill / SG spill | 当前 |
|---|---:|---:|
| compact，24组真实输入 | 0 / 0 / 8 | **0 / 0 / 0** |
| compact，15组low/high/short/share | 0 / 0 / 80 | **0 / 0 / 0** |
| compact，6组raw/ragged | 0 / 0 / 120 | **0 / 0 / 0** |
| compact，3组overbudget | 2064 / 515 / 194 | **0 / 0 / 0** |
| 原Torch bool reduce，真实12k | 12 / 0 / 0 | 被固定宽度检查替代，**0 / 0 / 0** |

54组full范围共**570个实际dispatch**：recover54、check_errors54、Torch async assert54、dense33、Torch fill48、scatter48、compact48、score_masks48、order_tasks48、union48、pack39、direct48，全部三零。全链范围是QSA本身，不包含外部indexer/KV写入/gather、reference或旧SGLang对照的实现。资源捕获只做数值/ELF验证，不要求空闲GPU，也不提供可比计时；每场仍保存设备快照。

最终正常套件导出131个attention编译对象/168个kernel metadata实例，以及164个planner/验证对象；三个字段全0。131个attention对象与上一轮相同shape/family的整ELF/资源全部一致；53场普通对照另有185对attention整ELF/`.text`/resources相同。未改变的其它Triton planner `.text`及资源相同，整ELF可能因源码行号/debug变化而不同，二者分开核验。正常编译矩阵中`_qsa_check_errors`VGPR5～16/SGPR16、compact VGPR23～80/SGPR34～98，均LDS0和三零；最大VGPR项包含测试直接调用C4096的合法资源特化，生产launch上限仍1024。

### 4. 普通性能：53组有效，60,928条raw；原失败不重试

物理GPU2/0000:a4:00.0、MI308X/gfx942/80CU，PyHIP .venv；Torch2.12.0+rocm7.14.0.git6bbd260、Triton3.8.0+rocm7.14.0.git4cff872、FlyDSL0.3.2。仍隔离system-site-packages，`pip check`通过。before是这轮改动前的完整3D快照，非零资源仅作为明确对照，不作为接受版本。

原`cudaPerf`、10个独立输入/输出buffer、每buffer2warmup、每实现128samples、AB/BA交错，全部raw/长尾保留。准备好的dense/union/direct和full都保持相同边界，direct含每次pack；新增isolated plan scope测rebuild，不含attention。入口/pre-sample/post门禁use≤5%、VRAM≤20%、PTL Enabled/VECTOR,F8，159个有效普通门禁通过。没有改timer、过滤时段或仅取最快样本。

首次矩阵前36组通过，第37组**TP0 L47 M11888/TP2**的pre-sample GPU use=100%、VRAM4%，停止且0raw，见[停止收据](../../../../mytest/mydata/qsa_zero_spill_20260928_01/performance_stop.json)。收到用户“继续”并一次确认GPU空闲后，仅执行[此前从未启动的17组](../../../../mytest/mydata/qsa_zero_spill_20260928_01/resume_unstarted_plan.json)，全部通过；失败场本身未重试，原停止收据仍保持其当时36/54状态。

最终**53组、238个scope、60,928raw**。其中53 full全部≤1.03，ratio范围**0.841917～1.015460**，配对ratio中位0.841832～1.015605；138个prepared attention scope也全部通过。47个isolated plan范围0.441123～1.501129，只有25个≤1.03：**planner不是每个形状都更快**，两遍扫描可增加小M建表成本，而移除Torch临时验证抵消部分完整开销。原验收是full≤1.03，不拿它掩饰plan的独立退化。

完整53组的所有分支、full/plan、有效/填充T与route见final_tables/final_analysis。主要full中位数如下，每格为**修复前→当前µs**：

| 输入 | TP2 | TP4 | TP8 |
|---|---:|---:|---:|
| dense2048 | 211.801→205.961 | 153.801→147.781 | 152.001→146.081 |
| dense2051 | 230.521→224.521 | 159.161→153.721 | 158.400→152.161 |
| low M2048/P30000 | 688.783→685.244 | 677.664→673.504 | 673.144→665.483 |
| high M2048/P30000 | 528.402→528.003 | 318.922→320.682 | 211.161→214.281 |
| short M64/P30000 | 146.320→144.320 | 144.240→142.040 | 142.640→140.541 |
| raw M33/P30000 | 471.902→468.262 | 286.662→283.222 | 285.882→283.021 |
| ragged HK2 | 179.641→178.581 | 178.181→177.641 | 177.221→176.161 |
| overbudget M68/NK65540 | 170.761→173.401 | 696.224→586.163 | 712.424→605.583 |
| TP0 L3 M12000 | 2927.335→2916.695 | 2465.713→2462.773 | 1629.448→1624.269 |
| TP0 L47 M12000 | 3054.097→3011.276 | 2444.833→2442.053 | 1599.249→1592.629 |
| TP0 L47 M11888 | **门禁失败，无有效计时** | 2383.313→2377.813 | 1607.648→1606.449 |

overbudget TP4/8 full改善约15.81%/15.00%，对应plan184.801→81.520µs、204.081→102.600µs；TP2 plan17.720→26.600µs，full反而+1.55%，如实保留。high TP8 plan49.160→58.201µs、full+1.48%，不能说全部配置更快。真实TP2 full L3中位之比0.996365，配对中位0.997938；L47为0.985979、0.997609，受不同时间分布影响，不能把前者全部当作kernel因果收益。

当前真实TP0 M12000 prepared分支（µs）：L3 TP2/4/8 union2789.975/1963.130/1150.226、direct2532.754/2476.633/2443.833；L47 union2694.314/1931.890/1126.766、direct2504.373/2474.533/2441.413。TP2 direct有效T为L3 **98.927**、L47 **100.048**，填充T133.351/134.862；不能沿用前轮两层均过100有效T。当前full有效T L3为94.770/56.119/42.545，L47为91.794/56.595/43.390；数学F、填充模型和选集保持原口径。

### 5. 分流、kernel占比与独立包

QSA选集与route未改：真实M12000每层dense2051行（17.092%）。L3 TP2 union3929/direct6020，TP4 union9341/direct608，TP8 union9949/direct0；L47对应4269/5680、9933/16、9949/0。TP8的gated direct/pack空工作launch仍存在，零行不等于没有kernel。合成share75/TP8仍可能full约1.315ms而forced direct约0.574ms，raw/ragged rho4尾组固定成本仍在；没有为通过资源验收调路由。

按原要求尝试独立的修复后逐kernel时间占比，但第一场入口use=34%、VRAM0，门禁失败，**未生成任何可用trace，未重试**；见[profile失败收据](../../../../mytest/mydata/qsa_zero_spill_20260928_01/profile_tp0_layer3_m12000_tp2/result.json)。所以本节能报告实际kernel数量与资源，**没有新的kernel时长百分比**，前节13-kernel占比保持历史身份，不能套给当前12-kernel链。普通53场的门禁及raw独立有效，不用失败profile否定或代替它们。

已按当前源码重建[八源码3D独立包manifest](../../../../mytest/mydata/qsa_zero_spill_20260928_01/plugin_zero_spill/pyhip_qsa_runtime/source_manifest.json)。[禁用入口](../../../../mytest/mydata/qsa_zero_spill_20260928_01/plugin_disabled.json)实际调用且不导入Torch/FlyDSL/Triton/SGLang/experiments；[启用验证](../../../../mytest/mydata/qsa_zero_spill_20260928_01/plugin_execution/result.json)通过原三个ABI哈希及五hook注册，并直接运行包内QSA处理L3/L47×TP2/4/8六份真实输入，原容差、重复bitexact、无experiments导入。包内[72个实际dispatch](../../../../mytest/mydata/qsa_zero_spill_20260928_01/plugin_execution/resource_audit.json)亦全部三零。只构建/注册/调用独立包，不启动整模型或宣称完整服务已部署；旧包需显式新建target替换，不能把冻结旧包重标为修复版。

### 6. 最终限制与完整性

本轮“全链零spill已解决”限定于上述当前ROCm/FlyDSL/Triton版本、gfx942、已验证QSA API和测试矩阵。53项完整时延≤3%与54项资源通过分开报告；唯一普通门禁失败与一次profile入口失败均保留。最大本轮current普通样本为overbudget/TP8 full5.387ms（中位0.606ms），其它多层TP2也有约3.9ms长尾，不因此宣布p99 SLA或所有历史绝对最快稳定重现。前轮MHA8192共同慢阶段不是本次任务，MHA源码未改，也未声称其已恢复。

所有新脚本、包、ELF、日志、JSON仅在本研究目录；本节追加前的opt284234字节旧正文及其SHA保留，没有新增Markdown报告。修改范围为两处运行时、原测试入口与既有说明文档，不动SGLang、原暂存项/HEAD/index、旧5D归档或原9078服务；无硬件写入、无自动切换Python、无模型部署。正常测试的Triton ELF门禁和禁止旧Torch归约的热路径/graph检查将持续防止该缺口回流。
