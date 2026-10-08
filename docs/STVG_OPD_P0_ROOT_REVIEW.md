# STVG-OPD 固定参数 P0：根复核与正式评测决定

本轮真实完成两个目标数据集各128个父来源、每父一个预锁query、原双序的P0。Full、Frozen-rollout和Shuffled-feedback共1536次适应，Frozen共512次匹配逻辑读数。两方向、全部适应臂封存后才读取GT。128父来源与各32个本轮调参来源互斥，但都有项目历史曝光；这是固定名单确认，不是fresh/unseen测试。没有重选参数。

| Target | Frozen m_vIoU % | Full m_vIoU % | Full−Frozen pp [95% parent CI] | Current pp [95% CI] | Inherited pp | >5 / >20 pp负尾父来源 |
|---|---:|---:|---|---|---:|---:|
| VidSTG test | 14.466 | 18.337 | +3.871 [2.430, 5.290] | +2.037 [0.753, 3.317] | +1.834 | 6 / 1 |
| HC2 validation | 22.156 | 22.469 | +0.313 [-1.236, 1.644] | −1.381 [-2.854, −0.133] | +1.694 | 16 / 3 |

vIoU>.3和>.5的HC正确→错误分别6/256和4/256到达，错误→正确分别8/256和12/256；Vid分别4/256和4/256，以及20/256和18/256。保留所有严重负尾，不能用小幅正均值掩盖当前query损伤。各128父来源一query双序，所以本阶段query和parent宏数值一致；这不替代P1全官方query主表。

## 任务收益与机制证据分开

Full−Shuffled：Vid +8.820pp [6.861,10.979]，HC +7.734pp [5.555,9.877]。这支持反馈与动作的对应关系对该设置有作用，不能单独证明Full优于不适应。

Full−Frozen-rollout：Vid +3.860pp [2.415,5.334]，HC +0.184pp [-1.281,1.479]。on-policy刷新优势目前只在Vid确认，HC没有建立。三个适应臂的Frozen输入/WHEN/观测均相同，分别重建各自完整在线状态。

HC Full的observed dense IoU当前变化平均−1.302pp，unobserved平均−3.308pp；Vid分别+9.078pp和+4.008pp。该观测是封存后帧几何诊断，缺失GT支持不填零。

## 已完成的失败归因

全体128父来源的描述性分组没有删掉负例。HC专家GT IoU<.3的24父来源，总Δv−6.574pp、当前−8.367pp；专家IoU≥.5的64父来源，总+2.934pp、当前+0.777pp。Vid对应17父来源的总−2.717pp/当前−4.156pp，以及60父来源的总+8.327pp/当前+5.206pp。质量分组是事后描述性关联，不能冒充因果gate或在线GT条件，不据此更改admission。

已实际目检三个公开统计图、四个不同的私有before/after帧对照，以及对应的sample→teacher→central output→GT标量链。

- HC严重负例匿名query1818/order1：Before框在GT人物，DINO Top1在另一位坐着的人；After明显转到DINO对应的人。四个有GT的观测位置上，central GT IoU从0.669降到0.015，central专家IoU从0.025升到0.823；当前vIoU−55.729pp，总−53.256pp。这里有实际人物混淆证据，代理反馈提高并不表示任务正确。
- Vid严重负例匿名query9622/order2：DINO局部脸框落在GT对应的同一人身上，GT标注却是整个人。学生由全身框向局部缩小。观测central GT IoU从0.761降到0.207，专家IoU从0.026升到0.140；当前−38.318pp，总−42.854pp。这个例子是部位/尺度失配，不能叫错身份。
- 正控制HC匿名query2157/order1与Vid匿名query10163/order2保留：正确专家支持可见纠正，当前分别+13.226pp和+31.271pp，观测和未观测GT空间读出都改善。

预设最大专家奖励/任务失配选择恰好与两个低质量严重负例相同，六个面板只有四个不同arrival，报告明确其重复，不称六个独立案例。所有完整行和采样诊断仍公开，不以案例代替平均效果。Native WHEN始终固定，各Frozen/Before/After tIoU逐行完全相同。

## 数学、状态、密集几何和实际成本

四真实query×三臂12fit GPU资格通过，两个数据集用独立worker，所有资格fit有实际更新。CPU对1536个正式fit复算detached奖励、Gaussian likelihood及KL梯度算术、Adam raw update、末轮选择和1792坐标各臂LN链。query和Adam每次重置，LN按目标固定alpha写回，没有跨臂状态混用。没有独立重写decoder Jacobian，不能把算术复核夸大成第二套模型反传复现。

根独立9936个统计/阈值/分解检查和10000 parent-paired bootstrap一致；额外4608个全部臂/全部序/Frozen-Before-After dense sIoU独立比较，最大绝对误差1.78e−15。独立opaque核1792文件共735400920bytes，覆盖完整，SHA/原runtime/收据先于global seal一致。新root字节检查最初假设input receipt有bytes字段而报KeyError；原件保存，修正为识别既有四字段input receipt，SHA仍逐字节精确绑定。这项元数据工程修复没有改预测、方法或评分。

Full每arrival平均GPU fit：Vid0.347秒，HC0.721秒；共享capture分别0.797/1.317秒，fit后另计的独立CPU算术分别0.0078/0.0189秒。DINO使用原Uniform4观测，匹配旧缓存逐输入/像素/native/interval核验后复用，本阶段新增DINO调用为0，以上不是冷启动完整端到端延迟。两source模型未改变，CURRENT及已选配置原字节保持。

## P0资格门与接续决定

预锁的“双方向paired CI下界>0才自动启动昂贵P1”门未通过，原P0_GATE保留失败结果。按照实验方案要求，先完整失败归因，再决定P1。根的决定是继续固定参数的全官方query评测：已存在可信Vid效果与HC当前损伤、正负案例和反馈机制差异，完整主表可以如实量化官方query分布及匹配baseline差距。不能把这个执行范围决定说成HC通过资格，不能再调参、加gate、去掉负例或把同一128集重新称独立确认。

新的P1 launcher revision只表达这一已完成根复核后的显式接续，不改P0自动资格门、模型、1792接口、Gaussian动作/likelihood/采样、lr/sigma/tau/steps/alpha或名单。P1实际启动还要求P0远端字节核验和ROOT_CLOSING_RECEIPT。旧baseline f15641a精确复用，旧IoU-energy Ours不更名。HC来源Fisher及EATA缺失方向仍用户暂停。

P1–P6在本报告写入时仍未执行完成；67个组件/预算/扰动CPU合同与34个P1聚合合同不替代真实GPU资格或实验执行。主方法仍是原注册OPD，不以本次确认分数改变CURRENT。
