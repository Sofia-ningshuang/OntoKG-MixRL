**Step 1 onto_label.py：转换本体为图**
读取src中的本体文件.ttl，输出graph形式json文件，作为提取标签与后续计算的基础：onto_graph.json

**Step 2 label_generator.py：标签生成器（此步骤独立）**
读取onto_graph.json，输出两种类型的标签xml对接Label Studio平台：no_level & two_level

**Step 3 label_ner.py: 拟合标签数据为实例节点**
读取src中的case01.json, 输出拟合实例数据的merged_graph.json（实例节点存有text切片）, 提供可视化形式（可独立使用visual_graph.py,可视化拟合后的图），拟合规则在src中的mapping_rule.md 

**Step 4 graph_rl_ini.py: 图结构与权重初始化**
权重初始规则在src中的weighting_rule.md，现阶段采用粗粒度权重

**Step 5 graph_rl_q_simple.py: 本体结构强化学习q-learning**
学习事故路径，主要针对本体的类节点

**Step 6 graph_rl_ins.py: 本体与实例节点混合强化学习q-learning**
(1) 建立instance-to-instance双向节点并默认连通，通过本体类节点进行权重传播，即将本体类节点连通的权重赋予实例节点，权重传播与边方向继承规则在src中的ins_weighting_rule.md
(2) 对全图进行强化学习并生成top rank路径

**Step 7 path_evi.py: 根据证据反馈的权重与路径更新**
根据提供的关键证据，更新路径与排序
