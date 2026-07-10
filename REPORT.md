# 项目报告：智能手机人体活动识别 (Human Activity Recognition)

## 1. 项目目标
根据腰部佩戴智能手机（加速度计 + 陀螺仪）采集并预处理得到的 **561 维特征向量**，
将受试者的日常行为分类为 6 种活动之一：

`WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`

## 2. 数据集
| 集合 | 样本数 | 特征数 | 受试者 |
|------|-------|--------|--------|
| train.csv | 7352 | 561 | 21 人 |
| test.csv  | 2947 | 561 | 9 人  |

- 无缺失值、无重复行。
- 类别分布较均衡（LAYING 最多 1407，WALKING_DOWNSTAIRS 最少 986）。
- 训练/测试按受试者划分（70% / 30%），保证被测者不重叠。

## 3. 方法流程 (`har_project.py`)
1. **EDA**：类别分布、每位受试者样本量、特征箱线图。
2. **预处理**：`StandardScaler` 标准化 + `LabelEncoder` 标签编码。
3. **降维可视化**：PCA 二维投影（前 2 主成分解释方差 ~57%），可见静态类
   (SITTING/STANDING/LAYING) 与动态类 (WALKING*) 明显分离。
4. **模型对比**：在训练集上做 5 折分层交叉验证，比较 5 种分类器。
5. **最终评估**：选出 CV 最优模型，在独立测试集上评估。

## 4. 模型对比结果（5 折交叉验证）
| 模型 | CV 准确率 |
|------|-----------|
| Logistic Regression | 0.9846 |
| Linear SVC | 0.9850 |
| **RBF SVM** | **0.9875** |
| Random Forest | 0.9808 |
| KNN | 0.9572 |

最优模型：**RBF 核 SVM**。

## 5. 测试集最终结果
- **准确率 (Accuracy): 0.9542**
- **宏平均 F1: 0.9533**

| 活动 | precision | recall | f1 |
|------|-----------|--------|-----|
| LAYING | 0.99 | 1.00 | 1.00 |
| SITTING | 0.96 | 0.89 | 0.93 |
| STANDING | 0.91 | 0.97 | 0.94 |
| WALKING | 0.95 | 0.97 | 0.96 |
| WALKING_DOWNSTAIRS | 0.97 | 0.92 | 0.94 |
| WALKING_UPSTAIRS | 0.93 | 0.96 | 0.95 |

**主要误差**来自 SITTING 与 STANDING 之间的混淆（两者都是静止姿态，传感器信号相近），
详见混淆矩阵。动态类（走路 / 上下楼）识别效果很好。

## 6. 输出文件（`outputs/` 目录）
| 文件 | 内容 |
|------|------|
| 01_class_distribution.png | 活动类别分布 |
| 02_samples_per_subject.png | 每位受试者样本量 |
| 03_feature_by_activity.png | 代表性特征按活动分布 |
| 04_pca_scatter.png | PCA 二维可视化 |
| 05_model_comparison.png | 各模型 CV 准确率对比 |
| 06_confusion_matrix.png | 最优模型测试集混淆矩阵 |
| metrics_report.txt | 文本版评估指标 |

## 7. 复现方法
```bash
python har_project.py
```
依赖：`pandas, numpy, scikit-learn, matplotlib, seaborn`（Anaconda 环境已具备）。
