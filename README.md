# ESNを用いたスパース信号再構成（非線形遷移モデル）

## 概要

本プロジェクトでは、Echo State
Network（ESN）を用いた信号再構成手法を実装しました。\
特に、リザバー状態の遷移をPyTorchによる非線形MLPでモデル化し、
スパース観測点から高解像度信号を復元することを目的としています。

2つのローレンツ型ディップを持つODMR様フォトルミネッセンス信号を生成し、
RMSE・nRMSE・R²を用いて再構成精度を評価しています。

------------------------------------------------------------------------

## 取り組んだ内容

-   ESNの状態更新ダイナミクスをゼロから実装
-   観測点削減（スパース測定）設定の設計
-   非線形状態遷移モデル（PyTorch MLP）の構築
-   Ridge回帰によるReadout学習
-   Reset付きロールアウト再構成アルゴリズムの実装
-   定量評価指標（RMSE / nRMSE / R²）による検証
-   再構成結果の可視化

------------------------------------------------------------------------

## 技術スタック

-   Python
-   NumPy
-   PyTorch
-   Matplotlib

------------------------------------------------------------------------

## 問題設定

計測システムにおいて、測定点数を増やすほど精度は向上しますが、
測定時間が増加するというトレードオフがあります。

本研究では、

-   測定点を削減しても信号を復元できるか？
-   非線形遷移モデルは再構成精度を向上させるか？

という点を検証しています。

------------------------------------------------------------------------

## 手法

1.  2ディップのODMR様信号を合成生成
2.  固定ESNに通してリザバー状態を取得
3.  以下を学習
    -   非線形遷移モデル：x\_{t+1} ≈ fθ(\[x_t, u_t\])
    -   信号再構成用Readout
4.  スパース観測点を用いたロールアウト再構成
5.  精度評価

------------------------------------------------------------------------

## プロジェクト構成

    esn-nonlinear-transition/
    │
    ├── src/
    │   ├── esn.py
    │   ├── signal.py
    │   ├── transition_model.py
    │   ├── readout.py
    │   ├── rollout.py
    │   ├── metrics.py
    │   └── visualization.py
    │
    ├── main.py
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## 実行方法

``` bash
pip install -r requirements.txt
python main.py
```

------------------------------------------------------------------------

## 工夫した点

-   ESNライブラリを使わず、自身で状態更新を実装
-   非線形モデル学習時に標準化処理を導入し安定化
-   状態生成・遷移学習・Readout学習を明確に分離
-   乱数seed固定による再現性の確保
-   可視化だけでなく定量評価を重視

------------------------------------------------------------------------

## 今後の改善

-   CLIによる実験設定管理
-   モデル保存機能
-   ハイパーパラメータ探索の自動化
-   実データへの適用

------------------------------------------------------------------------



# ESN-based Sparse Signal Reconstruction with Nonlinear Transition Model

## Overview

This project implements a signal reconstruction pipeline based on Echo
State Networks (ESN), where internal state transitions are modeled using
a nonlinear neural network (PyTorch MLP).

The goal is to reconstruct full-resolution signals from sparse
observations by learning reservoir state dynamics and readout mappings.

The experiment simulates ODMR-like photoluminescence (PL) signals with
two Lorentzian dips and evaluates reconstruction performance using RMSE,
nRMSE, and R².

------------------------------------------------------------------------

## Key Features

-   Custom implementation of a "true" ESN (reservoir state generation)
-   Sparse observation setting (measurement point reduction)
-   Nonlinear state transition model using PyTorch MLP
-   Ridge regression readout layers
-   Reset-based rollout reconstruction
-   Quantitative evaluation (RMSE / nRMSE / R²)
-   Visualization of reconstruction results

------------------------------------------------------------------------

## Technical Stack

-   Python
-   NumPy
-   PyTorch
-   Matplotlib

------------------------------------------------------------------------

## Problem Setting

In practical sensing systems, increasing measurement points improves
accuracy but increases acquisition time.

This project explores:

-   Can we reconstruct full signals from sparse measurements?
-   Can nonlinear state dynamics improve reconstruction accuracy?

------------------------------------------------------------------------

## Method

1.  Generate synthetic ODMR-like signals (2 Lorentzian dips).
2.  Pass signals through a fixed ESN to obtain reservoir states.
3.  Learn:
    -   Nonlinear transition model: x\_{t+1} ≈ fθ(\[x_t, u_t\])
    -   Readout model for signal reconstruction.
4.  Perform sparse rollout reconstruction.
5.  Evaluate reconstruction performance.

------------------------------------------------------------------------

## Project Structure

    esn-nonlinear-transition/
    │
    ├── src/
    │   ├── esn.py
    │   ├── signal.py
    │   ├── transition_model.py
    │   ├── readout.py
    │   ├── rollout.py
    │   ├── metrics.py
    │   └── visualization.py
    │
    ├── main.py
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## How to Run

``` bash
pip install -r requirements.txt
python main.py
```

------------------------------------------------------------------------

## What I Focused On

-   Implemented ESN dynamics from scratch
-   Designed standardized training pipeline for stable nonlinear
    learning
-   Separated state generation, transition modeling, and readout
    learning
-   Ensured reproducibility with controlled random seeds
-   Evaluated quantitatively rather than visually only

------------------------------------------------------------------------

## Author

Yusuke Shimozawa\
Kanazawa University\
Electrical and Information Engineering

