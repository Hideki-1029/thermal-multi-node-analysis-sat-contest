## 月周回熱解析対応ガイド（中心天体一般化の実装手順）

本ドキュメントは、本リポジトリの地球周回前提の熱環境モデルを「中心天体（Earth/Moon）を切り替え可能」に一般化し、月周回の熱解析を行えるようにする改築手順を示します。既存の地球向けシナリオを壊さず、設定で切替・拡張できる方針です。

### 対象と前提
- 現行の軌道・環境モデルは以下の簡易化を前提とします。
  - 円軌道（半径 = 天体半径 + 高度）、ベータ角で軌道面を定義
  - 太陽方向は慣性系で固定ベクトル
  - 影判定は円筒影（本影のみの近似）
  - アルベド・惑星赤外（IR）は一様場の近似＋平板ビューファクター
- 本改築では上記前提を維持したまま、中心天体パラメータ（半径・μ・アルベド・IR）を切り替え可能にします。
- 将来的な高度化（位相依存のアルベド、二次天体寄与、楕円軌道等）は「今後の拡張」に記します。

---

## 実装の全体像

1) 設定ファイルに天体カタログを追加し、`primary_body`（解析対象の中心天体）で切替可能にする。
2) 軌道計算・蝕判定・ビューファクター等の「Earth 固定値」を、設定から取得する「Body 可変値」に差し替える。
3) 関数／定数名の一般化（`earth_*` → `planet_*` / `body_*`）と、それに伴う呼び出し箇所の整合。
4) CLI・出力命名を天体に応じて可読化。
5) 回帰（Earth）とスモーク（Moon）の双方で検証。

---

## 1. 設定ファイルの一般化

ファイル: `settings/constants.yaml`

### 追加・変更方針
- 既存の `earth_albedo`, `earth_ir`, `enable_earth_ir` は当面残してもよいですが、新実装では `environment.primary_body` と `bodies[...]` を優先して参照します。
- 新しいキー構成案（例）:

```yaml
physical_constants:
  solar_constant: 1367.0
  stefan_boltzmann: 5.67e-8
  enable_albedo: true
  enable_planet_ir: true   # 旧: enable_earth_ir

environment:
  primary_body: earth      # earth | moon

bodies:
  earth:
    radius_km: 6378.137
    mu_km3_s2: 398600.4418
    albedo: 0.30
    planet_ir_w_m2: 221.0
  moon:
    radius_km: 1737.4
    mu_km3_s2: 4902.800066
    albedo: 0.12
    planet_ir_w_m2: 52.0   # 必要に応じて調整
```

注意:
- 旧キー（`earth_albedo`, `earth_ir`, `enable_earth_ir`）は、移行期間は読み取り継続可。新実装では新キー優先・旧キーはフォールバックとして扱うと安全です。

---

## 2. 軌道計算の一般化（半径・μ・蝕）

ファイル: `utils/orbit_utils.py`

### 変更ポイント
- `earth_radius`, `mu` のハードコードを廃止し、`environment.primary_body` と `bodies[...]` から取得。
- 蝕判定（円筒影）の閾値 `earth_radius` → `body_radius_km` に置換。
- 関数名は可能であれば一般化（後方互換のためラッパ導入可）。

対象関数（例）:
- `calculate_orbit_parameters(altitude, beta_angle)`
  - 使う定数: `radius_km`, `mu_km3_s2`
  - 返り値や β角処理は現状維持で可
- `calculate_satellite_position(time, period, altitude, orbit_normal, e1, e2)`
  - 使う定数: `radius_km`, `mu_km3_s2`
  - 蝕判定の半径を置換

推奨: 旧関数名を残す場合、内部で新実装を呼ぶ薄いラッパを用意（段階移行）。

---

## 3. 惑星IR・アルベド計算の一般化

ファイル: `utils/thermal_utils.py`, `utils/orbit_utils.py`

### 名称・引数の一般化
- `Surface.calculate_earth_heat(...)` → `Surface.calculate_planetary_heat(planet_vector, solar_constant, planet_albedo, planet_ir, altitude, sun_vector, orbit_normal)`
- `ThermalNode.calculate_heat_balance(...)` 内で参照する環境定数
  - `earth_albedo` → `bodies[primary_body].albedo`
  - `earth_ir` → `bodies[primary_body].planet_ir_w_m2`
  - `enable_earth_ir` → `enable_planet_ir`
  - 呼び出し引数 `earth_vector` → `planet_vector`

### ビューファクター関数の一般化
- `utils/orbit_utils.py` 内の以下を一般化し、半径は `radius_km` を使用。
  - `calculate_earth_ir_view_factor(...)` → `calculate_planet_ir_view_factor(...)`
  - `calculate_albedo_view_factor(...)` → `calculate_planet_albedo_view_factor(...)`
- 参照側（`thermal_utils.py` の import と呼出）も新名称へ変更。

注記:
- 現状の式は地球を想定した文献式に基づく近似です。半径置換により月へも一次近似は可能ですが、厳密な位相依存・地形依存は未考慮です。

---

## 4. 解析ループ／CLI の一般化

ファイル: `multi-node_analysis.py`

### ランタイム処理
- 関数 `run_earth_orbit_analysis(...)` を汎用関数へ集約
  - 例: `run_orbit_analysis(config, altitude, beta_angle, body='earth', duration=None)`
  - 旧関数は互換ラッパとして維持（内部で `body='earth'` を渡して新関数を呼ぶ）。
- ループ内の `earth_vector` は中心天体方向ベクトルなので `planet_vector` に改名。
  - 計算式自体は既存通り `planet_vector = R^T @ (-position / ||position||)` でOK。

### CLI 拡張
- 引数 `--body {earth,moon}` を追加（既定は `constants.yaml` の `environment.primary_body`）。
- 出力ディレクトリ命名: `earth_orbit_...` → `{body}_orbit_alt{...}_beta{...}`

---

## 5. 可視化の一般化

ファイル: `utils/plotting_utils.py`

### 変更ポイント
- 軌道可視化 `plot_orbit_visualization(...)` での半径・蝕判定に `body_radius_km` を使用。
- 引数に `body: str = None` を追加し、未指定なら設定の `primary_body` を使用。
- 図注・凡例の文言が Earth 固定なら、天体名が入るように調整。

---

## 6. 検証計画

### 回帰（Earth）
- 既存の Earth ケース（例: 高度 400–1000 km, β角 0/60/90°）を同条件で再実行。
- 変更前後で温度・熱入力プロファイルの差が微小であること（丸めやコード整理に伴う1–2%以内など）を確認。

### スモーク（Moon）
- 代表ケースの実行（例: 高度 100 km, β角 0°/60°/90°）。
- 期待感度:
  - 蝕（eclipse）の発生率は月の半径に依存して変化。
  - アルベド・IR は Earth より小さく、太陽直達寄与が支配的に。
- アルベド/IRのON/OFFにより温度や熱入力の差分が出ることを確認。

### パフォーマンス
- 半径・定数の参照先変更のみのため、大きな計算量増は想定なし。

---

## 7. 既知の制約と今後の拡張

制約（本改築時点）:
- 太陽方向は慣性固定ベクトル。月の自転・日照サイクルや、季節・位相に伴う空間分布は未考慮。
- アルベド・IR は一様定数。実際の月面は低アルベドで位相依存の散乱特性を持つ。
- 二次天体寄与（例: 月周回時の Earthshine）は未導入。

将来拡張の候補:
- 二次天体を `secondary_bodies: [...]` として設定に追加し、各寄与（アルベド・IR・視線幾何）を線形加算。
- 時間依存の太陽方向（公転・自転・季節）と、位相依存アルベドモデル。
- 影モデルの高度化（半影、円錐影、食の動的経路）。
- 楕円軌道・任意軌道要素（Keplerian）入力。

---

## 8. 実装順序（チェックリスト）

1) `settings/constants.yaml` に `environment.primary_body` と `bodies` セクションを追加（旧キーは残置）。
2) `utils/orbit_utils.py`
   - 半径・μ・影判定の半径を設定経由に置換
   - 可能なら関数名を一般化し、旧名はラッパで維持
3) `utils/orbit_utils.py` のビューファクター関数を一般化（半径置換、関数名統一）
4) `utils/thermal_utils.py`
   - `calculate_earth_heat` → `calculate_planetary_heat` に改名・引数名更新
   - `ThermalNode.calculate_heat_balance` で新キーを参照、`planet_vector` を渡す
5) `multi-node_analysis.py`
   - `run_orbit_analysis(..., body=...)` を新設、旧関数はラッパ化
   - CLI に `--body` 追加、出力ディレクトリ命名を `{body}_orbit_...` に
6) `utils/plotting_utils.py` の半径・蝕判定を `body_radius_km` に切替
7) Earth 回帰 → Moon スモークの順に検証

---

## 9. 実行例（改築後想定）

Earth（既定設定のまま）:
```bash
python multi-node_analysis.py --mode earth --altitude 400 --beta 60
```

Moon（中心天体を指定）:
```bash
python multi-node_analysis.py --mode earth --body moon --altitude 100 --beta 60
```

深宇宙（参考・従来通り）:
```bash
python multi-node_analysis.py --mode deep_space --sun_x 1 --sun_y 0 --sun_z 0
```

---

## 10. リスクとロールバック

リスク:
- 既存キーと新キーの整合が取れない場合、環境定数の参照が混在し結果が不安定になる可能性。
- 関数名・引数名変更に伴う参照漏れ。

緩和策:
- 新旧キーの両対応期間を設け、ログでどちらを参照したかをオプション表示できるようにする（`debug` フラグ）。
- 旧APIは薄いラッパで保持し、段階的に呼び出し側を移行。

ロールバック:
- 既存の Earth 固定ロジックはブランチで保全済み。問題発生時は当該ファイルの `earth_*` 参照版へ一時的に戻すことで可逆。

---

以上で、月周回熱解析を可能とする中心天体一般化の実装計画と作業手順を示しました。まずは設定の一般化と `orbit_utils.py` の置換から始め、段階的に関数名・引数名の一般化と参照更新を進めると、安全に移行できます。


