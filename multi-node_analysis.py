#!/usr/bin/env python3
import numpy as np
import os
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from utils.thermal_utils import (
    ThermalNode, Surface, PanelProperties, SurfaceOpticalProperties,
    load_surface_properties, load_material_properties, load_panel_material_assignments,
    load_constants, HeatInputRecord
)
from utils.satellite_config import SatelliteConfiguration
from utils.orbit_utils import (
    calculate_orbit_parameters, calculate_earth_parameters,
    calculate_satellite_attitude, calculate_sun_vector_in_satellite_frame,
    calculate_satellite_position
)
from utils.plotting_utils import (
    plot_temperature_profile, save_temperature_data, plot_heat_balance,
    save_heat_input_data, plot_heat_input_by_surface, plot_orbit_visualization
)
from utils.config_loader import load_power_modes

def create_satellite_surfaces(config: SatelliteConfiguration) -> List[Surface]:
    """衛星の各面を作成"""
    dims = config.dimensions
    surfaces = []
    
    # 各面の法線ベクトルと面積を定義
    surface_defs = [
        ('PX', np.array([1, 0, 0]), dims['length_y'] * dims['length_z']),  # +X
        ('MX', np.array([-1, 0, 0]), dims['length_y'] * dims['length_z']),  # -X
        ('PY', np.array([0, 1, 0]), dims['length_x'] * dims['length_z']),  # +Y
        ('MY', np.array([0, -1, 0]), dims['length_x'] * dims['length_z']),  # -Y
        ('PZ', np.array([0, 0, 1]), dims['length_x'] * dims['length_y']),  # +Z
        ('MZ', np.array([0, 0, -1]), dims['length_x'] * dims['length_y'])   # -Z
    ]
    
    for name, normal, area in surface_defs:
        # パネルの材料構成を読み込み
        panel_config = config.panel_material_assignments[name][0]  # パネルは単一材料
        panel_material = config.material_properties[panel_config['material']]
        panel_thickness = panel_config['thickness']
        
        # 表面光学特性を読み込み
        optical_configs = config.surface_optical_assignments[name]
        
        # 外側の表面光学特性
        outside_materials = []
        for opt_config in optical_configs['outside']:
            opt_name = opt_config['material']
            ratio = opt_config['ratio']
            outside_materials.append((
                config.surface_materials[opt_name],
                ratio
            ))
        
        # 内側の表面光学特性
        inside_materials = []
        for opt_config in optical_configs['inside']:
            opt_name = opt_config['material']
            ratio = opt_config['ratio']
            inside_materials.append((
                config.surface_materials[opt_name],
                ratio
            ))
        
        surfaces.append(Surface(
            name=name,
            normal=normal,
            area=area * 1e-6,  # mm^2 to m^2
            panel=PanelProperties(
                material=panel_material,
                thickness=panel_thickness
            ),
            optical_properties=SurfaceOpticalProperties(
                outside=outside_materials,
                inside=inside_materials
            )
        ))
    
    return surfaces

def run_earth_orbit_analysis(
    config: SatelliteConfiguration,
    altitude: float,
    beta_angle: float,
    duration: float = None,
    power_modes: Optional[Dict[str, Dict[str, bool]]] = None,
    power_mode_name: Optional[str] = None,
) -> Tuple[List[float], Dict[str, List[float]], List[HeatInputRecord], List[bool]]:
    """
    地球周回軌道での熱解析を実行
    
    Args:
        config: 衛星の設定
        altitude: 軌道高度 [km]
        beta_angle: ベータ角 [度]
        duration: 解析時間 [秒]（Noneの場合は1軌道周期）
    
    Returns:
        times: 時間リスト [秒]
        temperatures: 各面の温度履歴（キー：面の名前）
        heat_input_records: 熱入力記録
        eclipse_flags: 各時刻で蝕中かどうかのリスト
    """
    # 軌道パラメータの計算
    period, eclipse_fraction, beta_rad, orbit_normal, e1, e2 = calculate_orbit_parameters(altitude, beta_angle)
    if duration is None:
        duration = period
    
    # 時間ステップの設定
    time_step = load_constants()['analysis_parameters']['time_step']
    times = np.arange(0, duration, time_step)
    
    # 姿勢制御モードの取得
    attitude_config = load_constants()['satellite_attitude']
    attitude_mode = attitude_config['earth_orbit_mode']
    custom_attitude = attitude_config['custom_attitude'] if 'custom' in attitude_mode.values() else None
    
    # 熱ノードの作成
    node = ThermalNode(
        initial_temp=load_constants()['analysis_parameters']['initial_temperature']
    )
    if power_modes is not None:
        node.power_modes = power_modes
    if power_mode_name is not None:
        node.power_mode_name = power_mode_name
    
    # 面の追加と内部発熱の設定
    for surface in create_satellite_surfaces(config):
        node.add_surface(surface)
        node.set_internal_heat(surface.name, config.internal_heat[surface.name])
    
    # コンポーネントの追加
    for comp_key, component in config.components.items():
        node.add_component(comp_key, component)
    
    # コンダクタンス行列の設定
    node.set_conductance_matrix(config.conductance_matrix, config.enable_conductance)
    
    # 温度履歴の記録
    temperatures = {surface_name: [node.get_temperature(surface_name)] 
                   for surface_name in node.surfaces.keys()}
    # MLIノードの温度も記録
    for surface_name, surface in node.surfaces.items():
        if surface.has_mli:
            temperatures[f"{surface_name}_MLI"] = [surface.mli_node.temperature]
    # コンポーネントの温度も記録
    for component_name in node.components.keys():
        temperatures[component_name] = [node.get_component_temperature(component_name)]
    
    # 蝕フラグの記録
    eclipse_flags = [False]
    
    # 時間積分
    for t in times[1:]:
        # 衛星の位置・速度ベクトルと蝕の状態を計算
        position, velocity, in_eclipse = calculate_satellite_position(
            time=t,
            period=period,
            altitude=altitude,
            orbit_normal=orbit_normal,
            e1=e1,
            e2=e2
        )
        
        # 姿勢行列を計算
        rotation_matrix = calculate_satellite_attitude(
            position=position,
            velocity=velocity,
            attitude_config=attitude_mode
        )
        
        # 太陽方向ベクトル（衛星固定系）
        sun_vector = calculate_sun_vector_in_satellite_frame(
            time=t,
            period=period,
            beta_angle=beta_rad,
            rotation_matrix=rotation_matrix
        )
        
        # 地球方向ベクトルとビューファクターを計算
        earth_vector = rotation_matrix.T @ (-position / np.linalg.norm(position))  # 衛星固定座標系に変換
        # 熱収支の計算と温度更新
        heat_balances = node.calculate_heat_balance(
            sun_vector=sun_vector,
            earth_vector=earth_vector,
            in_eclipse=in_eclipse,
            time=t,
            altitude=altitude,
            orbit_normal=orbit_normal
        )
        node.update_temperature(heat_balances, time_step)
        
        # 温度履歴の記録
        for surface_name in node.surfaces.keys():
            temperatures[surface_name].append(node.get_temperature(surface_name))
            # MLIノードの温度も記録
            if node.surfaces[surface_name].has_mli:
                temperatures[f"{surface_name}_MLI"].append(node.surfaces[surface_name].mli_node.temperature)
        # コンポーネントの温度も記録
        for component_name in node.components.keys():
            temperatures[component_name].append(node.get_component_temperature(component_name))
        # 蝕フラグの記録
        eclipse_flags.append(bool(in_eclipse))
    
    return times.tolist(), temperatures, node.heat_input_records, eclipse_flags

def run_deep_space_analysis(
    config: SatelliteConfiguration,
    sun_vector: np.ndarray,
    duration: float = None,
    power_modes: Optional[Dict[str, Dict[str, bool]]] = None,
    power_mode_name: Optional[str] = None,
) -> Tuple[List[float], Dict[str, List[float]], List[HeatInputRecord], List[bool]]:
    """
    深宇宙探査機の非定常熱解析を実行
    
    Args:
        config: 衛星の設定
        sun_vector: 太陽方向ベクトル（衛星固定座標系）
        duration: 解析時間 [秒]（Noneの場合はデフォルトで1軌道周期相当）
    
    Returns:
        times: 時間リスト [秒]
        temperatures: 各面の温度履歴（キー：面の名前）
        heat_input_records: 熱入力記録
        eclipse_flags: 各時刻で蝕中かどうかのリスト（深宇宙では常にFalse）
    """
    # 時間ステップの設定
    time_step = load_constants()['analysis_parameters']['time_step']
    if duration is None:
        duration = 6000.0  # デフォルトで6000秒（例）
    times = np.arange(0, duration, time_step)

    # 熱ノードの作成
    node = ThermalNode(
        initial_temp=load_constants()['analysis_parameters']['initial_temperature']
    )
    if power_modes is not None:
        node.power_modes = power_modes
    if power_mode_name is not None:
        node.power_mode_name = power_mode_name
    # 面の追加と内部発熱の設定
    for surface in create_satellite_surfaces(config):
        node.add_surface(surface)
        node.set_internal_heat(surface.name, config.internal_heat[surface.name])

    # コンポーネントの追加
    for comp_key, component in config.components.items():
        node.add_component(comp_key, component)

    # コンダクタンス行列の設定
    node.set_conductance_matrix(config.conductance_matrix, config.enable_conductance)
    
    # 温度履歴の記録
    temperatures = {surface_name: [node.get_temperature(surface_name)] 
                   for surface_name in node.surfaces.keys()}
    # MLIノードの温度も記録
    for surface_name, surface in node.surfaces.items():
        if surface.has_mli:
            temperatures[f"{surface_name}_MLI"] = [surface.mli_node.temperature]
    # コンポーネントの温度も記録
    for component_name in node.components.keys():
        temperatures[component_name] = [node.get_component_temperature(component_name)]
    
    # 蝕フラグの記録（深宇宙では常にFalse）
    eclipse_flags = [False]

    # 時間積分
    for t in times[1:]:
        heat_balances = node.calculate_heat_balance(
            sun_vector=sun_vector,
            time=t
        )
        node.update_temperature(heat_balances, time_step)
        for surface_name in node.surfaces.keys():
            temperatures[surface_name].append(node.get_temperature(surface_name))
            # MLIノードの温度も記録
            if node.surfaces[surface_name].has_mli:
                temperatures[f"{surface_name}_MLI"].append(node.surfaces[surface_name].mli_node.temperature)
        # コンポーネントの温度も記録
        for component_name in node.components.keys():
            temperatures[component_name].append(node.get_component_temperature(component_name))
        eclipse_flags.append(False)

    return times.tolist(), temperatures, node.heat_input_records, eclipse_flags

def copy_settings_files(output_dir: str):
    """
    設定ファイルを結果出力フォルダにコピーする関数
    
    Args:
        output_dir (str): 出力ディレクトリのパス
    """
    settings_dir = 'settings'
    settings_output_dir = os.path.join(output_dir, 'settings')
    
    # 出力ディレクトリ内にsettingsディレクトリを作成
    os.makedirs(settings_output_dir, exist_ok=True)
    
    # settings/配下の全てのファイルをコピー
    for file in os.listdir(settings_dir):
        if file.endswith(('.yaml', '.yml')):
            src_path = os.path.join(settings_dir, file)
            dst_path = os.path.join(settings_output_dir, file)
            shutil.copy2(src_path, dst_path)
            print(f'設定ファイルをコピーしました: {dst_path}')

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='衛星の熱解析プログラム')
    parser.add_argument('--mode', choices=['earth', 'deep_space', 'orbit'], default='earth',
                      help='解析モード: earth (地球周回軌道) または deep_space (深宇宙)')
    parser.add_argument('--body', choices=['earth', 'moon'], help='中心天体（未指定時は設定ファイルの primary_body を使用）')
    parser.add_argument('--altitude', type=float, help='軌道高度 [km]')
    parser.add_argument('--beta', type=float, help='ベータ角 [度]')
    parser.add_argument('--output_dir', type=str, default='output',
                      help='出力ディレクトリ')
    parser.add_argument('--sun_x', type=float, help='太陽方向ベクトルX成分（深宇宙モード用）')
    parser.add_argument('--sun_y', type=float, help='太陽方向ベクトルY成分（深宇宙モード用）')
    parser.add_argument('--sun_z', type=float, help='太陽方向ベクトルZ成分（深宇宙モード用）')
    parser.add_argument('--num_orbits', type=int, default=1, help='解析する周回数（デフォルト: 1）')
    parser.add_argument('--duration', type=float, help='解析時間 [秒]（指定しない場合は地球周回は1軌道×num_orbits、深宇宙は6000秒）')
    parser.add_argument('--temp-grid-interval', type=float, default=10.0,
                      help='温度プロファイルの等温線の間隔 [°C] (デフォルト: 10.0)')
    parser.add_argument('--plot-components', type=str, nargs='*',
                      help='temperature_components.png に表示するコンポーネント名のリスト（例: BAT RW XTRP）')
    parser.add_argument('--power-mode', type=str, default='nominal',
                      help='power_modes.csv で定義した電源モード名（デフォルト: nominal）')
    args = parser.parse_args()

    # 衛星の設定を読み込み
    config = SatelliteConfiguration.from_config_files()
    # 電源モードを読み込み
    power_modes = load_power_modes()
    # --body 指定がある場合は環境変数で一時上書き
    if args.body:
        os.environ['PRIMARY_BODY_OVERRIDE'] = args.body
    
    # orbit は earth と等価に扱う
    if args.mode in ['earth', 'orbit']:
        # 地球周回軌道解析
        altitude = args.altitude or load_constants()['orbit_parameters']['default_altitude']
        beta_angle = args.beta or load_constants()['orbit_parameters']['default_beta']
        
        # 周回数に応じたdurationを計算
        period, _, _, _, _, _ = calculate_orbit_parameters(altitude, beta_angle)
        if args.duration is not None:
            duration = args.duration
        else:
            duration = period * args.num_orbits
        # --num_orbitsが明示的に指定されていた場合は優先
        if args.mode == 'earth' and args.duration is not None and 'num_orbits' in vars(args) and args.num_orbits != 1:
            duration = period * args.num_orbits
        
        # 出力ディレクトリの作成（中心天体名・高度・ベータ角を含む）
        body_name = os.environ.get('PRIMARY_BODY_OVERRIDE') or load_constants().get('environment',{}).get('primary_body','earth')
        output_subdir = f"{body_name}_orbit_alt{altitude:.1f}_beta{beta_angle:.1f}"
        output_path = os.path.join(args.output_dir, output_subdir)
        os.makedirs(output_path, exist_ok=True)
        # 設定ファイルのコピー
        copy_settings_files(output_path)
        # 電源モード適用（存在すれば）
        if power_modes:
            # ThermalNodeは run_earth_orbit_analysis 内部で生成されるため、
            # ここでは後でプロット前にモード名を可視化にのみ反映
            pass
        
        times, temperatures, heat_input_records, eclipse_flags = run_earth_orbit_analysis(
            config=config,
            altitude=altitude,
            beta_angle=beta_angle,
            duration=duration,
            power_modes=power_modes,
            power_mode_name=args.power_mode
        )
        # 電源モードに応じた成分名を図タイトルに付記
        
        # ビューファクター行列（Rij）をCSV出力
        node_for_vf = ThermalNode(initial_temp=load_constants()['analysis_parameters']['initial_temperature'])
        for surface in create_satellite_surfaces(config):
            node_for_vf.add_surface(surface)
        vf_csv_path = os.path.join(output_path, 'view_factor_matrix.csv')
        node_for_vf.view_factor_matrix.to_csv(vf_csv_path)
        # RijマトリクスもCSV出力
        node_for_vf.save_rij_matrix(output_path)
        
        # 結果のプロットと保存
        plot_temperature_profile(
            times, temperatures, output_path,
            eclipse_flags, temp_grid_interval=args.temp_grid_interval,
            selected_components=args.plot_components
        )
        save_temperature_data(times, temperatures, output_path)
        plot_heat_balance(heat_input_records, output_path)
        plot_heat_input_by_surface(heat_input_records, output_path)
        save_heat_input_data(heat_input_records, output_path)
        
        # 軌道の可視化
        plot_orbit_visualization(altitude, beta_angle, output_path)
        
    else:  # deep_space
        # 深宇宙解析
        if not all(v is not None for v in [args.sun_x, args.sun_y, args.sun_z]):
            parser.error("深宇宙モードでは --sun_x, --sun_y, --sun_z の全てを指定してください")
        
        # 太陽方向ベクトルを正規化
        sun_vector = np.array([args.sun_x, args.sun_y, args.sun_z])
        sun_vector_normalized = sun_vector / np.linalg.norm(sun_vector)  # 正規化
        
        # 出力ディレクトリの作成（正規化した太陽方向ベクトルを使用）
        sun_dir = f"deep_space_sun_{sun_vector_normalized[0]:.3f}_{sun_vector_normalized[1]:.3f}_{sun_vector_normalized[2]:.3f}"
        output_path = os.path.join(args.output_dir, sun_dir)
        os.makedirs(output_path, exist_ok=True)
        # 設定ファイルのコピー
        copy_settings_files(output_path)
        
        if args.duration is not None:
            duration = args.duration
        else:
            duration = None
        times, temperatures, heat_input_records, eclipse_flags = run_deep_space_analysis(
            config=config,
            sun_vector=sun_vector_normalized,  # 正規化したベクトルを使用
            duration=duration,
            power_modes=power_modes,
            power_mode_name=args.power_mode
        )
        # ビューファクター行列（Rij）をCSV出力
        node_for_vf = ThermalNode(initial_temp=load_constants()['analysis_parameters']['initial_temperature'])
        for surface in create_satellite_surfaces(config):
            node_for_vf.add_surface(surface)
        vf_csv_path = os.path.join(output_path, 'view_factor_matrix.csv')
        node_for_vf.view_factor_matrix.to_csv(vf_csv_path)
        # RijマトリクスもCSV出力
        node_for_vf.save_rij_matrix(output_path)
        # 結果のプロットと保存（地球周回と同じ関数を使う）
        plot_temperature_profile(
            times, temperatures, output_path,
            eclipse_flags, temp_grid_interval=args.temp_grid_interval,
            selected_components=args.plot_components
        )
        save_temperature_data(times, temperatures, output_path)
        plot_heat_balance(heat_input_records, output_path)
        plot_heat_input_by_surface(heat_input_records, output_path)
        save_heat_input_data(heat_input_records, output_path)

if __name__ == '__main__':
    main()
