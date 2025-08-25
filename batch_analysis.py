import pandas as pd
import os
import argparse
import subprocess
from datetime import datetime
from typing import List, Dict

def create_analysis_config_template(output_file: str = 'analysis_config_template.csv'):
    """
    解析設定のテンプレートファイルを作成する関数
    
    Args:
        output_file (str): 出力ファイルのパス
    """
    template_df = pd.DataFrame({
        'mode': ['orbit'],  # orbit または deep_space（orbit は周回軌道解析）
        'body': [None],     # 中心天体（earth | moon）未指定なら設定のprimary_body
        'altitude': [500.0],  # 地球周回軌道の場合のみ使用 [km]
        'beta': [60.0],  # 地球周回軌道の場合のみ使用 [度]
        'sun_x': [None],  # 深宇宙の場合のみ使用
        'sun_y': [None],  # 深宇宙の場合のみ使用
        'sun_z': [None],  # 深宇宙の場合のみ使用
        'duration': [40010.0],  # 解析時間 [秒]
        'num_orbits': [None],  # 周回数（指定時はdurationより優先）
        'temp_grid_interval': [5.0],  # 温度データの出力間隔 [秒]
        'plot_components': [None],  # 表示するコンポ（空白/未指定で全て）。スペース区切りで列挙
        'power_mode': ['nominal'],  # 電源モード（settings/power_modes.csv の列名）
        'output_dir': ['output']  # 出力ディレクトリ
    })
    template_df.to_csv(output_file, index=False)
    print(f'解析設定テンプレートを作成しました: {output_file}')

def load_analysis_config(config_file: str) -> List[Dict]:
    """
    解析設定を読み込む関数
    
    Args:
        config_file (str): 解析設定CSVファイルのパス
    
    Returns:
        List[Dict]: 解析設定のリスト
    """
    config_df = pd.read_csv(config_file)
    required_columns = ['mode', 'duration', 'output_dir']
    
    # 必須カラムの存在確認
    if not all(col in config_df.columns for col in required_columns):
        raise ValueError(f'設定ファイルには以下のカラムが必要です: {required_columns}')
    
    # 設定を辞書のリストに変換
    configs = config_df.to_dict('records')
    return configs

def write_analysis_log(log_file: str, config: Dict, status: str, error_msg: str = None):
    """
    解析実行のログを記録する関数
    
    Args:
        log_file (str): ログファイルのパス
        config (Dict): 解析設定
        status (str): 実行状態（'success' または 'error'）
        error_msg (str, optional): エラーメッセージ
    """
    # ログファイルのパスを絶対パスに変換
    log_file = os.path.abspath(log_file)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f'\n=== 解析実行時刻: {timestamp} ===\n')
        mode_str = str(config["mode"]).lower()
        f.write(f'モード: {mode_str}\n')
        if mode_str in ("earth", "orbit"):
            f.write(f'軌道高度: {config["altitude"]} km\n')
            f.write(f'ベータ角: {config["beta"]} 度\n')
        else:
            f.write(f'太陽方向ベクトル: [{config["sun_x"]}, {config["sun_y"]}, {config["sun_z"]}]\n')
        f.write(f'解析時間: {config["duration"]} 秒\n')
        if pd.notna(config.get("num_orbits")):
            f.write(f'周回数: {config["num_orbits"]}\n')
        f.write(f'温度データ出力間隔: {config["temp_grid_interval"]} 秒\n')
        f.write(f'出力ディレクトリ: {config["output_dir"]}\n')
        f.write(f'実行状態: {status}\n')
        if error_msg:
            f.write(f'エラー: {error_msg}\n')
        f.write('-' * 50 + '\n')

def execute_analysis(config: Dict, log_file: str) -> bool:
    """
    単一の解析を実行する関数
    
    Args:
        config (Dict): 解析設定
        log_file (str): ログファイルのパス
    
    Returns:
        bool: 実行が成功したかどうか
    """
    # コマンドライン引数の構築
    cmd = ['python', 'multi-node_analysis.py']
    
    # モードに応じた引数の設定（orbit は earth と等価に扱う）
    mode = str(config['mode']).lower()
    # 入力検証
    if mode not in ('orbit', 'earth', 'deep_space'):
        write_analysis_log(log_file, config, 'error', f"不明なmodeが指定されました: {config['mode']}")
        print(f"エラー: 不明なmodeが指定されました: {config['mode']}")
        return False
    if mode == 'orbit':
        mode = 'earth'
    cmd.extend(['--mode', mode])
    # 中心天体が指定されていれば明示的に渡す
    if 'body' in config and pd.notna(config.get('body')):
        cmd.extend(['--body', str(config['body'])])

    # 正規化後の mode に基づいて引数を付与
    if mode == 'earth':
        # CSVで値が指定されている場合のみ付与（未指定なら multi-node 側のデフォルトに委ねる）
        if pd.notna(config.get('altitude')):
            cmd.extend(['--altitude', str(config['altitude'])])
        if pd.notna(config.get('beta')):
            cmd.extend(['--beta', str(config['beta'])])
    elif mode == 'deep_space':
        # 深宇宙は sun ベクトルが必須
        sun_vals = (config.get('sun_x'), config.get('sun_y'), config.get('sun_z'))
        if not all(pd.notna(v) for v in sun_vals):
            write_analysis_log(log_file, config, 'error', 'deep_spaceでは sun_x, sun_y, sun_z を全て指定してください')
            print('エラー: deep_spaceでは sun_x, sun_y, sun_z を全て指定してください')
            return False
        cmd.extend(['--sun_x', str(config['sun_x'])])
        cmd.extend(['--sun_y', str(config['sun_y'])])
        cmd.extend(['--sun_z', str(config['sun_z'])])
    
    # 共通の引数
    if pd.notna(config.get('num_orbits')):
        cmd.extend(['--num_orbits', str(config['num_orbits'])])
    else:
        cmd.extend(['--duration', str(config['duration'])])
    
    cmd.extend(['--temp-grid-interval', str(config['temp_grid_interval'])])
    cmd.extend(['--output_dir', config['output_dir']])

    # プロット対象コンポーネント（スペース区切りで列挙）
    plot_components = config.get('plot_components')
    if isinstance(plot_components, str) and plot_components.strip():
        names = plot_components.split()
        cmd.extend(['--plot-components', *names])

    # 電源モード
    if isinstance(config.get('power_mode'), str) and config.get('power_mode').strip():
        cmd.extend(['--power-mode', config['power_mode'].strip()])
    
    try:
        # 解析の実行
        # NOTE: capturing large stdout/stderr can block the child process. Do not capture.
        result = subprocess.run(cmd, check=True)
        write_analysis_log(log_file, config, 'success')
        print(f'解析が成功しました: {config["output_dir"]}')
        return True
    except subprocess.CalledProcessError as e:
        error_msg = f'コマンド実行エラー: {e.stderr}'
        write_analysis_log(log_file, config, 'error', error_msg)
        print(f'エラー: {config["output_dir"]} の解析中にエラーが発生しました: {error_msg}')
        return False

def batch_analysis(config_file: str, log_file: str = 'analysis_log.log'):
    """
    複数の解析を一括実行する関数
    
    Args:
        config_file (str): 解析設定CSVファイルのパス
        log_file (str): ログファイルのパス
    """
    # 設定の読み込み
    configs = load_analysis_config(config_file)
    
    # 各設定に対して解析を実行
    success_count = 0
    for config in configs:
        if execute_analysis(config, log_file):
            success_count += 1
    
    # 実行結果のサマリー
    print(f'\n解析実行完了: {success_count}/{len(configs)} 成功')

def main():
    parser = argparse.ArgumentParser(description='複数の解析条件を一括実行します。')
    subparsers = parser.add_subparsers(dest='command', help='実行するコマンド')
    
    # 解析設定のテンプレートを作成するコマンド
    template_parser = subparsers.add_parser('create-template', help='解析設定のテンプレートファイルを作成')
    template_parser.add_argument('--output-file', default='analysis_config_template.csv',
                               help='出力ファイルのパス（デフォルト: analysis_config_template.csv）')
    
    # 複数の解析を一括実行するコマンド
    batch_parser = subparsers.add_parser('batch', help='複数の解析を一括実行')
    batch_parser.add_argument('config_file', help='解析設定CSVファイルのパス')
    batch_parser.add_argument('--log-file', default=os.path.join(os.getcwd(), 'analysis_log.log'),
                            help='ログファイルのパス（デフォルト: ./analysis_log.log）')
    
    args = parser.parse_args()
    
    if args.command == 'create-template':
        create_analysis_config_template(args.output_file)
    elif args.command == 'batch':
        batch_analysis(args.config_file, args.log_file)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 