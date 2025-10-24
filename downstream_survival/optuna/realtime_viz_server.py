#!/usr/bin/env python3
"""
实时可视化服务器
在训练过程中提供网页版的可视化界面
"""

import os
import sys
import argparse
import optuna
import threading
import time
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver

class OptunaVizHandler(SimpleHTTPRequestHandler):
    """自定义HTTP处理器"""
    
    def __init__(self, study, *args, **kwargs):
        self.study = study
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # 生成HTML页面
            html = self.generate_viz_html()
            self.wfile.write(html.encode())
        else:
            super().do_GET()
    
    def generate_viz_html(self):
        """生成可视化HTML页面"""
        import optuna.visualization as vis
        
        # 生成图表
        history_fig = vis.plot_optimization_history(self.study)
        importance_fig = vis.plot_param_importances(self.study) if len(self.study.trials) > 10 else None
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Optuna 实时可视化</title>
            <meta charset="utf-8">
            <meta http-equiv="refresh" content="30">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
                .stat {{ background: #e8f4fd; padding: 15px; border-radius: 5px; }}
                .chart {{ margin: 20px 0; }}
                .auto-refresh {{ color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 Optuna 实时优化监控</h1>
                <p class="auto-refresh">页面每30秒自动刷新</p>
            </div>
            
            <div class="stats">
                <div class="stat">
                    <h3>📊 试验进度</h3>
                    <p>已完成: {len(self.study.trials)} 试验</p>
                </div>
                <div class="stat">
                    <h3>🏆 最佳AUC</h3>
                    <p>{self.study.best_value:.4f}</p>
                </div>
                <div class="stat">
                    <h3>⚙️ 最佳参数</h3>
                    <p>{', '.join([f'{k}: {v}' for k, v in list(self.study.best_params.items())[:3]])}</p>
                </div>
            </div>
            
            <div class="chart">
                <h2>📈 优化历史</h2>
                {history_fig.to_html(include_plotlyjs='cdn', div_id="history")}
            </div>
            
            {f'''
            <div class="chart">
                <h2>🎯 参数重要性</h2>
                {importance_fig.to_html(include_plotlyjs=False, div_id="importance")}
            </div>
            ''' if importance_fig else ''}
            
            <script>
                // 自动刷新页面
                setTimeout(function() {{
                    location.reload();
                }}, 30000);
            </script>
        </body>
        </html>
        """
        return html

def start_viz_server(study_path: str, port: int = 8080):
    """启动可视化服务器"""
    
    # 加载研究
    study = optuna.load_study(
        study_name=Path(study_path).stem,
        storage=f"sqlite:///{study_path}"
    )
    
    # 创建自定义处理器
    def handler(*args, **kwargs):
        return OptunaVizHandler(study, *args, **kwargs)
    
    # 启动服务器
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"🌐 Optuna 实时可视化服务器已启动")
        print(f"📊 访问地址: http://localhost:{port}")
        print(f"💡 在浏览器中打开上述地址查看实时优化进度")
        print(f"🔄 页面每30秒自动刷新")
        print(f"⏹️  按 Ctrl+C 停止服务器")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print(f"\n🛑 服务器已停止")

def main():
    parser = argparse.ArgumentParser(description='Optuna 实时可视化服务器')
    parser.add_argument('--study_path', type=str, required=True,
                       help='研究数据库路径 (.db 文件)')
    parser.add_argument('--port', type=int, default=8080,
                       help='服务器端口 (default: 8080)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.study_path):
        print(f"❌ 文件不存在: {args.study_path}")
        return
    
    start_viz_server(args.study_path, args.port)

if __name__ == "__main__":
    main()
