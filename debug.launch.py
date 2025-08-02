
import sys
import os
import traceback
import logging
from flask import Flask, render_template, request, jsonify
import threading
import time

# ログ設定を最初に行う
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug.log')
    ]
)
logger = logging.getLogger(__name__)

class MockROS2Interface:
    """ROS2が利用できない場合のモックインターフェース"""
    
    def __init__(self):
        logger.info("MockROS2Interface initialized")
        self.robot_status = {
            'position': {'x': 0, 'y': 0, 'z': 0},
            'velocity': {'linear': 0, 'angular': 0},
            'laser_data': {'min_distance': 2.5, 'ranges_count': 360},
            'last_update': time.time(),
            'mode': 'mock'
        }
    
    def move_robot(self, linear_x, angular_z):
        logger.info(f"Mock move: linear={linear_x}, angular={angular_z}")
        self.robot_status['velocity']['linear'] = linear_x
        self.robot_status['velocity']['angular'] = angular_z
        return True
    
    def stop_robot(self):
        logger.info("Mock stop")
        return self.move_robot(0.0, 0.0)
    
    def send_custom_command(self, command):
        logger.info(f"Mock command: {command}")
        return True
    
    def get_robot_status(self):
        return self.robot_status.copy()

def create_ros2_interface():
    """ROS2インターフェースを作成（フォールバック付き）"""
    try:
        logger.info("Attempting to import ROS2...")
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import Twist
        from std_msgs.msg import String
        from sensor_msgs.msg import LaserScan
        
        logger.info("ROS2 imports successful")
        
        class ROS2Interface(Node):
            def __init__(self):
                logger.info("Initializing ROS2 node...")
                super().__init__('flask_robot_controller')
                
                # Publisher設定
                logger.info("Setting up publishers...")
                self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
                self.command_pub = self.create_publisher(String, '/robot_command', 10)
                
                # Subscriber設定
                logger.info("Setting up subscribers...")
                self.laser_sub = self.create_subscription(
                    LaserScan, '/scan', self.laser_callback, 10
                )
                
                # ロボットの状態
                self.robot_status = {
                    'position': {'x': 0, 'y': 0, 'z': 0},
                    'velocity': {'linear': 0, 'angular': 0},
                    'laser_data': {'min_distance': 0, 'ranges_count': 0},
                    'last_update': time.time(),
                    'mode': 'ros2'
                }
                
                logger.info("ROS2Interface initialized successfully")
            
            def laser_callback(self, msg):
                try:
                    if msg.ranges:
                        min_distance = min([r for r in msg.ranges if r > 0])
                        self.robot_status['laser_data'] = {
                            'min_distance': min_distance,
                            'ranges_count': len(msg.ranges)
                        }
                        self.robot_status['last_update'] = time.time()
                except Exception as e:
                    logger.error(f"Laser callback error: {e}")
            
            def move_robot(self, linear_x, angular_z):
                try:
                    twist = Twist()
                    twist.linear.x = float(linear_x)
                    twist.angular.z = float(angular_z)
                    self.cmd_vel_pub.publish(twist)
                    
                    self.robot_status['velocity']['linear'] = linear_x
                    self.robot_status['velocity']['angular'] = angular_z
                    
                    logger.info(f"Robot moved: linear={linear_x}, angular={angular_z}")
                    return True
                except Exception as e:
                    logger.error(f"Move robot error: {e}")
                    return False
            
            def stop_robot(self):
                return self.move_robot(0.0, 0.0)
            
            def send_custom_command(self, command):
                try:
                    msg = String()
                    msg.data = command
                    self.command_pub.publish(msg)
                    logger.info(f"Command sent: {command}")
                    return True
                except Exception as e:
                    logger.error(f"Send command error: {e}")
                    return False
            
            def get_robot_status(self):
                return self.robot_status.copy()
        
        # ROS2初期化を試行
        logger.info("Initializing ROS2...")
        rclpy.init()
        
        ros2_interface = ROS2Interface()
        logger.info("ROS2 interface created successfully")
        
        return ros2_interface, True
        
    except ImportError as e:
        logger.warning(f"ROS2 not available: {e}")
        logger.info("Using mock interface")
        return MockROS2Interface(), False
    except Exception as e:
        logger.error(f"ROS2 initialization failed: {e}")
        logger.info("Falling back to mock interface")
        return MockROS2Interface(), False

def create_flask_app(ros2_interface, is_real_ros2):
    """Flaskアプリを作成"""
    try:
        logger.info("Creating Flask app...")
        app = Flask(__name__)
        app.config['DEBUG'] = True
        app.config['TESTING'] = True
        
        @app.errorhandler(Exception)
        def handle_exception(e):
            logger.error(f"Unhandled exception: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }), 500
        
        @app.route('/')
        def index():
            try:
                logger.info("Index route accessed")
                return render_template('index.html')
            except Exception as e:
                logger.error(f"Index route error: {e}")
                # テンプレートが見つからない場合の簡単なHTML
                return f"""
                <!DOCTYPE html>
                <html>
                <head><title>ROS2 Robot Controller</title></head>
                <body>
                    <h1>ROS2 Robot Controller (Debug Mode)</h1>
                    <p>Mode: {'Real ROS2' if is_real_ros2 else 'Mock Mode'}</p>
                    <p>Error: Template not found. Using fallback HTML.</p>
                    <div>
                        <button onclick="testAPI()">Test API</button>
                        <div id="result"></div>
                    </div>
                    <script>
                        function testAPI() {{
                            fetch('/api/status')
                                .then(response => response.json())
                                .then(data => {{
                                    document.getElementById('result').innerHTML = 
                                        '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                                }})
                                .catch(error => {{
                                    document.getElementById('result').innerHTML = 'Error: ' + error;
                                }});
                        }}
                    </script>
                </body>
                </html>
                """
        
        @app.route('/api/status', methods=['GET'])
        def get_status():
            try:
                logger.info("Status API called")
                status = ros2_interface.get_robot_status()
                logger.info(f"Status retrieved: {status}")
                return jsonify({
                    'success': True,
                    'data': status,
                    'mode': 'real_ros2' if is_real_ros2 else 'mock'
                })
            except Exception as e:
                logger.error(f"Status API error: {e}")
                logger.error(traceback.format_exc())
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @app.route('/api/move', methods=['POST'])
        def move_robot():
            try:
                logger.info("Move API called")
                data = request.get_json()
                logger.info(f"Move data received: {data}")
                
                if not data:
                    return jsonify({
                        'success': False,
                        'error': 'No JSON data received'
                    }), 400
                
                linear_x = data.get('linear_x', 0)
                angular_z = data.get('angular_z', 0)
                
                success = ros2_interface.move_robot(linear_x, angular_z)
                
                return jsonify({
                    'success': success,
                    'message': f'Robot moved: linear={linear_x}, angular={angular_z}'
                })
            except Exception as e:
                logger.error(f"Move API error: {e}")
                logger.error(traceback.format_exc())
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @app.route('/api/stop', methods=['POST'])
        def stop_robot():
            try:
                logger.info("Stop API called")
                success = ros2_interface.stop_robot()
                return jsonify({
                    'success': success,
                    'message': 'Robot stopped'
                })
            except Exception as e:
                logger.error(f"Stop API error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @app.route('/api/command', methods=['POST'])
        def send_command():
            try:
                logger.info("Command API called")
                data = request.get_json()
                
                if not data or 'command' not in data:
                    return jsonify({
                        'success': False,
                        'error': 'Command is required'
                    }), 400
                
                command = data['command']
                success = ros2_interface.send_custom_command(command)
                
                return jsonify({
                    'success': success,
                    'message': f'Command sent: {command}'
                })
            except Exception as e:
                logger.error(f"Command API error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @app.route('/api/debug', methods=['GET'])
        def debug_info():
            """デバッグ情報を返す"""
            try:
                return jsonify({
                    'success': True,
                    'debug_info': {
                        'ros2_available': is_real_ros2,
                        'python_version': sys.version,
                        'flask_version': Flask.__version__,
                        'current_directory': os.getcwd(),
                        'environment_variables': dict(os.environ),
                        'sys_path': sys.path
                    }
                })
            except Exception as e:
                logger.error(f"Debug API error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        logger.info("Flask app created successfully")
        return app
        
    except Exception as e:
        logger.error(f"Flask app creation failed: {e}")
        logger.error(traceback.format_exc())
        raise

def ros2_spin_thread(ros2_interface, is_real_ros2):
    """ROS2スピン処理"""
    if is_real_ros2:
        try:
            import rclpy
            logger.info("Starting ROS2 spin thread")
            rclpy.spin(ros2_interface)
        except Exception as e:
            logger.error(f"ROS2 spin error: {e}")
    else:
        logger.info("Mock mode - no ROS2 spinning needed")

def main():
    """メイン関数"""
    logger.info("Starting Flask ROS2 Robot Controller (Debug Mode)")
    
    try:
        # Step 1: ROS2インターフェース作成
        logger.info("Step 1: Creating ROS2 interface...")
        ros2_interface, is_real_ros2 = create_ros2_interface()
        logger.info(f"ROS2 interface created. Real ROS2: {is_real_ros2}")
        
        # Step 2: ROS2スピンスレッド開始
        if is_real_ros2:
            logger.info("Step 2: Starting ROS2 spin thread...")
            ros2_thread = threading.Thread(
                target=ros2_spin_thread,
                args=(ros2_interface, is_real_ros2),
                daemon=True
            )
            ros2_thread.start()
            logger.info("ROS2 spin thread started")
        
        # Step 3: Flaskアプリ作成
        logger.info("Step 3: Creating Flask app...")
        app = create_flask_app(ros2_interface, is_real_ros2)
        
        # Step 4: サーバー起動
        logger.info("Step 4: Starting Flask server...")
        print("\n" + "="*50)
        print("Flask ROS2 Robot Controller (Debug Mode)")
        print(f"ROS2 Mode: {'Real' if is_real_ros2 else 'Mock'}")
        print("Access: http://localhost:5000")
        print("Debug API: http://localhost:5000/api/debug")
        print("Status API: http://localhost:5000/api/status")
        print("="*50 + "\n")
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=False  # デバッグモードでリローダーを無効化
        )
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        print("\nShutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        print(f"\nFatal error: {e}")
        sys.exit(1)
    finally:
        try:
            if is_real_ros2 and 'ros2_interface' in locals():
                logger.info("Destroying ROS2 node...")
                ros2_interface.destroy_node()
                import rclpy
                rclpy.shutdown()
                logger.info("ROS2 shutdown complete")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

if __name__ == '__main__':
    main()