// robot_controller.js
// ROS2 Robot Controller のメインJavaScriptファイル

// 設定
const API_BASE = '';
let statusUpdateInterval;

// 初期化
document.addEventListener('DOMContentLoaded', function() {
    console.log('Robot Controller initialized');
    startStatusUpdates();
    updateConnectionStatus();
});

// アラート表示
function showAlert(message, type = 'success') {
    const alertContainer = document.getElementById('alert-container');
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;
    
    alertContainer.appendChild(alertDiv);
    
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 3000);
}

// 接続状態更新
function updateConnectionStatus() {
    const statusEl = document.getElementById('connection-status');
    
    fetch(`${API_BASE}/api/status`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                statusEl.textContent = 'ROS2接続中';
                statusEl.className = 'connection-status connected';
            } else {
                statusEl.textContent = '接続エラー';
                statusEl.className = 'connection-status disconnected';
            }
        })
        .catch(error => {
            console.error('Connection status error:', error);
            statusEl.textContent = '切断';
            statusEl.className = 'connection-status disconnected';
        });
}

// 状態更新を開始
function startStatusUpdates() {
    updateStatus();
    statusUpdateInterval = setInterval(updateStatus, 1000);
    setInterval(updateConnectionStatus, 5000);
}

// ロボット状態を更新
async function updateStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        const data = await response.json();
        
        if (data.success) {
            const status = data.data;
            
            // 状態表示を更新
            document.getElementById('linear-velocity').textContent = 
                `${status.velocity.linear.toFixed(2)} m/s`;
            document.getElementById('angular-velocity').textContent = 
                `${status.velocity.angular.toFixed(2)} rad/s`;
            document.getElementById('min-distance').textContent = 
                `${status.laser_data.min_distance.toFixed(2)} m`;
            
            // 最終更新時刻を表示
            const lastUpdate = new Date(status.last_update * 1000);
            document.getElementById('last-update').textContent = 
                lastUpdate.toLocaleTimeString();
        }
    } catch (error) {
        console.error('Status update failed:', error);
    }
}

// 速度取得関数
function getLinearSpeed() {
    const value = parseFloat(document.getElementById('linear-speed').value);
    return isNaN(value) ? 0 : value;
}

function getAngularSpeed() {
    const value = parseFloat(document.getElementById('angular-speed').value);
    return isNaN(value) ? 0 : value;
}

// ロボット移動
async function moveRobot(linearX, angularZ) {
    try {
        console.log(`Moving robot: linear=${linearX}, angular=${angularZ}`);
        
        const response = await fetch(`${API_BASE}/api/move`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                linear_x: linearX,
                angular_z: angularZ
            })
        });

        const data = await response.json();
        
        if (data.success) {
            showAlert(data.message, 'success');
        } else {
            showAlert(data.error || 'Movement failed', 'error');
        }
    } catch (error) {
        console.error('Move robot error:', error);
        showAlert('通信エラー: ' + error.message, 'error');
    }
}

// ロボット停止
async function stopRobot() {
    try {
        console.log('Stopping robot');
        
        const response = await fetch(`${API_BASE}/api/stop`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        const data = await response.json();
        
        if (data.success) {
            showAlert('ロボットを停止しました', 'success');
        } else {
            showAlert(data.error || 'Stop failed', 'error');
        }
    } catch (error) {
        console.error('Stop robot error:', error);
        showAlert('通信エラー: ' + error.message, 'error');
    }
}

// カスタムコマンド送信
async function sendCustomCommand() {
    const commandInput = document.getElementById('command-input');
    const command = commandInput.value.trim();
    
    if (!command) {
        showAlert('コマンドを入力してください', 'error');
        return;
    }

    try {
        console.log(`Sending command: ${command}`);
        
        const response = await fetch(`${API_BASE}/api/command`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                command: command
            })
        });

        const data = await response.json();
        
        if (data.success) {
            showAlert(data.message, 'success');
            commandInput.value = '';
        } else {
            showAlert(data.error || 'Command failed', 'error');
        }
    } catch (error) {
        console.error('Send command error:', error);
        showAlert('通信エラー: ' + error.message, 'error');
    }
}

// プリセットコマンド送信
async function sendPresetCommand(command) {
    document.getElementById('command-input').value = command;
    await sendCustomCommand();
}

// Enterキーでコマンド送信
function handleCommandEnter(event) {
    if (event.key === 'Enter') {
        sendCustomCommand();
    }
}

// キーボード制御
document.addEventListener('keydown', function(event) {
    // テキスト入力中は無視
    if (document.activeElement.type === 'text' || 
        document.activeElement.type === 'number' ||
        document.activeElement.tagName === 'TEXTAREA') {
        return;
    }
    
    let handled = false;
    
    switch(event.key.toLowerCase()) {
        case 'arrowup':
        case 'w':
            event.preventDefault();
            moveRobot(getLinearSpeed(), 0);
            handled = true;
            break;
            
        case 'arrowdown':
        case 's':
            event.preventDefault();
            moveRobot(-getLinearSpeed(), 0);
            handled = true;
            break;
            
        case 'arrowleft':
        case 'a':
            event.preventDefault();
            moveRobot(0, getAngularSpeed());
            handled = true;
            break;
            
        case 'arrowright':
        case 'd':
            event.preventDefault();
            moveRobot(0, -getAngularSpeed());
            handled = true;
            break;
            
        case ' ':
            event.preventDefault();
            stopRobot();
            handled = true;
            break;
    }
    
    if (handled) {
        console.log(`Keyboard control: ${event.key}`);
    }
});

// ページ離脱時の処理
window.addEventListener('beforeunload', function() {
    if (statusUpdateInterval) {
        clearInterval(statusUpdateInterval);
    }
});

// デバッグ用関数
function debugInfo() {
    console.log('Robot Controller Debug Info:');
    console.log('API Base:', API_BASE);
    console.log('Linear Speed:', getLinearSpeed());
    console.log('Angular Speed:', getAngularSpeed());
}

// グローバルスコープに公開（HTMLから呼び出し可能にする）
window.moveRobot = moveRobot;
window.stopRobot = stopRobot;
window.sendCustomCommand = sendCustomCommand;
window.sendPresetCommand = sendPresetCommand;
window.handleCommandEnter = handleCommandEnter;
window.debugInfo = debugInfo;