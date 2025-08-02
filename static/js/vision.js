const API_BASE = '';
let availableCameras = {};
let currentCamera = null;
let updateInterval = null;
let cameraUpdateInterval = null;

// 初期化
document.addEventListener('DOMContentLoaded', function() {
    console.log('Vision system with camera switching initialized');
    initializeSystem();
});

// システム初期化
async function initializeSystem() {
    try {
        await loadCameras();
        startStatusUpdates();
        updateConnectionStatus();
        setupEventListeners();
    } catch (error) {
        console.error('System initialization failed:', error);
        showAlert('システム初期化に失敗しました', 'error');
    }
}

// イベントリスナー設定
function setupEventListeners() {
    // ビデオストリームエラーハンドリング
    const videoStream = document.getElementById('video-stream');
    if (videoStream) {
        videoStream.addEventListener('error', handleVideoError);
        videoStream.addEventListener('load', handleVideoLoad);
    }

    // カメラ選択ドロップダウン
    const cameraSelect = document.getElementById('camera-select');
    if (cameraSelect) {
        cameraSelect.addEventListener('change', handleCameraChange);
    }

    // リサイズ対応
    window.addEventListener('resize', handleResize);
}

// カメラ管理機能
async function loadCameras() {
    try {
        showAlert('カメラ情報を読み込み中...', 'info');
        
        const response = await fetch(`${API_BASE}/api/vision/cameras`);
        const data = await response.json();
        
        if (data.success) {
            availableCameras = data.cameras;
            currentCamera = data.current_camera;
            updateCameraUI();
            
            console.log('Cameras loaded:', Object.keys(availableCameras));
            showAlert(`${Object.keys(availableCameras).length}台のカメラが利用可能です`, 'success');
        } else {
            throw new Error(data.error || 'カメラ情報の取得に失敗しました');
        }
    } catch (error) {
        console.error('Camera loading error:', error);
        showAlert('カメラ情報の取得でエラーが発生しました: ' + error.message, 'error');
        
        // フォールバック: 空のカメラリストで初期化
        availableCameras = {};
        updateCameraUI();
    }
}

async function rescanCameras() {
    try {
        showAlert('カメラを再スキャンしています...', 'info');
        
        // スキャン中の状態を表示
        const cameraSelect = document.getElementById('camera-select');
        if (cameraSelect) {
            cameraSelect.disabled = true;
            cameraSelect.innerHTML = '<option>スキャン中...</option>';
        }
        
        const response = await fetch(`${API_BASE}/api/vision/rescan_cameras`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            availableCameras = data.cameras;
            updateCameraUI();
            showAlert(`カメラスキャンが完了しました (${Object.keys(availableCameras).length}台発見)`, 'success');
        } else {
            throw new Error(data.error || 'カメラスキャンに失敗しました');
        }
    } catch (error) {
        console.error('Camera rescan error:', error);
        showAlert('カメラスキャンでエラーが発生しました: ' + error.message, 'error');
    } finally {
        // スキャン完了後の状態復元
        const cameraSelect = document.getElementById('camera-select');
        if (cameraSelect) {
            cameraSelect.disabled = false;
        }
    }
}

async function handleCameraChange() {
    const cameraSelect = document.getElementById('camera-select');
    const selectedCameraId = parseInt(cameraSelect.value);
    
    if (isNaN(selectedCameraId)) {
        return;
    }

    // 既に選択されているカメラの場合は何もしない
    if (currentCamera && currentCamera.id == selectedCameraId) {
        return;
    }

    try {
        showAlert(`カメラ ${selectedCameraId} に切り替え中...`, 'info');
        
        // UI要素を無効化
        cameraSelect.disabled = true;
        
        const response = await fetch(`${API_BASE}/api/vision/switch_camera`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                camera_id: selectedCameraId
            })
        });

        const data = await response.json();
        
        if (data.success) {
            currentCamera = data.current_camera;
            updateCameraUI();
            refreshVideoStream();
            showAlert(`カメラ ${selectedCameraId} に切り替えました`, 'success');
            
            // 切り替え後に状態を更新
            setTimeout(() => {
                updateVisionStatus();
            }, 1000);
        } else {
            throw new Error(data.error || `カメラ ${selectedCameraId} への切り替えに失敗しました`);
        }
    } catch (error) {
        console.error('Camera switch error:', error);
        showAlert('カメラ切り替えでエラーが発生しました: ' + error.message, 'error');
        
        // エラー時は元のカメラに戻す
        if (currentCamera) {
            cameraSelect.value = currentCamera.id;
        }
    } finally {
        // UI要素を有効化
        cameraSelect.disabled = false;
    }
}

function refreshVideoStream() {
    const videoStream = document.getElementById('video-stream');
    if (videoStream) {
        // キャッシュを回避してビデオストリームを更新
        const timestamp = new Date().getTime();
        videoStream.src = `${API_BASE}/api/vision/video_feed?t=${timestamp}`;
    }
}

function updateCameraUI() {
    updateCameraDropdown();
    updateCurrentCameraInfo();
    updateCameraList();
}

function updateCameraDropdown() {
    const cameraSelect = document.getElementById('camera-select');
    if (!cameraSelect) return;
    
    cameraSelect.innerHTML = '';
    
    if (Object.keys(availableCameras).length === 0) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'カメラが見つかりません';
        option.disabled = true;
        cameraSelect.appendChild(option);
        cameraSelect.disabled = true;
        return;
    }
    
    cameraSelect.disabled = false;
    
    for (const [cameraId, cameraInfo] of Object.entries(availableCameras)) {
        const option = document.createElement('option');
        option.value = cameraId;
        option.textContent = `${cameraInfo.name} (${cameraInfo.width}x${cameraInfo.height})`;
        
        if (currentCamera && currentCamera.id == cameraId) {
            option.selected = true;
        }
        
        cameraSelect.appendChild(option);
    }
}

function updateCurrentCameraInfo() {
    const currentCameraInfo = document.getElementById('current-camera-info');
    if (!currentCameraInfo) return;
    
    if (currentCamera && currentCamera.name) {
        const resolution = currentCamera.width && currentCamera.height ? 
            `(${currentCamera.width}x${currentCamera.height})` : '';
        currentCameraInfo.textContent = `${currentCamera.name} ${resolution}`;
    } else {
        currentCameraInfo.textContent = 'カメラ情報なし';
    }
}

function updateCameraList() {
    const container = document.getElementById('camera-list-container');
    if (!container) return;
    
    container.innerHTML = '';
    
    if (Object.keys(availableCameras).length === 0) {
        container.innerHTML = '<p style="color: #a0aec0; text-align: center;">カメラが見つかりません</p>';
        return;
    }
    
    for (const [cameraId, cameraInfo] of Object.entries(availableCameras)) {
        const cameraItem = document.createElement('div');
        cameraItem.className = 'camera-item';
        
        if (currentCamera && currentCamera.id == cameraId) {
            cameraItem.classList.add('active');
        }
        
        cameraItem.onclick = () => {
            const cameraSelect = document.getElementById('camera-select');
            if (cameraSelect && !cameraSelect.disabled) {
                cameraSelect.value = cameraId;
                handleCameraChange();
            }
        };
        
        const statusText = cameraInfo.status === 'active' ? '使用中' : '利用可能';
        const statusClass = cameraInfo.status === 'active' ? 'color: #68d391; font-weight: bold;' : '';
        
        cameraItem.innerHTML = `
            <div>
                <div class="camera-name">${cameraInfo.name}</div>
                <div class="camera-details">${cameraInfo.width}x${cameraInfo.height} @ ${cameraInfo.fps}fps</div>
            </div>
            <span class="camera-status" style="${statusClass}">${statusText}</span>
        `;
        
        container.appendChild(cameraItem);
    }
}

// 状態更新機能
function startStatusUpdates() {
    updateVisionStatus();
    updateDetectionHistory();
    
    // 定期更新を設定
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    updateInterval = setInterval(() => {
        updateVisionStatus();
        updateDetectionHistory();
    }, 1000);
    
    if (cameraUpdateInterval) {
        clearInterval(cameraUpdateInterval);
    }
    cameraUpdateInterval = setInterval(() => {
        updateConnectionStatus();
        // カメラ状態も定期的に更新（頻度は低めに）
        loadCameras();
    }, 10000);
}

async function updateConnectionStatus() {
    const statusEl = document.getElementById('connection-text');
    const dotEl = document.getElementById('connection-dot');
    const navStatusEl = document.getElementById('nav-connection-status');
    const navDotEl = document.getElementById('nav-connection-dot');
    
    try {
        const response = await fetch(`${API_BASE}/api/vision/status`);
        const data = await response.json();
        
        if (data.success && data.active) {
            const statusText = 'ビジョンシステム接続中';
            if (statusEl) statusEl.textContent = statusText;
            if (navStatusEl) navStatusEl.textContent = statusText;
            
            if (dotEl) {
                dotEl.style.background = '#68d391';
                dotEl.classList.remove('disconnected');
            }
            if (navDotEl) {
                navDotEl.style.background = '#68d391';
            }
        } else {
            throw new Error('Vision system not active');
        }
    } catch (error) {
        const statusText = 'ビジョンシステム切断';
        if (statusEl) statusEl.textContent = statusText;
        if (navStatusEl) navStatusEl.textContent = statusText;
        
        if (dotEl) {
            dotEl.style.background = '#e53e3e';
            dotEl.classList.add('disconnected');
        }
        if (navDotEl) {
            navDotEl.style.background = '#e53e3e';
        }
    }
}

async function updateVisionStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/vision/status`);
        const data = await response.json();
        
        if (data.success) {
            const stats = data.stats;
            const detections = data.detections;
            
            // 統計情報を更新
            updateElement('fps-value', stats.fps.toFixed(1));
            updateElement('person-count', data.current_detections);
            updateElement('total-frames', stats.total_frames);
            
            // 稼働時間を更新
            const uptime = Math.floor(stats.uptime);
            const hours = Math.floor(uptime / 3600);
            const minutes = Math.floor((uptime % 3600) / 60);
            const seconds = uptime % 60;
            const uptimeText = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            updateElement('uptime', uptimeText);
            
            // 現在の検出結果を更新
            updateCurrentDetections(detections);
            
            // 行動統計を更新
            updateDirectionStats(detections);
            
            // カメラ情報が含まれている場合は更新
            if (stats.current_camera) {
                currentCamera = stats.current_camera;
                updateCurrentCameraInfo();
            }
        }
    } catch (error) {
        console.error('Vision status update failed:', error);
    }
}

function updateCurrentDetections(detections) {
    const container = document.getElementById('current-detections');
    if (!container) return;
    
    if (!detections || detections.length === 0) {
        container.innerHTML = '<p style="color: #a0aec0; text-align: center;">検出結果なし</p>';
        return;
    }
    
    container.innerHTML = '';
    
    detections.forEach(detection => {
        const detectionItem = document.createElement('div');
        detectionItem.className = 'detection-item';
        
        const directionMap = {
            "moving_right": "右移動", "moving_left": "左移動", 
            "moving_toward": "前進", "moving_away": "後退", 
            "stationary": "静止", "unknown": "不明"
        };
        
        const directionText = directionMap[detection.direction] || detection.direction;
        const confidence = (detection.confidence * 100).toFixed(1);
        
        detectionItem.innerHTML = `
            <div class="person-id">${detection.person_id}</div>
            <div class="direction">方向: ${directionText}</div>
            <div class="confidence">信頼度: ${confidence}%</div>
        `;
        
        container.appendChild(detectionItem);
    });
}

function updateDirectionStats(detections) {
    const stats = {
        'moving_toward': 0,
        'moving_away': 0,
        'moving_left': 0,
        'moving_right': 0,
        'stationary': 0
    };
    
    detections.forEach(detection => {
        if (stats.hasOwnProperty(detection.direction)) {
            stats[detection.direction]++;
        }
    });
    
    updateElement('count-toward', stats.moving_toward);
    updateElement('count-away', stats.moving_away);
    updateElement('count-left', stats.moving_left);
    updateElement('count-right', stats.moving_right);
    updateElement('count-stationary', stats.stationary);
}

async function updateDetectionHistory() {
    try {
        const response = await fetch(`${API_BASE}/api/vision/history`);
        const data = await response.json();
        
        if (data.success) {
            const history = data.history;
            const container = document.getElementById('detection-history');
            if (!container) return;
            
            if (!history || history.length === 0) {
                container.innerHTML = '<p style="color: #a0aec0; text-align: center;">検出履歴なし</p>';
                return;
            }
            
            container.innerHTML = '';
            
            // 最新の10件を表示
            const recentHistory = history.slice(-10).reverse();
            
            recentHistory.forEach(record => {
                const historyItem = document.createElement('div');
                historyItem.className = 'history-item';
                
                const timestamp = new Date(record.timestamp).toLocaleTimeString();
                const detectionCount = record.detections.length;
                const cameraInfo = record.camera_id !== undefined ? `カメラ ${record.camera_id}` : '';
                
                const directionMap = {
                    "moving_right": "右移動", "moving_left": "左移動", 
                    "moving_toward": "前進", "moving_away": "後退", 
                    "stationary": "静止", "unknown": "不明"
                };
                
                const detectionsText = record.detections.map(d => {
                    return `${d.person_id}: ${directionMap[d.direction] || d.direction}`;
                }).join(', ');
                
                historyItem.innerHTML = `
                    <div class="history-timestamp">${timestamp} - ${cameraInfo} - フレーム ${record.frame_number}</div>
                    <div class="history-detections">${detectionCount}人検出: ${detectionsText}</div>
                `;
                
                container.appendChild(historyItem);
            });
        }
    } catch (error) {
        console.error('History update failed:', error);
    }
}

// ユーティリティ関数
function updateElement(id, text) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = text;
    }
}

function showAlert(message, type = 'success') {
    const alertContainer = document.getElementById('alert-container');
    if (!alertContainer) {
        console.log(`${type.toUpperCase()}: ${message}`);
        return;
    }
    
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;
    
    alertContainer.appendChild(alertDiv);
    
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// イベントハンドラー
function handleVideoError() {
    const placeholder = document.getElementById('video-placeholder');
    if (placeholder) {
        placeholder.style.display = 'block';
        placeholder.textContent = 'ビデオストリーム接続エラー';
    }
    console.error('Video stream error');
}

function handleVideoLoad() {
    const placeholder = document.getElementById('video-placeholder');
    if (placeholder) {
        placeholder.style.display = 'none';
    }
}

function handleResize() {
    // リサイズ時の処理（必要に応じて実装）
    console.log('Window resized');
}

function toggleMobileMenu() {
    const navMenu = document.getElementById('nav-menu');
    if (navMenu) {
        navMenu.classList.toggle('active');
    }
}

// グローバルスコープに公開（HTMLから呼び出し可能にする）
window.rescanCameras = rescanCameras;
window.handleCameraChange = handleCameraChange;
window.toggleMobileMenu = toggleMobileMenu;
window.refreshVideoStream = refreshVideoStream;

// ページ離脱時のクリーンアップ
window.addEventListener('beforeunload', function() {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    if (cameraUpdateInterval) {
        clearInterval(cameraUpdateInterval);
    }
});

// デバッグ用関数
function debugCameraInfo() {
    console.log('Available Cameras:', availableCameras);
    console.log('Current Camera:', currentCamera);
    console.log('Update Intervals:', { updateInterval, cameraUpdateInterval });
}

window.debugCameraInfo = debugCameraInfo;