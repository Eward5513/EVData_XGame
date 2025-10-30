# 地理数据可视化平台

一个基于Cesium的地理数据可视化系统，支持车辆轨迹数据的实时可视化展示。

## 功能特性

- ✨ 基于Cesium的3D地理可视化
- 🚗 车辆轨迹数据展示
- 🎨 根据速度进行颜色编码
- 📊 实时数据统计信息
- 🔄 前后端分离架构
- 📱 响应式界面设计

## 技术栈

### 前端
- **Cesium 1.127** - 3D地理可视化框架
- **HTML5 + CSS3** - 现代Web界面
- **JavaScript ES6+** - 交互逻辑

### 后端
- **Python 3.7+** - 后端服务
- **Flask** - Web框架
- **Pandas** - 数据处理
- **Flask-CORS** - 跨域支持

## 快速开始

### 1. 后端服务启动

首先安装Python依赖：

```bash
cd visualization/backend
pip install -r requirements.txt
```

启动后端服务：

```bash
python server.py
```

服务将在 `http://127.0.0.1:5555` 启动

### 2. 前端页面访问

使用Web服务器打开前端页面（不能直接双击HTML文件，需要通过HTTP服务访问）：

```bash
# 方法1: 使用Python内置服务器
cd visualization/frontend
python -m http.server 8080

# 方法2: 使用Node.js http-server
npm install -g http-server
cd visualization/frontend
http-server -p 8080

# 方法3: 使用Live Server (VS Code扩展)
# 在VS Code中安装Live Server扩展，右键HTML文件选择"Open with Live Server"
```

然后访问: `http://localhost:8080/html/index.html`

## API接口文档

### 获取车辆数据
```
GET /api/vehicle/data?vehicle_id=1&limit=1000
```

**参数:**
- `vehicle_id`: 车辆ID (默认: 1)
- `limit`: 数据点限制 (可选)

**响应:**
```json
{
    "status": "success",
    "vehicle_id": 1,
    "total_points": 23479,
    "data": [
        {
            "vehicle_id": 1,
            "collectiontime": 1718787136000,
            "time_stamp": "08:52:16",
            "road_id": "A0003",
            "longitude": 123.146328,
            "latitude": 32.344848,
            "speed": 48.2,
            "acceleratorpedal": 13,
            "brakestatus": 0
        }
    ]
}
```

### 获取数据摘要
```
GET /api/vehicle/summary?vehicle_id=1
```

**响应:**
```json
{
    "status": "success",
    "vehicle_id": 1,
    "total_points": 23479,
    "time_range": {
        "start": "08:52:16",
        "end": "17:03:26"
    },
    "coordinate_bounds": {
        "longitude_min": 123.146328,
        "longitude_max": 123.151635,
        "latitude_min": 32.344848,
        "latitude_max": 32.345088
    },
    "speed_stats": {
        "min": 0.0,
        "max": 60.5,
        "avg": 25.3
    }
}
```

## 使用指南

1. **启动服务**: 确保后端服务在5555端口运行
2. **打开页面**: 通过HTTP服务器访问前端页面
3. **选择参数**: 在控制面板中选择车辆ID和数据点数量
4. **加载数据**: 点击"加载数据"按钮获取并显示轨迹
5. **查看详情**: 点击地图上的点查看详细信息
6. **清除数据**: 点击"清除数据"按钮清空地图

## 速度颜色编码

- 🟢 **绿色**: 0-10 km/h (慢速)
- 🟡 **黄色**: 10-30 km/h (中速)
- 🟠 **橙色**: 30-50 km/h (快速)
- 🔴 **红色**: 50+ km/h (高速)

## 目录结构

```
visualization/
├── backend/                 # 后端服务
│   ├── server.py           # Flask服务器
│   └── requirements.txt    # Python依赖
├── frontend/               # 前端文件
│   ├── html/
│   │   └── index.html     # 主页面
│   ├── js/
│   │   └── app.js         # 应用逻辑
│   ├── css/
│   │   └── style.css      # 样式文件
│   └── Cesium-1.127/      # Cesium框架
└── README.md              # 说明文档
```

## 数据格式要求

CSV文件需要包含以下字段：
- `vehicle_id`: 车辆ID
- `collectiontime`: 收集时间戳(毫秒)
- `time_stamp`: 时间字符串
- `road_id`: 道路ID
- `longitude`: 经度
- `latitude`: 纬度
- `speed`: 速度(km/h)
- `acceleratorpedal`: 油门踏板(%)
- `brakestatus`: 刹车状态(0/1)

## 故障排除

### 常见问题

1. **页面空白**: 确保通过HTTP服务器访问，不要直接打开HTML文件
2. **数据加载失败**: 检查后端服务是否正常运行在5555端口
3. **跨域错误**: 确保后端已启用CORS支持
4. **Cesium加载失败**: 检查Cesium文件路径是否正确

### 调试方法

1. 打开浏览器开发者工具查看控制台错误
2. 检查Network选项卡查看API请求状态
3. 确认后端服务日志输出
4. 验证CSV文件路径和格式

## 扩展功能

### 可添加的功能
- 时间轴播放
- 多车辆对比
- 路况热力图
- 实时数据流
- 数据导出功能
- 路径规划

### 性能优化
- 数据分页加载
- 点聚合显示
- 瓦片缓存
- 后端数据缓存

## 许可证

本项目仅供学习和研究使用。

## 联系支持

如有问题或建议，请通过Issue反馈。
