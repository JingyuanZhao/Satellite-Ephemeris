# 卫星星历计算器 - Android APK 构建说明

## 项目结构

```
satellite_ephemeris/
├── main.py                 # Kivy版本主程序入口
├── buildozer.spec          # Buildozer配置文件
├── sat_gui.py             # 原始tkinter版本（桌面版）
├── .github/
│   └── workflows/
│       └── build-apk.yml  # GitHub Actions工作流
└── README_APK.md          # 本文件
```

## 使用 GitHub Actions 自动构建 APK

### 步骤 1: 推送代码到 GitHub

1. 在 GitHub 上创建一个新仓库
2. 将本地代码推送到 GitHub:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/你的用户名/仓库名.git
git push -u origin main
```

### 步骤 2: 触发构建

推送到 main 分支后，GitHub Actions 会自动开始构建 APK。

你也可以手动触发：
1. 进入 GitHub 仓库页面
2. 点击 "Actions" 标签
3. 选择 "Build Android APK" 工作流
4. 点击 "Run workflow" 手动触发

### 步骤 3: 下载 APK

构建完成后（约 10-20 分钟）：
1. 进入 GitHub 仓库的 "Actions" 页面
2. 点击最新的工作流运行记录
3. 在 "Artifacts" 部分下载 `satellite-tracker-apk`
4. 解压下载的文件即可获得 APK

## 本地构建（需要 Linux 环境）

如果你希望在本地构建 APK，需要：

### 系统要求
- Ubuntu 20.04+ 或其他 Linux 发行版
- Python 3.8+
- 至少 4GB 可用磁盘空间

### 安装依赖
```bash
# 安装系统依赖
sudo apt-get update
sudo apt-get install -y \
    python3-pip build-essential git ffmpeg \
    libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
    libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev \
    zlib1g-dev libgstreamer1.0 gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good autoconf automake libtool pkg-config

# 安装 Python 依赖
pip3 install buildozer cython
```

### 构建 APK
```bash
buildozer -v android debug
```

构建完成后，APK 文件位于 `bin/` 目录下。

## 功能说明

### Kivy 版本功能
- 参数输入（观测地点、卫星信息、计算参数）
- TLE 数据获取（开发中）
- 星历计算（开发中）
- 结果显示

### 与桌面版的区别
Kivy 版本是为 Android 优化的简化版本，主要特点：
- 使用 Kivy 框架替代 tkinter
- 适配触屏操作
- 响应式布局
- 部分高级功能可能需要后续开发

## 注意事项

1. **首次构建时间较长**：GitHub Actions 首次构建可能需要 15-30 分钟，因为需要下载和配置 Android SDK/NDK。

2. **缓存机制**：后续构建会利用缓存，时间缩短到 5-10 分钟。

3. **APK 大小**：生成的 APK 文件约 20-50MB，包含 Python 运行时和所有依赖库。

4. **权限**：APK 需要以下权限：
   - INTERNET（网络访问）
   - ACCESS_NETWORK_STATE（网络状态）

## 故障排除

### 构建失败
1. 检查 `buildozer.spec` 配置是否正确
2. 查看 GitHub Actions 日志获取详细错误信息
3. 确保所有依赖库都在 `requirements` 中列出

### 运行时崩溃
1. 检查是否缺少必要的权限
2. 查看 Android 设备日志：`adb logcat`
3. 确保 Kivy 版本与 Android 版本兼容

## 更新日志

### v2.0
- 初始 Android 版本
- 基础 UI 框架
- 参数输入界面

## 许可证

与主项目相同
