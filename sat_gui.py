import tkinter as tk
from tkinter import ttk, messagebox
import requests
import math
import numpy as np
import re
from datetime import datetime, timedelta
from skyfield.api import load, wgs84, EarthSatellite
from skyfield import almanac
import sys
import json
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.dates as mdates

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 配置区 ======================
# 默认本征星等（当未手动设置且网络获取失败时使用）
DEFAULT_STD_MAG = None   # 常用的默认本征星等值

# Earth equatorial radius used by MPC (km)
EARTH_RADIUS_KM = 6378.14

# 配置文件路径
CONFIG_FILE = "mpc_codes.json"

# ====================================================

def load_config():
    """加载MPC代码配置文件"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # 确保配置结构完整
                if "manual_location" not in config:
                    config["manual_location"] = {
                        "latitude": 39.9,
                        "longitude": 116.4,
                        "height_m": 0
                    }
                if "saved_locations" not in config:
                    config["saved_locations"] = {}
                return config
        except Exception as e:
            print(f"加载配置文件失败: {e}")
    # 返回默认配置
    return {
        "manual_location": {
            "latitude": 39.9,
            "longitude": 116.4,
            "height_m": 0
        },
        "saved_locations": {}
    }

def save_config(config):
    """保存MPC代码配置文件"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"保存配置文件失败: {e}")
        return False

def get_observer_from_mpc(obscode, builtin_codes=None):
    """从HTML页面读取MPC天文台代码，返回Skyfield观测点和数据字典"""
    obs_code_upper = obscode.strip().upper()
    
    # 使用传入的内置代码列表或默认列表
    if builtin_codes is None:
        builtin_codes = {}
    
    if obs_code_upper in builtin_codes:
        data = builtin_codes[obs_code_upper]
        lon = float(data["longitude"])
        lat = float(data["latitude"])
        height_m = float(data["height_m"])  # 使用内置的高度
        name = data.get("name", "未知地点")  # 获取地点名称
        lat_dms = decimal_to_dms(lat, is_latitude=True)
        lon_dms = decimal_to_dms(lon, is_latitude=False)
        return wgs84.latlon(lat, lon, height_m), f"使用内置数据读取MPC {obs_code_upper}：名称: {name}，纬度 {lat_dms}，经度 {lon_dms}", None
    
    # 如果内置代码中没有，从 HTML 页面提取数据
    html_url = "https://www.minorplanetcenter.net/iau/lists/ObsCodesF.html"
    try:
        resp = requests.get(html_url, timeout=15)
        resp.raise_for_status()
        content = resp.text
        
        # 查找包含 MPC 代码的行
        lines = content.split('\n')
        for line in lines:
            # 查找以 MPC 代码开头的行
            stripped_line = line.strip()
            if stripped_line.startswith(obs_code_upper):
                # 解析行数据
                parts = stripped_line.split()
                if len(parts) >= 4:
                    try:
                        code = parts[0]
                        if code == obs_code_upper:
                            lon = float(parts[1])
                            rhoc = float(parts[2])
                            rhos = float(parts[3])
                            rho = math.sqrt(rhoc**2 + rhos**2)
                            phi_gc = math.degrees(math.atan2(rhos, rhoc))
                            height_m = 0  # 统一使用0米高度
                            # 处理名称部分，移除HTML链接标签
                            name_parts = parts[4:] if len(parts) > 4 else []
                            name = ' '.join(name_parts)
                            # 移除HTML链接标签，只保留纯文本
                            name = re.sub(r'<a[^>]*>(.*?)</a>', r'\1', name)
                            name = name.strip()
                            # 返回数据字典，用于保存到内置列表
                            data_dict = {
                                "longitude": lon,
                                "latitude": phi_gc,
                                "height_m": height_m,
                                "name": name
                            }
                            lat_dms = decimal_to_dms(phi_gc, is_latitude=True)
                            lon_dms = decimal_to_dms(lon, is_latitude=False)
                            return wgs84.latlon(phi_gc, lon, height_m), f"已从HTML页面读取 {code}：名称: {name}，纬度 {lat_dms}，经度 {lon_dms}", data_dict
                    except ValueError:
                        continue
        return None, "在HTML页面中未找到该MPC代码", None
    except Exception as e:
        return None, f"从HTML页面获取数据失败: {e}", None

def compute_phase_angle(t, sat, observer, eph):
    """计算相位角（°）"""
    earth = eph["earth"]
    sun = eph["sun"]
    sat_gc = sat.at(t)
    obs_gc = observer.at(t)
    sun_from_earth = earth.at(t).observe(sun)
    vec_sat_to_sun = sun_from_earth.position.km - sat_gc.position.km
    vec_sat_to_obs = obs_gc.position.km - sat_gc.position.km
    dot = np.dot(vec_sat_to_sun, vec_sat_to_obs)
    norm_s = np.linalg.norm(vec_sat_to_sun)
    norm_o = np.linalg.norm(vec_sat_to_obs)
    cos_phi = np.clip(dot / (norm_s * norm_o), -1.0, 1.0)
    return math.degrees(math.acos(cos_phi))

def compute_earth_shadow_factor(t, sat, eph):
    """
    计算地球影子对卫星亮度的影响因子
    
    返回:
        shadow_factor: 亮度因子 (0.0 - 1.0)
                       1.0 = 完全光照
                       0.0 = 完全在本影中
                       0.0-1.0 = 在半影中
    """
    earth = eph["earth"]
    sun = eph["sun"]
    
    # 获取太阳、地球和卫星的位置
    sun_pos = sun.at(t).position.km
    earth_pos = earth.at(t).position.km
    sat_pos = sat.at(t).position.km
    
    # 太阳半径 (km)
    SUN_RADIUS_KM = 696340.0
    # 地球半径 (km)
    EARTH_RADIUS_KM = 6378.14
    
    # 向量计算
    # 太阳到地球的向量
    sun_to_earth = earth_pos - sun_pos
    # 太阳到卫星的向量
    sun_to_sat = sat_pos - sun_pos
    # 地球到卫星的向量
    earth_to_sat = sat_pos - earth_pos
    
    # 距离
    dist_sun_earth = np.linalg.norm(sun_to_earth)
    dist_sun_sat = np.linalg.norm(sun_to_sat)
    dist_earth_sat = np.linalg.norm(earth_to_sat)
    
    # 计算太阳-地球-卫星的夹角（太阳方向和地球-卫星方向的夹角）
    cos_angle = np.dot(sun_to_earth, earth_to_sat) / (dist_sun_earth * dist_earth_sat)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_sun_earth_sat = math.degrees(math.acos(cos_angle))
    
    # 计算太阳-卫星-地球的夹角
    cos_angle2 = np.dot(-sun_to_sat, earth_to_sat) / (dist_sun_sat * dist_earth_sat)
    cos_angle2 = np.clip(cos_angle2, -1.0, 1.0)
    angle_sat_earth_sun = math.degrees(math.acos(cos_angle2))
    
    # 如果太阳-地球-卫星夹角 > 90度，卫星在地球背向太阳的一侧
    if angle_sun_earth_sat <= 90:
        # 卫星在太阳那一侧，不可能在地影中
        return 1.0
    
    # 计算卫星视角中地球遮挡太阳的角半径
    # 使用余弦定理：地球半径 / 卫星到地球中心的距离
    earth_angular_radius = math.asin(EARTH_RADIUS_KM / dist_earth_sat)
    
    # 计算太阳的角半径
    sun_angular_radius = math.asin(SUN_RADIUS_KM / dist_sun_sat)
    
    # 计算卫星看到地球遮挡太阳的程度
    # 本影：地球完全遮挡太阳
    # 角度差 = 太阳-卫星-地球夹角 - 太阳角半径
    # 如果这个角度差 < 地球角半径，说明卫星在本影中
    
    # 修正后的几何计算
    # 在卫星位置，太阳和地球的张角
    sun_earth_angle = angle_sat_earth_sun
    
    # 本影半径（从太阳边缘到地球中心连线形成的角度）
    # 太阳的半角 - 地球的半角
    umbra_angular_radius = sun_angular_radius - earth_angular_radius
    
    # 半影半径
    penumbra_angular_radius = sun_angular_radius + earth_angular_radius
    
    # 判断卫星是否在地影中
    if sun_earth_angle < umbra_angular_radius:
        # 在本影中
        return 0.0
    elif sun_earth_angle > penumbra_angular_radius:
        # 在半影外，完全光照
        return 1.0
    else:
        # 在半影中
        # 从本影边缘(0)到半影边缘(1)线性变化
        shadow_factor = (sun_earth_angle - umbra_angular_radius) / (penumbra_angular_radius - umbra_angular_radius)
        return np.clip(shadow_factor, 0.0, 1.0)

def compute_satellite_angular_distance(t, sat, observer, eph):
    """
    计算卫星与太阳、月亮的角距离。
    
    参数:
        t: Skyfield时间对象
        sat: 卫星对象
        observer: 观测者位置（通过 earth + topos 创建）
        eph: 星历数据
    
    返回:
        tuple: (卫星与太阳角距离（度）, 卫星与月亮角距离（度）)
    """
    earth = eph["earth"]
    sun = eph["sun"]
    moon = eph["moon"]
    
    # 获取卫星位置（相对于观测者）
    sat_diff = sat - observer
    sat_pos = sat_diff.at(t)
    vector_sat = sat_pos.position.km
    norm_sat = np.linalg.norm(vector_sat)
    if norm_sat < 1e-6:
        return 0.0, 0.0
    unit_sat = vector_sat / norm_sat
    
    # 从地球中心观测太阳和月亮（视位置）
    # 使用 earth.at(t).observe() 获取天体方向
    sun_obs = earth.at(t).observe(sun).apparent()
    vector_sun = sun_obs.position.km
    unit_sun = vector_sun / np.linalg.norm(vector_sun)
    
    moon_obs = earth.at(t).observe(moon).apparent()
    vector_moon = moon_obs.position.km
    unit_moon = vector_moon / np.linalg.norm(vector_moon)
    
    # 计算与太阳的角距离
    cos_sun = np.clip(np.dot(unit_sat, unit_sun), -1.0, 1.0)
    angle_sun_deg = np.degrees(np.arccos(cos_sun))
    
    # 计算与月亮的角距离
    cos_moon = np.clip(np.dot(unit_sat, unit_moon), -1.0, 1.0)
    angle_moon_deg = np.degrees(np.arccos(cos_moon))
    
    return angle_sun_deg, angle_moon_deg

def format_radec(ra, dec):
    """格式化赤道坐标为 hms / dms"""
    ra_hms = ra.hms()
    dec_dms = dec.dms()
    ra_str = f"{int(ra_hms[0]):02d}h {int(ra_hms[1]):02d}m {ra_hms[2]:05.2f}s"
    sign = "+" if dec_dms[0] >= 0 else "-"
    dec_str = f"{sign}{int(abs(dec_dms[0])):02d}° {int(abs(dec_dms[1])):02d}' {abs(dec_dms[2]):04.1f}\""
    return ra_str, dec_str

def decimal_to_dms(degrees, is_latitude=True):
    """将十进制度数转换为度分秒格式，使用N/S/E/W表示方向"""
    if is_latitude:
        direction = "N" if degrees >= 0 else "S"
    else:
        direction = "E" if degrees >= 0 else "W"
    degrees = abs(degrees)
    d = int(degrees)
    m_float = (degrees - d) * 60
    m = int(m_float)
    s = (m_float - m) * 60
    return f"{d}°{m}'{s:.2f}\"{direction}"

def fetch_tle_from_celestrak(sat_id):
    """
    从CelesTrak获取TLE数据
    
    参数:
        sat_id: 卫星NORAD ID (如 25544 表示ISS)
    
    返回:
        (tle_line1, tle_line2, name, message)
        如果失败则返回 (None, None, None, error_type)
        error_type: "not_found" (卫星不存在), "timeout" (超时), "other" (其他错误)
    """
    try:
        # CelesTrak API URL
        url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={sat_id}&FORMAT=tle"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # 解析TLE数据
        lines = response.text.strip().split('\n')
        
        # 过滤空行
        lines = [line.strip() for line in lines if line.strip()]
        
        if len(lines) >= 2:
            # 第一行可能是卫星名称
            if lines[0].startswith('1 '):
                tle_line1 = lines[0]
                tle_line2 = lines[1]
                name = f"Satellite {sat_id}"
            elif len(lines) >= 3:
                # 第一行是名称，后面两行是TLE
                name = lines[0]
                tle_line1 = lines[1]
                tle_line2 = lines[2]
            else:
                return None, None, None, "other"
            
            # 验证TLE格式
            if tle_line1.startswith('1 ') and tle_line2.startswith('2 '):
                return tle_line1, tle_line2, name, "success"
            else:
                # TLE格式不正确，可能是卫星不存在
                return None, None, None, "not_found"
        else:
            # 未找到卫星数据
            return None, None, None, "not_found"
            
    except requests.exceptions.Timeout:
        return None, None, None, "timeout"
    except requests.exceptions.RequestException as e:
        # 检查是否是404错误（卫星不存在）
        if hasattr(e, 'response') and e.response is not None and hasattr(e.response, 'status_code'):
            if e.response.status_code == 404:
                return None, None, None, "not_found"
            # CelesTrak对于不存在的卫星可能返回400错误（No TLE found）
            elif e.response.status_code == 400:
                # 检查响应内容是否包含"No TLE found"
                try:
                    content = e.response.text if hasattr(e.response, 'text') else ""
                    if "No TLE found" in content or "No data" in content:
                        return None, None, None, "not_found"
                except:
                    pass
                return None, None, None, "other"
            # 403错误（IP被阻止）
            elif e.response.status_code == 403:
                return None, None, None, "forbidden"
            # 服务器内部错误（500）
            elif e.response.status_code == 500:
                return None, None, None, "server_error"
        return None, None, None, "other"
    except Exception as e:
        return None, None, None, "other"

def fetch_tle_from_n2yo(sat_id):
    """
    从n2yo.com获取TLE数据
    
    参数:
        sat_id: 卫星NORAD ID (如 25544 表示ISS)
    
    返回:
        (tle_line1, tle_line2, name, message)
        如果失败则返回 (None, None, None, error_type)
        error_type: "not_found" (卫星不存在), "timeout" (超时), "other" (其他错误)
    """
    try:
        # n2yo.com 轨道数据页面
        url = f"https://www.n2yo.com/satellite/?s={sat_id}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # 解析页面获取TLE数据
        content = response.text
        
        # 导入正则表达式模块
        import re
        
        # 先提取TLE数据，以便后续可能从TLE中提取名称
        tle_pattern = r'TLE[^<]*<pre[^>]*>(.*?)</pre>'
        tle_match = re.search(tle_pattern, content, re.DOTALL | re.IGNORECASE)
        
        # 提取卫星名称
        name = f"Satellite {sat_id}"
        
        # 尝试多种方式提取卫星名称
        # 1. 尝试从h1标签提取
        name_match = re.search(r'<h1[^>]*>(.*?)</h1>', content, re.DOTALL)
        if name_match:
            name = re.sub(r'<[^>]+>', '', name_match.group(1)).strip()
            # 清理名称，移除多余的空格和数字ID
            name = re.sub(r'\s*\(.*?\)\s*', '', name)
            name = re.sub(r'\s+', ' ', name).strip()
        
        # 2. 尝试从卫星信息区域提取（更灵活的选择器）
        if name == f"Satellite {sat_id}":
            # 尝试多种可能的class名称
            info_patterns = [
                r'<div[^>]*class=["\']satName["\'][^>]*>(.*?)</div>',
                r'<div[^>]*class=["\']sat-name["\'][^>]*>(.*?)</div>',
                r'<div[^>]*class=["\']name["\'][^>]*>(.*?)</div>',
                r'<div[^>]*class=["\']satellite-name["\'][^>]*>(.*?)</div>'
            ]
            for pattern in info_patterns:
                info_match = re.search(pattern, content, re.DOTALL)
                if info_match:
                    name = re.sub(r'<[^>]+>', '', info_match.group(1)).strip()
                    name = re.sub(r'\s+', ' ', name).strip()
                    if name != f"Satellite {sat_id}":
                        break
        
        # 3. 尝试从页面标题提取
        if name == f"Satellite {sat_id}":
            title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.DOTALL)
            if title_match:
                title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()
                # 移除各种可能的后缀
                title = re.sub(r'\s*-\s*Satellites\s*-\s*n2yo\.com.*$', '', title, flags=re.IGNORECASE)
                title = re.sub(r'\s*-\s*n2yo\.com.*$', '', title, flags=re.IGNORECASE)
                title = re.sub(r'\s+', ' ', title).strip()
                if title:
                    name = title
        
        # 4. 尝试从面包屑导航提取
        if name == f"Satellite {sat_id}":
            breadcrumb_patterns = [
                r'<div[^>]*class=["\']breadcrumb["\'][^>]*>(.*?)</div>',
                r'<nav[^>]*class=["\']breadcrumb["\'][^>]*>(.*?)</nav>',
                r'<ol[^>]*class=["\']breadcrumb["\'][^>]*>(.*?)</ol>'
            ]
            for pattern in breadcrumb_patterns:
                breadcrumb_match = re.search(pattern, content, re.DOTALL)
                if breadcrumb_match:
                    breadcrumb = re.sub(r'<[^>]+>', '', breadcrumb_match.group(1)).strip()
                    # 提取最后一部分作为名称
                    parts = breadcrumb.split('>')
                    if len(parts) > 1:
                        last_part = parts[-1].strip()
                        if last_part and last_part != sat_id:
                            name = last_part
                            break
        
        # 5. 尝试从元数据提取
        if name == f"Satellite {sat_id}":
            meta_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\'](.*?)["\']', content, re.DOTALL)
            if meta_match:
                description = meta_match.group(1).strip()
                # 从描述中提取名称
                name_match = re.search(r'^.*?of\s+(.*?)\s+\(NORAD', description, re.IGNORECASE)
                if name_match:
                    name = name_match.group(1).strip()
        
        # 6. 特殊处理常见卫星
        special_satellites = {
            "25544": "ISS (International Space Station)",
            "43204": "Tiangong 2",
            "48274": "Tiangong 3",
            "49026": "Shenzhou 13",
            "50412": "Shenzhou 14",
            "54339": "Shenzhou 16"
        }
        if sat_id in special_satellites:
            name = special_satellites[sat_id]
        
        # 7. 尝试从TLE数据中提取名称（如果页面中没有其他信息）
        if name == f"Satellite {sat_id}" and tle_match:
            # 有时TLE数据前面会有名称
            tle_text = tle_match.group(1)
            lines = tle_text.split('\n')
            # 检查第一行是否是名称（不是以1或2开头）
            if lines:
                first_line = lines[0].strip()
                if not first_line.startswith('1 ') and not first_line.startswith('2 '):
                    name = first_line.strip()
        
        # 最终清理
        name = re.sub(r'\s+', ' ', name).strip()
        # 确保名称不是空的
        if not name:
            name = f"Satellite {sat_id}"
        
        if tle_match:
            tle_text = tle_match.group(1)
            tle_lines = [line.strip() for line in tle_text.split('\n') if line.strip()]
            
            # 过滤掉可能的标题行，只保留以1或2开头的TLE行
            tle_lines = [line for line in tle_lines if (line.startswith('1 ') or line.startswith('2 '))]
            
            if len(tle_lines) >= 2:
                tle_line1 = tle_lines[0]
                tle_line2 = tle_lines[1]
                
                # 验证TLE格式
                if tle_line1.startswith('1 ') and tle_line2.startswith('2 '):
                    return tle_line1, tle_line2, name, "success"
                else:
                    # TLE格式不正确，可能是卫星不存在
                    return None, None, None, "not_found"
            else:
                # 没有足够的TLE行，可能是卫星不存在
                return None, None, None, "not_found"
        else:
            # 未找到TLE数据，可能是卫星不存在
            return None, None, None, "not_found"
            
    except requests.exceptions.Timeout:
        return None, None, None, "timeout"
    except requests.exceptions.RequestException as e:
        # 检查是否是404错误（卫星不存在）
        if hasattr(e, 'response') and e.response is not None and hasattr(e.response, 'status_code') and e.response.status_code == 404:
            return None, None, None, "not_found"
        return None, None, None, "other"
    except Exception as e:
        return None, None, None, "other"

def fetch_std_mag_from_heavens_above(sat_id):
    """
    从Heavens-Above获取本征星等
    
    参数:
        sat_id: 卫星NORAD ID (如 25544 表示ISS)
    
    返回:
        (std_mag, status)
        std_mag: 本征星等值，失败为None
        status: "success", "timeout", "not_provided"
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        # 从satinfo.aspx获取本征星等
        info_url = f"https://www.heavens-above.com/satinfo.aspx?satid={sat_id}"
        info_response = requests.get(info_url, headers=headers, timeout=10)
        info_response.raise_for_status()
        info_html = info_response.text
        
        # 提取本征星等 (Intrinsic brightness / 固有亮度)
        # 定义：1000km距离、相位角90度（50%被照亮）时的视星等
        std_mag = None
        mag_patterns = [
            r'固有亮度[^\d]{0,20}?([+-]?\d+\.?\d+)',
            r'Intrinsic\s+brightness[^\d]{0,20}?([+-]?\d+\.?\d+)',
            r'(?:本征|固有|Intrinsic)[^\d]{0,30}?亮度[^\d]{0,10}?([+-]?\d+\.?\d+)',
            r'(?:brightness|magnitude)[^\d]{0,20}?([+-]?\d+\.?\d+)',
        ]
        
        for pattern in mag_patterns:
            mag_match = re.search(pattern, info_html, re.IGNORECASE | re.DOTALL)
            if mag_match:
                try:
                    candidate = float(mag_match.group(1))
                    if -10 <= candidate <= 20:
                        std_mag = candidate
                        break
                except ValueError:
                    continue
        
        if std_mag is not None:
            return std_mag, "success"
        else:
            return None, "not_provided"
            
    except requests.exceptions.Timeout:
        return None, "timeout"
    except requests.exceptions.RequestException as e:
        return None, "timeout"
    except Exception as e:
        return None, "timeout"

def fetch_satellite_from_heavens_above(sat_id):
    """
    从Heavens-Above网站获取卫星数据
    
    参数:
        sat_id: 卫星NORAD ID (如 25544 表示ISS)
    
    返回:
        (tle_line1, tle_line2, std_mag, name, status)
        如果失败则返回 (None, None, None, None, error_type)
        error_type: "not_found" (卫星不存在), "timeout" (超时), "other" (其他错误)
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # 从orbit.aspx获取TLE轨道根数
        orbit_url = f"https://www.heavens-above.com/orbit.aspx?satid={sat_id}"
        orbit_response = requests.get(orbit_url, headers=headers, timeout=10)
        orbit_response.raise_for_status()
        orbit_html = orbit_response.text
        
        # 从satinfo.aspx获取本征星等和卫星名称
        info_url = f"https://www.heavens-above.com/satinfo.aspx?satid={sat_id}"
        info_response = requests.get(info_url, headers=headers, timeout=10)
        info_response.raise_for_status()
        info_html = info_response.text
        
        # 提取卫星名称（从satinfo页面或orbit页面）
        name = "Unknown Satellite"
        
        # 尝试多种方式提取卫星名称
        # 方式1: 从<title>标签提取
        title_match = re.search(r'<title[^>]*>(.*?)</title>', info_html, re.IGNORECASE | re.DOTALL)
        if title_match:
            title_text = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()
            # 移除 "- Heavens-Above"、"- Satellite Information" 等后缀
            title_text = re.sub(r'\s*[-|]\s*(Heavens-Above|Satellite Information).*$', '', title_text, flags=re.IGNORECASE)
            if title_text:
                name = title_text
        
        # 方式2: 如果title提取失败，尝试从<h1>标签提取
        if name == "Unknown Satellite":
            name_match = re.search(r'<h1[^>]*>(.*?)</h1>', info_html, re.IGNORECASE | re.DOTALL)
            if name_match:
                name = re.sub(r'<[^>]+>', '', name_match.group(1)).strip()
            else:
                # 尝试从orbit页面提取
                name_match = re.search(r'<h1[^>]*>(.*?)</h1>', orbit_html, re.IGNORECASE | re.DOTALL)
                if name_match:
                    name = re.sub(r'<[^>]+>', '', name_match.group(1)).strip()
        
        # 方式3: 从TLE第一行提取卫星名称（如果其他方式都失败）
        if name == "Unknown Satellite" and tle_line1:
            # TLE第一行格式: 1 NNNNNU NNNNNAAA NNNNN.NNNNNNNN +.NNNNNNNN +NNNNN-N +NNNNN-N N NNNNN
            # 名称通常在TLE数据之前
            tle_name_match = re.search(r'(\d{5})\s+([A-Z])\s+(\d{4}-\d{3}[A-Z])', tle_line1)
            if tle_name_match:
                sat_num = tle_name_match.group(1)
                name = f"Satellite {sat_num}"
        
        # 从orbit页面提取TLE数据
        tle_line1 = None
        tle_line2 = None
        
        # 标准TLE格式匹配
        tle_pattern = r'1\s+\d{5}[A-Z]?\s+.*?(?:\n|\r\n?)2\s+\d{5}\s+.*?(?:\n|\r\n?)'
        tle_match = re.search(tle_pattern, orbit_html, re.DOTALL)
        
        if tle_match:
            tle_text = tle_match.group(0).strip()
            tle_lines = [line.strip() for line in tle_text.split('\n') if line.strip()]
            if len(tle_lines) >= 2:
                tle_line1 = tle_lines[0]
                tle_line2 = tle_lines[1]
        
        # 如果没有找到，尝试查找轨道根数区域
        if not tle_line1 or not tle_line2:
            # 查找包含TLE的pre标签或文本区域
            tle_pre_pattern = r'<pre[^>]*>(.*?)</pre>'
            pre_matches = re.findall(tle_pre_pattern, orbit_html, re.DOTALL | re.IGNORECASE)
            for pre_content in pre_matches:
                pre_text = re.sub(r'<[^>]+>', '', pre_content)
                lines = [line.strip() for line in pre_text.split('\n') if line.strip()]
                for i, line in enumerate(lines):
                    if line.startswith('1 ') and i + 1 < len(lines) and lines[i + 1].startswith('2 '):
                        tle_line1 = line
                        tle_line2 = lines[i + 1]
                        break
                if tle_line1 and tle_line2:
                    break
        
        # 从satinfo页面提取本征星等 (Intrinsic brightness / 固有亮度)
        std_mag = None
        # 网页格式: "固有亮度 (食分) -1.8 (在千公里距离，50%照亮时)"
        # 定义：1000km距离、相位角90度（50%被照亮）时的视星等
        # 尝试多种可能的格式
        mag_patterns = [
            # 匹配中文 "固有亮度" 后面跟着数字（允许中间有其他字符）
            r'固有亮度[^\d]{0,20}?([+-]?\d+\.?\d+)',
            # 匹配英文 "Intrinsic brightness" 后面跟着数字
            r'Intrinsic\s+brightness[^\d]{0,20}?([+-]?\d+\.?\d+)',
            # 匹配 "亮度" 前面有"本征"或"固有"或"Intrinsic"
            r'(?:本征|固有|Intrinsic)[^\d]{0,30}?亮度[^\d]{0,10}?([+-]?\d+\.?\d+)',
            # 匹配 "brightness" 或 "magnitude" 后面跟着数字
            r'(?:brightness|magnitude)[^\d]{0,20}?([+-]?\d+\.?\d+)',
        ]
        
        for pattern in mag_patterns:
            mag_match = re.search(pattern, info_html, re.IGNORECASE | re.DOTALL)
            if mag_match:
                try:
                    candidate = float(mag_match.group(1))
                    # 本征星等通常在-10到20之间，过滤掉不合理的值
                    if -10 <= candidate <= 20:
                        std_mag = candidate
                        break
                except ValueError:
                    continue
        
        # 如果找到了TLE数据，返回成功
        if tle_line1 and tle_line2:
            return tle_line1, tle_line2, std_mag, name, "success"
        else:
            # 未找到TLE数据，可能是卫星不存在
            return None, None, None, None, "not_found"
            
    except requests.exceptions.Timeout:
        return None, None, None, None, "timeout"
    except requests.exceptions.RequestException as e:
        # 检查是否是404错误（卫星不存在）
        if hasattr(e, 'response') and e.response is not None and hasattr(e.response, 'status_code'):
            if e.response.status_code == 404:
                return None, None, None, None, "not_found"
            # Heavens-Above对于不存在的卫星可能返回500错误
            elif e.response.status_code == 500:
                # 检查响应内容是否包含错误信息
                try:
                    content = e.response.text if hasattr(e.response, 'text') else ""
                    # 如果内容包含错误信息，认为是卫星不存在
                    if "error" in content.lower() or "not found" in content.lower() or "不存在" in content:
                        return None, None, None, None, "not_found"
                except:
                    pass
                return None, None, None, None, "other"
        return None, None, None, None, "other"
    except Exception as e:
        return None, None, None, None, "other"

def parse_coordinate(coord_str):
    """解析度分秒格式的坐标，返回小数值
    支持的格式：
    - 小数格式：39.9
    - 度分秒格式：39°54'30" 或 39 54 30
    - 带方向的度分秒格式：39°54'30"N 或 116°24'10"E
    """
    coord_str = coord_str.strip()
    
    # 检查是否是小数格式
    try:
        return float(coord_str)
    except ValueError:
        pass
    
    # 处理度分秒格式
    # 移除方向符号（N/S/E/W）
    direction = 1
    if coord_str.endswith(('N', 'E')):
        coord_str = coord_str[:-1].strip()
    elif coord_str.endswith(('S', 'W')):
        coord_str = coord_str[:-1].strip()
        direction = -1
    
    # 移除度分秒符号
    coord_str = coord_str.replace('°', ' ').replace("'", ' ').replace('"', ' ')
    
    # 分割成度分秒
    parts = list(filter(None, coord_str.split()))
    if len(parts) == 3:
        degrees = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        decimal = degrees + minutes/60 + seconds/3600
        return decimal * direction
    elif len(parts) == 2:
        degrees = float(parts[0])
        minutes = float(parts[1])
        decimal = degrees + minutes/60
        return decimal * direction
    else:
        raise ValueError("无效的坐标格式")

def parse_dms_coordinate(deg, min, sec, direction):
    """解析度分秒格式的坐标，返回小数值
    """
    try:
        degrees = float(deg)
        minutes = float(min)
        seconds = float(sec)
        decimal = degrees + minutes/60 + seconds/3600
        if direction in ('S', 'W'):
            decimal = -decimal
        return decimal
    except ValueError:
        raise ValueError("无效的度分秒格式")

def compute_sun_altitude(t, observer, eph):
    """计算观测地点的太阳高度角（度）
    
    参数:
        t: skyfield时间对象
        observer: 观测者位置（Topos对象）
        eph: 星历数据
    
    返回:
        太阳高度角（度），负数表示太阳在地平线以下
    """
    # 优先使用 pyephem 库来计算太阳高度角
    try:
        import ephem
        
        # 获取观测者的经纬度和高度
        lat, lon, elevation = observer.latitude.degrees, observer.longitude.degrees, observer.elevation.m
        
        # 创建观测者
        obs = ephem.Observer()
        obs.lat = str(lat)
        obs.lon = str(lon)
        obs.elevation = elevation
        
        # 设置时间
        utc_dt = t.utc_datetime()
        obs.date = ephem.Date(utc_dt)
        
        # 创建太阳
        sun_ephem = ephem.Sun()
        sun_ephem.compute(obs)
        
        # 计算太阳高度角（弧度转度）
        import math
        sun_alt = math.degrees(sun_ephem.alt)
        return sun_alt
    except Exception as e:
        # 如果 pyephem 方法失败，使用 Skyfield 的标准方法
        try:
            sun = eph["sun"]
            
            # 计算观测者在给定时间的位置
            observer_at_t = observer.at(t)
            
            # 从观测者位置观测太阳
            astrometric = observer_at_t.observe(sun)
            
            # 计算地平坐标
            alt, az, distance = astrometric.altaz()
            
            # 返回太阳高度角（度）
            return alt.degrees
        except Exception as e:
            # 如果所有方法都失败，返回 0.0
            return 0.0

def compute_moon_altitude(t, observer, eph):
    """计算观测地点的月亮高度角（度）
    
    参数:
        t: skyfield时间对象
        observer: 观测者位置（Topos对象）
        eph: 星历数据
    
    返回:
        月亮高度角（度），负数表示月亮在地平线以下
    """
    # 优先使用 pyephem 库来计算月亮高度角
    try:
        import ephem
        
        # 获取观测者的经纬度和高度
        lat, lon, elevation = observer.latitude.degrees, observer.longitude.degrees, observer.elevation.m
        
        # 创建观测者
        obs = ephem.Observer()
        obs.lat = str(lat)
        obs.lon = str(lon)
        obs.elevation = elevation
        
        # 设置时间
        utc_dt = t.utc_datetime()
        obs.date = ephem.Date(utc_dt)
        
        # 创建月亮
        moon_ephem = ephem.Moon()
        moon_ephem.compute(obs)
        
        # 计算月亮高度角（弧度转度）
        import math
        moon_alt = math.degrees(moon_ephem.alt)
        return moon_alt
    except Exception as e:
        # 如果 pyephem 方法失败，使用 Skyfield 的标准方法
        try:
            moon = eph["moon"]
            
            # 计算观测者在给定时间的位置
            observer_at_t = observer.at(t)
            
            # 从观测者位置观测月亮
            astrometric = observer_at_t.observe(moon)
            
            # 计算地平坐标
            alt, az, distance = astrometric.altaz()
            
            # 返回月亮高度角（度）
            return alt.degrees
        except Exception as e:
            # 如果所有方法都失败，返回 0.0
            return 0.0



def compute_motion_pa_speed(t, sat, observer, ts, dt_minutes=1.0):
    """计算运动角速度（"/min）和位置角PA（°）"""
    # 获取 UTC datetime 对象
    utc_dt = t.utc_datetime()
    # 计算 t2
    t2_dt = utc_dt + timedelta(minutes=dt_minutes)
    # 使用传入的timescale对象来创建t2
    t2 = ts.utc(t2_dt.year, t2_dt.month, t2_dt.day, t2_dt.hour, t2_dt.minute, t2_dt.second)
    
    diff1 = (sat - observer).at(t)
    diff2 = (sat - observer).at(t2)
    
    # 计算角距离（度）
    sep = diff1.separation_from(diff2).degrees
    speed = sep * 3600 / dt_minutes   # 转为角秒/分钟
    
    # 使用赤道坐标系计算位置角PA
    # 0°正北，90°正东
    ra1, dec1, _ = diff1.radec()
    ra2, dec2, _ = diff2.radec()
    
    dra = ra2.hours - ra1.hours
    # 调整RA差值到-12到12小时
    dra = (dra + 12) % 24 - 12
    dra_rad = math.radians(dra * 15)  # 转换为弧度（1小时=15度）
    
    dec1_rad = math.radians(dec1.degrees)
    dec2_rad = math.radians(dec2.degrees)
    
    # 计算位置角
    numerator = math.sin(dra_rad)
    denominator = math.cos(dec1_rad) * math.tan(dec2_rad) - math.sin(dec1_rad) * math.cos(dra_rad)
    
    pa = math.degrees(math.atan2(numerator, denominator))
    pa = (pa + 360) % 360
    
    return speed, pa

class SatelliteEphemerisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("人造卫星星历计算器")
        self.root.geometry("750x450")
        self.root.resizable(True, True)
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建输入框架
        self.input_tab = ttk.Frame(self.main_frame)
        self.input_tab.pack(fill=tk.BOTH, expand=True)
        
        # 全局变量
        self.eph = None
        self.ts = None
        self.ephemeris_data = []
        
        # 状态变量
        self.status_var = tk.StringVar(value="就绪")
        
        # 结果窗口变量
        self.result_window = None
        self.result_tree = None
        
        # 从配置文件加载配置
        self.config = load_config()
        # 提取MPC代码
        self.builtin_codes = {k: v for k, v in self.config.items() if k != "manual_location"}
        
        # 当前从HTML读取的MPC数据
        self.current_mpc_data = None
        
        # 初始化输入界面
        self.init_input_tab()
    
    def create_context_menu(self, widget):
        """为输入控件创建右键菜单"""
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="复制", command=lambda: widget.event_generate("<<Copy>>"))
        menu.add_command(label="粘贴", command=lambda: widget.event_generate("<<Paste>>"))
        menu.add_command(label="剪切", command=lambda: widget.event_generate("<<Cut>>"))
        
        # 根据控件类型选择全选方法
        if isinstance(widget, tk.Text):
            menu.add_command(label="全选", command=lambda: widget.tag_add("sel", "1.0", tk.END))
        else:
            menu.add_command(label="全选", command=lambda: widget.select_range(0, tk.END))
        
        def show_menu(event):
            menu.post(event.x_root, event.y_root)
        
        widget.bind("<Button-3>", show_menu)
    
    def setup_decimal_input(self, entry_widget):
        """
        为输入控件设置中文标点自动转换为英文标点的功能
        支持中文输入法下的小数点、负号等符号的自动转换
        """
        # 中文标点到英文标点的映射
        # 包括：中文输入法下各种可能输入的标点符号
        punctuation_map = {
            # 小数点相关
            '。': '.',          # 中文句号 (字母区域)
            '\uff0e': '.',      # 全角句号 (数字键盘区域)
            '\u3002': '.',      # 中文句号 (CJK)
            '．': '.',          # 全角小数点
            '\uff61': '.',      # 半角片假名句号
            # 减号/负号相关
            '－': '-',          # 中文破折号/减号
            '\uff0d': '-',      # 全角减号
            '\u2212': '-',      # 数学减号
            '\u2013': '-',      # 短破折号
            '\u2014': '-',      # 长破折号
            # 逗号
            '，': ',',          # 中文逗号
            '\uff0c': ',',      # 全角逗号
            # 其他常用标点
            '：': ':',          # 中文冒号
            '\uff1a': ':',      # 全角冒号
            '；': ';',          # 中文分号
            '\uff1b': ';',      # 全角分号
            '（': '(',          # 中文左括号
            '\uff08': '(',      # 全角左括号
            '）': ')',          # 中文右括号
            '\uff09': ')',      # 全角右括号
            '【': '[',          # 中文左方括号
            '\uff3b': '[',      # 全角左方括号
            '】': ']',          # 中文右方括号
            '\uff3d': ']',      # 全角右方括号
        }
        
        def on_key_release(event):
            """按键释放时检查并转换中文标点"""
            # 获取当前光标位置
            try:
                cursor_pos = entry_widget.index(tk.INSERT)
                current_text = entry_widget.get()
                
                # 检查是否有需要转换的字符
                new_text = current_text
                for cn_char, en_char in punctuation_map.items():
                    if cn_char in new_text:
                        new_text = new_text.replace(cn_char, en_char)
                
                # 如果文本有变化，更新内容并恢复光标位置
                if new_text != current_text:
                    # 计算光标位置的偏移（因为替换可能导致长度变化）
                    old_len = len(current_text)
                    entry_widget.delete(0, tk.END)
                    entry_widget.insert(0, new_text)
                    # 尝试恢复光标位置
                    try:
                        new_pos = min(cursor_pos, len(new_text))
                        entry_widget.icursor(new_pos)
                    except:
                        pass
            except Exception:
                pass
        
        # 绑定键盘释放事件
        entry_widget.bind("<KeyRelease>", on_key_release)
    
    def init_input_tab(self):
        # 创建输入表单框架
        form_frame = ttk.LabelFrame(self.input_tab, text="观测参数", padding="10")
        form_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建网格布局
        form_frame.grid_columnconfigure(0, weight=1)
        form_frame.grid_columnconfigure(1, weight=3)
        
        # 观测地点选择
        ttk.Label(form_frame, text="观测地点:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.loc_type_var = tk.StringVar(value="1")
        loc_frame = ttk.Frame(form_frame)
        loc_frame.grid(row=0, column=1, sticky=tk.W)
        ttk.Radiobutton(loc_frame, text="MPC代码", variable=self.loc_type_var, value="1", command=self.toggle_loc_input).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(loc_frame, text="经纬度", variable=self.loc_type_var, value="2", command=self.toggle_loc_input).pack(side=tk.LEFT, padx=10)
        
        # MPC代码输入
        self.mpc_label = ttk.Label(form_frame, text="MPC天文台代码:")
        self.mpc_label.grid(row=1, column=0, sticky=tk.W, pady=5)
        mpc_sat_frame = ttk.Frame(form_frame)
        mpc_sat_frame.grid(row=1, column=1, sticky=tk.W, pady=5)
        self.mpc_code_var = tk.StringVar()
        self.mpc_entry = ttk.Entry(mpc_sat_frame, textvariable=self.mpc_code_var, width=15)
        self.mpc_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.create_context_menu(self.mpc_entry)
        
        # 卫星NORAD ID（在MPC代码同一行）
        self.sat_id_var = tk.StringVar()
        ttk.Label(mpc_sat_frame, text="卫星NORAD ID:").pack(side=tk.LEFT, padx=(40, 5))
        self.sat_id_entry = ttk.Entry(mpc_sat_frame, textvariable=self.sat_id_var, width=15)
        self.sat_id_entry.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(mpc_sat_frame, text="(如: 25544表示ISS)", foreground="gray").pack(side=tk.LEFT, padx=(0, 10))
        
        # 经纬度输入方式选择
        ttk.Label(form_frame, text="经纬度输入方式:").grid(row=2, column=0, sticky=tk.W, pady=5)
        coord_sat_frame = ttk.Frame(form_frame)
        coord_sat_frame.grid(row=2, column=1, sticky=tk.W, pady=5)
        self.coord_input_type = tk.StringVar(value="decimal")
        # 添加输入方式切换的回调函数
        self.coord_input_type.trace_add("write", self.toggle_coord_input)
        ttk.Radiobutton(coord_sat_frame, text="小数格式", variable=self.coord_input_type, value="decimal").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(coord_sat_frame, text="度分秒格式", variable=self.coord_input_type, value="dms").pack(side=tk.LEFT, padx=10)
        
        # 卫星NORAD ID（在经纬度输入方式同一行）
        ttk.Label(coord_sat_frame, text="卫星NORAD ID:").pack(side=tk.LEFT, padx=(40, 5))
        self.sat_id_entry2 = ttk.Entry(coord_sat_frame, textvariable=self.sat_id_var, width=15)
        self.sat_id_entry2.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(coord_sat_frame, text="(如: 25544表示ISS)", foreground="gray").pack(side=tk.LEFT, padx=(0, 10))
        
        # 小数格式输入（同一行）
        self.lat_label = ttk.Label(form_frame, text="纬度（北纬为正，南纬为负）:")
        self.lon_label = ttk.Label(form_frame, text="经度（东经为正，西经为负）:")
        self.height_label = ttk.Label(form_frame, text="海拔高度（米）:")
        
        # 创建统一的输入框架
        coord_frame = ttk.Frame(form_frame)
        coord_frame.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # 纬度输入
        ttk.Label(coord_frame, text="纬度:").pack(side=tk.LEFT, padx=(0, 5))
        self.lat_var = tk.StringVar(value=str(self.config.get("manual_location", {}).get("latitude", 39.9)))
        self.lat_entry = ttk.Entry(coord_frame, textvariable=self.lat_var, width=15)
        self.lat_entry.pack(side=tk.LEFT, padx=(70, 15))
        self.create_context_menu(self.lat_entry)
        self.setup_decimal_input(self.lat_entry)
        
        # 经度输入
        ttk.Label(coord_frame, text="经度:").pack(side=tk.LEFT, padx=(40, 5))
        self.lon_var = tk.StringVar(value=str(self.config.get("manual_location", {}).get("longitude", 116.4)))
        self.lon_entry = ttk.Entry(coord_frame, textvariable=self.lon_var, width=15)
        self.lon_entry.pack(side=tk.LEFT, padx=(0, 15))
        self.create_context_menu(self.lon_entry)
        self.setup_decimal_input(self.lon_entry)
        
        # 海拔高度输入
        ttk.Label(coord_frame, text="海拔高度（米）:").pack(side=tk.LEFT, padx=(40, 5))
        height_val = self.config.get("manual_location", {}).get("height_m", 0)
        # 确保高度值有效，默认为0.0
        try:
            height_float = float(height_val) if height_val != "" else 0.0
        except (ValueError, TypeError):
            height_float = 0.0
        self.height_var = tk.StringVar(value=f"{height_float:.1f}")
        self.height_entry = ttk.Entry(coord_frame, textvariable=self.height_var, width=15)
        self.height_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.create_context_menu(self.height_entry)
        self.setup_decimal_input(self.height_entry)
        
        # 添加说明文字
        self.coord_note = ttk.Label(form_frame, text="北纬为正，南纬为负；东经为正，西经为负。", foreground="gray")
        self.coord_note.grid(row=4, column=1, sticky=tk.W, pady=2)
        
        # 度分秒格式输入（保持原样）
        self.lat_dms_label = ttk.Label(form_frame, text="纬度:")
        self.lat_dms_label.grid(row=5, column=0, sticky=tk.W, pady=5)
        self.lat_dms_frame = ttk.Frame(form_frame)
        self.lat_dms_frame.grid(row=5, column=1, sticky=tk.W)
        self.lat_deg_var = tk.StringVar(value="39")
        self.lat_deg_entry = ttk.Entry(self.lat_dms_frame, textvariable=self.lat_deg_var, width=5)
        self.lat_deg_entry.pack(side=tk.LEFT, padx=5)
        self.setup_decimal_input(self.lat_deg_entry)
        ttk.Label(self.lat_dms_frame, text="°").pack(side=tk.LEFT)
        self.lat_min_var = tk.StringVar(value="54")
        self.lat_min_entry = ttk.Entry(self.lat_dms_frame, textvariable=self.lat_min_var, width=5)
        self.lat_min_entry.pack(side=tk.LEFT, padx=5)
        self.setup_decimal_input(self.lat_min_entry)
        ttk.Label(self.lat_dms_frame, text="'").pack(side=tk.LEFT)
        self.lat_sec_var = tk.StringVar(value="0")
        self.lat_sec_entry = ttk.Entry(self.lat_dms_frame, textvariable=self.lat_sec_var, width=5)
        self.lat_sec_entry.pack(side=tk.LEFT, padx=5)
        self.setup_decimal_input(self.lat_sec_entry)
        ttk.Label(self.lat_dms_frame, text='"').pack(side=tk.LEFT)
        # 纬度方向选择（直接列出南北选项）
        self.lat_dir_var = tk.StringVar(value="N")
        lat_dir_frame = ttk.Frame(self.lat_dms_frame)
        lat_dir_frame.pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(lat_dir_frame, text="N", variable=self.lat_dir_var, value="N").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(lat_dir_frame, text="S", variable=self.lat_dir_var, value="S").pack(side=tk.LEFT, padx=2)
        
        self.lon_dms_label = ttk.Label(form_frame, text="经度:")
        self.lon_dms_label.grid(row=5, column=0, sticky=tk.W, pady=5)
        self.lon_dms_frame = ttk.Frame(form_frame)
        self.lon_dms_frame.grid(row=5, column=1, sticky=tk.W)
        self.lon_deg_var = tk.StringVar(value="116")
        self.lon_deg_entry = ttk.Entry(self.lon_dms_frame, textvariable=self.lon_deg_var, width=5)
        self.lon_deg_entry.pack(side=tk.LEFT, padx=5)
        self.setup_decimal_input(self.lon_deg_entry)
        ttk.Label(self.lon_dms_frame, text="°").pack(side=tk.LEFT)
        self.lon_min_var = tk.StringVar(value="24")
        self.lon_min_entry = ttk.Entry(self.lon_dms_frame, textvariable=self.lon_min_var, width=5)
        self.lon_min_entry.pack(side=tk.LEFT, padx=5)
        self.setup_decimal_input(self.lon_min_entry)
        ttk.Label(self.lon_dms_frame, text="'").pack(side=tk.LEFT)
        self.lon_sec_var = tk.StringVar(value="0")
        self.lon_sec_entry = ttk.Entry(self.lon_dms_frame, textvariable=self.lon_sec_var, width=5)
        self.lon_sec_entry.pack(side=tk.LEFT, padx=5)
        self.setup_decimal_input(self.lon_sec_entry)
        ttk.Label(self.lon_dms_frame, text='"').pack(side=tk.LEFT)
        # 经度方向选择（直接列出东西选项）
        self.lon_dir_var = tk.StringVar(value="E")
        lon_dir_frame = ttk.Frame(self.lon_dms_frame)
        lon_dir_frame.pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(lon_dir_frame, text="E", variable=self.lon_dir_var, value="E").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(lon_dir_frame, text="W", variable=self.lon_dir_var, value="W").pack(side=tk.LEFT, padx=2)
        
        # 海拔高度输入（度分秒格式）
        ttk.Label(self.lon_dms_frame, text="海拔高度（米）:").pack(side=tk.LEFT, padx=(40, 5))
        height_val_dms = self.config.get("manual_location", {}).get("height_m", 0)
        # 确保高度值有效，默认为0.0
        try:
            height_float_dms = float(height_val_dms) if height_val_dms != "" else 0.0
        except (ValueError, TypeError):
            height_float_dms = 0.0
        self.height_var_dms = tk.StringVar(value=f"{height_float_dms:.1f}")
        self.height_entry_dms = ttk.Entry(self.lon_dms_frame, textvariable=self.height_var_dms, width=10)
        self.height_entry_dms.pack(side=tk.LEFT, padx=(0, 10))
        self.create_context_menu(self.height_entry_dms)
        self.setup_decimal_input(self.height_entry_dms)
        
        # 从Heavens-Above获取数据按钮（保持在TLE输入框上方）
        ttk.Label(form_frame, text="获取TLE数据：").grid(row=6, column=0, sticky=tk.W, pady=5)
        sat_btn_frame = ttk.Frame(form_frame)
        sat_btn_frame.grid(row=6, column=1, sticky=tk.W, pady=5)
        self.fetch_sat_btn = ttk.Button(sat_btn_frame, text="Heavens-Above", command=self.fetch_satellite_data)
        self.fetch_sat_btn.pack(side=tk.LEFT, padx=(0, 20))
        self.fetch_celestrak_btn = ttk.Button(sat_btn_frame, text="CelesTrak", command=self.fetch_tle_from_celestrak)
        self.fetch_celestrak_btn.pack(side=tk.LEFT, padx=(0, 20))
        self.fetch_n2yo_btn = ttk.Button(sat_btn_frame, text="N2YO.com", command=self.fetch_tle_from_n2yo)
        self.fetch_n2yo_btn.pack(side=tk.LEFT, padx=(0, 20))
        
        # 本征星等输入
        ttk.Label(sat_btn_frame, text="本征星等:").pack(side=tk.LEFT, padx=(30, 5))
        self.mag_var = tk.StringVar(value=str(DEFAULT_STD_MAG))
        self.mag_entry = ttk.Entry(sat_btn_frame, textvariable=self.mag_var, width=10)
        self.mag_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.create_context_menu(self.mag_entry)
        # 添加中文标点自动转换功能
        self.setup_decimal_input(self.mag_entry)
        
        # TLE输入（合并为一个文本框）
        ttk.Label(form_frame, text="TLE数据（两行）:").grid(row=7, column=0, sticky=tk.W, pady=5)
        self.tle_text = tk.Text(form_frame, height=2, width=60)
        self.tle_text.grid(row=7, column=1, sticky=tk.EW, pady=5)
        self.create_context_menu(self.tle_text)
        
        # 人造卫星名称和开始时间在同一行
        name_time_frame = ttk.Frame(form_frame)
        name_time_frame.grid(row=8, column=1, sticky=tk.EW, pady=5)
        
        # 人造卫星名称
        ttk.Label(form_frame, text="人造卫星名称:").grid(row=8, column=0, sticky=tk.W, pady=5)
        self.name_var = tk.StringVar(value="Unknown Satellite")
        self.name_entry = ttk.Entry(name_time_frame, textvariable=self.name_var, width=30)
        self.name_entry.pack(side=tk.LEFT, padx=(0, 15))
        self.create_context_menu(self.name_entry)
        
        # 开始时间
        ttk.Label(name_time_frame, text="开始时间（UTC+8）:").pack(side=tk.LEFT, padx=(45, 5))
        # 获取当前UTC+8时间
        from datetime import datetime, timezone, timedelta
        utc8_now = datetime.now(timezone(timedelta(hours=8)))
        self.start_var = tk.StringVar(value=utc8_now.strftime("%Y-%m-%d %H:%M:%S"))
        self.start_entry = ttk.Entry(name_time_frame, textvariable=self.start_var, width=20)
        self.start_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.create_context_menu(self.start_entry)
        
        # 计算时长
        ttk.Label(form_frame, text="计算时长:").grid(row=9, column=0, sticky=tk.W, pady=5)
        duration_frame = ttk.Frame(form_frame)
        duration_frame.grid(row=9, column=1, sticky=tk.EW, pady=5)
        self.duration_var = tk.StringVar(value="24")
        self.duration_entry = ttk.Entry(duration_frame, textvariable=self.duration_var, width=10)
        self.duration_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.create_context_menu(self.duration_entry)
        self.setup_decimal_input(self.duration_entry)
        
        # 计算时长单位选项
        self.duration_unit_var = tk.StringVar(value="小时")
        duration_unit_frame = ttk.Frame(duration_frame)
        duration_unit_frame.pack(side=tk.LEFT)
        ttk.Radiobutton(duration_unit_frame, text="秒", variable=self.duration_unit_var, value="秒").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(duration_unit_frame, text="分钟", variable=self.duration_unit_var, value="分钟").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(duration_unit_frame, text="小时", variable=self.duration_unit_var, value="小时").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(duration_unit_frame, text="天", variable=self.duration_unit_var, value="天").pack(side=tk.LEFT, padx=5)
        
        # 时间步长
        ttk.Label(form_frame, text="时间步长:").grid(row=10, column=0, sticky=tk.W, pady=5)
        step_frame = ttk.Frame(form_frame)
        step_frame.grid(row=10, column=1, sticky=tk.EW, pady=5)
        self.step_var = tk.StringVar(value="1")
        self.step_entry = ttk.Entry(step_frame, textvariable=self.step_var, width=10)
        self.step_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.create_context_menu(self.step_entry)
        self.setup_decimal_input(self.step_entry)
        
        # 时间步长单位选项
        self.step_unit_var = tk.StringVar(value="分钟")
        step_unit_frame = ttk.Frame(step_frame)
        step_unit_frame.pack(side=tk.LEFT)
        ttk.Radiobutton(step_unit_frame, text="秒", variable=self.step_unit_var, value="秒").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(step_unit_frame, text="分钟", variable=self.step_unit_var, value="分钟").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(step_unit_frame, text="小时", variable=self.step_unit_var, value="小时").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(step_unit_frame, text="天", variable=self.step_unit_var, value="天").pack(side=tk.LEFT, padx=5)
        

        

        
        # MPC代码解析结果显示
        self.mpc_result_var = tk.StringVar(value="")
        self.mpc_result_frame = ttk.LabelFrame(form_frame, text="MPC代码解析结果", padding="10")
        self.mpc_result_frame.grid(row=11, column=0, columnspan=2, sticky=tk.EW, pady=5)
        # 使用Text控件替代Label，以便用户可以选择和复制文本
        self.mpc_result_text = tk.Text(self.mpc_result_frame, height=2, wrap=tk.WORD, state=tk.DISABLED)
        self.mpc_result_text.pack(fill=tk.X)
        # 创建右键菜单
        self.create_context_menu(self.mpc_result_text)
        
        # 底部框架，包含状态和按钮
        bottom_frame = ttk.Frame(self.input_tab)
        bottom_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # 状态标签（左侧）
        self.status_label = ttk.Label(bottom_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # 按钮（右侧）
        button_frame = ttk.Frame(bottom_frame)
        button_frame.pack(side=tk.RIGHT, padx=5)
        
        # 计算按钮
        self.calculate_btn = ttk.Button(button_frame, text="开始计算", command=self.calculate)
        self.calculate_btn.pack(side=tk.RIGHT, padx=5)
        
        # 添加MPC代码验证按钮
        self.verify_mpc_btn = ttk.Button(button_frame, text="验证MPC代码", command=self.verify_mpc_code)
        self.verify_mpc_btn.pack(side=tk.RIGHT, padx=5)
        
        # 添加保存MPC代码按钮（初始隐藏）
        self.save_mpc_btn = ttk.Button(button_frame, text="保存到内置列表", command=self.save_mpc_to_builtin)
        # 初始不显示，只有在从HTML读取数据后才显示
        
        # 添加保存观测地点按钮
        self.save_location_as_btn = ttk.Button(button_frame, text="保存观测地点", command=self.save_location_as)
        
        # 添加管理观测地点按钮
        self.manage_locations_btn = ttk.Button(button_frame, text="管理观测地点", command=self.manage_locations)
        
        # 初始状态
        self.toggle_loc_input()
        
        # 添加窗口关闭事件处理，自动保存位置
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def toggle_coord_input(self, *args):
        """切换经纬度输入方式"""
        # 辅助函数：检查框架中是否包含指定文本的组件
        def has_text_in_frame(frame, text):
            for widget in frame.winfo_children():
                if hasattr(widget, "cget"):
                    try:
                        widget_text = widget.cget("text")
                        if text in widget_text:
                            return True
                    except:
                        pass
            return False
        
        if self.coord_input_type.get() == "decimal":
            # 显示小数格式输入，隐藏度分秒格式输入
            # 同步海拔高度值（从度分秒格式到小数格式）
            try:
                height_dms = float(self.height_var_dms.get())
                self.height_var.set(f"{height_dms:.1f}")
            except (ValueError, TypeError):
                pass
            
            # 显示统一的经纬度输入框架
            for child in self.input_tab.winfo_children():
                if isinstance(child, ttk.LabelFrame):
                    for grandchild in child.winfo_children():
                        # 显示统一的经纬度输入框架
                        if isinstance(grandchild, ttk.Frame) and has_text_in_frame(grandchild, "纬度:"):
                            grandchild.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)
            
            # 隐藏度分秒格式输入
            self.lat_dms_label.grid_remove()
            self.lon_dms_label.grid_remove()
            self.lat_dms_frame.grid_remove()
            self.lon_dms_frame.grid_remove()
        else:
            # 显示度分秒格式输入，隐藏小数格式输入
            # 同步海拔高度值（从小数格式到度分秒格式）
            try:
                height_decimal = float(self.height_var.get())
                self.height_var_dms.set(f"{height_decimal:.1f}")
            except (ValueError, TypeError):
                pass
            
            # 隐藏统一的经纬度输入框架
            for child in self.input_tab.winfo_children():
                if isinstance(child, ttk.LabelFrame):
                    for grandchild in child.winfo_children():
                        # 隐藏统一的经纬度输入框架
                        if isinstance(grandchild, ttk.Frame) and has_text_in_frame(grandchild, "纬度:"):
                            grandchild.grid_remove()
            
            # 显示度分秒格式输入
            self.lat_dms_label.grid(row=3, column=0, sticky=tk.W, pady=5)
            self.lon_dms_label.grid(row=4, column=0, sticky=tk.W, pady=5)
            self.lat_dms_frame.grid(row=3, column=1, sticky=tk.W)
            self.lon_dms_frame.grid(row=4, column=1, sticky=tk.W)
    
    def toggle_loc_input(self):
        # 辅助函数：检查框架中是否包含指定文本的组件
        def has_text_in_frame(frame, text):
            for widget in frame.winfo_children():
                if hasattr(widget, "cget"):
                    try:
                        widget_text = widget.cget("text")
                        if text in widget_text:
                            return True
                    except:
                        pass
            return False
        
        if self.loc_type_var.get() == "1":
            # MPC代码模式
            # 显示MPC代码输入框和卫星NORAD ID
            self.mpc_label.grid(row=1, column=0, sticky=tk.W, pady=5)
            # 重新创建MPC和卫星ID的框架
            for child in self.input_tab.winfo_children():
                if isinstance(child, ttk.LabelFrame):
                    for grandchild in child.winfo_children():
                        if isinstance(grandchild, ttk.Frame) and hasattr(grandchild, 'winfo_children'):
                            # 检查是否包含MPC输入框
                            has_mpc_entry = any(isinstance(w, ttk.Entry) and w == self.mpc_entry for w in grandchild.winfo_children())
                            if has_mpc_entry:
                                grandchild.grid(row=1, column=1, sticky=tk.W, pady=5)
            # 隐藏经纬度和海拔高度输入框
            # 输入方式选择
            self.coord_input_type.set("decimal")
            # 隐藏所有经纬度相关控件
            # 隐藏经纬度输入方式选择
            for child in self.input_tab.winfo_children():
                if isinstance(child, ttk.LabelFrame):
                    for grandchild in child.winfo_children():
                        # 隐藏经纬度输入方式选择
                        if isinstance(grandchild, ttk.Label):
                            try:
                                if "经纬度输入方式" in grandchild.cget("text"):
                                    grandchild.grid_remove()
                            except:
                                pass
                        elif isinstance(grandchild, ttk.Frame) and has_text_in_frame(grandchild, "小数格式"):
                            # 检查是否包含经纬度输入方式选择
                            has_coord_radio = has_text_in_frame(grandchild, "小数格式")
                            if has_coord_radio:
                                grandchild.grid_remove()
                        # 隐藏度分秒格式输入相关控件
                        elif isinstance(grandchild, ttk.Label):
                            try:
                                if "纬度:" in grandchild.cget("text") or "经度:" in grandchild.cget("text"):
                                    grandchild.grid_remove()
                            except:
                                pass
                        elif isinstance(grandchild, ttk.Frame) and has_text_in_frame(grandchild, "°"):
                            grandchild.grid_remove()
                        # 隐藏统一的经纬度输入框架
                        elif isinstance(grandchild, ttk.Frame) and has_text_in_frame(grandchild, "纬度:"):
                            grandchild.grid_remove()
            # 小数格式输入
            self.lat_label.grid_remove()
            self.lat_entry.grid_remove()
            self.lon_label.grid_remove()
            self.lon_entry.grid_remove()
            # 度分秒格式输入
            self.lat_dms_label.grid_remove()
            self.lon_dms_label.grid_remove()
            self.lat_dms_frame.grid_remove()
            self.lon_dms_frame.grid_remove()
            # 海拔高度
            self.height_label.grid_remove()
            self.height_entry.grid_remove()
            self.height_entry_dms.grid_remove()
            # 隐藏其他可能的经纬度相关控件
            if hasattr(self, 'format_note'):
                self.format_note.grid_remove()
            # 隐藏经纬度说明文字
            if hasattr(self, 'coord_note'):
                self.coord_note.grid_remove()
            self.verify_mpc_btn.config(state=tk.NORMAL)
            # 显示验证MPC代码按钮
            if hasattr(self, 'verify_mpc_btn'):
                self.verify_mpc_btn.pack(side=tk.RIGHT, padx=5)
            # 显示MPC代码解析结果框架
            if hasattr(self, 'mpc_result_frame'):
                self.mpc_result_frame.grid(row=11, column=0, columnspan=2, sticky=tk.EW, pady=5)

            # 隐藏手动位置相关按钮
            self.save_location_as_btn.config(state=tk.DISABLED)
            self.save_location_as_btn.pack_forget()
            self.manage_locations_btn.config(state=tk.DISABLED)
            self.manage_locations_btn.pack_forget()
        else:
            # 经纬度模式
            # 隐藏MPC代码输入框和验证按钮
            self.mpc_label.grid_remove()
            # 隐藏MPC相关框架
            for child in self.input_tab.winfo_children():
                if isinstance(child, ttk.LabelFrame):
                    for grandchild in child.winfo_children():
                        if isinstance(grandchild, ttk.Frame) and hasattr(grandchild, 'winfo_children'):
                            # 检查是否包含MPC输入框
                            has_mpc_entry = any(isinstance(w, ttk.Entry) and w == self.mpc_entry for w in grandchild.winfo_children())
                            if has_mpc_entry:
                                grandchild.grid_remove()
            # 隐藏验证MPC代码按钮
            if hasattr(self, 'verify_mpc_btn'):
                self.verify_mpc_btn.pack_forget()
            # 隐藏MPC代码解析结果框架
            if hasattr(self, 'mpc_result_frame'):
                self.mpc_result_frame.grid_remove()
            # 显示经纬度输入方式选择和卫星NORAD ID
            for child in self.input_tab.winfo_children():
                if isinstance(child, ttk.LabelFrame):
                    for grandchild in child.winfo_children():
                        if isinstance(grandchild, ttk.Label):
                            try:
                                if "经纬度输入方式" in grandchild.cget("text"):
                                    grandchild.grid(row=2, column=0, sticky=tk.W, pady=5)
                            except:
                                pass
                        elif isinstance(grandchild, ttk.Frame) and hasattr(grandchild, 'winfo_children'):
                            # 检查是否包含经纬度输入方式选择
                            has_coord_radio = has_text_in_frame(grandchild, "小数格式")
                            if has_coord_radio:
                                grandchild.grid(row=2, column=1, sticky=tk.W, pady=5)
            # 显示经纬度和海拔高度输入框
            # 显示统一的经纬度输入框架
            for child in self.input_tab.winfo_children():
                if isinstance(child, ttk.LabelFrame):
                    for grandchild in child.winfo_children():
                        # 显示统一的经纬度输入框架
                        if isinstance(grandchild, ttk.Frame) and has_text_in_frame(grandchild, "纬度:"):
                            grandchild.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)
            # 根据当前选择的输入方式显示相应的输入框
            self.toggle_coord_input()
            # 启用输入
            self.lat_entry.config(state=tk.NORMAL)
            self.lon_entry.config(state=tk.NORMAL)
            self.height_entry.config(state=tk.NORMAL)
            self.height_entry_dms.config(state=tk.NORMAL)
            self.verify_mpc_btn.config(state=tk.DISABLED)
            # 清空MPC代码解析结果
            self.mpc_result_text.config(state=tk.NORMAL)
            self.mpc_result_text.delete(1.0, tk.END)
            self.mpc_result_text.config(state=tk.DISABLED)
            # 显示手动位置相关按钮
            self.save_location_as_btn.config(state=tk.NORMAL)
            self.save_location_as_btn.pack(side=tk.RIGHT, padx=5)
            self.manage_locations_btn.config(state=tk.NORMAL)
            self.manage_locations_btn.pack(side=tk.RIGHT, padx=5)
            # 显示经纬度说明文字
            if hasattr(self, 'coord_note'):
                self.coord_note.grid(row=4, column=1, sticky=tk.W, pady=2)
        # 隐藏保存MPC按钮
        if hasattr(self, 'save_mpc_btn'):
            self.save_mpc_btn.pack_forget()
        self.current_mpc_data = None
    
    def verify_mpc_code(self):
        """验证MPC代码并显示解析结果"""
        mpc_code = self.mpc_code_var.get().strip()
        if not mpc_code:
            self.mpc_result_var.set("请输入MPC代码")
            self.current_mpc_data = None
            return
        
        observer, msg, data_dict = get_observer_from_mpc(mpc_code, self.builtin_codes)
        if observer is None:
            # 更新Text控件
            self.mpc_result_text.config(state=tk.NORMAL)
            self.mpc_result_text.delete(1.0, tk.END)
            self.mpc_result_text.insert(tk.END, f"错误: {msg}")
            self.mpc_result_text.config(state=tk.DISABLED)
            self.current_mpc_data = None
        else:
            # 更新Text控件
            self.mpc_result_text.config(state=tk.NORMAL)
            self.mpc_result_text.delete(1.0, tk.END)
            self.mpc_result_text.insert(tk.END, msg)
            self.mpc_result_text.config(state=tk.DISABLED)
            # 如果是从HTML读取的数据，保存到current_mpc_data
            if data_dict is not None:
                self.current_mpc_data = {
                    "code": mpc_code.upper(),
                    "data": data_dict
                }
                # 显示保存按钮
                if hasattr(self, 'save_mpc_btn'):
                    self.save_mpc_btn.pack(side=tk.RIGHT, padx=5)
            else:
                self.current_mpc_data = None
                # 隐藏保存按钮
                if hasattr(self, 'save_mpc_btn'):
                    self.save_mpc_btn.pack_forget()
    
    def save_mpc_to_builtin(self):
        """将当前从HTML读取的MPC代码保存到内置列表和配置文件"""
        if self.current_mpc_data is None:
            messagebox.showinfo("提示", "没有可保存的MPC数据")
            return
        
        code = self.current_mpc_data["code"]
        data = self.current_mpc_data["data"]
        
        # 保存到内置列表
        self.builtin_codes[code] = {
            "longitude": data["longitude"],
            "latitude": data["latitude"],
            "height_m": data["height_m"],
            "name": data.get("name", "未知地点")
        }
        
        # 保存到配置文件
        if save_config(self.builtin_codes):
            messagebox.showinfo("成功", f"MPC代码 {code} 已保存到内置列表\n"
                               f"地点: {data.get('name', '未知地点')}\n"
                               f"经度: {data['longitude']:.5f}°\n"
                               f"纬度: {data['latitude']:.5f}°\n"
                               f"高度: {data['height_m']:.1f} m\n\n"
                               f"已永久保存到配置文件，下次启动时仍可使用。")
        else:
            messagebox.showinfo("成功", f"MPC代码 {code} 已保存到内置列表\n"
                               f"地点: {data.get('name', '未知地点')}\n"
                               f"经度: {data['longitude']:.5f}°\n"
                               f"纬度: {data['latitude']:.5f}°\n"
                               f"高度: {data['height_m']:.1f} m\n\n"
                               f"但保存到配置文件失败，重启程序后将丢失此数据。")
        
        # 隐藏保存按钮
        if hasattr(self, 'save_mpc_btn'):
            self.save_mpc_btn.pack_forget()
        self.current_mpc_data = None
    
    def fetch_satellite_data(self):
        """从Heavens-Above获取卫星数据"""
        sat_id = self.sat_id_var.get().strip()
        if not sat_id:
            messagebox.showerror("错误", "请输入卫星NORAD ID")
            return
        
        try:
            # 验证输入是否为数字
            sat_id_num = int(sat_id)
            if sat_id_num <= 0:
                messagebox.showerror("错误", "卫星ID必须是正整数")
                return
        except ValueError:
            messagebox.showerror("错误", "卫星ID必须是数字")
            return
        
        # 更新状态
        self.status_var.set(f"正在从Heavens-Above获取卫星 {sat_id} 的数据...")
        self.root.update()
        
        # 在后台线程中获取数据，避免阻塞GUI
        def fetch_in_thread():
            try:
                tle_line1, tle_line2, std_mag, name, status = fetch_satellite_from_heavens_above(sat_id)
                
                if tle_line1 and tle_line2:
                    # 成功获取数据
                    if std_mag is not None:
                        status_message = "成功从Heavens-Above获取TLE数据和本征星等"
                    else:
                        status_message = "成功从Heavens-Above获取TLE数据；Heavens-Above未提供本征星等"
                        self.root.after(0, lambda: self.mag_var.set("None"))
                    
                    # 在主线程中更新GUI
                    self.root.after(0, lambda: self._update_satellite_data(tle_line1, tle_line2, std_mag, name, status_message))
                else:
                    # TLE获取失败，根据错误类型显示不同消息
                    if status == "not_found":
                        error_status = f"从Heavens-Above获取TLE数据失败：卫星NORAD ID {sat_id} 不存在"
                    elif status == "timeout":
                        error_status = "从Heavens-Above获取TLE数据失败：网站超时"
                    else:
                        error_status = "从Heavens-Above获取TLE数据失败"
                    self.root.after(0, lambda: self.status_var.set(error_status))
                    self.root.after(0, lambda: self.tle_text.delete(1.0, tk.END))
                    self.root.after(0, lambda: self.tle_text.insert(1.0, "获取失败"))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set("从Heavens-Above获取TLE数据失败"))
                self.root.after(0, lambda: self.tle_text.delete(1.0, tk.END))
                self.root.after(0, lambda: self.tle_text.insert(1.0, "获取失败"))

        import threading
        thread = threading.Thread(target=fetch_in_thread)
        thread.daemon = True
        thread.start()
    
    def fetch_tle_from_celestrak(self):
        """从CelesTrak获取TLE，从Heavens-Above获取本征星等"""
        sat_id = self.sat_id_var.get().strip()
        if not sat_id:
            messagebox.showerror("错误", "请输入卫星NORAD ID")
            return
        
        try:
            # 验证输入是否为数字
            sat_id_num = int(sat_id)
            if sat_id_num <= 0:
                messagebox.showerror("错误", "卫星ID必须是正整数")
                return
        except ValueError:
            messagebox.showerror("错误", "卫星ID必须是数字")
            return
        
        # 更新状态
        self.status_var.set(f"正在从CelesTrak获取卫星 {sat_id} 的TLE数据...")
        self.root.update()
        
        # 在后台线程中获取数据，避免阻塞GUI
        def fetch_in_thread():
            try:
                # 从CelesTrak获取TLE
                tle_line1, tle_line2, name, tle_message = fetch_tle_from_celestrak(sat_id)
                
                if tle_line1 and tle_line2:
                    # 从Heavens-Above获取本征星等
                    self.root.after(0, lambda: self.status_var.set(f"正在从Heavens-Above获取本征星等..."))
                    std_mag, mag_status = fetch_std_mag_from_heavens_above(sat_id)
                    
                    # 根据本征星等获取状态构建消息
                    if mag_status == "success":
                        status_message = "成功从CelesTrak获取TLE数据；成功从Heavens-Above获取本征星等"
                    elif mag_status == "timeout":
                        status_message = "成功从CelesTrak获取TLE数据；Heavens-Above超时，本征星等获取失败"
                        self.root.after(0, lambda: self.mag_var.set("None"))
                    else:  # not_provided
                        status_message = "成功从CelesTrak获取TLE数据；Heavens-Above未提供本征星等"
                        self.root.after(0, lambda: self.mag_var.set("None"))
                    
                    # 在主线程中更新GUI
                    self.root.after(0, lambda: self._update_satellite_data(tle_line1, tle_line2, std_mag, name, status_message))
                else:
                    # TLE获取失败，根据错误类型显示不同消息
                    if tle_message == "not_found":
                        error_status = f"从CelesTrak获取TLE数据失败：卫星NORAD ID {sat_id} 不存在"
                    elif tle_message == "timeout":
                        error_status = "从CelesTrak获取TLE数据失败：网站超时"
                    elif tle_message == "forbidden":
                        error_status = "从CelesTrak获取TLE数据失败：访问被拒绝（IP可能被限制）"
                    elif tle_message == "server_error":
                        error_status = "从CelesTrak获取TLE数据失败：服务器错误"
                    else:
                        error_status = "从CelesTrak获取TLE数据失败"
                    self.root.after(0, lambda: self.status_var.set(error_status))
                    self.root.after(0, lambda: self.tle_text.delete(1.0, tk.END))
                    self.root.after(0, lambda: self.tle_text.insert(1.0, "获取失败"))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set("从CelesTrak获取TLE数据失败"))
                self.root.after(0, lambda: self.tle_text.delete(1.0, tk.END))
                self.root.after(0, lambda: self.tle_text.insert(1.0, "获取失败"))

        import threading
        thread = threading.Thread(target=fetch_in_thread)
        thread.daemon = True
        thread.start()

    def fetch_tle_from_n2yo(self):
        """从n2yo.com获取TLE，从Heavens-Above获取本征星等"""
        sat_id = self.sat_id_var.get().strip()
        if not sat_id:
            messagebox.showerror("错误", "请输入卫星NORAD ID")
            return
        
        try:
            # 验证输入是否为数字
            sat_id_num = int(sat_id)
            if sat_id_num <= 0:
                messagebox.showerror("错误", "卫星ID必须是正整数")
                return
        except ValueError:
            messagebox.showerror("错误", "卫星ID必须是数字")
            return
        
        # 更新状态
        self.status_var.set(f"正在从n2yo.com获取卫星 {sat_id} 的TLE数据...")
        self.root.update()
        
        # 在后台线程中获取数据，避免阻塞GUI
        def fetch_in_thread():
            try:
                # 从n2yo.com获取TLE
                tle_line1, tle_line2, name, tle_message = fetch_tle_from_n2yo(sat_id)
                
                if tle_line1 and tle_line2:
                    # 从Heavens-Above获取本征星等
                    self.root.after(0, lambda: self.status_var.set(f"正在从Heavens-Above获取本征星等..."))
                    std_mag, mag_status = fetch_std_mag_from_heavens_above(sat_id)
                    
                    # 根据本征星等获取状态构建消息
                    if mag_status == "success":
                        status_message = "成功从n2yo.com获取TLE数据；成功从Heavens-Above获取本征星等"
                    elif mag_status == "timeout":
                        status_message = "成功从n2yo.com获取TLE数据；Heavens-Above超时，本征星等获取失败"
                        self.root.after(0, lambda: self.mag_var.set("None"))
                    else:  # not_provided
                        status_message = "成功从n2yo.com获取TLE数据；Heavens-Above未提供本征星等"
                        self.root.after(0, lambda: self.mag_var.set("None"))
                    
                    # 在主线程中更新GUI
                    self.root.after(0, lambda: self._update_satellite_data(tle_line1, tle_line2, std_mag, name, status_message))
                else:
                    # TLE获取失败，根据错误类型显示不同消息
                    if tle_message == "not_found":
                        error_status = f"从n2yo.com获取TLE数据失败：卫星NORAD ID {sat_id} 不存在"
                    elif tle_message == "timeout":
                        error_status = "从n2yo.com获取TLE数据失败：网站超时"
                    else:
                        error_status = "从n2yo.com获取TLE数据失败"
                    self.root.after(0, lambda: self.status_var.set(error_status))
                    self.root.after(0, lambda: self.tle_text.delete(1.0, tk.END))
                    self.root.after(0, lambda: self.tle_text.insert(1.0, "获取失败"))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set("从n2yo.com获取TLE数据失败"))
                self.root.after(0, lambda: self.tle_text.delete(1.0, tk.END))
                self.root.after(0, lambda: self.tle_text.insert(1.0, "获取失败"))

        import threading
        thread = threading.Thread(target=fetch_in_thread)
        thread.daemon = True
        thread.start()

    def _update_satellite_data(self, tle_line1, tle_line2, std_mag, name, status_message):
        """更新卫星数据到GUI"""
        # 更新TLE数据
        self.tle_text.delete(1.0, tk.END)
        self.tle_text.insert(1.0, f"{tle_line1}\n{tle_line2}")
        
        # 更新卫星名称
        self.name_var.set(name)
        
        # 更新本征星等
        if std_mag is not None:
            self.mag_var.set(str(std_mag))
        
        # 更新状态栏（不显示卫星名称）
        self.status_var.set(status_message)
    
    def _handle_fetch_error(self, error_message):
        """处理获取数据时的错误"""
        # 在TLE输入框显示"获取失败"
        self.tle_text.delete(1.0, tk.END)
        self.tle_text.insert(1.0, "获取失败")
        self.status_var.set(f"获取数据失败: {error_message}")
    
    def save_manual_location(self):
        """保存手动输入的经纬度到配置文件"""
        try:
            # 获取输入的经纬度和高度
            if self.coord_input_type.get() == "decimal":
                # 小数格式输入
                lat = parse_coordinate(self.lat_var.get())
                lon = parse_coordinate(self.lon_var.get())
            else:
                # 度分秒格式输入
                lat = parse_dms_coordinate(
                    self.lat_deg_var.get(),
                    self.lat_min_var.get(),
                    self.lat_sec_var.get(),
                    self.lat_dir_var.get()
                )
                lon = parse_dms_coordinate(
                    self.lon_deg_var.get(),
                    self.lon_min_var.get(),
                    self.lon_sec_var.get(),
                    self.lon_dir_var.get()
                )
            # 根据当前输入方式获取海拔高度
            if self.coord_input_type.get() == "decimal":
                height_m = float(self.height_var.get())
            else:
                height_m = float(self.height_var_dms.get())
            
            # 更新配置
            self.config["manual_location"] = {
                "latitude": lat,
                "longitude": lon,
                "height_m": height_m
            }
            
            # 保存到配置文件
            if save_config(self.config):
                messagebox.showinfo("成功", f"手动位置已保存\n"
                                   f"纬度: {lat:.5f}°\n"
                                   f"经度: {lon:.5f}°\n"
                                   f"高度: {height_m:.1f} m\n\n"
                                   f"下次启动时将自动加载这些值。")
            else:
                messagebox.showinfo("成功", f"手动位置已保存到内存\n"
                                   f"纬度: {lat:.5f}°\n"
                                   f"经度: {lon:.5f}°\n"
                                   f"高度: {height_m:.1f} m\n\n"
                                   f"但保存到配置文件失败，重启程序后将丢失此数据。")
        except ValueError as e:
            messagebox.showerror("错误", f"请输入有效的经纬度和高度值: {e}")
    
    def save_location_as(self):
        """将当前位置保存为命名位置"""
        try:
            # 获取输入的经纬度和高度
            if self.coord_input_type.get() == "decimal":
                # 小数格式输入
                lat = parse_coordinate(self.lat_var.get())
                lon = parse_coordinate(self.lon_var.get())
            else:
                # 度分秒格式输入
                lat = parse_dms_coordinate(
                    self.lat_deg_var.get(),
                    self.lat_min_var.get(),
                    self.lat_sec_var.get(),
                    self.lat_dir_var.get()
                )
                lon = parse_dms_coordinate(
                    self.lon_deg_var.get(),
                    self.lon_min_var.get(),
                    self.lon_sec_var.get(),
                    self.lon_dir_var.get()
                )
            # 根据当前输入方式获取海拔高度
            if self.coord_input_type.get() == "decimal":
                height_m = float(self.height_var.get())
            else:
                height_m = float(self.height_var_dms.get())
            
            # 创建一个新窗口用于输入位置名称
            save_window = tk.Toplevel(self.root)
            save_window.title("保存位置")
            save_window.geometry("300x150")
            save_window.transient(self.root)
            save_window.grab_set()
            
            # 添加标签和输入框
            ttk.Label(save_window, text="位置名称:").pack(pady=10)
            name_var = tk.StringVar()
            ttk.Entry(save_window, textvariable=name_var, width=30).pack(pady=5)
            
            # 保存按钮
            def on_save():
                name = name_var.get().strip()
                if not name:
                    messagebox.showerror("错误", "请输入位置名称")
                    return
                
                # 保存位置
                self.config["saved_locations"][name] = {
                    "latitude": lat,
                    "longitude": lon,
                    "height_m": height_m
                }
                
                # 保存到配置文件
                if save_config(self.config):
                    messagebox.showinfo("成功", f"位置 '{name}' 已保存\n"
                                       f"纬度: {lat:.5f}°\n"
                                       f"经度: {lon:.5f}°\n"
                                       f"高度: {height_m:.1f} m")
                else:
                    messagebox.showinfo("成功", f"位置 '{name}' 已保存到内存\n"
                                       f"但保存到配置文件失败，重启程序后将丢失此数据。")
                
                save_window.destroy()
            
            ttk.Button(save_window, text="保存", command=on_save).pack(pady=10)
            
        except ValueError as e:
            messagebox.showerror("错误", f"请输入有效的经纬度和高度值: {e}")
    
    def on_close(self):
        """窗口关闭时自动保存位置"""
        try:
            # 只有在经纬度模式下才保存
            if self.loc_type_var.get() == "2":
                # 获取当前输入的经纬度和高度
                if self.coord_input_type.get() == "decimal":
                    # 小数格式输入
                    lat = parse_coordinate(self.lat_var.get())
                    lon = parse_coordinate(self.lon_var.get())
                else:
                    # 度分秒格式输入
                    lat = parse_dms_coordinate(
                        self.lat_deg_var.get(),
                        self.lat_min_var.get(),
                        self.lat_sec_var.get(),
                        self.lat_dir_var.get()
                    )
                    lon = parse_dms_coordinate(
                        self.lon_deg_var.get(),
                        self.lon_min_var.get(),
                        self.lon_sec_var.get(),
                        self.lon_dir_var.get()
                    )
                # 根据当前输入方式获取海拔高度
                if self.coord_input_type.get() == "decimal":
                    height_m = float(self.height_var.get())
                else:
                    height_m = float(self.height_var_dms.get())
                
                # 更新配置
                self.config["manual_location"] = {
                    "latitude": lat,
                    "longitude": lon,
                    "height_m": height_m
                }
                
                # 保存到配置文件
                save_config(self.config)
        except ValueError:
            # 如果输入无效，不保存
            pass
        finally:
            # 关闭窗口
            self.root.destroy()
    
    def manage_locations(self):
        """管理保存的位置"""
        # 创建一个新窗口
        manage_window = tk.Toplevel(self.root)
        manage_window.title("管理位置")
        manage_window.geometry("400x400")  # 进一步增加窗口高度
        manage_window.transient(self.root)
        manage_window.grab_set()
        
        # 创建列表框
        list_frame = ttk.LabelFrame(manage_window, text="保存的位置", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建树状视图
        tree = ttk.Treeview(list_frame, columns=("name", "lat", "lon", "height"), show="headings")
        tree.heading("name", text="名称")
        tree.heading("lat", text="纬度")
        tree.heading("lon", text="经度")
        tree.heading("height", text="高度")
        
        tree.column("name", width=100)
        tree.column("lat", width=80)
        tree.column("lon", width=80)
        tree.column("height", width=80)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)
        
        # 填充数据
        for name, data in self.config["saved_locations"].items():
            tree.insert("", tk.END, values=(name, f"{data['latitude']:.5f}", f"{data['longitude']:.5f}", f"{data['height_m']:.1f}"))
        
        # 按钮框架
        button_frame = ttk.Frame(manage_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 选择按钮
        def on_select():
            selected = tree.selection()
            if not selected:
                messagebox.showinfo("提示", "请选择一个位置")
                return
            
            item = tree.item(selected[0])
            name = item["values"][0]
            data = self.config["saved_locations"][name]
            
            # 更新输入框
            self.lat_var.set(str(data["latitude"]))
            self.lon_var.set(str(data["longitude"]))
            self.height_var.set(str(data["height_m"]))
            self.height_var_dms.set(str(data["height_m"]))
            
            manage_window.destroy()
            messagebox.showinfo("成功", f"已选择位置: {name}")
        
        # 删除按钮
        def on_delete():
            selected = tree.selection()
            if not selected:
                messagebox.showinfo("提示", "请选择一个位置")
                return
            
            item = tree.item(selected[0])
            name = item["values"][0]
            
            if messagebox.askyesno("确认", f"确定要删除位置 '{name}' 吗？"):
                del self.config["saved_locations"][name]
                save_config(self.config)
                tree.delete(selected[0])
                messagebox.showinfo("成功", f"位置 '{name}' 已删除")
        
        ttk.Button(button_frame, text="选择", command=on_select).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="删除", command=on_delete).pack(side=tk.LEFT, padx=5)
    
    def show_result_window(self):
        if self.result_window and self.result_window.winfo_exists():
            self.update_result_window()
            self.result_window.lift()
            return
        
        self.result_window = tk.Toplevel(self.root)
        self.result_window.title("计算结果")
        self.result_window.geometry("1400x850")
        
        main_frame = ttk.Frame(self.result_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        info_frame = ttk.LabelFrame(main_frame, text="观测信息", padding="10")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        satellite_name = self.name_var.get().strip() or "Unknown Satellite"
        ttk.Label(info_frame, text="卫星名称:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(info_frame, text=satellite_name).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(info_frame, text="观测地点:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        
        if self.loc_type_var.get() == "1":
            mpc_code = self.mpc_code_var.get().strip()
            observer, msg, data_dict = get_observer_from_mpc(mpc_code, self.builtin_codes)
            if data_dict and "name" in data_dict:
                lat_dms = decimal_to_dms(float(data_dict['latitude']), is_latitude=True)
                lon_dms = decimal_to_dms(float(data_dict['longitude']), is_latitude=False)
                location_info = f"{data_dict['name']}，纬度: {lat_dms}，经度: {lon_dms}"
            else:
                if mpc_code.upper() in self.builtin_codes and "name" in self.builtin_codes[mpc_code.upper()]:
                    data = self.builtin_codes[mpc_code.upper()]
                    lat_dms = decimal_to_dms(float(data['latitude']), is_latitude=True)
                    lon_dms = decimal_to_dms(float(data['longitude']), is_latitude=False)
                    location_info = f"{data['name']}，纬度: {lat_dms}，经度: {lon_dms}"
                else:
                    location_info = f"MPC代码: {mpc_code}, {msg}"
        else:
            if self.coord_input_type.get() == "decimal":
                lat = parse_coordinate(self.lat_var.get())
                lon = parse_coordinate(self.lon_var.get())
            else:
                lat = parse_dms_coordinate(
                    self.lat_deg_var.get(),
                    self.lat_min_var.get(),
                    self.lat_sec_var.get(),
                    self.lat_dir_var.get()
                )
                lon = parse_dms_coordinate(
                    self.lon_deg_var.get(),
                    self.lon_min_var.get(),
                    self.lon_sec_var.get(),
                    self.lon_dir_var.get()
                )
            # 根据当前输入方式获取海拔高度
            if self.coord_input_type.get() == "decimal":
                height_m = float(self.height_var.get())
            else:
                height_m = float(self.height_var_dms.get())
            lat_dms = decimal_to_dms(lat, is_latitude=True)
            lon_dms = decimal_to_dms(lon, is_latitude=False)
            location_info = f"纬度: {lat_dms}，经度: {lon_dms}，海拔高度: {height_m}m"
        
        ttk.Label(info_frame, text=location_info, wraplength=1300).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2, columnspan=3)
        
        self.result_notebook = ttk.Notebook(main_frame)
        self.result_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.data_tab = ttk.Frame(self.result_notebook)
        self.result_notebook.add(self.data_tab, text="星历数据")
        
        self.chart_tab = ttk.Frame(self.result_notebook)
        self.result_notebook.add(self.chart_tab, text="曲线图表")
        
        # 绑定标签页切换事件，根据当前页面显示/隐藏导出Excel按钮
        self.result_notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        
        self._setup_data_tab()
        self._setup_chart_tab()
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        self._bottom_export_btn = ttk.Button(button_frame, text="导出Excel", command=self._export_excel)
        self._bottom_export_btn.pack(side=tk.RIGHT, padx=5)
        
        # 创建状态标签，显示计算结果统计信息
        self.result_status_label = ttk.Label(button_frame, text="")
        self.result_status_label.pack(side=tk.LEFT, padx=5)
        
        self.update_result_window()
    
    def _update_result_status(self):
        """更新结果窗口底部的状态标签"""
        if hasattr(self, 'result_status_label') and self.result_status_label:
            data_count = len(self.ephemeris_data)
            duration_str = ""
            if hasattr(self, 'calc_duration'):
                duration_str = f"计算完成！用时 {self.calc_duration:.2f} 秒，"
            self.result_status_label.config(
                text=f"{duration_str}输出 {data_count} 条数据。提示：地平高度 > 10° 且太阳高度 < -12° 为较佳观测条件。"
            )
    
    def _setup_data_tab(self):
        result_frame = ttk.LabelFrame(self.data_tab, text="星历数据", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 筛选区域
        filter_frame = ttk.LabelFrame(result_frame, text="筛选条件", padding="5")
        filter_frame.pack(fill=tk.X, padx=5, pady=5)

        # 地平高度筛选
        ttk.Label(filter_frame, text="地平高度 ≥").grid(row=0, column=0, padx=5, pady=2)
        self.filter_alt_var = tk.StringVar(value="")
        self.filter_alt_entry = ttk.Entry(filter_frame, textvariable=self.filter_alt_var, width=8)
        self.filter_alt_entry.grid(row=0, column=1, padx=5, pady=2)
        self.setup_decimal_input(self.filter_alt_entry)
        ttk.Label(filter_frame, text="°").grid(row=0, column=2, padx=2, pady=2)

        # 太阳高度筛选
        ttk.Label(filter_frame, text="太阳高度 ≤").grid(row=0, column=3, padx=5, pady=2)
        self.filter_sun_alt_var = tk.StringVar(value="")
        self.filter_sun_alt_entry = ttk.Entry(filter_frame, textvariable=self.filter_sun_alt_var, width=8)
        self.filter_sun_alt_entry.grid(row=0, column=4, padx=5, pady=2)
        self.setup_decimal_input(self.filter_sun_alt_entry)
        ttk.Label(filter_frame, text="°").grid(row=0, column=5, padx=2, pady=2)

        # 亮度筛选
        ttk.Label(filter_frame, text="星等 ≤").grid(row=0, column=6, padx=5, pady=2)
        self.filter_mag_var = tk.StringVar(value="")
        self.filter_mag_entry = ttk.Entry(filter_frame, textvariable=self.filter_mag_var, width=8)
        self.filter_mag_entry.grid(row=0, column=7, padx=5, pady=2)
        self.setup_decimal_input(self.filter_mag_entry)
        
        # 筛选和重置按钮
        ttk.Button(filter_frame, text="应用筛选", command=self._apply_filter).grid(row=0, column=8, padx=10, pady=2)
        ttk.Button(filter_frame, text="重置", command=self._reset_filter).grid(row=0, column=9, padx=5, pady=2)
        
        # 筛选结果显示标签
        self.filter_result_label = ttk.Label(filter_frame, text="")
        self.filter_result_label.grid(row=0, column=10, padx=10, pady=2)
        
        columns = ("time", "ra", "dec", "mag", "alt", "az", "speed", "pa", "range", "surf_range", "sun_alt", "sun_sep", "moon_alt", "moon_sep")
        self.result_tree = ttk.Treeview(result_frame, columns=columns, show="headings")
        
        self.result_tree.heading("time", text="Date (UTC+8)")
        self.result_tree.heading("ra", text="RA")
        self.result_tree.heading("dec", text="Dec")
        self.result_tree.heading("mag", text="Mag")
        self.result_tree.heading("alt", text="Alt (°)")
        self.result_tree.heading("az", text="Az (°)")
        self.result_tree.heading("speed", text="Speed (''/min)")
        self.result_tree.heading("pa", text="PA (°)")
        self.result_tree.heading("range", text="Obs Dist (km)")
        self.result_tree.heading("surf_range", text="Orb Alt (km)")
        self.result_tree.heading("sun_alt", text="Sun Alt (°)")
        self.result_tree.heading("sun_sep", text="Sun Sep (°)")
        self.result_tree.heading("moon_alt", text="Moon Alt (°)")
        self.result_tree.heading("moon_sep", text="Moon Sep (°)")
        
        self.result_tree.column("time", width=120, anchor=tk.CENTER)
        self.result_tree.column("ra", width=100, anchor=tk.CENTER)
        self.result_tree.column("dec", width=100, anchor=tk.CENTER)
        self.result_tree.column("mag", width=60, anchor=tk.CENTER)
        self.result_tree.column("alt", width=60, anchor=tk.CENTER)
        self.result_tree.column("az", width=60, anchor=tk.CENTER)
        self.result_tree.column("speed", width=80, anchor=tk.CENTER)
        self.result_tree.column("pa", width=60, anchor=tk.CENTER)
        self.result_tree.column("range", width=80, anchor=tk.CENTER)
        self.result_tree.column("surf_range", width=80, anchor=tk.CENTER)
        self.result_tree.column("sun_alt", width=60, anchor=tk.CENTER)
        self.result_tree.column("sun_sep", width=80, anchor=tk.CENTER)
        self.result_tree.column("moon_alt", width=60, anchor=tk.CENTER)
        self.result_tree.column("moon_sep", width=80, anchor=tk.CENTER)
        
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_tree.yview)
        self.result_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_tree.pack(fill=tk.BOTH, expand=True)
        
        current_cell = [None]
        selection_start = [None]
        selection_end = [None]
        is_selecting = [False]
        
        def copy_selected_cell():
            if not current_cell[0]:
                return
            item, column = current_cell[0]
            column_index = int(column[1:]) - 1
            values = self.result_tree.item(item, "values")
            if column_index < len(values):
                cell_value = str(values[column_index])
                self.result_window.clipboard_clear()
                self.result_window.clipboard_append(cell_value)
        
        def copy_selected_row():
            selected = self.result_tree.selection()
            if not selected:
                return
            row_data = []
            for item in selected:
                values = self.result_tree.item(item, "values")
                row_data.append("\t".join(str(v) for v in values))
            self.result_window.clipboard_clear()
            self.result_window.clipboard_append("\n".join(row_data))
        
        def copy_all_data():
            if not self.ephemeris_data:
                return
            headers = ["时间(UTC+8)", "RA", "Dec", "Mag", "Alt (°)", "Az (°)", "Sun Alt (°)", "Moon Alt (°)", "Sat-Moon (°)", "Sat-Sun (°)", "Speed (''/min)", "PA (°)"]
            header_str = "\t".join(headers)
            data_rows = []
            for item in self.result_tree.get_children():
                values = self.result_tree.item(item, "values")
                data_rows.append("\t".join(str(v) for v in values))
            all_data = header_str + "\n" + "\n".join(data_rows)
            self.result_window.clipboard_clear()
            self.result_window.clipboard_append(all_data)
        
        def export_excel():
            if not self.ephemeris_data:
                messagebox.showinfo("提示", "没有数据可导出")
                return
            try:
                try:
                    import pandas as pd
                except ImportError:
                    import subprocess
                    import sys
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "openpyxl"])
                    import pandas as pd
                from tkinter import filedialog
                filename = filedialog.asksaveasfilename(
                    defaultextension=".xlsx",
                    filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                    initialfile="ephemeris.xlsx"
                )
                if filename:
                    df = pd.DataFrame(self.ephemeris_data)
                    df.to_excel(filename, index=False)
                    messagebox.showinfo("成功", f"Excel文件导出成功：{filename}")
            except Exception as e:
                messagebox.showerror("错误", f"Excel导出失败: {e}")
        
        tree_menu = tk.Menu(self.result_tree, tearoff=0)
        tree_menu.add_command(label="复制单元格", command=copy_selected_cell)
        tree_menu.add_command(label="复制选中行", command=copy_selected_row)
        tree_menu.add_command(label="复制所有数据", command=copy_all_data)
        
        def show_tree_menu(event):
            region = self.result_tree.identify_region(event.x, event.y)
            if region == "cell":
                item = self.result_tree.identify_row(event.y)
                column = self.result_tree.identify_column(event.x)
                current_cell[0] = (item, column)
                tree_menu.post(event.x_root, event.y_root)
        
        def on_mouse_press(event):
            region = self.result_tree.identify_region(event.x, event.y)
            if region == "cell":
                item = self.result_tree.identify_row(event.y)
                column = self.result_tree.identify_column(event.x)
                if item in self.result_tree.selection():
                    self.result_tree.selection_remove(item)
                else:
                    selection_start[0] = (item, column)
                    is_selecting[0] = True
                    self.result_tree.selection_remove(self.result_tree.selection())
                    self.result_tree.selection_add(item)
        
        def on_mouse_motion(event):
            if is_selecting[0]:
                region = self.result_tree.identify_region(event.x, event.y)
                if region == "cell":
                    item = self.result_tree.identify_row(event.y)
                    column = self.result_tree.identify_column(event.x)
                    selection_end[0] = (item, column)
                    all_items = self.result_tree.get_children()
                    start_idx = all_items.index(selection_start[0][0])
                    end_idx = all_items.index(item)
                    if start_idx <= end_idx:
                        selected_items = all_items[start_idx:end_idx+1]
                    else:
                        selected_items = all_items[end_idx:start_idx+1]
                    self.result_tree.selection_remove(self.result_tree.selection())
                    for it in selected_items:
                        self.result_tree.selection_add(it)
        
        def on_mouse_release(event):
            is_selecting[0] = False
        
        self.result_tree.bind("<Button-3>", show_tree_menu)
        self.result_tree.bind("<Button-1>", on_mouse_press)
        self.result_tree.bind("<B1-Motion>", on_mouse_motion)
        self.result_tree.bind("<ButtonRelease-1>", on_mouse_release)
    
    def _apply_filter(self):
        """应用筛选条件"""
        if not self.ephemeris_data:
            return
        
        # 获取筛选条件
        alt_min = self.filter_alt_var.get().strip()
        sun_alt_max = self.filter_sun_alt_var.get().strip()
        mag_max = self.filter_mag_var.get().strip()
        
        # 清空当前显示
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        
        # 筛选并显示数据
        filtered_count = 0
        for data in self.ephemeris_data:
            try:
                # 检查地平高度条件
                if alt_min:
                    alt_val = float(data["Alt"].strip())
                    if alt_val < float(alt_min):
                        continue
                
                # 检查太阳高度条件
                if sun_alt_max:
                    sun_alt_val = float(data["Sun Alt"].strip())
                    if sun_alt_val > float(sun_alt_max):
                        continue
                
                # 检查星等条件
                if mag_max:
                    mag_str = data["Mag"].strip()
                    if mag_str in ["N/A", "不可见", "在地影中", "暗"]:
                        continue
                    mag_val = float(mag_str)
                    if mag_val > float(mag_max):
                        continue
                
                # 通过筛选，添加到表格
                self.result_tree.insert("", tk.END, values=(
                    data["Date(UTC+8)"],
                    data["RA"],
                    data["Dec"],
                    data["Mag"],
                    data["Alt"],
                    data["Az"],
                    data["Speed"],
                    data["PA"],
                    data["Obs Dist"],
                    data["Orbit Alt"],
                    data["Sun Alt"],
                    data["Sun Sep"],
                    data["Moon Alt"],
                    data["Moon Sep"]
                ))
                filtered_count += 1
            except:
                continue
        
        # 更新筛选结果显示
        total_count = len(self.ephemeris_data)
        self.filter_result_label.config(text=f"显示 {filtered_count} / {total_count} 条")
    
    def _reset_filter(self):
        """重置筛选条件"""
        self.filter_alt_var.set("")
        self.filter_sun_alt_var.set("")
        self.filter_mag_var.set("")
        self.filter_result_label.config(text="")
        self.update_result_window()
    
    def _on_tab_changed(self, event):
        """标签页切换时控制导出Excel按钮的显示"""
        current_tab = self.result_notebook.select()
        if current_tab == str(self.chart_tab):
            # 曲线图表页面，隐藏导出Excel按钮
            self._bottom_export_btn.pack_forget()
        else:
            # 星历数据页面，显示导出Excel按钮
            self._bottom_export_btn.pack(side=tk.RIGHT, padx=5)
    
    def _setup_chart_tab(self):
        chart_frame = ttk.Frame(self.chart_tab, padding="5")
        chart_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig = Figure(figsize=(12, 4.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        control_frame = ttk.LabelFrame(chart_frame, text="曲线选择", padding="10")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.curve_vars = {}
        curve_options = [
            ("sat_mag", "卫星亮度", True),
            ("sat_alt", "卫星地平高度", False),
            ("earth_dist", "与地球表面距离", False),
            ("obs_dist", "与观测地点距离", False),
            ("sun_alt", "太阳地平高度", False),
            ("sun_sep", "太阳角距离", False),
            ("moon_alt", "月亮地平高度", False),
            ("moon_sep", "月亮角距离", False),
        ]
        
        # 颜色定义
        cb_colors = {
            "sat_mag": "#000000",
            "sat_alt": "#ff7f0e",
            "earth_dist": "#2ca02c",
            "obs_dist": "#d62728",
            "sun_alt": "#9467bd",
            "sun_sep": "#8c564b",
            "moon_alt": "#e377c2",
            "moon_sep": "#7f7f7f",
        }
        
        # 创建复选框容器框架，用于居中
        cb_container = ttk.Frame(control_frame)
        cb_container.grid(row=0, column=0, columnspan=8, pady=5)
        
        for i, (key, label, default) in enumerate(curve_options):
            var = tk.BooleanVar(value=default)
            self.curve_vars[key] = var
            
            # 创建复选框，使用指定颜色显示文字
            cb = tk.Checkbutton(cb_container, text=label, variable=var, 
                                command=self._update_chart,
                                fg=cb_colors[key])  # 文字颜色
            cb.pack(side=tk.LEFT, padx=10, pady=5)
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=1, column=0, columnspan=8, pady=10)
        
        ttk.Button(btn_frame, text="全选", command=self._select_all_curves).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消全选", command=self._deselect_all_curves).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="框选放大", command=self._chart_zoom_rect).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="后退缩小", command=self._chart_back).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="刷新", command=self._update_chart).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="保存为PNG", command=self._chart_save_png).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="保存为JPG", command=self._chart_save_jpg).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="保存为PDF", command=self._chart_save_pdf).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="导出Excel", command=self._export_excel).pack(side=tk.LEFT, padx=5)
        
        # 让复选框容器和按钮框架在control_frame中居中
        control_frame.grid_columnconfigure(0, weight=1)
        
        self._update_chart()
    
    def _update_chart(self):
        if not self.ephemeris_data:
            return
        
        # 清空整个figure
        self.fig.clear()
        
        times = []
        for data in self.ephemeris_data:
            try:
                dt = datetime.strptime(data["Date(UTC+8)"], "%Y-%m-%d %H:%M:%S")
                times.append(dt)
            except:
                continue
        
        if not times:
            return
        
        # 定义曲线分组 - 每组使用一个Y轴
        # 地平高度：太阳地平高度、月亮地平高度、卫星地平高度
        # 角距离：太阳角距离、月亮角距离
        # 星等：卫星星等（使用曲线颜色作为Y轴颜色）
        # 距离1：与地球表面距离
        # 距离2：与观测地点距离
        curve_groups = {
            "地平高度": {
                "keys": ["sat_alt", "sun_alt", "moon_alt"],
                "color": "#1f77b4",
                "label": "地平高度",
                "ylabel": "地平高度(°)"
            },
            "角距离": {
                "keys": ["sun_sep", "moon_sep"],
                "color": "#ff7f0e",
                "label": "角距离",
                "ylabel": "角距离(°)"
            },
            "星等": {
                "keys": ["sat_mag"],
                "color": None,  # 使用曲线自身的颜色
                "label": "星等",
                "ylabel": "星等"
            },
            "距离1": {
                "keys": ["earth_dist"],
                "color": "#2ca02c",
                "label": "与地球表面距离",
                "ylabel": "与地球表面距离(km)"
            },
            "距离2": {
                "keys": ["obs_dist"],
                "color": "#d62728",
                "label": "与观测地点距离",
                "ylabel": "与观测地点距离(km)"
            }
        }
        
        colors = {
            "sat_mag": "#000000",  # 黑色
            "sat_alt": "#ff7f0e",
            "earth_dist": "#2ca02c",
            "obs_dist": "#d62728",
            "sun_alt": "#9467bd",
            "sun_sep": "#8c564b",
            "moon_alt": "#e377c2",
            "moon_sep": "#7f7f7f",
        }
        
        labels = {
            "sat_mag": "卫星星等",
            "sat_alt": "卫星地平高度",
            "earth_dist": "与地球表面距离",
            "obs_dist": "与观测地点距离",
            "sun_alt": "太阳地平高度",
            "sun_sep": "太阳角距离",
            "moon_alt": "月亮地平高度",
            "moon_sep": "月亮角距离",
        }
        
        active_curves = [k for k, v in self.curve_vars.items() if v.get()]
        
        if not active_curves:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlabel("时间 (UTC+8)")
            self.ax.set_ylabel("数值")
            self.ax.set_title("请选择要显示的曲线")
            self._all_axes = [self.ax]
            self.canvas.draw()
            return
        
        # 确定哪些组有激活的曲线
        active_groups = []
        for group_name, group_info in curve_groups.items():
            if any(k in active_curves for k in group_info["keys"]):
                active_groups.append(group_name)
        
        if not active_groups:
            return
        
        # 如果星等被选中，确保它排在第一位（显示在左侧）
        if "星等" in active_groups:
            active_groups.remove("星等")
            active_groups.insert(0, "星等")
        
        # 创建子图，每个组一个Y轴
        if len(active_groups) == 1:
            # 只有一个组，使用单Y轴
            self.ax = self.fig.add_subplot(111)
            axes = {active_groups[0]: self.ax}
            self._all_axes = [self.ax]  # 保存所有坐标轴
        else:
            # 多个组，使用多Y轴
            self.ax = self.fig.add_subplot(111)
            axes = {active_groups[0]: self.ax}
            self._all_axes = [self.ax]  # 保存所有坐标轴
            for i, group_name in enumerate(active_groups[1:], 1):
                ax = self.ax.twinx()
                ax.spines["right"].set_position(("outward", 60 * (i - 1)))
                axes[group_name] = ax
                self._all_axes.append(ax)  # 添加到所有坐标轴列表
        
        lines = []
        
        # 绘制夜间灰色区域
        sun_alt_values = [data["_sun_alt"] for data in self.ephemeris_data]
        
        # 灰色级别定义（从浅到深）
        # 太阳高度 > 0°: 白天（无灰色）
        # -6° < 太阳高度 ≤ 0°: 民用曙暮光（浅灰）
        # -12° < 太阳高度 ≤ -6°: 航海曙暮光（中灰）
        # -18° < 太阳高度 ≤ -12°: 天文曙暮光（深灰）
        # 太阳高度 ≤ -18°: 深夜（最深灰）
        
        gray_levels = [
            (0, None, "white"),           # 白天
            (-6, 0, "#e8e8e8"),          # 浅灰 - 民用曙暮光
            (-12, -6, "#c0c0c0"),        # 中灰 - 航海曙暮光
            (-18, -12, "#909090"),        # 深灰 - 天文曙暮光
            (None, -18, "#606060"),       # 最深灰 - 深夜
        ]
        
        # 绘制灰色区域
        for i in range(len(times) - 1):
            sun_alt = sun_alt_values[i]
            if sun_alt is None:
                continue
            
            t1 = times[i]
            t2 = times[i + 1]
            
            # 确定当前时间点属于哪个灰色级别
            for level_min, level_max, color in gray_levels:
                if level_max is None and sun_alt > level_min:
                    # 白天，不绘制
                    break
                if level_min is None and sun_alt <= level_max:
                    # 最深灰
                    self.ax.axvspan(t1, t2, facecolor=color, alpha=0.3, edgecolor='none', zorder=0)
                    break
                if level_min < sun_alt <= level_max:
                    self.ax.axvspan(t1, t2, facecolor=color, alpha=0.3, edgecolor='none', zorder=0)
                    break
        
        # 按组绘制曲线
        for group_name in active_groups:
            group_info = curve_groups[group_name]
            ax = axes[group_name]
            group_lines = []
            
            for curve_key in group_info["keys"]:
                if curve_key not in active_curves:
                    continue
                
                values = []
                for data in self.ephemeris_data:
                    try:
                        if curve_key == "sat_mag":
                            # 使用高精度星等值
                            if data["_mag"] is not None:
                                values.append(data["_mag"])
                            else:
                                values.append(None)
                        elif curve_key == "sat_alt":
                            values.append(data["_alt"])
                        elif curve_key == "earth_dist":
                            values.append(data["_surface"])
                        elif curve_key == "obs_dist":
                            values.append(data["_range"])
                        elif curve_key == "sun_alt":
                            values.append(data["_sun_alt"])
                        elif curve_key == "sun_sep":
                            values.append(data["_sun_sep"])
                        elif curve_key == "moon_alt":
                            values.append(data["_moon_alt"])
                        elif curve_key == "moon_sep":
                            values.append(data["_moon_sep"])
                    except Exception as e:
                        values.append(None)
                
                valid_times = [t for t, v in zip(times, values) if v is not None]
                valid_values = [v for v in values if v is not None]
                
                if valid_times:
                    # 星等只显示数据点，其他曲线显示连线
                    if curve_key == "sat_mag":
                        line_width = 0
                        zorder = 10  # 星等曲线显示在最上层
                    else:
                        line_width = 1.5
                        zorder = 1
                    line, = ax.plot(valid_times, valid_values, 
                                   color=colors[curve_key], 
                                   label=labels[curve_key],
                                   linewidth=line_width,
                                   marker='o',
                                   markersize=3,
                                   zorder=zorder)
                    lines.append(line)
                    group_lines.append(line)
            
            # 设置该组的Y轴标签
            if group_lines:
                # 所有Y轴统一使用黑色
                y_axis_color = "#000000"  # 黑色
                
                ax.set_ylabel(group_info["ylabel"], color=y_axis_color)
                ax.tick_params(axis='y', labelcolor=y_axis_color)
                
                # 如果是星等组，反转Y轴（小的数字在上，大的数字在下）
                if group_name == "星等":
                    ax.invert_yaxis()
        
        self.ax.set_xlabel("时间 (UTC+8)")
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # 设置X轴标签水平显示（不旋转）
        for label in self.ax.get_xticklabels():
            label.set_rotation(0)
            label.set_horizontalalignment('center')
        
        # 不显示图注，颜色标识已在复选框处显示
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _select_all_curves(self):
        for var in self.curve_vars.values():
            var.set(True)
        self._update_chart()

    def _deselect_all_curves(self):
        for var in self.curve_vars.values():
            var.set(False)
        self._update_chart()

    def _show_mag_only(self):
        for key, var in self.curve_vars.items():
            var.set(key == "sat_mag")
        self._update_chart()

    def _save_view(self):
        """保存当前视图到历史记录"""
        if not hasattr(self, '_view_history'):
            self._view_history = []
            self._view_index = -1
        
        # 保存当前视图范围
        current_view = {
            'xlim': self.ax.get_xlim(),
            'ylim': self.ax.get_ylim()
        }
        
        # 如果当前不是最后一个视图，删除后面的历史
        if self._view_index < len(self._view_history) - 1:
            self._view_history = self._view_history[:self._view_index + 1]
        
        # 添加新视图
        self._view_history.append(current_view)
        self._view_index += 1
        
        # 限制历史记录数量
        if len(self._view_history) > 20:
            self._view_history.pop(0)
            self._view_index -= 1

    def _chart_back(self):
        """后退到上一个视图"""
        if hasattr(self, '_view_history') and self._view_index > 0:
            self._view_index -= 1
            view = self._view_history[self._view_index]
            self.ax.set_xlim(view['xlim'])
            self.ax.set_ylim(view['ylim'])
            self.canvas.draw()



    def _chart_zoom_rect(self):
        """切换框选放大模式"""
        # 检查当前是否已启用框选模式
        if hasattr(self, '_rect_select_mode') and self._rect_select_mode:
            # 如果已启用，则禁用
            self._disable_zoom_rect()
        else:
            # 如果未启用，则启用
            # 保存当前视图
            self._save_view()
            
            # 启用框选模式标志
            self._rect_select_mode = True
            
            # 连接鼠标事件
            self._rect_cid_press = self.canvas.mpl_connect('button_press_event', self._on_rect_press)
            self._rect_cid_motion = self.canvas.mpl_connect('motion_notify_event', self._on_rect_motion)
            self._rect_cid_release = self.canvas.mpl_connect('button_release_event', self._on_rect_release)
    
    def _on_rect_press(self, event):
        """框选开始"""
        if not hasattr(self, '_rect_select_mode') or not self._rect_select_mode:
            return
        if event.button != 1:
            return
        
        # 保存像素位置（而不是数据坐标）
        self._rect_start_pixel = (event.x, event.y)
        self._rect_patch = None
    
    def _on_rect_motion(self, event):
        """框选中 - 绘制矩形"""
        if not hasattr(self, '_rect_select_mode') or not self._rect_select_mode:
            return
        if not hasattr(self, '_rect_start_pixel') or self._rect_start_pixel is None:
            return
        
        # 删除之前的矩形
        self._remove_rect_patch()
        
        # 获取像素位置
        x1_pixel, y1_pixel = self._rect_start_pixel
        x2_pixel, y2_pixel = event.x, event.y
        
        # 将像素位置转换为数据坐标
        inv_transform = self.ax.transData.inverted()
        start_data = inv_transform.transform((x1_pixel, y1_pixel))
        end_data = inv_transform.transform((x2_pixel, y2_pixel))
        
        x1, y1 = start_data
        x2, y2 = end_data
        
        # 绘制矩形
        from matplotlib.patches import Rectangle
        self._rect_patch = Rectangle(
            (min(x1, x2), min(y1, y2)),
            abs(x2 - x1),
            abs(y2 - y1),
            fill=False,
            edgecolor='red',
            linestyle='--',
            linewidth=0.8,
            alpha=0.7
        )
        self.ax.add_patch(self._rect_patch)
        self.canvas.draw_idle()
    
    def _remove_rect_patch(self):
        """删除矩形"""
        if hasattr(self, '_rect_patch') and self._rect_patch is not None:
            try:
                self._rect_patch.remove()
            except (NotImplementedError, ValueError):
                pass
            self._rect_patch = None
    
    def _on_rect_release(self, event):
        """框选结束"""
        if not hasattr(self, '_rect_select_mode') or not self._rect_select_mode:
            return
        if not hasattr(self, '_rect_start_pixel') or self._rect_start_pixel is None:
            return
        if event.button != 1:
            return
        
        # 获取像素位置
        x1_pixel, y1_pixel = self._rect_start_pixel
        x2_pixel, y2_pixel = event.x, event.y
        
        # 删除矩形
        self._remove_rect_patch()
        
        # 将像素位置转换为数据坐标
        inv_transform = self.ax.transData.inverted()
        start_data = inv_transform.transform((x1_pixel, y1_pixel))
        end_data = inv_transform.transform((x2_pixel, y2_pixel))
        
        x1, y1 = start_data
        x2, y2 = end_data
        
        # 设置新的显示范围
        if abs(x2 - x1) > 0:
            # 设置X轴范围（所有Y轴共享X轴）
            self.ax.set_xlim(min(x1, x2), max(x1, x2))
        
        # 设置Y轴范围（主Y轴）
        if abs(y2 - y1) > 0:
            self.ax.set_ylim(min(y1, y2), max(y1, y2))
        
        self.canvas.draw()
        # 保存新视图
        self._save_view()
        
        # 清理，但保持框选模式启用
        self._rect_start_pixel = None
        # 注意：不设置 _rect_select_mode = False，保持框选模式启用
        # 不断开事件连接，允许继续框选
        
    def _disable_zoom_rect(self):
        """禁用框选放大模式"""
        self._rect_select_mode = False
        self._rect_start_pixel = None
        
        # 删除矩形
        self._remove_rect_patch()
        
        # 删除十字线
        self._remove_crosshair()
        
        # 断开事件连接
        if hasattr(self, '_rect_cid_press'):
            self.canvas.mpl_disconnect(self._rect_cid_press)
            delattr(self, '_rect_cid_press')
        if hasattr(self, '_rect_cid_motion'):
            self.canvas.mpl_disconnect(self._rect_cid_motion)
            delattr(self, '_rect_cid_motion')
        if hasattr(self, '_rect_cid_release'):
            self.canvas.mpl_disconnect(self._rect_cid_release)
            delattr(self, '_rect_cid_release')

    def _chart_save_png(self):
        """保存图表为PNG图片"""
        # 获取软件所在文件夹
        app_dir = os.path.dirname(os.path.abspath(__file__))
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialfile="chart.png",
            initialdir=app_dir
        )
        if filename:
            # 保存前添加图例
            self._add_legend_before_save()
            self.fig.savefig(filename, dpi=150, bbox_inches='tight', format='png')
            # 保存后移除图例
            self._remove_legend_after_save()
    
    def _chart_save_jpg(self):
        """保存图表为JPG图片"""
        # 获取软件所在文件夹
        app_dir = os.path.dirname(os.path.abspath(__file__))
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")],
            initialfile="chart.jpg",
            initialdir=app_dir
        )
        if filename:
            # 保存前添加图例
            self._add_legend_before_save()
            self.fig.savefig(filename, dpi=150, bbox_inches='tight', format='jpeg')
            # 保存后移除图例
            self._remove_legend_after_save()
    
    def _chart_save_pdf(self):
        """保存图表为PDF文件"""
        # 获取软件所在文件夹
        app_dir = os.path.dirname(os.path.abspath(__file__))
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            initialfile="chart.pdf",
            initialdir=app_dir
        )
        if filename:
            # 保存前添加图例
            self._add_legend_before_save()
            self.fig.savefig(filename, bbox_inches='tight', format='pdf')
            # 保存后移除图例
            self._remove_legend_after_save()
    
    def _add_legend_before_save(self):
        """保存图片前添加图例和标题"""
        # 获取卫星名称
        satellite_name = self.name_var.get().strip() or "Unknown Satellite"
        
        # 获取观测地点信息
        if self.loc_type_var.get() == "1":
            mpc_code = self.mpc_code_var.get().strip()
            observer, msg, data_dict = get_observer_from_mpc(mpc_code, self.builtin_codes)
            if data_dict and "name" in data_dict:
                lat_dms = decimal_to_dms(float(data_dict['latitude']), is_latitude=True)
                lon_dms = decimal_to_dms(float(data_dict['longitude']), is_latitude=False)
                location_info = f"{data_dict['name']}（纬度: {lat_dms}，经度: {lon_dms}）"
            else:
                if mpc_code.upper() in self.builtin_codes and "name" in self.builtin_codes[mpc_code.upper()]:
                    data = self.builtin_codes[mpc_code.upper()]
                    lat_dms = decimal_to_dms(float(data['latitude']), is_latitude=True)
                    lon_dms = decimal_to_dms(float(data['longitude']), is_latitude=False)
                    location_info = f"{data['name']}（纬度: {lat_dms}，经度: {lon_dms}）"
                else:
                    location_info = f"MPC代码: {mpc_code}"
        else:
            if self.coord_input_type.get() == "decimal":
                lat = parse_coordinate(self.lat_var.get())
                lon = parse_coordinate(self.lon_var.get())
            else:
                lat = parse_dms_coordinate(
                    self.lat_deg_var.get(),
                    self.lat_min_var.get(),
                    self.lat_sec_var.get(),
                    self.lat_dir_var.get()
                )
                lon = parse_dms_coordinate(
                    self.lon_deg_var.get(),
                    self.lon_min_var.get(),
                    self.lon_sec_var.get(),
                    self.lon_dir_var.get()
                )
            if self.coord_input_type.get() == "decimal":
                height_m = float(self.height_var.get())
            else:
                height_m = float(self.height_var_dms.get())
            lat_dms = decimal_to_dms(lat, is_latitude=True)
            lon_dms = decimal_to_dms(lon, is_latitude=False)
            location_info = f"纬度: {lat_dms}，经度: {lon_dms}，海拔高度: {height_m}m"
        
        # 创建标题
        title_text = f"{satellite_name} - {location_info}"
        self._chart_title = self.ax.set_title(title_text, fontsize=10)
        
        # 获取当前图表中所有线条
        handles = []
        labels = []
        for ax in self._all_axes:
            for line in ax.lines:
                label = line.get_label()
                if label and not label.startswith('_'):
                    handles.append(line)
                    labels.append(label)
        
        if handles:
            # 添加图例，显示在X坐标轴下方，居中，无边框
            self._chart_legend = self.ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8, frameon=False)
            self.canvas.draw_idle()
    
    def _remove_legend_after_save(self):
        """保存图片后移除图例和标题"""
        # 清除标题
        if hasattr(self, '_chart_title'):
            self.ax.set_title("")
            delattr(self, '_chart_title')
        # 移除图例
        if hasattr(self, '_chart_legend'):
            self._chart_legend.remove()
            delattr(self, '_chart_legend')
            self.canvas.draw_idle()
    
    def _export_excel(self):
        if not self.ephemeris_data:
            messagebox.showinfo("提示", "没有数据可导出")
            return
        try:
            try:
                import pandas as pd
            except ImportError:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "openpyxl"])
                import pandas as pd
            # 获取软件所在文件夹作为默认目录
            app_dir = os.path.dirname(os.path.abspath(__file__))
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                initialfile="ephemeris.xlsx",
                initialdir=app_dir
            )
            if filename:
                df = pd.DataFrame(self.ephemeris_data)
                df.to_excel(filename, index=False)
        except Exception as e:
            messagebox.showerror("错误", f"Excel导出失败: {e}")
    
    def update_result_window(self):
        """更新结果窗口的数据"""
        if not self.result_tree:
            return
        
        # 清空现有数据
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        
        # 填充新数据
        for data in self.ephemeris_data:
            self.result_tree.insert("", tk.END, values=(
                data["Date(UTC+8)"],
                data["RA"],
                data["Dec"],
                data["Mag"],
                data["Alt"],
                data["Az"],
                data["Speed"],
                data["PA"],
                data["Obs Dist"],
                data["Orbit Alt"],
                data["Sun Alt"],
                data["Sun Sep"],
                data["Moon Alt"],
                data["Moon Sep"]
            ))
        
        # 更新状态标签
        self._update_result_status()
    
    def calculate(self):
        try:
            self.status_var.set("正在加载星历数据...")
            self.root.update()
            
            # 加载Skyfield数据
            global ts
            self.ts = load.timescale()
            self.eph = load("de421.bsp")
            
            # 获取观测点
            if self.loc_type_var.get() == "1":
                mpc_code = self.mpc_code_var.get().strip()
                if not mpc_code:
                    messagebox.showerror("错误", "请输入MPC代码")
                    self.status_var.set("就绪")
                    return
                observer, msg, data_dict = get_observer_from_mpc(mpc_code, self.builtin_codes)
                if observer is None:
                    messagebox.showerror("错误", msg)
                    self.status_var.set("就绪")
                    return
                self.status_var.set(msg)
                # 更新MPC代码解析结果
                self.mpc_result_text.config(state=tk.NORMAL)
                self.mpc_result_text.delete(1.0, tk.END)
                self.mpc_result_text.insert(tk.END, msg)
                self.mpc_result_text.config(state=tk.DISABLED)
                # 如果是从HTML读取的数据，保存到current_mpc_data并显示保存按钮
                if data_dict is not None:
                    self.current_mpc_data = {
                        "code": mpc_code.upper(),
                        "data": data_dict
                    }
                    if hasattr(self, 'save_mpc_btn'):
                        self.save_mpc_btn.pack(side=tk.RIGHT, padx=5)
                else:
                    self.current_mpc_data = None
                    if hasattr(self, 'save_mpc_btn'):
                        self.save_mpc_btn.pack_forget()
            else:
                try:
                    if self.coord_input_type.get() == "decimal":
                        # 小数格式输入
                        lat = parse_coordinate(self.lat_var.get())
                        lon = parse_coordinate(self.lon_var.get())
                    else:
                        # 度分秒格式输入
                        lat = parse_dms_coordinate(
                            self.lat_deg_var.get(),
                            self.lat_min_var.get(),
                            self.lat_sec_var.get(),
                            self.lat_dir_var.get()
                        )
                        lon = parse_dms_coordinate(
                            self.lon_deg_var.get(),
                            self.lon_min_var.get(),
                            self.lon_sec_var.get(),
                            self.lon_dir_var.get()
                        )
                    # 根据当前输入方式获取海拔高度
                    if self.coord_input_type.get() == "decimal":
                        height_m = float(self.height_var.get())
                    else:
                        height_m = float(self.height_var_dms.get())
                    observer = wgs84.latlon(lat, lon, height_m)
                    self.status_var.set(f"手动地点：{lat}°N, {lon}°E, {height_m}m")
                except ValueError as e:
                    messagebox.showerror("错误", f"经纬度输入格式错误: {e}")
                    self.status_var.set("就绪")
                    return
            
            self.root.update()
            
            # 获取TLE数据
            tle_content = self.tle_text.get("1.0", tk.END).strip()
            lines = [line.strip() for line in tle_content.split('\n') if line.strip()]
            
            if len(lines) < 2:
                messagebox.showerror("错误", "请输入完整的TLE数据（两行）")
                self.status_var.set("就绪")
                return
            
            line1 = lines[0]
            line2 = lines[1]
            name = self.name_var.get().strip() or "Unknown Satellite"
            
            try:
                satellite = EarthSatellite(line1, line2, name, self.ts)
                # 存储TLE数据以便pyephem使用
                satellite.line1 = line1
                satellite.line2 = line2
                self.status_var.set(f"TLE加载成功：{name} (NORAD ID: {satellite.model.satnum})")
            except Exception as e:
                messagebox.showerror("错误", f"TLE加载失败: {e}")
                self.status_var.set("就绪")
                return
            
            self.root.update()
            
            # 获取计算参数
            start_str = self.start_var.get().strip()
            if not start_str:
                t_start = self.ts.now()
            else:
                try:
                    y, m, d, hh, mm, ss = map(int, start_str.replace("-", " ").replace(":", " ").split())
                    # 将UTC+8时间转换为UTC时间
                    from datetime import datetime, timezone, timedelta
                    utc8_time = datetime(y, m, d, hh, mm, ss, tzinfo=timezone(timedelta(hours=8)))
                    utc_time = utc8_time.astimezone(timezone.utc)
                    t_start = self.ts.utc(utc_time.year, utc_time.month, utc_time.day, 
                                   utc_time.hour, utc_time.minute, utc_time.second)
                except Exception as e:
                    messagebox.showerror("错误", f"时间格式错误: {e}")
                    self.status_var.set("就绪")
                    return
            
            try:
                duration_value = float(self.duration_var.get())
                # 根据选择的单位转换为小时
                duration_unit = self.duration_unit_var.get()
                if duration_unit == "秒":
                    duration_hours = duration_value / 3600
                elif duration_unit == "分钟":
                    duration_hours = duration_value / 60
                elif duration_unit == "小时":
                    duration_hours = duration_value
                elif duration_unit == "天":
                    duration_hours = duration_value * 24
                else:
                    duration_hours = duration_value
            except ValueError:
                messagebox.showerror("错误", "无效的计算时长")
                self.status_var.set("就绪")
                return
            
            try:
                step_value = float(self.step_var.get())
                # 根据选择的单位转换为分钟
                step_unit = self.step_unit_var.get()
                if step_unit == "秒":
                    step_min = step_value / 60
                elif step_unit == "分钟":
                    step_min = step_value
                elif step_unit == "小时":
                    step_min = step_value * 60
                elif step_unit == "天":
                    step_min = step_value * 1440
                else:
                    step_min = step_value
            except ValueError:
                messagebox.showerror("错误", "无效的时间步长")
                self.status_var.set("就绪")
                return
            
            std_mag_str = self.mag_var.get().strip()
            if std_mag_str:
                if std_mag_str.lower() == "none":
                    std_mag = DEFAULT_STD_MAG
                else:
                    try:
                        std_mag = float(std_mag_str)
                    except ValueError:
                        messagebox.showerror("错误", "无效的本征星等")
                        self.status_var.set("就绪")
                        return
            else:
                std_mag = DEFAULT_STD_MAG
            
            self.status_var.set("正在计算星历...")
            self.root.update()
            
            # 记录开始时间
            import time
            calc_start_time = time.time()
            
            self.ephemeris_data = []
            
            # 生成星历
            t = t_start
            step = timedelta(minutes=step_min)
            end_time = t.utc_datetime() + timedelta(hours=duration_hours)
            
            while t.utc_datetime() <= end_time:
                # 时间（UTC+8）
                utc_dt = t.utc_datetime()
                bj_dt = utc_dt + timedelta(hours=8)
                time_str = bj_dt.strftime("%Y-%m-%d %H:%M:%S")

                # 人造卫星位置
                difference = satellite - observer
                topo = difference.at(t)
                # 对于卫星位置，使用altaz()直接计算地平坐标
                # 可以添加大气折射参数来提高精度
                alt, az, distance = topo.altaz(temperature_C=15.0, pressure_mbar=1013.25)
                alt_deg = alt.degrees
                az_deg = az.degrees
                range_km = distance.km

                # 计算卫星与地球表面的距离（地心距离 - 地球半径）
                earth = self.eph["earth"]
                sat_gc = satellite.at(t)
                sat_position = sat_gc.position.km
                sat_to_earth_center = np.linalg.norm(sat_position)
                surface_distance_km = sat_to_earth_center - EARTH_RADIUS_KM

                # 计算太阳高度角
                sun_alt = compute_sun_altitude(t, observer, self.eph)
                
                # 计算月亮高度角
                moon_alt = compute_moon_altitude(t, observer, self.eph)

                # 赤道坐标
                ra_dec = topo.radec()
                ra_str, dec_str = format_radec(ra_dec[0], ra_dec[1])

                # 亮度
                mag_val = None  # 高精度星等值
                if std_mag is not None:
                    # 如果卫星在地平线下，不可见
                    if alt_deg < 0:
                        mag_str = "不可见"
                    else:
                        phase_deg = compute_phase_angle(t, satellite, observer, self.eph)
                        # 相位因子：0（完全不被照亮）到1（完全被照亮）
                        phase_factor = (1 + math.cos(math.radians(phase_deg))) / 2
                        
                        # 计算地球影子对亮度的影响
                        shadow_factor = compute_earth_shadow_factor(t, satellite, self.eph)
                        
                        # 如果在本影中(shadow_factor=0)，显示"在本影中"
                        if shadow_factor == 0.0:
                            mag_str = "在本影中"
                        elif phase_factor > 0:
                            # Heavens-Above的本征星等定义是在1000km距离、相位角90度（50%被照亮，phase_factor=0.5）时的星等
                            # 因此需要将phase_factor归一化到0.5作为参考
                            mag = std_mag + 5 * math.log10(range_km / 1000.0) - 2.5 * math.log10(phase_factor / 0.5)
                            
                            # 如果在半影中，根据影子因子调整亮度
                            if shadow_factor < 1.0:
                                # 在半影中，亮度逐渐下降
                                # shadow_factor从0到1，亮度从很暗到正常
                                # 使用对数关系调整星等
                                mag += 2.5 * math.log10(1.0 / shadow_factor)
                            
                            # 大气消光修正
                            # 当卫星接近地平线时，星光穿过更多大气层，亮度会衰减
                            # 使用简化的大气质量公式：X = 1/cos(z)，z是天顶距
                            # 消光系数k约0.2-0.3等/大气质量（取0.25作为平均值）
                            if alt_deg < 90:
                                # 计算天顶距（90° - 地平高度）
                                zenith_distance = math.radians(90 - alt_deg)
                                # 大气质量（简化公式，当高度>10°时较准确）
                                if alt_deg > 10:
                                    air_mass = 1.0 / math.cos(zenith_distance)
                                else:
                                    # 低高度时使用更精确的公式
                                    # Rozenberg公式：X = 1/(sin(h) + 0.025 * exp(-11 * sin(h)))
                                    sin_h = math.sin(math.radians(alt_deg))
                                    air_mass = 1.0 / (sin_h + 0.025 * math.exp(-11 * sin_h))
                                # 消光系数（等/大气质量）
                                extinction_coeff = 0.25
                                # 消光量
                                extinction = extinction_coeff * air_mass
                                # 修正星等（消光使星等变大，即变暗）
                                mag += extinction
                            
                            # 存储高精度星等值
                            mag_val = mag
                            
                            # 如果星等太暗（大于20等），标记为不可见
                            if mag > 20:
                                mag_str = "不可见"
                            else:
                                mag_str = f"{mag:.2f}"
                        else:
                            mag_str = "不可见"
                else:
                    mag_str = "N/A"

                speed, pa = compute_motion_pa_speed(t, satellite, observer, self.ts, dt_minutes=1.0)
                
                # 计算卫星与太阳、月亮的角距离
                sun_sep, moon_sep = compute_satellite_angular_distance(t, satellite, observer, self.eph)

                # 收集数据（存储高精度数值用于曲线图，显示时格式化2位小数）
                self.ephemeris_data.append({
                    "Date(UTC+8)": time_str,
                    "RA": ra_str,
                    "Dec": dec_str,
                    "Mag": mag_str,
                    "Alt": f"{alt_deg:.2f}",
                    "Az": f"{az_deg:.2f}",
                    "Speed": f"{speed:.2f}",
                    "PA": f"{pa:.2f}",
                    "Obs Dist": f"{range_km:.2f}",
                    "Orbit Alt": f"{surface_distance_km:.2f}",
                    "Sun Sep": f"{sun_sep:.2f}",
                    "Moon Sep": f"{moon_sep:.2f}",
                    "Sun Alt": f"{sun_alt:.2f}",
                    "Moon Alt": f"{moon_alt:.2f}",
                    # 高精度数值（用于曲线图和鼠标悬停）
                    "_mag": mag_val,
                    "_alt": alt_deg,
                    "_az": az_deg,
                    "_speed": speed,
                    "_pa": pa,
                    "_range": range_km,
                    "_surface": surface_distance_km,
                    "_sun_sep": sun_sep,
                    "_moon_sep": moon_sep,
                    "_sun_alt": sun_alt,
                    "_moon_alt": moon_alt,
                })

                t = self.ts.utc(utc_dt + step)
                
                # 每10条记录更新一次界面
                if len(self.ephemeris_data) % 10 == 0:
                    self.root.update()
            
            # 计算用时
            self.calc_duration = time.time() - calc_start_time
            data_count = len(self.ephemeris_data)
            
            self.status_var.set(f"计算完成！用时 {self.calc_duration:.2f} 秒，输出 {data_count} 条数据。")
            # 弹出结果窗口
            self.show_result_window()
            
        except Exception as e:
            messagebox.showerror("错误", f"计算过程中出错: {e}")
            self.status_var.set("就绪")

if __name__ == "__main__":
    root = tk.Tk()
    app = SatelliteEphemerisGUI(root)
    root.mainloop()