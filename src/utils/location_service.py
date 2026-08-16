"""
地理位置服务 - 自动获取当前位置信息
支持多种方式：IP定位、浏览器Geolocation、手动配置
同时提供实时天气信息
"""

import json
import os
import httpx
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple

# 配置文件路径
CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
LOCATION_CONFIG_FILE = CONFIG_DIR / "deployment.json"


def get_location_from_ip() -> Optional[Dict]:
    """
    通过IP地址获取地理位置
    优先级：高德IP定位 > ip-api.com（备用）
    返回: {"province": "广东省", "city": "深圳市", "district": "南山区", ...}
    """
    amap_key = os.getenv("AMAP_KEY", "")

    # 方案1: 高德IP定位 API（精度高，国内稳定）
    if amap_key:
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(
                    "https://restapi.amap.com/v3/ip",
                    params={"key": amap_key}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "1" and data.get("city"):
                        province = data.get("province", "")
                        city = data.get("city", "")
                        # 高德IP定位还返回 adcode，可以直接用于天气查询
                        adcode = data.get("adcode", "")
                        location = {
                            "province": province if isinstance(province, str) else "",
                            "city": city if isinstance(city, str) else "",
                            "district": "",
                            "place": "家中",
                            "country": "中国",
                            "adcode": adcode,
                            "source": "amap-ip"
                        }
                        print("[LocationService] ✅ 高德IP定位成功")
                        return location
        except Exception as e:
            print(f"[LocationService] ⚠️ 高德IP定位失败: {type(e).__name__}")

    # 方案2: ip-api.com（备用，免费无需key）
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get("http://ip-api.com/json/?lang=zh-CN")
            if response.status_code == 200:
                r = response.json()
                location = {
                    "province": r.get("regionName", ""),
                    "city": r.get("city", ""),
                    "district": "",
                    "place": "家中",
                    "country": r.get("country", ""),
                    "lat": r.get("lat"),
                    "lon": r.get("lon"),
                    "source": "ip-api.com"
                }
                if location.get("city"):
                    print("[LocationService] ✅ IP定位(备用)成功")
                    return location
    except Exception as e:
        print(f"[LocationService] ⚠️ ip-api.com 失败: {type(e).__name__}")
    
    return None


def get_location_from_config() -> Optional[Dict]:
    """从配置文件读取位置"""
    try:
        if LOCATION_CONFIG_FILE.exists():
            with open(LOCATION_CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get("location")
    except Exception as e:
        print(f"[LocationService] ⚠️ 读取配置失败: {type(e).__name__}")
    return None


def save_location_to_config(location: Dict) -> bool:
    """保存位置到配置文件"""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        config = {
            "location": location,
            "updated_at": datetime.now().isoformat(),
            "auto_detected": True
        }
        
        with open(LOCATION_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        
        print(f"[LocationService] 💾 位置已保存到配置文件")
        return True
    except Exception as e:
        print(f"[LocationService] ❌ 保存配置失败: {type(e).__name__}")
        return False


def get_current_location(force_refresh: bool = False) -> Dict:
    """
    获取当前位置（优先级：配置文件 > IP定位 > 默认值）
    
    Args:
        force_refresh: 是否强制重新获取（忽略缓存配置）
    
    Returns:
        位置信息字典
    """
    # 1. 如果不强制刷新，先尝试从配置文件读取
    if not force_refresh:
        cached = get_location_from_config()
        if cached and cached.get("city"):
            print("[LocationService] 📍 使用缓存位置")
            return cached
    
    # 2. 尝试IP定位
    location = get_location_from_ip()
    if location:
        # 保存到配置文件（缓存）
        save_location_to_config(location)
        return location
    
    # 3. 返回默认值
    print("[LocationService] ⚠️ 无法获取位置，使用默认值")
    return {
        "province": "未知",
        "city": "未知",
        "district": "",
        "place": "家中",
        "source": "default"
    }


def update_location_manually(province: str, city: str, district: str = "", place: str = "") -> Dict:
    """手动更新位置"""
    location = {
        "province": province,
        "city": city,
        "district": district,
        "place": place or "家中",
        "source": "manual"
    }
    save_location_to_config(location)
    return location


def _amap_get_adcode(city: str, key: str) -> Optional[str]:
    """通过高德地理编码 API 将城市名转为 adcode"""
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(
                "https://restapi.amap.com/v3/geocode/geo",
                params={"key": key, "address": city}
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "1" and data.get("geocodes"):
                    adcode = data["geocodes"][0].get("adcode", "")
                    if adcode:
                        print("[WeatherService] 📍 高德地理编码成功")
                        return adcode
    except Exception as e:
        print(f"[WeatherService] ⚠️ 高德 geocode 失败: {type(e).__name__}")
    return None


# 缓存 adcode 避免每次都查
_adcode_cache: Dict[str, str] = {}


def _season_inferred_weather(city: str = None, lat: float = None) -> Optional[Dict]:
    """根据季节给出天气占位，避免外部天气 API 影响主流程。"""
    if not (city or lat):
        return None
    month = datetime.now().month
    if month in [12, 1, 2]:
        weather_desc = "冬季寒冷"
    elif month in [3, 4, 5]:
        weather_desc = "春季温和"
    elif month in [6, 7, 8]:
        weather_desc = "夏季炎热"
    else:
        weather_desc = "秋季凉爽"

    print("[WeatherService] ⚠️ 降级为季节推断")
    return {
        "temperature": "未知",
        "weather": weather_desc,
        "humidity": "未知",
        "source": "season-inferred"
    }


def get_weather(city: str = None, lat: float = None, lon: float = None, adcode: str = None) -> Optional[Dict]:
    """
    获取实时天气信息（高德天气 API）
    
    Args:
        city: 城市名称（中文）
        lat: 纬度（备用）
        lon: 经度（备用）
        adcode: 高德城市编码（如已有则跳过 geocoding）
    
    Returns:
        天气信息字典，失败返回None
    """
    amap_key = os.getenv("AMAP_KEY", "")
    print("[WeatherService] 📍 已获取位置上下文")
    
    # 高德天气 API
    if amap_key and (city or adcode):
        try:
            # 优先用传入的 adcode，否则通过 geocoding 获取
            _adcode = adcode or _adcode_cache.get(city)
            if not _adcode and city:
                _adcode = _amap_get_adcode(city, amap_key)
                if _adcode:
                    _adcode_cache[city] = _adcode
            adcode = _adcode

            if adcode:
                with httpx.Client(timeout=3.0) as client:
                    resp = client.get(
                        "https://restapi.amap.com/v3/weather/weatherInfo",
                        params={"key": amap_key, "city": adcode, "extensions": "base"}
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("status") == "1" and data.get("lives"):
                            live = data["lives"][0]
                            result = {
                                "temperature": f"{live.get('temperature', '未知')}°C",
                                "weather": live.get("weather", "未知"),
                                "humidity": f"{live.get('humidity', '未知')}%",
                                "wind": f"{live.get('winddirection', '')}风{live.get('windpower', '')}级",
                                "source": "amap"
                            }
                            print("[WeatherService] ✅ 高德天气获取成功")
                            return result
        except Exception as e:
            print(f"[WeatherService] ⚠️ 高德天气API失败: {type(e).__name__}")
    
    # 降级：根据月份推断季节性天气
    fallback_weather = _season_inferred_weather(city=city, lat=lat)
    if fallback_weather:
        return fallback_weather
    
    print(f"[WeatherService] ❌ 所有方法均失败")
    return None


# 全局缓存
_cached_location = None
_cached_weather = None
_weather_update_time = None


def get_deployment_location() -> Dict:
    """
    获取部署位置（供其他模块调用的主接口）
    首次调用时自动获取，之后使用缓存
    """
    global _cached_location
    
    if _cached_location is None:
        _cached_location = get_current_location()
    
    return _cached_location


def refresh_location() -> Dict:
    """强制刷新位置"""
    global _cached_location
    _cached_location = get_current_location(force_refresh=True)
    return _cached_location


def get_realtime_context(fetch_weather: bool = True) -> Dict:
    """
    获取完整的实时上下文信息（位置、时间、天气）
    用于MMSE定向力评估
    
    Returns:
        包含位置、时间、天气的完整字典
    """
    global _cached_weather, _weather_update_time
    
    # 1. 获取位置
    location = get_deployment_location()
    
    # 2. 获取当前时间
    now = datetime.now()
    weekday_names = ["一", "二", "三", "四", "五", "六", "日"]
    
    def get_season(month):
        if month in [3, 4, 5]: return "春季"
        elif month in [6, 7, 8]: return "夏季"
        elif month in [9, 10, 11]: return "秋季"
        else: return "冬季"
    
    time_info = {
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "weekday": weekday_names[now.weekday()],
        "season": get_season(now.month),
        "hour": now.hour,
        "minute": now.minute
    }
    
    # 3. 获取天气（缓存10分钟）
    weather = None
    if _cached_weather and _weather_update_time:
        elapsed = (datetime.now() - _weather_update_time).total_seconds()
        if elapsed < 600:  # 10分钟内使用缓存
            weather = _cached_weather
            print(f"[RealtimeContext] 🌤️ 使用缓存天气 ({elapsed:.0f}秒前)")
    
    if not weather:
        # 尝试获取新天气（如果位置已有 adcode 则直接传入，跳过 geocoding）
        city = location.get('city', '')
        lat = location.get('lat')
        lon = location.get('lon')
        adcode = location.get('adcode', '')

        if fetch_weather:
            weather = get_weather(city=city, lat=lat, lon=lon, adcode=adcode)
        else:
            print("[RealtimeContext] ⏭️ 跳过启动期天气联网请求")
            weather = _season_inferred_weather(city=city, lat=lat)
        if not weather:
            weather = {
                "temperature": "未知",
                "weather": "未知",
                "humidity": "未知",
                "source": "unavailable"
            }
        # 无论成功或失败都缓存，避免失败时反复超时
        _cached_weather = weather
        _weather_update_time = datetime.now()
    
    # 组合所有信息
    context = {
        "location": location,
        "time": time_info,
        "weather": weather,
        "timestamp": now.isoformat()
    }
    
    return context


# 测试
if __name__ == "__main__":
    print("测试实时上下文获取...")
    ctx = get_realtime_context()
    print(f"上下文键数量: {len(ctx)}")
