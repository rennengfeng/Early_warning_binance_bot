#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import asyncio
import aiohttp
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from telegram import (
    ReplyKeyboardMarkup,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    Update
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
    Application
)
from config import TOKEN, CHAT_IDS, BINANCE_API_KEY, BINANCE_API_SECRET, DEFAULT_INTERVAL

# 设置日志
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 配置文件路径
USER_DATA_DIR = "user_data"
os.makedirs(USER_DATA_DIR, exist_ok=True)

# K线参数配置
INTERVALS = {
    "5m": "5分钟",
    "15m": "15分钟",
    "60m": "60分钟",
    "240m": "4小时"
}

# Binance API实际支持的间隔
BINANCE_INTERVALS = {
    "5m": "5m",
    "15m": "15m",
    "60m": "1h",  # Binance使用1h表示60分钟
    "240m": "4h"  # Binance使用4h表示4小时
}

# 市场类型中文名
MARKET_TYPE_NAMES = {
    "spot": "现货",
    "contract": "合约"
}

# 监控类型中文名
MONITOR_TYPE_NAMES = {
    "price": "价格异动",
    "macd": "MACD交叉",
    "ma": "MA交叉"
}

# 主菜单
main_menu = [
    ["1️⃣ 添加币种", "2️⃣ 删除币种"],
    ["3️⃣ 开启监控", "4️⃣ 停止监控"],
    ["5️⃣ 查看状态", "6️⃣ 帮助"]
]
reply_markup = ReplyKeyboardMarkup(main_menu, resize_keyboard=True)
 
# 返回菜单
back_menu = [["↩️ 返回主菜单", "❌ 取消"]]
back_markup = ReplyKeyboardMarkup(back_menu, resize_keyboard=True)  # 修复变量名

# 服务器时间偏移量（用于与Binance API时间同步）
time_offset = 0

# --- 数据管理 ---
def get_user_file(user_id):
    return os.path.join(USER_DATA_DIR, f"{user_id}.json")

def load_user_data(user_id):
    file_path = get_user_file(user_id)
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                # 确保数据结构兼容
                if "monitors" not in data:
                    data["monitors"] = {
                        "price": {"enabled": False},
                        "macd": {"enabled": False},
                        "ma": {"enabled": False}
                    }
                if "active" not in data:
                    data["active"] = False
                    
                # 迁移旧数据结构 - 确保每个监控项都是独立字典
                new_symbols = []
                for symbol in data.get("symbols", []):
                    if isinstance(symbol, dict):
                        # 如果是字典格式，直接使用
                        new_symbols.append(symbol)
                    else:
                        # 如果是旧格式（字符串），转换为字典
                        new_symbols.append({
                            "symbol": symbol,
                            "type": "spot",
                            "monitor": "price",
                            "interval": "15m",
                            "threshold": 5.0
                        })
                data["symbols"] = new_symbols
                return data
        except Exception as e:
            logger.error(f"加载用户数据失败: {e}")
            # 创建新的数据结构
            return {
                "symbols": [],
                "monitors": {
                    "price": {"enabled": False},
                    "macd": {"enabled": False},
                    "ma": {"enabled": False}
                },
                "active": False
            }
    # 默认配置
    return {
        "symbols": [],
        "monitors": {
            "price": {"enabled": False},
            "macd": {"enabled": False},
            "ma": {"enabled": False}
        },
        "active": False
    }

def save_user_data(user_id, data):
    file_path = get_user_file(user_id)
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"保存用户数据失败: {e}")

# --- 时间同步函数 ---
async def sync_binance_time():
    """同步Binance服务器时间"""
    global time_offset
    url = "https://api.binance.com/api/v3/time"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    server_time = data['serverTime']
                    local_time = int(time.time() * 1000)
                    time_offset = server_time - local_time
                    logger.info(f"时间同步完成，服务器时间偏移: {time_offset}ms")
                else:
                    logger.error(f"时间同步失败: {resp.status}")
        except Exception as e:
            logger.error(f"时间同步出错: {e}")

# --- Binance API 辅助函数 ---
async def get_klines(symbol, interval, market_type="spot", limit=100):
    """获取K线数据"""
    # 确保symbol是大写的
    symbol = symbol.upper()
    
    # 检查合约市场是否需要USDT后缀
    if market_type == "contract" and not symbol.endswith("USDT"):
        symbol += "USDT"
    
    # 使用正确的API端点
    base_url = "https://api.binance.com/api/v3"
    if market_type == "contract":
        base_url = "https://fapi.binance.com/fapi/v1"
    
    # 将用户间隔转换为Binance API间隔
    binance_interval = BINANCE_INTERVALS.get(interval, interval)
    
    url = f"{base_url}/klines?symbol={symbol}&interval={binance_interval}&limit={limit}"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    error_text = await resp.text()
                    logger.error(f"获取K线失败: {symbol} {binance_interval} {resp.status} - {error_text}")
                    return None
        except Exception as e:
            logger.error(f"获取K线数据失败: {e}")
            return None

def klines_to_dataframe(klines):
    """将K线数据转换为DataFrame"""
    if not klines:
        return None
        
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    # 转换数据类型
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
    
    # 转换时间戳（应用时间偏移）
    df['timestamp'] = pd.to_datetime(df['timestamp'] + time_offset, unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df

# --- 技术指标计算 ---
def calculate_ma(df, window=20):
    """计算移动平均线"""
    return df['close'].rolling(window=window).mean()

def calculate_ema(df, window):
    """计算指数移动平均线"""
    return df['close'].ewm(span=window, adjust=False).mean()

def calculate_macd(df, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    ema_fast = calculate_ema(df, fast)
    ema_slow = calculate_ema(df, slow)
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

# --- 监控任务 ---
class MonitorTask:
    def __init__(self, app):
        self.app = app
        self.price_history = {}
        self.macd_history = {}
        self.ma_history = {}
        self.active = True
        self.task = None

    async def run(self):
        """监控主循环"""
        logger.info("监控任务已启动")
        # 首次运行前同步时间
        await sync_binance_time()
        
        while self.active:
            try:
                # 每15分钟同步一次时间
                if datetime.now().minute % 15 == 0:
                    await sync_binance_time()
                
                # 获取所有用户文件
                user_files = [f for f in os.listdir(USER_DATA_DIR) if f.endswith('.json')]
                
                for user_file in user_files:
                    try:
                        user_id = int(user_file.split('.')[0])
                        user_data = load_user_data(user_id)
                        
                        # 检查用户是否启用监控
                        if not user_data.get('active', False):
                            continue
                        
                        # 处理用户的所有币种
                        for symbol_info in user_data['symbols']:
                            symbol = symbol_info['symbol']
                            market_type = symbol_info['type']
                            monitor_type = symbol_info.get('monitor', 'price')  # 默认价格监控
                            
                            # 只处理启用状态的监控
                            if not user_data['monitors'][monitor_type]['enabled']:
                                continue
                            
                            # 根据监控类型执行检查
                            if monitor_type == "price":
                                await self.check_price_change(user_id, symbol, market_type, symbol_info)
                            elif monitor_type == "macd":
                                await self.check_macd(user_id, symbol, market_type)
                            elif monitor_type == "ma":
                                await self.check_ma_cross(user_id, symbol, market_type)
                    except Exception as e:
                        logger.error(f"处理用户 {user_file} 时出错: {e}")
                        continue
                
                # 每分钟检查一次
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"监控任务出错: {e}")
                await asyncio.sleep(10)

    async def check_price_change(self, user_id, symbol, market_type, symbol_info):
        """检查价格异动"""
        try:
            # 获取币种的监控周期
            interval = symbol_info.get('interval', '15m')
            threshold = symbol_info.get('threshold', 5.0)
            
            # 获取当前价格
            klines = await get_klines(symbol, interval, market_type, limit=2)
            if not klines or len(klines) < 2:
                logger.warning(f"无法获取足够的K线数据: {symbol} {interval}")
                return
                
            current_price = float(klines[-1][4])
            prev_price = float(klines[-2][4])
            
            # 计算价格变化百分比
            change_percent = ((current_price - prev_price) / prev_price) * 100
            
            # 检查是否超过阈值
            if abs(change_percent) > threshold:
                direction = "上涨" if change_percent > 0 else "下跌"
                message = (
                    f"🚨 价格异动警报: {symbol} ({MARKET_TYPE_NAMES[market_type]}) - {INTERVALS.get(interval, interval)}\n"
                    f"• 变化: {abs(change_percent):.2f}% ({direction})\n"
                    f"• 前价: {prev_price:.4f}\n"
                    f"• 现价: {current_price:.4f}\n"
                    f"• 阈值: {threshold}%\n"
                    f"• 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                await self.app.bot.send_message(chat_id=user_id, text=message)
                
        except Exception as e:
            logger.error(f"价格异动监控出错: {e}")

    async def check_macd(self, user_id, symbol, market_type):
        """检查MACD交叉 - 使用配置的DEFAULT_INTERVAL"""
        try:
            # 使用配置的DEFAULT_INTERVAL获取K线数据
            klines = await get_klines(symbol, DEFAULT_INTERVAL, market_type, limit=100)
            if not klines or len(klines) < 50:
                logger.warning(f"无法获取足够的K线数据: {symbol} {DEFAULT_INTERVAL}")
                return
                
            df = klines_to_dataframe(klines)
            if df is None or len(df) < 50:
                return
                
            # 计算MACD
            macd, signal, _ = calculate_macd(df)
            
            # 检查是否形成交叉
            key = f"{user_id}_{symbol}_{market_type}"
            
            # 获取最后两个数据点
            if len(macd) < 2 or len(signal) < 2:
                return
                
            current_macd = macd.iloc[-1]
            prev_macd = macd.iloc[-2]
            current_signal = signal.iloc[-1]
            prev_signal = signal.iloc[-2]
            
            # 金叉检测：MACD从下方穿越信号线
            if prev_macd < prev_signal and current_macd > current_signal:
                message = (
                    f"📈 MACD金叉信号: {symbol} ({MARKET_TYPE_NAMES[market_type]}) - {INTERVALS.get(DEFAULT_INTERVAL, DEFAULT_INTERVAL)}\n"
                    f"• MACD: {current_macd:.4f}\n"
                    f"• 信号线: {current_signal:.4f}\n"  # 修复变量名
                    f"• 价格: {df['close'].iloc[-1]:.4f}\n"
                    f"• 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                await self.app.bot.send_message(chat_id=user_id, text=message)
            
            # 死叉检测：MACD从上方穿越信号线
            elif prev_macd > prev_signal and current_macd < current_signal:
                message = (
                    f"📉 MACD死叉信号: {symbol} ({MARKET_TYPE_NAMES[market_type]}) - {INTERVALS.get(DEFAULT_INTERVAL, DEFAULT_INTERVAL)}\n"
                    f"• MACD: {current_macd:.4f}\n"
                    f"• 信号线: {current_signal:.4f}\n"  # 修复变量名
                    f"• 价格: {df['close'].iloc[-1]:.4f}\n"
                    f"• 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                await self.app.bot.send_message(chat_id=user_id, text=message)
            
        except Exception as e:
            logger.error(f"MACD监控出错: {e}")

    async def check_ma_cross(self, user_id, symbol, market_type):
        """检查MA交叉 - 使用配置的DEFAULT_INTERVAL"""
        try:
            # 使用配置的DEFAULT_INTERVAL获取K线数据
            klines = await get_klines(symbol, DEFAULT_INTERVAL, market_type, limit=100)
            if not klines or len(klines) < 30:
                logger.warning(f"无法获取足够的K线数据: {symbol} {DEFAULT_INTERVAL}")
                return
                
            df = klines_to_dataframe(klines)
            if df is None or len(df) < 30:
                return
                
            # 计算MA指标
            ma9 = calculate_ma(df, 9)
            ma26 = calculate_ma(df, 26)
            
            # 检查是否形成交叉
            key = f"{user_id}_{symbol}_{market_type}"
            
            # 获取最后两个数据点
            if len(ma9) < 2 or len(ma26) < 2:
                return
                
            current_ma9 = ma9.iloc[-1]
            prev_ma9 = ma9.iloc[-2]
            current_ma26 = ma26.iloc[-1]
            prev_ma26 = ma26.iloc[-2]
            
            # 金叉检测：MA9从下方穿越MA26
            if prev_ma9 < prev_ma26 and current_ma9 > current_ma26:
                message = (
                    f"📈 MA金叉信号: {symbol} ({MARKET_TYPE_NAMES[market_type]}) - {INTERVALS.get(DEFAULT_INTERVAL, DEFAULT_INTERVAL)}\n"
                    f"• MA9: {current_ma9:.4f}\n"
                    f"• MA26: {current_ma26:.4f}\n"
                    f"• 价格: {df['close'].iloc[-1]:.4f}\n"
                    f"• 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                await self.app.bot.send_message(chat_id=user_id, text=message)
            
            # 死叉检测：MA9从上方穿越MA26
            elif prev_ma9 > prev_ma26 and current_ma9 < current_ma26:
                message = (
                    f"📉 MA死叉信号: {symbol} ({MARKET_TYPE_NAMES[market_type]}) - {INTERVALS.get(DEFAULT_INTERVAL, DEFAULT_INTERVAL)}\n"
                    f"• MA9: {current_ma9:.4f}\n"
                    f"• MA26: {current_ma26:.4f}\n"
                    f"• 价格: {df['close'].iloc[-1]:.4f}\n"
                    f"• 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                await self.app.bot.send_message(chat_id=user_id, text=message)
            
        except Exception as e:
            logger.error(f"MA交叉监控出错: {e}")


# --- 监控管理 ---
monitor_task = None

async def start_monitor(app):
    """启动监控任务"""
    global monitor_task
    if monitor_task is None or not monitor_task.active:
        monitor_task = MonitorTask(app)
        # 在事件循环中启动监控任务
        monitor_task.task = asyncio.create_task(monitor_task.run())
        logger.info("监控任务已启动")
        return True
    return False

async def stop_monitor():
    """停止监控任务"""
    global monitor_task
    if monitor_task:
        monitor_task.active = False
        # 等待任务完成
        if monitor_task.task:
            await monitor_task.task
        monitor_task = None
        logger.info("监控任务已停止")
        return True
    return False

# --- 用户状态管理 ---
user_states = {}

def set_user_state(user_id, state, data=None):
    """设置用户状态"""
    if data is None:
        data = {}
    user_states[user_id] = {"state": state, "data": data}

def get_user_state(user_id):
    """获取用户状态"""
    return user_states.get(user_id, {})

def clear_user_state(user_id):
    """清除用户状态"""
    if user_id in user_states:
        del user_states[user_id]

# --- 按钮回调 ---
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    data_parts = query.data.split(":")

    # 确保用户已授权
    if user_id not in CHAT_IDS:
        await query.message.reply_text("您未获得使用此机器人的授权")
        return

    try:
        if data_parts[0] == "select_type":
            symbol = data_parts[1]
            market_type = data_parts[2]
            monitor_type = data_parts[3]
            
            # 保存到用户状态，不立即添加
            set_user_state(user_id, "add_symbol_config", {
                "symbol": symbol.upper(),
                "type": market_type,
                "monitor": monitor_type
            })
            
            await query.edit_message_text(f"已选择 {symbol} ({MARKET_TYPE_NAMES[market_type]}) 的{MONITOR_TYPE_NAMES.get(monitor_type, monitor_type)}监控")
            
            # 根据监控类型进行下一步
            if monitor_type == "price":
                # 价格异动需要选择周期
                keyboard = [
                    [InlineKeyboardButton("5分钟", callback_data=f"select_interval:5m")],
                    [InlineKeyboardButton("15分钟", callback_data=f"select_interval:15m")],
                    [InlineKeyboardButton("60分钟", callback_data=f"select_interval:60m")]
                ]
                await query.message.reply_text(
                    f"请选择 {symbol} 的价格异动监控周期:",
                    reply_markup=InlineKeyboardMarkup(keyboard))
            else:
                # 非价格监控，直接完成配置
                state_info = get_user_state(user_id)
                config = state_info.get("data", {})
                
                # 添加到监控列表
                user_data = load_user_data(user_id)
                user_data["symbols"].append(config)
                save_user_data(user_id, user_data)
                
                # 询问是否继续添加
                keyboard = [
                    [InlineKeyboardButton("✅ 继续添加", callback_data=f"continue_add:{monitor_type}")],
                    [InlineKeyboardButton("❌ 完成添加", callback_data=f"finish_add:{monitor_type}")]
                ]
                await query.message.reply_text(
                    "是否继续添加交易对?",
                    reply_markup=InlineKeyboardMarkup(keyboard))
        
        elif data_parts[0] == "select_interval":
            interval = data_parts[1]
            state_info = get_user_state(user_id)
            config = state_info.get("data", {})
            config["interval"] = interval
            
            # 更新用户状态
            set_user_state(user_id, "set_price_threshold", config)
            await query.edit_message_text(f"已选择 {INTERVALS[interval]}周期")
            
            await query.message.reply_text(
                "请输入价格异动的阈值百分比（例如：0.5）:",
                reply_markup=back_markup)
        
        # 处理继续添加回调
        elif data_parts[0] == "continue_add":
            monitor_type = data_parts[1]
            set_user_state(user_id, f"add_symbol:{monitor_type}")
            await query.message.reply_text(
                f"请输入要添加{MONITOR_TYPE_NAMES.get(monitor_type, monitor_type)}监控的交易对（例如：BTCUSDT）:",
                reply_markup=back_markup)
        
        elif data_parts[0] == "finish_add":
            monitor_type = data_parts[1]
            user_data = load_user_data(user_id)
            
            # 询问是否立即开启监控
            keyboard = [
                [InlineKeyboardButton("✅ 立即开启", callback_data=f"enable_now:{monitor_type}")],
                [InlineKeyboardButton("❌ 稍后开启", callback_data="back_to_main")]
            ]
            
            # 获取已添加的币种列表（带详细信息）
            symbols_list = []
            for s in user_data['symbols']:
                if s['monitor'] == monitor_type:
                    if monitor_type == "price":
                        symbols_list.append(
                            f"• {s['symbol']}（{MARKET_TYPE_NAMES[s['type']]}）周期: {INTERVALS.get(s.get('interval', '15m'), '15分钟')} 阈值: {s.get('threshold', 5.0)}%"
                        )
                    else:
                        symbols_list.append(
                            f"• {s['symbol']}（{MARKET_TYPE_NAMES[s['type']]}）"
                        )
            
            symbols_list_text = "\n".join(symbols_list) if symbols_list else "无"
            
            await query.message.reply_text(
                f"已添加以下{MONITOR_TYPE_NAMES.get(monitor_type, monitor_type)}监控的交易对:\n{symbols_list_text}\n\n是否立即开启{MONITOR_TYPE_NAMES.get(monitor_type, monitor_type)}监控?",
                reply_markup=InlineKeyboardMarkup(keyboard))
        
        elif data_parts[0] == "enable_now":
            monitor_type = data_parts[1]
            user_data = load_user_data(user_id)  # 修复拼写错误
            
            # 启用监控
            user_data['monitors'][monitor_type]['enabled'] = True
            user_data['active'] = True
            save_user_data(user_id, user_data)
            
            # 获取监控币种列表（带详细信息）
            symbols_list = []
            for s in user_data['symbols']:
                if s['monitor'] == monitor_type:
                    if monitor_type == "price":
                        symbols_list.append(
                            f"• {s['symbol']}（{MARKET_TYPE_NAMES[s['type']]}）周期: {INTERVALS.get(s.get('interval', '15m'), '15分钟')} 阈值: {s.get('threshold', 5.0)}%"
                        )
                    else:
                        symbols_list.append(
                            f"• {s['symbol']}（{MARKET_TYPE_NAMES[s['type']]}）"
                        )
            
            symbols_list_text = "\n".join(symbols_list) if symbols_list else "无"
            
            await query.message.reply_text(
                f"✅ 已开启{MONITOR_TYPE_NAMES.get(monitor_type, monitor_type)}监控\n\n监控币种列表:\n{symbols_list_text}",
                reply_markup=reply_markup)
            clear_user_state(user_id)
        
        elif data_parts[0] == "back_to_main":
            clear_user_state(user_id)
            await query.message.reply_text(
                "已返回主菜单",
                reply_markup=reply_markup)
        
        elif data_parts[0] == "select_monitor":
            monitor_type = data_parts[1]
            set_user_state(user_id, f"add_symbol:{monitor_type}")
            await query.message.reply_text(
                f"请输入要添加{MONITOR_TYPE_NAMES.get(monitor_type, monitor_type)}监控的交易对（例如：BTCUSDT）:",
                reply_markup=back_markup)
        
        elif data_parts[0] == "remove_monitor":
            monitor_type = data_parts[1]
            user_data = load_user_data(user_id)
            
            # 获取指定监控类型的交易对
            symbols = [s for s in user_data['symbols'] if s['monitor'] == monitor_type]
            
            if not symbols:
                await query.message.reply_text(
                    f"当前没有{MONITOR_TYPE_NAMES.get(monitor_type, monitor_type)}监控的交易对",
                    reply_markup=reply_markup)
                return
            
            # 显示交易对列表（带详细信息）
            symbols_list = []
            for i, s in enumerate(symbols):
                if monitor_type == "price":
                    symbols_list.append(
                        f"{i+1}. {s['symbol']}（{MARKET_TYPE_NAMES[s['type']]}）周期: {INTERVALS.get(s.get('interval', '15m'), '15分钟')} 阈值: {s.get('threshold', 5.0)}%"
                    )
                else:
                    symbols_list.append(
                        f"{i+1}. {s['symbol']}（{MARKET_TYPE_NAMES[s['type']]}）"
                    )
            
            symbols_list_text = "\n".join(symbols_list)
            
            set_user_state(user_id, f"remove_symbol:{monitor_type}")
            await query.message.reply_text(
                f"请选择要删除的交易对:\n{symbols_list_text}\n\n请输入编号:",
                reply_markup=back_markup)
        
        elif data_parts[0] == "enable_monitor":
            monitor_type = data_parts[1]
            user_data = load_user_data(user_id)
            
            if monitor_type == "all":
                # 启用所有监控
                for mt in user_data["monitors"]:
                    user_data["monitors"][mt]["enabled"] = True
                user_data["active"] = True
                monitor_type_str = "全部监控"
            else:
                user_data["monitors"][monitor_type]["enabled"] = True
                user_data["active"] = True
                monitor_type_str = MONITOR_TYPE_NAMES.get(monitor_type, monitor_type)
            
            save_user_data(user_id, user_data)
            
            # 获取监控币种列表（按类型分组显示）
            groups = {"price": [], "macd": [], "ma": []}
            for s in user_data['symbols']:
                if monitor_type == "all" or s['monitor'] == monitor_type:
                    groups[s['monitor']].append(s)
            
            symbols_list_text = ""
            for mt, symbols in groups.items():
                if symbols:
                    symbols_list_text += f"\n{MONITOR_TYPE_NAMES[mt]}监控:\n"
                    for s in symbols:
                        if mt == "price":
                            symbols_list_text += f"• {s['symbol']}（{MARKET_TYPE_NAMES[s['type']]}）周期: {INTERVALS.get(s.get('interval', '15m'), '15分钟')} 阈值: {s.get('threshold', 5.0)}%\n"
                        else:
                            symbols_list_text += f"• {s['symbol']}（{MARKET_TYPE_NAMES[s['type']]}）\n"
            
            if not symbols_list_text:
                symbols_list_text = "无"
            
            await query.message.reply_text(
                f"✅ 已开启{monitor_type_str}监控\n\n监控币种列表:{symbols_list_text}",
                reply_markup=reply_markup)
            clear_user_state(user_id)
        
        elif data_parts[0] == "disable_monitor":
            monitor_type = data_parts[1]
            user_data = load_user_data(user_id)
            
            if monitor_type == "all":
                # 禁用所有监控
                for mt in user_data["monitors"]:
                    user_data["monitors"][mt]["enabled"] = False
                user_data["active"] = False
                monitor_type_str = "全部监控"
                await query.message.reply_text(
                    f"✅ 已关闭{monitor_type_str}",
                    reply_markup=reply_markup)
            else:
                user_data["monitors"][monitor_type]["enabled"] = False
                monitor_type_str = MONITOR_TYPE_NAMES.get(monitor_type, monitor_type)
                
                # 检查是否还有任何监控启用
                any_enabled = any(user_data["monitors"][mt]["enabled"] for mt in user_data["monitors"])
                user_data["active"] = any_enabled
            
                save_user_data(user_id, user_data)
                await query.message.reply_text(
                    f"✅ 已关闭{monitor_type_str}监控",
                    reply_markup=reply_markup)
            clear_user_state(user_id)
    
    except Exception as e:
        logger.error(f"按钮回调处理出错: {e}", exc_info=True)
        await query.message.reply_text(
            "处理您的请求时出错，请重试",
            reply_markup=reply_markup)

# --- 命令处理 ---
async def start(update, context):
    user_id = update.effective_chat.id
    
    # 检查用户是否授权
    if user_id not in CHAT_IDS:
        await update.message.reply_text("您未获得使用此机器人的授权")
        return
    
    try:
        # 初始化用户数据
        user_data = load_user_data(user_id)
        save_user_data(user_id, user_data)
        
        await update.message.reply_text(
            "欢迎使用币安监控机器人\n请使用下方菜单开始操作",
            reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"启动命令出错: {e}")
        await update.message.reply_text("初始化失败，请稍后再试")

async def add_symbol(update, context):
    user_id = update.effective_chat.id
    
    # 选择监控类型
    keyboard = [
        [InlineKeyboardButton("1. 价格异动监控", callback_data="select_monitor:price")],
        [InlineKeyboardButton("2. MACD交叉监控", callback_data="select_monitor:macd")],
        [InlineKeyboardButton("3. MA交叉监控", callback_data="select_monitor:ma")],
        [InlineKeyboardButton("↩️ 返回主菜单", callback_data="back_to_main")]
    ]
    
    await update.message.reply_text(
        "请选择要添加的监控类型:",
        reply_markup=InlineKeyboardMarkup(keyboard))

async def remove_symbol(update, context):
    user_id = update.effective_chat.id
    user_data = load_user_data(user_id)
    
    if not user_data["symbols"]:
        await update.message.reply_text("当前没有监控的交易对", reply_markup=reply_markup)
        return
    
    # 选择监控类型
    keyboard = [
        [InlineKeyboardButton("1. 价格异动监控", callback_data="remove_monitor:price")],
        [InlineKeyboardButton("2. MACD交叉监控", callback_data="remove_monitor:macd")],
        [InlineKeyboardButton("3. MA交叉监控", callback_data="remove_monitor:ma")],
        [InlineKeyboardButton("↩️ 返回主菜单", callback_data="back_to_main")]
    ]
    
    await update.message.reply_text(
        "请选择要删除的监控类型:",
        reply_markup=InlineKeyboardMarkup(keyboard))

async def enable_monitoring(update, context):
    user_id = update.effective_chat.id
    user_data = load_user_data(user_id)
    
    if not user_data["symbols"]:
        await update.message.reply_text("请先添加交易对", reply_markup=reply_markup)
        return
    
    # 创建监控类型选择键盘
    keyboard = [
        [InlineKeyboardButton("1. 价格异动监控", callback_data="enable_monitor:price")],
        [InlineKeyboardButton("2. MACD交叉监控", callback_data="enable_monitor:macd")],
        [InlineKeyboardButton("3. MA交叉监控", callback_data="enable_monitor:ma")],
        [InlineKeyboardButton("4. 全部监控", callback_data="enable_monitor:all")],
        [InlineKeyboardButton("↩️ 返回主菜单", callback_data="back_to_main")]
    ]
    
    await update.message.reply_text(
        "请选择要开启的监控类型:",
        reply_markup=InlineKeyboardMarkup(keyboard))

async def disable_monitoring(update, context):
    user_id = update.effective_chat.id
    user_data = load_user_data(user_id)
    
    if not user_data["active"]:
        await update.message.reply_text("监控尚未开启", reply_markup=reply_markup)
        return
    
    # 创建监控类型选择键盘
    keyboard = [
        [InlineKeyboardButton("1. 价格异动监控", callback_data="disable_monitor:price")],  # 修复回调数据
        [InlineKeyboardButton("2. MACD交叉监控", callback_data="disable_monitor:macd")],  # 修复回调数据
        [InlineKeyboardButton("3. MA交叉监控", callback_data="disable_monitor:ma")],  # 修复回调数据
        [InlineKeyboardButton("4. 全部监控", callback_data="disable_monitor:all")],  # 修复回调数据
        [InlineKeyboardButton("↩️ 返回主菜单", callback_data="back_to_main")]
    ]
    
    await update.message.reply_text(
        "请选择要关闭的监控类型:",
        reply_markup=InlineKeyboardMarkup(keyboard))

async def show_status(update, context):
    user_id = update.effective_chat.id
    try:
        user_data = load_user_data(user_id)
        
        status = "🔴 监控已停止"
        if user_data["active"]:
            status = "🟢 监控运行中"
        
        # 按监控类型分组
        monitor_groups = {
            "price": [],
            "macd": [],
            "ma": []
        }

        for s in user_data['symbols']:
            monitor_type = s.get('monitor', 'price')
            if monitor_type in monitor_groups:
                monitor_groups[monitor_type].append(s)
        
        # 价格异动监控状态
        price_status = "🟢 已启用" if user_data["monitors"]["price"]["enabled"] else "🔴 已禁用"
        price_list = "\n".join([
            f"  • {s['symbol']}（{MARKET_TYPE_NAMES[s['type']]}）周期: {INTERVALS.get(s.get('interval', '15m'), '15分钟')} 阈值: {s.get('threshold', 5.0)}%"
            for s in monitor_groups["price"]
        ]) if monitor_groups["price"] else "  无"
        
        # MACD监控状态
        macd_status = "🟢 已启用" if user_data["monitors"]["macd"]["enabled"] else "🔴 已禁用"
        macd_list = "\n".join([
            f"  • {s['symbol']}（{MARKET_TYPE_NAMES[s['type']]}）" 
            for s in monitor_groups["macd"]
        ]) if monitor_groups["macd"] else "  无"
        
        # MA监控状态
        ma_status = "🟢 已启用" if user_data["monitors"]["ma"]["enabled"] else "🔴 已禁用"
        ma_list = "\n".join([
            f"  • {s['symbol']}（{MARKET_TYPE_NAMES[s['type']]}）" 
            for s in monitor_groups["ma"]
        ]) if monitor_groups["ma"] else "  无"
        
        # 显示当前使用的K线周期
        ma_macd_interval = INTERVALS.get(DEFAULT_INTERVAL, DEFAULT_INTERVAL)
        
        message = (
            f"📊 监控状态: {status}\n\n"
            f"1️⃣ 价格异动监控: {price_status}\n"
            f"   监控币种:\n{price_list}\n\n"
            f"2️⃣ MACD交叉监控: {macd_status}\n"
            f"   监控币种:\n{macd_list}\n\n"
            f"3️⃣ MA交叉监控: {ma_status}\n"
            f"   监控币种:\n{ma_list}\n\n"
            f"📈 MACD和MA监控使用 {ma_macd_interval} K线周期"
        )
        
        await update.message.reply_text(message, reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"显示状态出错: {e}", exc_info=True)
        await update.message.reply_text("获取状态失败，请稍后再试")

async def show_help(update, context):
    # 获取当前配置的MA/MACD周期
    ma_macd_interval = INTERVALS.get(DEFAULT_INTERVAL, DEFAULT_INTERVAL)
    
    help_text = (
        "📚 币安监控机器人使用指南\n\n"
        "1️⃣ 添加币种 - 添加新的监控币种\n"
        "2️⃣ 删除币种 - 删除现有监控币种\n"
        "3️⃣ 开启监控 - 启动价格监控\n"
        "4️⃣ 停止监控 - 暂停价格监控\n"
        "5️⃣ 查看状态 - 查看当前监控配置\n"
        "6️⃣ 帮助 - 显示使用指南\n\n"
        "监控类型说明:\n"
        "• 价格异动监控: 检测指定周期内价格波动超过设定阈值\n"
        "• MACD交叉监控: 检测MACD指标的金叉/死叉信号（基于{ma_macd_interval}K线）\n"
        "• MA交叉监控: 检测MA9和MA26的交叉信号（基于{ma_macd_interval}K线）\n\n"
        "🔄 服务器时间每15分钟与Binance同步一次\n"
        "⏱ 所有监控数据每分钟刷新一次"
    ).format(ma_macd_interval=ma_macd_interval)
    
    await update.message.reply_text(help_text, reply_markup=reply_markup)

async def handle_message(update, context):
    user_id = update.effective_chat.id
    text = update.message.text.strip()
    
    # 检查用户是否授权
    if user_id not in CHAT_IDS:
        await update.message.reply_text("您未获得使用此机器人的授权")
        return
    
    try:
        # 处理取消/返回命令
        if text in ["❌ 取消", "取消", "↩️ 返回主菜单"]:
            clear_user_state(user_id)
            await update.message.reply_text("已返回主菜单", reply_markup=reply_markup)
            return
        
        # 处理用户状态
        state_info = get_user_state(user_id)
        state = state_info.get("state", "")
        
        # 优先处理状态中的输入
        if state.startswith("add_symbol:"):
            monitor_type = state.split(":")[1]
            
            # 验证交易对格式
            if not (len(text) >= 5 and text.isalnum()):
                await update.message.reply_text("无效的交易对格式，请重新输入（例如：BTCUSDT）")
                return
            
            # 创建市场类型选择键盘
            keyboard = [
                [InlineKeyboardButton("现货", callback_data=f"select_type:{text.upper()}:spot:{monitor_type}")],
                [InlineKeyboardButton("合约", callback_data=f"select_type:{text.upper()}:contract:{monitor_type}")]
            ]
            
            await update.message.reply_text(
                f"请选择 {text.upper()} 的市场类型:",
                reply_markup=InlineKeyboardMarkup(keyboard))
            
            clear_user_state(user_id)
            return
        
        elif state.startswith("remove_symbol:"):
            monitor_type = state.split(":")[1]
            try:
                idx = int(text) - 1
                user_data = load_user_data(user_id)
                
                # 获取指定监控类型的交易对
                symbols = [s for s in user_data['symbols'] if s['monitor'] == monitor_type]
                
                if 0 <= idx < len(symbols):
                    # 从原始列表中删除
                    symbol_to_remove = symbols[idx]
                    user_data['symbols'] = [s for s in user_data['symbols'] if s != symbol_to_remove]
                    
                    save_user_data(user_id, user_data)
                    await update.message.reply_text(
                        f"已删除 {symbol_to_remove['symbol']}（{MARKET_TYPE_NAMES[symbol_to_remove['type']]}）",
                        reply_markup=reply_markup)
                    
                    # 检查是否还有交易对
                    symbols = [s for s in user_data['symbols'] if s['monitor'] == monitor_type]
                    if not symbols:
                        clear_user_state(user_id)
                        await update.message.reply_text("当前无监控交易对，已返回主菜单", reply_markup=reply_markup)
                        return
                    
                    # 显示剩余交易对（带详细信息）
                    symbols_list = []
                    for i, s in enumerate(symbols):
                        if monitor_type == "price":
                            symbols_list.append(
                                f"{i+1}. {s['symbol']}（{MARKET_TYPE_NAMES[s['type']]}）周期: {INTERVALS.get(s.get('interval', '15m'), '15分钟')} 阈值: {s.get('threshold', 5.0)}%"
                            )
                        else:
                            symbols_list.append(
                                f"{i+1}. {s['symbol']}（{MARKET_TYPE_NAMES[s['type']]}）"
                            )
                    
                    await update.message.reply_text(
                        f"剩余{MONITOR_TYPE_NAMES.get(monitor_type, monitor_type)}监控的交易对:\n\n" + "\n".join(symbols_list) + "\n\n请输入编号删除或输入'取消'返回主菜单:",
                        reply_markup=back_markup)
                else:
                    await update.message.reply_text("无效的编号，请重新输入")
            except ValueError:
                await update.message.reply_text("请输入有效的编号（例如：1）")
            return
        
        elif state == "set_price_threshold":
            try:
                threshold = float(text)
                if threshold <= 0 or threshold > 50:
                    await update.message.reply_text("阈值必须在0.1到50之间，请重新输入")
                    return
                    
                user_data = load_user_data(user_id)
                state_info = get_user_state(user_id)
                config = state_info.get("data", {})
                config["threshold"] = threshold
                
                # 添加完整的监控配置
                user_data["symbols"].append(config)
                save_user_data(user_id, user_data)
                logger.info(f"用户 {user_id} 添加 {config['symbol']} 监控: 周期{config.get('interval','')} 阈值{threshold}%")
                
                # 询问是否继续添加
                keyboard = [
                    [InlineKeyboardButton("✅ 继续添加", callback_data=f"continue_add:price")],
                    [InlineKeyboardButton("❌ 完成添加", callback_data=f"finish_add:price")]
                ]
                reply_markup_kb = InlineKeyboardMarkup(keyboard)
                await update.message.reply_text(
                    f"已为 {config['symbol']}（{MARKET_TYPE_NAMES[config['type']]}）添加价格异动监控: 周期{INTERVALS[config.get('interval','15m')]} 阈值{threshold}%\n\n是否继续添加交易对?",
                    reply_markup=reply_markup_kb)
                # 清除状态
                clear_user_state(user_id)
            except ValueError:
                await update.message.reply_text("请输入有效的数字（例如：0.5）")
            return
        
        # 处理主菜单命令（仅在无状态时处理）
        if text in ["1️⃣ 添加币种", "1"]:
            await add_symbol(update, context)
        elif text in ["2️⃣ 删除币种", "2"]:
            await remove_symbol(update, context)
        elif text in ["3️⃣ 开启监控", "3"]:
            await enable_monitoring(update, context)
        elif text in ["4️⃣ 停止监控", "4"]:
            await disable_monitoring(update, context)
        elif text in ["5️⃣ 查看状态", "5"]:
            await show_status(update, context)
        elif text in ["6️⃣ 帮助", "6"]:
            await show_help(update, context)
        else:
            await update.message.reply_text("无法识别的命令，请使用菜单操作", reply_markup=reply_markup)
            
    except Exception as e:
        logger.error(f"消息处理出错: {e}", exc_info=True)
        await update.message.reply_text("处理您的消息时出错，请重试")

# 错误处理器
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("处理更新时出现异常", exc_info=context.error)
    if update and isinstance(update, Update) and update.message:
        await update.message.reply_text("处理您的请求时出错，请稍后再试")

# 在应用启动后启动监控任务
async def on_startup(application):
    await start_monitor(application)
    logger.info("应用初始化完成，监控任务已启动")

# 在应用停止时停止监控任务
async def on_shutdown(application):
    await stop_monitor()
    logger.info("应用已停止")

# --- 主程序 ---
if __name__ == "__main__":
    # 创建应用实例，并设置启动和停止回调
    application = (
        Application.builder()
        .token(TOKEN)
        .post_init(on_startup)  # 启动回调
        .post_shutdown(on_shutdown)  # 关闭回调
        .build()
    )
    
    # 添加错误处理器
    application.add_error_handler(error_handler)
    
    # 添加命令处理器
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # 添加快捷命令
    application.add_handler(MessageHandler(filters.Regex(r'^1️⃣ 添加币种$|^1$'), add_symbol))
    application.add_handler(MessageHandler(filters.Regex(r'^2️⃣ 删除币种$|^2$'), remove_symbol))
    application.add_handler(MessageHandler(filters.Regex(r'^3️⃣ 开启监控$|^3$'), enable_monitoring))
    application.add_handler(MessageHandler(filters.Regex(r'^4️⃣ 停止监控$|^4$'), disable_monitoring))
    application.add_handler(MessageHandler(filters.Regex(r'^5️⃣ 查看状态$|^5$'), show_status))
    application.add_handler(MessageHandler(filters.Regex(r'^6️⃣ 帮助$|^6$'), show_help))
    
    logger.info("机器人已启动")
    application.run_polling()
