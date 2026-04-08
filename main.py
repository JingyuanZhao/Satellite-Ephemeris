#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卫星星历计算器 - Kivy版本
用于Android APK构建
"""

import os
import sys

# 设置Kivy环境变量
os.environ['KIVY_NO_CONSOLELOG'] = '1'
os.environ['KIVY_WINDOW'] = 'sdl2'

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelHeader
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.core.window import Window
from threading import Thread

# 导入核心计算模块
import json
import requests
import math
from datetime import datetime, timedelta

# 配置文件路径
CONFIG_FILE = "mpc_codes.json"
DEFAULT_STD_MAG = None
EARTH_RADIUS_KM = 6378.14


class SatelliteTrackerApp(App):
    """卫星星历计算器主应用"""

    def build(self):
        """构建应用界面"""
        self.title = '卫星星历计算器'
        Window.clearcolor = (0.95, 0.95, 0.95, 1)

        # 主布局
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # 标题
        title_label = Label(
            text='卫星星历计算器',
            font_size='24sp',
            size_hint_y=None,
            height=50,
            color=(0.1, 0.1, 0.5, 1)
        )
        main_layout.add_widget(title_label)

        # 创建标签页
        tab_panel = TabbedPanel(do_default_tab=False)

        # 输入标签页
        input_tab = TabbedPanelHeader(text='参数输入')
        input_content = self._create_input_tab()
        input_tab.content = input_content
        tab_panel.add_widget(input_tab)

        # 结果标签页
        result_tab = TabbedPanelHeader(text='计算结果')
        self.result_content = self._create_result_tab()
        result_tab.content = self.result_content
        tab_panel.add_widget(result_tab)

        main_layout.add_widget(tab_panel)

        return main_layout

    def _create_input_tab(self):
        """创建输入标签页"""
        scroll = ScrollView()
        layout = GridLayout(cols=1, spacing=10, padding=10, size_hint_y=None)
        layout.bind(minimum_height=layout.setter('height'))

        # 观测地点输入
        layout.add_widget(Label(text='观测地点', font_size='18sp', size_hint_y=None, height=30))

        # 纬度
        lat_box = BoxLayout(size_hint_y=None, height=40)
        lat_box.add_widget(Label(text='纬度:', size_hint_x=0.3))
        self.lat_input = TextInput(text='39.9', multiline=False, size_hint_x=0.7, input_filter='float')
        lat_box.add_widget(self.lat_input)
        layout.add_widget(lat_box)

        # 经度
        lon_box = BoxLayout(size_hint_y=None, height=40)
        lon_box.add_widget(Label(text='经度:', size_hint_x=0.3))
        self.lon_input = TextInput(text='116.4', multiline=False, size_hint_x=0.7, input_filter='float')
        lon_box.add_widget(self.lon_input)
        layout.add_widget(lon_box)

        # 海拔
        height_box = BoxLayout(size_hint_y=None, height=40)
        height_box.add_widget(Label(text='海拔(m):', size_hint_x=0.3))
        self.height_input = TextInput(text='0', multiline=False, size_hint_x=0.7, input_filter='float')
        height_box.add_widget(self.height_input)
        layout.add_widget(height_box)

        # 卫星信息
        layout.add_widget(Label(text='卫星信息', font_size='18sp', size_hint_y=None, height=30))

        # NORAD ID
        sat_id_box = BoxLayout(size_hint_y=None, height=40)
        sat_id_box.add_widget(Label(text='NORAD ID:', size_hint_x=0.3))
        self.sat_id_input = TextInput(text='25544', multiline=False, size_hint_x=0.7, input_filter='int')
        sat_id_box.add_widget(self.sat_id_input)
        layout.add_widget(sat_id_box)

        # 卫星名称
        sat_name_box = BoxLayout(size_hint_y=None, height=40)
        sat_name_box.add_widget(Label(text='卫星名称:', size_hint_x=0.3))
        self.sat_name_input = TextInput(text='ISS', multiline=False, size_hint_x=0.7)
        sat_name_box.add_widget(self.sat_name_input)
        layout.add_widget(sat_name_box)

        # 本征星等
        mag_box = BoxLayout(size_hint_y=None, height=40)
        mag_box.add_widget(Label(text='本征星等:', size_hint_x=0.3))
        self.mag_input = TextInput(text='-2.0', multiline=False, size_hint_x=0.7)
        mag_box.add_widget(self.mag_input)
        layout.add_widget(mag_box)

        # TLE数据
        layout.add_widget(Label(text='TLE数据 (可选)', font_size='18sp', size_hint_y=None, height=30))
        self.tle_input = TextInput(
            text='',
            multiline=True,
            size_hint_y=None,
            height=100,
            hint_text='输入TLE两行数据...'
        )
        layout.add_widget(self.tle_input)

        # 计算参数
        layout.add_widget(Label(text='计算参数', font_size='18sp', size_hint_y=None, height=30))

        # 计算时长
        duration_box = BoxLayout(size_hint_y=None, height=40)
        duration_box.add_widget(Label(text='时长:', size_hint_x=0.3))
        self.duration_input = TextInput(text='24', multiline=False, size_hint_x=0.4, input_filter='float')
        duration_box.add_widget(self.duration_input)
        self.duration_unit = Spinner(
            text='小时',
            values=['秒', '分钟', '小时', '天'],
            size_hint_x=0.3
        )
        duration_box.add_widget(self.duration_unit)
        layout.add_widget(duration_box)

        # 时间步长
        step_box = BoxLayout(size_hint_y=None, height=40)
        step_box.add_widget(Label(text='步长:', size_hint_x=0.3))
        self.step_input = TextInput(text='1', multiline=False, size_hint_x=0.4, input_filter='float')
        step_box.add_widget(self.step_input)
        self.step_unit = Spinner(
            text='分钟',
            values=['秒', '分钟', '小时', '天'],
            size_hint_x=0.3
        )
        step_box.add_widget(self.step_unit)
        layout.add_widget(step_box)

        # 按钮区域
        button_box = BoxLayout(size_hint_y=None, height=50, spacing=10)

        fetch_btn = Button(text='获取TLE数据')
        fetch_btn.bind(on_press=self._fetch_tle_data)
        button_box.add_widget(fetch_btn)

        calc_btn = Button(text='开始计算')
        calc_btn.bind(on_press=self._start_calculation)
        button_box.add_widget(calc_btn)

        layout.add_widget(button_box)

        scroll.add_widget(layout)
        return scroll

    def _create_result_tab(self):
        """创建结果标签页"""
        scroll = ScrollView()
        self.result_layout = GridLayout(cols=1, spacing=5, padding=10, size_hint_y=None)
        self.result_layout.bind(minimum_height=self.result_layout.setter('height'))

        # 默认提示
        self.result_layout.add_widget(
            Label(text='请先输入参数并点击"开始计算"', font_size='16sp')
        )

        scroll.add_widget(self.result_layout)
        return scroll

    def _show_popup(self, title, message):
        """显示弹出窗口"""
        popup = Popup(
            title=title,
            content=Label(text=message),
            size_hint=(None, None),
            size=(400, 200)
        )
        popup.open()

    def _fetch_tle_data(self, instance):
        """获取TLE数据"""
        sat_id = self.sat_id_input.text.strip()
        if not sat_id:
            self._show_popup('错误', '请输入NORAD ID')
            return

        def fetch():
            try:
                # 这里简化处理，实际应该调用CelesTrak或N2YO API
                Clock.schedule_once(lambda dt: self._show_popup('提示', f'正在获取卫星 {sat_id} 的TLE数据...'), 0)
            except Exception as e:
                Clock.schedule_once(lambda dt: self._show_popup('错误', f'获取TLE数据失败: {str(e)}'), 0)

        Thread(target=fetch).start()

    def _start_calculation(self, instance):
        """开始计算"""
        try:
            # 获取输入参数
            lat = float(self.lat_input.text)
            lon = float(self.lon_input.text)
            height = float(self.height_input.text)
            sat_id = self.sat_id_input.text.strip()
            sat_name = self.sat_name_input.text.strip()

            # 清空结果区域
            self.result_layout.clear_widgets()

            # 添加计算结果标题
            self.result_layout.add_widget(Label(
                text=f'计算结果 - {sat_name}',
                font_size='18sp',
                size_hint_y=None,
                height=30
            ))

            # 添加示例结果
            self.result_layout.add_widget(Label(
                text=f'观测地点: 纬度 {lat}, 经度 {lon}, 海拔 {height}m',
                font_size='14sp',
                size_hint_y=None,
                height=25
            ))

            self.result_layout.add_widget(Label(
                text=f'卫星ID: {sat_id}',
                font_size='14sp',
                size_hint_y=None,
                height=25
            ))

            self.result_layout.add_widget(Label(
                text='计算功能开发中...',
                font_size='14sp',
                size_hint_y=None,
                height=25
            ))

        except ValueError as e:
            self._show_popup('错误', f'输入参数无效: {str(e)}')
        except Exception as e:
            self._show_popup('错误', f'计算失败: {str(e)}')


if __name__ == '__main__':
    SatelliteTrackerApp().run()