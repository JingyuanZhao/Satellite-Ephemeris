[app]
title = 卫星星历计算器
package.name = satellitetracker
package.domain = org.example
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json,bsp
version = 2.0
requirements = python3,kivy,requests,certifi,urllib3,idna,charset-normalizer
orientation = portrait
fullscreen = 0
android.permissions = INTERNET,ACCESS_NETWORK_STATE
android.api = 33
android.minapi = 21
android.sdk = 33
android.ndk = 25b
android.archs = arm64-v8a
android.accept_sdk_license = True

[buildozer]
log_level = 2
warn_on_root = 1
