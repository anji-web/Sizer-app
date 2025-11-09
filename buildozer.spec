[app]
title = Body Measure AI WHB
package.name = whbbodymeasure
package.domain = org.cgc
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite
exclude_dirs = venv, __pycache__, .git
version = 1.0

requirements = python3,kivy,opencv-python,mediapipe,numpy,openssl,cython==0.29.33

orientation = portrait
fullscreen = 0

android.permissions = CAMERA,INTERNET
android.api = 33
android.minapi = 26

# SDK & NDK
android.sdk_path = /Users/anjidananto/.buildozer/android/platform/android-sdk
android.ndk_path = /Users/anjidananto/.buildozer/android/platform/android-ndk-r25b
android.ndk = 25b
android.ndk_api = 26
android.accept_sdk_license = True
android.enable_androidx = True
android.copy_libs = 1
android.archs = arm64-v8a

# python-for-android (p4a)
p4a.bootstrap = sdl2
p4a.local_recipes = ./recipes
p4a.extra_args = --enable-androidx
p4a.gradle_version = 8.1


# Gradle & Android Build Tools setup
android.gradle_plugin_version = 8.1.1
android.build_tools_version = 33.0.2
android.gradle_dependencies = com.android.tools.build:gradle:8.1.1

[buildozer]
log_level = 2
warn_on_root = 1
build_dir = ./.buildozer
bin_dir = ./bin
