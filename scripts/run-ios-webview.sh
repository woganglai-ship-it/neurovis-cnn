#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8001}"
PROJECT="$ROOT/ios/NeuroVisWebView/NeuroVisWebView.xcodeproj"
DERIVED_DATA="$ROOT/ios/NeuroVisWebView/build"
BUNDLE_ID="com.local.NeuroVisWebView"

if ! lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Starting web server on http://127.0.0.1:$PORT"
  (cd "$ROOT" && nohup python3 -m http.server "$PORT" >/tmp/neurovis-webview-server.log 2>&1 &)
  sleep 1
fi

if ! xcrun simctl list devices booted | grep -q "(Booted)"; then
  echo "No booted iPhone Simulator found. Open Simulator first, then run this again."
  exit 1
fi

xcodebuild \
  -project "$PROJECT" \
  -scheme NeuroVisWebView \
  -configuration Debug \
  -sdk iphonesimulator \
  -derivedDataPath "$DERIVED_DATA" \
  CODE_SIGNING_ALLOWED=NO \
  build

APP_PATH="$DERIVED_DATA/Build/Products/Debug-iphonesimulator/NeuroVisWebView.app"
xcrun simctl terminate booted "$BUNDLE_ID" >/dev/null 2>&1 || true
xcrun simctl uninstall booted "$BUNDLE_ID" >/dev/null 2>&1 || true
xcrun simctl install booted "$APP_PATH"
xcrun simctl terminate booted com.apple.mobilesafari >/dev/null 2>&1 || true
xcrun simctl launch booted "$BUNDLE_ID"

osascript >/dev/null 2>&1 <<'APPLESCRIPT' || true
tell application "Simulator" to activate
tell application "System Events"
  tell process "Simulator"
    tell menu bar 1
      tell menu bar item "Device"
        tell menu "Device"
          if exists menu item "Orientation" then
            tell menu item "Orientation"
              tell menu 1
                if exists menu item "Landscape Right" then click menu item "Landscape Right"
                if exists menu item "横排右" then click menu item "横排右"
                if exists menu item "横向右" then click menu item "横向右"
              end tell
            end tell
          end if
        end tell
      end tell
      tell menu bar item "Window"
        tell menu "Window"
          if exists menu item "Stay On Top" then
            if (value of attribute "AXMenuItemMarkChar" of menu item "Stay On Top") is missing value then click menu item "Stay On Top"
          else if exists menu item "保持在最前" then
            if (value of attribute "AXMenuItemMarkChar" of menu item "保持在最前") is missing value then click menu item "保持在最前"
          else if exists menu item "置于顶层" then
            if (value of attribute "AXMenuItemMarkChar" of menu item "置于顶层") is missing value then click menu item "置于顶层"
          end if
        end tell
      end tell
    end tell
  end tell
end tell
APPLESCRIPT

echo "Opened NeuroVis WebView preview in the booted Simulator."
