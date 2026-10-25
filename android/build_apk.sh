#!/usr/bin/env bash
# Build + sign the Pronunciation Coach APK with the Android SDK command-line
# tools only (aapt2 -> javac -> d8 -> zipalign -> apksigner). No Gradle, no
# third-party libraries: everything that goes into the APK is in this folder.
#
#   ANDROID_SDK  path to the SDK (needs platforms/android-36, build-tools/36.1.0)
#   JAVA_HOME    a JDK 17
#   KEYSTORE     your release keystore; created on first run if missing
#   KS_PASS      its password (first run: generated and saved next to it)
set -euo pipefail
cd "$(dirname "$0")"
ANDROID_SDK="${ANDROID_SDK:-$HOME/Android/sdk}"
BT="$ANDROID_SDK/build-tools/36.1.0"
JAR="$ANDROID_SDK/platforms/android-36/android.jar"
KEYSTORE="${KEYSTORE:-$HOME/Android/keys/coach-release.jks}"
EXE=""; [ -f "$BT/aapt2.exe" ] && EXE=".exe"
BAT=""; [ -f "$BT/d8.bat" ] && BAT=".bat"
OUT=build; rm -rf "$OUT"; mkdir -p "$OUT/gen" "$OUT/classes" "$OUT/dex"
VERSION_CODE="${VERSION_CODE:-2}"; VERSION_NAME="${VERSION_NAME:-1.1}"

echo "== resources"
"$BT/aapt2$EXE" compile --dir app/res -o "$OUT/res.zip"
"$BT/aapt2$EXE" link -o "$OUT/base.apk" -I "$JAR" --manifest app/AndroidManifest.xml -A app/assets \
  --java "$OUT/gen" --min-sdk-version 24 --target-sdk-version 36 \
  --version-code "$VERSION_CODE" --version-name "$VERSION_NAME" "$OUT/res.zip"

echo "== java"
find app/src "$OUT/gen" -name '*.java' > "$OUT/sources.txt"
"$JAVA_HOME/bin/javac" -nowarn -source 11 -target 11 -Xlint:-options -encoding UTF-8 \
  -classpath "$JAR" -d "$OUT/classes" @"$OUT/sources.txt"

echo "== dex"
find "$OUT/classes" -name '*.class' > "$OUT/classes.txt"
"$BT/d8$BAT" --release --min-api 24 --lib "$JAR" --output "$OUT/dex" $(cat "$OUT/classes.txt")
cp "$OUT/base.apk" "$OUT/unsigned.apk"
(cd "$OUT/dex" && "$JAVA_HOME/bin/jar" uf ../unsigned.apk classes.dex)

echo "== align + sign"
"$BT/zipalign$EXE" -p -f 4 "$OUT/unsigned.apk" "$OUT/aligned.apk"
if [ ! -f "$KEYSTORE" ]; then
  mkdir -p "$(dirname "$KEYSTORE")"
  KS_PASS="$(head -c 24 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 24)"
  "$JAVA_HOME/bin/keytool" -genkeypair -keystore "$KEYSTORE" -alias coach -keyalg RSA -keysize 3072 -validity 10000 \
    -storepass "$KS_PASS" -keypass "$KS_PASS" -dname "CN=Pronunciation Coach, O=Personal" >/dev/null
  printf 'KS_PASS=%s\n' "$KS_PASS" > "${KEYSTORE%.jks}.password"
  echo "created signing key $KEYSTORE (password saved next to it -- back both up, never commit them)"
fi
KS_PASS="${KS_PASS:-$(sed -n 's/^KS_PASS=//p' "${KEYSTORE%.jks}.password")}"
"$BT/apksigner$BAT" sign --ks "$KEYSTORE" --ks-key-alias coach --ks-pass "pass:$KS_PASS" --key-pass "pass:$KS_PASS" \
  --out "$OUT/PronunciationCoach.apk" "$OUT/aligned.apk"
"$BT/apksigner$BAT" verify --verbose "$OUT/PronunciationCoach.apk" | head -4
echo "== done: android/$OUT/PronunciationCoach.apk"
