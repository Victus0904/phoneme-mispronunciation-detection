# Pronunciation Coach for Android

A small (about 50 KB) Android app that opens **your own** coach server, the
`server.py` app in this repo shared over HTTPS (for example with
`run_public.bat`'s free Cloudflare link). The speech model runs on your
computer, not on the phone.

## Why it's safe to install

- **Two permissions only:** internet and microphone, plus
  `MODIFY_AUDIO_SETTINGS`, which Android needs for in-app recording. No
  storage, contacts, location, ads, analytics or third-party libraries.
- **HTTPS only:** no cleartext traffic, no user-installed certificates, and
  no mixed content.
- **Stays on your server:** links to anywhere else open in your normal
  browser. The microphone is only granted to your server's address.
- **Locked-down settings:** the "change server" bridge only works on the
  app's own built-in setup screen, so a web page can't redirect the app.
- **Fully readable source:** everything in the APK is in this folder. That's
  one Java file (`app/src/.../MainActivity.java`), one setup page, icons
  and the manifest.
- **Signed with your own key**, which `build_apk.sh` creates outside the
  repo. Check the file you install against the SHA-256 the build prints.

## Install

1. Download `PronunciationCoach.apk` from the
   [latest release](https://github.com/Victus0904/phoneme-mispronunciation-detection/releases/latest)
   on your phone, or copy your own build over (USB, Google Drive, or email).
   The release notes list its SHA-256 and the signing certificate's
   fingerprint, so you can check the file before installing.
2. Open it. Android will ask you to allow installs from that app (Files,
   Drive…) once. This is normal for apps not from the Play Store.
3. On your computer, run `run_public.bat` and copy the
   `https://….trycloudflare.com` link.
4. Open the app, paste the link, and press **Connect**. When the link
   changes, use **Settings → Change coach link**.

## Build it yourself

Needs JDK 17 plus the Android SDK command-line tools with
`platforms/android-36` and `build-tools/36.1.0`. All are free.

```
JAVA_HOME=/path/to/jdk17 ANDROID_SDK=/path/to/sdk bash android/build_apk.sh
```

It produces `android/build/PronunciationCoach.apk`. The first build creates
a signing key in `~/Android/keys/`. Back it up, because updates must be
signed with the same key.
