package com.pronunciationcoach.app;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.webkit.JavascriptInterface;
import android.webkit.PermissionRequest;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import java.util.Locale;

/**
 * Pronunciation Coach for Android: a minimal WebView shell around the user's
 * own coach server (the FastAPI app in this repo, reached over HTTPS, e.g.
 * through a Cloudflare tunnel).
 *
 * Security choices, all deliberate:
 *  - Only https:// servers; no cleartext, no mixed content, no file access.
 *  - Navigation outside the configured server opens in the system browser.
 *  - The microphone is granted only to the configured server's origin.
 *  - The settings bridge (CoachApp.setServer) only works while the bundled
 *    setup page is showing, so a remote page can't change the server.
 *  - AndroidTTS only speaks text (the phone's own text-to-speech engine),
 *    because WebView has no speechSynthesis.
 */
public class MainActivity extends Activity {
    private static final String SETUP = "file:///android_asset/setup.html";
    private static final int REQ_MIC = 1, REQ_FILE = 2;

    private WebView web;
    private SharedPreferences prefs;
    private TextToSpeech tts;
    private boolean ttsReady = false;
    private volatile String currentUrl = "";
    private PermissionRequest pendingPermission;
    private ValueCallback<Uri[]> fileCallback;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        prefs = getSharedPreferences("coach", MODE_PRIVATE);
        web = new WebView(this);
        setContentView(web);

        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setMediaPlaybackRequiresUserGesture(false);
        s.setAllowFileAccess(false);          // bundled assets stay readable; device files don't
        s.setAllowContentAccess(false);
        s.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
        s.setSupportMultipleWindows(false);

        web.addJavascriptInterface(new Tts(), "AndroidTTS");
        web.addJavascriptInterface(new App(), "CoachApp");
        web.setWebViewClient(new Client());
        web.setWebChromeClient(new Chrome());

        tts = new TextToSpeech(this, status -> {
            ttsReady = status == TextToSpeech.SUCCESS;
            if (ttsReady) {
                tts.setLanguage(Locale.US);
                tts.setOnUtteranceProgressListener(new UtteranceProgressListener() {
                    @Override public void onStart(String id) { js("window.__ttsStart && window.__ttsStart()"); }
                    @Override public void onDone(String id) { js("window.__ttsEnd && window.__ttsEnd()"); }
                    @Override public void onError(String id) { js("window.__ttsEnd && window.__ttsEnd()"); }
                });
            }
        });

        String server = prefs.getString("server", "");
        web.loadUrl(server.isEmpty() ? SETUP : server);
    }

    private String server() { return prefs.getString("server", ""); }

    private static String origin(String url) {
        Uri u = Uri.parse(url);
        return u.getScheme() + "://" + u.getAuthority();
    }

    private boolean isServer(String url) {
        String s = server();
        return !s.isEmpty() && url != null && url.startsWith("https://") && origin(url).equals(origin(s));
    }

    private void js(final String code) {
        runOnUiThread(() -> web.evaluateJavascript(code, null));
    }

    private void showSetup(final String error) {
        runOnUiThread(() -> web.loadUrl(SETUP + (error == null ? "" : "#error=" + Uri.encode(error))));
    }

    // ------------------------------------------------------------ JS bridges
    private class Tts {
        @JavascriptInterface
        public void speak(String text, float rate) {
            if (!ttsReady || text == null) { js("window.__ttsEnd && window.__ttsEnd()"); return; }
            tts.setSpeechRate(Math.max(0.5f, Math.min(1.6f, rate)));
            Bundle b = new Bundle();
            tts.speak(text.length() > 3000 ? text.substring(0, 3000) : text, TextToSpeech.QUEUE_FLUSH, b, "coach");
        }

        @JavascriptInterface
        public void stop() { if (ttsReady) tts.stop(); }
    }

    private class App {
        @JavascriptInterface
        public boolean setServer(String url) {
            if (!currentUrl.startsWith(SETUP)) return false;   // only from the bundled setup page
            if (url == null) return false;
            url = url.trim();
            if (url.endsWith("/")) url = url.substring(0, url.length() - 1);
            if (!url.startsWith("https://") || Uri.parse(url).getHost() == null) return false;
            prefs.edit().putString("server", url + "/").apply();
            final String target = url + "/";
            runOnUiThread(() -> web.loadUrl(target));
            return true;
        }

        @JavascriptInterface
        public String getServer() { return currentUrl.startsWith(SETUP) ? server() : ""; }

        @JavascriptInterface
        public void openSetup() { showSetup(null); }
    }

    // ------------------------------------------------------------ navigation
    private class Client extends WebViewClient {
        @Override
        public void onPageStarted(WebView view, String url, Bitmap favicon) { currentUrl = url == null ? "" : url; }

        @Override
        public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest req) {
            String url = req.getUrl().toString();
            if (url.startsWith(SETUP) || isServer(url)) return false;
            if (url.startsWith("https://") || url.startsWith("http://") || url.startsWith("mailto:")) {
                try { startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse(url))); } catch (Exception ignored) { }
            }
            return true;
        }

        @Override
        public void onReceivedError(WebView view, WebResourceRequest req, WebResourceError err) {
            if (req.isForMainFrame() && !req.getUrl().toString().startsWith(SETUP)) {
                showSetup("Couldn't reach " + origin(req.getUrl().toString()) + " (" + err.getDescription() + "). Is the coach server running?");
            }
        }
    }

    private class Chrome extends WebChromeClient {
        @Override
        public void onPermissionRequest(final PermissionRequest request) {
            runOnUiThread(() -> {
                boolean wantsMic = false;
                for (String r : request.getResources()) if (PermissionRequest.RESOURCE_AUDIO_CAPTURE.equals(r)) wantsMic = true;
                if (!wantsMic || !isServer(request.getOrigin().toString())) { request.deny(); return; }
                if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                    request.grant(new String[]{PermissionRequest.RESOURCE_AUDIO_CAPTURE});
                } else {
                    pendingPermission = request;
                    requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, REQ_MIC);
                }
            });
        }

        @Override
        public boolean onShowFileChooser(WebView view, ValueCallback<Uri[]> cb, FileChooserParams params) {
            if (fileCallback != null) fileCallback.onReceiveValue(null);
            fileCallback = cb;
            Intent i = new Intent(Intent.ACTION_GET_CONTENT);
            i.addCategory(Intent.CATEGORY_OPENABLE);
            i.setType("audio/*");
            try { startActivityForResult(Intent.createChooser(i, "Choose a recording"), REQ_FILE); }
            catch (Exception e) { fileCallback = null; return false; }
            return true;
        }
    }

    @Override
    public void onRequestPermissionsResult(int code, String[] perms, int[] results) {
        if (code == REQ_MIC && pendingPermission != null) {
            if (results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED)
                pendingPermission.grant(new String[]{PermissionRequest.RESOURCE_AUDIO_CAPTURE});
            else pendingPermission.deny();
            pendingPermission = null;
        }
    }

    @Override
    protected void onActivityResult(int code, int result, Intent data) {
        if (code == REQ_FILE && fileCallback != null) {
            Uri[] uris = (result == RESULT_OK && data != null && data.getData() != null) ? new Uri[]{data.getData()} : null;
            fileCallback.onReceiveValue(uris);
            fileCallback = null;
        }
    }

    @Override
    public void onBackPressed() {
        if (web.canGoBack()) web.goBack(); else super.onBackPressed();
    }

    @Override
    protected void onDestroy() {
        if (tts != null) tts.shutdown();
        web.destroy();
        super.onDestroy();
    }
}
