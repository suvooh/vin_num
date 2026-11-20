package com.tieitup.vinplate_tester

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import android.os.Build
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "vinplate/sys").setMethodCallHandler { call, result ->
            if (call.method != "lastExits") {
                result.notImplemented()
                return@setMethodCallHandler
            }
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) {
                result.success(emptyList<String>())
                return@setMethodCallHandler
            }
            try {
                val am = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                val fmt = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
                val out = am.getHistoricalProcessExitReasons(packageName, 0, 3).map { e ->
                    val reason = when (e.reason) {
                        ApplicationExitInfo.REASON_CRASH -> "CRASH (Java/Kotlin exception)"
                        ApplicationExitInfo.REASON_CRASH_NATIVE -> "NATIVE CRASH (segfault in a C/C++ library, e.g. ONNX Runtime)"
                        ApplicationExitInfo.REASON_ANR -> "ANR (app not responding)"
                        ApplicationExitInfo.REASON_LOW_MEMORY -> "KILLED: low memory"
                        ApplicationExitInfo.REASON_SIGNALED -> "KILLED by signal ${e.status} (often out-of-memory)"
                        ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE -> "KILLED: excessive resource usage"
                        ApplicationExitInfo.REASON_USER_REQUESTED -> "closed by user"
                        ApplicationExitInfo.REASON_USER_STOPPED -> "force-stopped by user"
                        ApplicationExitInfo.REASON_EXIT_SELF -> "exited normally"
                        else -> "reason code ${e.reason}"
                    }
                    "${fmt.format(Date(e.timestamp))}  $reason  | ${e.description ?: ""}  | rss ${e.rss / 1024} MB, pss ${e.pss / 1024} MB"
                }
                result.success(out)
            } catch (t: Throwable) {
                result.success(listOf("exit info unavailable: ${t.message}"))
            }
        }
    }
}
