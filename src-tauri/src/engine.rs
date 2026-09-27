//! Runs the Python engine as a child process and relays JSON-lines requests to it.
//!
//! Each request gets an id; a reader thread matches responses back to the waiting
//! command and forwards unsolicited events (status messages) to the UI.

use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter};
use tokio::sync::oneshot;

type Pending = Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>;

pub struct Engine {
    child: Mutex<Child>,
    stdin: Mutex<ChildStdin>,
    pending: Pending,
    next_id: AtomicU64,
}

/// Where the engine's uv project lives. Dev builds use the repo checkout;
/// RETOUCH_ENGINE_DIR overrides it. (Packaging will bundle it — not done yet.)
fn engine_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("RETOUCH_ENGINE_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../engine")
}

impl Engine {
    pub fn spawn(app: AppHandle) -> Result<Self, String> {
        let dir = engine_dir();
        let mut child = Command::new("uv")
            .args(["run", "--project"])
            .arg(&dir)
            .args(["retouch-engine", "serve"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("couldn't start the engine in {}: {e}", dir.display()))?;

        let stdin = child.stdin.take().ok_or("engine has no stdin")?;
        let stdout = child.stdout.take().ok_or("engine has no stdout")?;
        let pending: Pending = Arc::default();

        let reader_pending = pending.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                let Ok(msg) = serde_json::from_str::<Value>(&line) else {
                    eprintln!("engine: unparseable line: {line}");
                    continue;
                };
                if let Some(id) = msg.get("id").and_then(Value::as_u64) {
                    if let Some(tx) = reader_pending.lock().unwrap().remove(&id) {
                        let _ = tx.send(msg);
                    }
                } else if msg.get("event").is_some() {
                    let _ = app.emit("engine-event", msg);
                }
            }
            // Engine exited: fail everything still waiting rather than hang forever.
            for (_, tx) in reader_pending.lock().unwrap().drain() {
                let _ = tx.send(json!({"error": {"message": "the engine stopped unexpectedly"}}));
            }
            let _ = app.emit("engine-event", json!({"event": "stopped"}));
        });

        Ok(Self {
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
            pending,
            next_id: AtomicU64::new(1),
        })
    }

    pub async fn call(&self, method: &str, params: Value) -> Result<Value, Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().unwrap().insert(id, tx);
        let line = json!({"id": id, "method": method, "params": params}).to_string();
        {
            let mut stdin = self.stdin.lock().unwrap();
            if let Err(e) = writeln!(stdin, "{line}").and_then(|_| stdin.flush()) {
                self.pending.lock().unwrap().remove(&id);
                return Err(json!({"message": format!("couldn't reach the engine: {e}")}));
            }
        }
        let msg = rx
            .await
            .map_err(|_| json!({"message": "the engine stopped unexpectedly"}))?;
        match msg.get("error") {
            Some(err) => Err(err.clone()),
            None => Ok(msg.get("result").cloned().unwrap_or(Value::Null)),
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
        }
    }
}
