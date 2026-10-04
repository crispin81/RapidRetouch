//! Runs the Python engine as a child process and relays JSON-lines requests to it.
//!
//! Each request gets an id; a reader thread matches responses back to the waiting
//! command and forwards unsolicited events (status messages) to the UI.
//!
//! The engine starts in the background, so the window opens at once and calls
//! simply wait until it's up. In a released app it's set up on first launch:
//! the app ships uv (a single small binary) and the engine's source, and uv
//! downloads Python and the engine's libraries into the user's app data
//! folder, with PyTorch to suit the computer (NVIDIA GPU, Apple GPU or CPU).
//! The UI is told what's happening as "setup" events. A later version of the
//! app redoes it (quickly, from the download cache) when the engine changes.

use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager};
use tokio::sync::{oneshot, watch};

type Pending = Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>;

/// The running engine process.
struct Running {
    child: Mutex<Child>,
    stdin: Mutex<ChildStdin>,
    pending: Pending,
}

#[derive(Clone)]
enum State {
    Starting,
    Ready(Arc<Running>),
    Failed(String),
}

pub struct Engine {
    state: watch::Receiver<State>,
    next_id: AtomicU64,
}

/// Start helpers clean: without the variables a Linux AppImage's launcher sets
/// for its own bundled libraries (the engine's Python, inheriting them, looked
/// for its standard library in the wrong place and wouldn't start), and on
/// Windows without flashing a console window.
fn quiet(cmd: &mut Command) -> &mut Command {
    for var in ["PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "LD_LIBRARY_PATH", "LD_PRELOAD"] {
        cmd.env_remove(var);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

fn emit_setup(app: &AppHandle, message: &str, fraction: Option<f64>) {
    let _ = app.emit(
        "engine-event",
        json!({"event": "setup", "message": message, "fraction": fraction}),
    );
}

/// "cuda" on a PC with a working NVIDIA driver, else "cpu" (a Mac's build
/// has Apple GPU support either way).
fn backend() -> &'static str {
    if cfg!(target_os = "macos") {
        return "cpu";
    }
    let found = quiet(Command::new("nvidia-smi").arg("-L"))
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .map(|o| o.status.success() && String::from_utf8_lossy(&o.stdout).contains("GPU"))
        .unwrap_or(false);
    if found {
        "cuda"
    } else {
        "cpu"
    }
}

/// The engine's Python in its environment.
fn env_python(env: &Path) -> PathBuf {
    if cfg!(windows) {
        env.join("Scripts").join("python.exe")
    } else {
        env.join("bin").join("python")
    }
}

/// Sizes uv prints, e.g. "Downloading torch (812.6MiB)": in MiB.
fn size_mib(line: &str) -> Option<f64> {
    let inner = line.rsplit_once('(')?.1.strip_suffix(')')?;
    let (num, unit) = inner.split_at(inner.find(|c: char| c.is_ascii_alphabetic())?);
    let n: f64 = num.trim().parse().ok()?;
    Some(match unit {
        "GiB" => n * 1024.0,
        "MiB" => n,
        "KiB" => n / 1024.0,
        _ => return None,
    })
}

fn downloading(done: f64, total: f64) -> String {
    format!("Downloading the AI engine… {:.0} of {:.0} MB", done, total)
}

/// Released app: set up the engine if this version hasn't yet, and give the
/// command that starts it.
fn prepare_bundled(app: &AppHandle) -> Result<Command, String> {
    let exe_dir = std::env::current_exe()
        .map_err(|e| e.to_string())?
        .parent()
        .ok_or("no app folder")?
        .to_path_buf();
    let uv = exe_dir.join(if cfg!(windows) { "uv.exe" } else { "uv" });
    let source = app
        .path()
        .resource_dir()
        .map_err(|e| e.to_string())?
        .join("engine");
    let data = app.path().app_local_data_dir().map_err(|e| e.to_string())?;
    let env = data.join("engine");
    let backend = backend();
    let version = app.package_info().version.to_string();
    let stamp = format!("{version} {backend}");
    let marker = env.join("rapidretouch-setup.txt");

    if std::fs::read_to_string(&marker).ok().as_deref() != Some(stamp.as_str()) {
        let first = !env.exists();
        emit_setup(
            app,
            if first {
                "Setting up RapidRetouch's AI engine. This happens once and needs an internet connection."
            } else {
                "Updating RapidRetouch's AI engine."
            },
            Some(0.0),
        );
        let mut child = quiet(&mut Command::new(&uv))
            .args(["sync", "--no-config", "--frozen", "--no-editable", "--no-default-groups"])
            .args(["--group", backend, "--python", "3.12", "--project"])
            .arg(&source)
            .env("UV_PROJECT_ENVIRONMENT", &env)
            .env("UV_PYTHON_INSTALL_DIR", data.join("python"))
            .env("UV_CACHE_DIR", data.join("download-cache"))
            .env("UV_PYTHON_PREFERENCE", "only-managed")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("couldn't run the engine setup ({}): {e}", uv.display()))?;
        // Progress from uv's messages: bytes downloaded of those announced so far.
        let (mut total, mut done) = (0.0_f64, 0.0_f64);
        let mut sizes: HashMap<String, f64> = HashMap::new();
        let mut last = String::new();
        for line in BufReader::new(child.stderr.take().ok_or("no setup output")?).lines() {
            let Ok(line) = line else { break };
            let text = line.trim();
            if text.is_empty() {
                continue;
            }
            last = text.to_string();
            if let Some(rest) = text.strip_prefix("Downloading ") {
                if let Some(mib) = size_mib(rest) {
                    let name = rest.split(" (").next().unwrap_or(rest).to_string();
                    total += mib;
                    sizes.insert(name.clone(), mib);
                    emit_setup(app, &downloading(done, total), Some(done / total.max(1.0)));
                }
            } else if let Some(name) = text.strip_prefix("Downloaded ") {
                done += sizes.get(name).copied().unwrap_or(0.0);
                emit_setup(app, &downloading(done, total), Some(done / total.max(1.0)));
            } else if text.starts_with("Prepared") || text.starts_with("Installed") {
                emit_setup(app, "Installing the AI engine…", Some(0.97));
            }
        }
        let status = child.wait().map_err(|e| e.to_string())?;
        if !status.success() {
            return Err(format!(
                "Setting up the AI engine didn't finish. Check the internet connection and open RapidRetouch again. ({last})"
            ));
        }
        std::fs::write(&marker, &stamp).map_err(|e| e.to_string())?;
        emit_setup(app, "Ready", Some(1.0));
    }
    let mut cmd = Command::new(env_python(&env));
    cmd.args(["-m", "rapidretouch_engine.cli", "serve"]);
    Ok(cmd)
}

/// Development: the repo checkout, through the developer's own uv.
/// RAPIDRETOUCH_ENGINE_DIR overrides where it is.
fn dev_command() -> Command {
    let dir = std::env::var("RAPIDRETOUCH_ENGINE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../engine"));
    let mut cmd = Command::new("uv");
    cmd.args(["run", "--project"]).arg(dir).args(["rapidretouch-engine", "serve"]);
    cmd
}

fn start(app: &AppHandle) -> Result<Running, String> {
    let mut cmd = if cfg!(debug_assertions) || std::env::var("RAPIDRETOUCH_ENGINE_DIR").is_ok() {
        dev_command()
    } else {
        prepare_bundled(app)?
    };
    let mut child = quiet(&mut cmd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("couldn't start the engine: {e}"))?;

    let stdin = child.stdin.take().ok_or("engine has no stdin")?;
    let stdout = child.stdout.take().ok_or("engine has no stdout")?;
    let pending: Pending = Arc::default();

    let reader_pending = pending.clone();
    let app = app.clone();
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

    Ok(Running {
        child: Mutex::new(child),
        stdin: Mutex::new(stdin),
        pending,
    })
}

impl Engine {
    /// Start the engine (setting it up first if needed) on a thread of its
    /// own; calls made meanwhile wait for it.
    pub fn spawn(app: AppHandle) -> Self {
        let (tx, rx) = watch::channel(State::Starting);
        std::thread::spawn(move || {
            let state = match start(&app) {
                Ok(running) => State::Ready(Arc::new(running)),
                Err(e) => {
                    let _ = app.emit("engine-event", json!({"event": "setup", "error": e}));
                    State::Failed(e)
                }
            };
            let _ = tx.send(state);
        });
        Self {
            state: rx,
            next_id: AtomicU64::new(1),
        }
    }

    async fn running(&self) -> Result<Arc<Running>, Value> {
        let mut rx = self.state.clone();
        loop {
            match &*rx.borrow_and_update() {
                State::Ready(r) => return Ok(r.clone()),
                State::Failed(e) => return Err(json!({ "message": e })),
                State::Starting => {}
            }
            rx.changed()
                .await
                .map_err(|_| json!({"message": "the engine didn't start"}))?;
        }
    }

    pub async fn call(&self, method: &str, params: Value) -> Result<Value, Value> {
        let engine = self.running().await?;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        engine.pending.lock().unwrap().insert(id, tx);
        let line = json!({"id": id, "method": method, "params": params}).to_string();
        {
            let mut stdin = engine.stdin.lock().unwrap();
            if let Err(e) = writeln!(stdin, "{line}").and_then(|_| stdin.flush()) {
                engine.pending.lock().unwrap().remove(&id);
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

impl Drop for Running {
    fn drop(&mut self) {
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
        }
    }
}
