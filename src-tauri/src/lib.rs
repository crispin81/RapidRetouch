pub mod engine;

use engine::Engine;
use serde_json::Value;
use tauri::{Manager, State};

/// Forward a call to the Python engine. Errors come back as the engine's error
/// object (message, plus `kind: "licence"` details when a model needs acceptance).
#[tauri::command]
async fn engine_call(engine: State<'_, Engine>, method: String, params: Value) -> Result<Value, Value> {
    engine.call(&method, params).await
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let engine = Engine::spawn(app.handle().clone())?;
            app.manage(engine);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![engine_call])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
