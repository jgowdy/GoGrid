use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // System tray will be set up here
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
