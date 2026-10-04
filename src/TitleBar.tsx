import { getCurrentWindow } from "@tauri-apps/api/window";
import { Minus, Square, X } from "lucide-react";
import { useAppVersion } from "./version";

const appWindow = getCurrentWindow();

/** Custom title bar, matching RapidCulling's (the window has no system frame). */
export default function TitleBar() {
  const version = useAppVersion();
  return (
    <div className="titlebar" data-tauri-drag-region>
      <span className="titlebar__title" data-tauri-drag-region>
        <span className="titlebar__title-rapid">Rapid</span>
        <span className="titlebar__title-retouch">Retouch</span>{" "}
        {version && <span className="titlebar__version">v{version}</span>}
      </span>
      <div className="titlebar__controls">
        <button type="button" onClick={() => appWindow.minimize()} title="Minimize">
          <Minus size={13} />
        </button>
        <button type="button" onClick={() => appWindow.toggleMaximize()} title="Maximize">
          <Square size={11} />
        </button>
        <button type="button" className="titlebar__close" onClick={() => appWindow.close()} title="Close">
          <X size={13} />
        </button>
      </div>
    </div>
  );
}
