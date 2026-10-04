import { Check, FolderOpen, X } from "lucide-react";

export type ExportFormat = "tiff" | "jpeg";
/** JPEG size: full resolution, or scaled down to a long edge in pixels. */
export interface ExportSize {
  mode: "full" | "long";
  px: number;
}

export interface ExportRow {
  path: string;
  thumb?: string;
  state: "waiting" | "exporting" | "done" | "failed" | "cancelled";
  fraction: number; // 0..1
  message?: string; // the file written, or what went wrong
}

const name = (path: string) => path.split("/").pop() ?? path;
export const MIN_LONG_EDGE = 100;
export const MAX_LONG_EDGE = 20000;

/**
 * Export: the photos selected in the film strip (or just the one being
 * edited), each with its own progress bar. Every photo is exported with its
 * own settings, one after another.
 */
export default function ExportDialog({
  rows,
  format,
  size,
  folder,
  running,
  cancelling,
  onFormat,
  onSize,
  onChooseFolder,
  onSameFolder,
  onStart,
  onCancel,
  onClose,
}: {
  rows: ExportRow[];
  format: ExportFormat;
  size: ExportSize;
  folder: string | null; // null: next to each original
  running: boolean;
  cancelling: boolean;
  onFormat: (f: ExportFormat) => void;
  onSize: (s: ExportSize) => void;
  onChooseFolder: () => void;
  onSameFolder: () => void;
  onStart: () => void;
  onCancel: () => void;
  onClose: () => void;
}) {
  const started = rows.some((r) => r.state !== "waiting");
  const finished = started && !running;
  const done = rows.filter((r) => r.state === "done").length;
  const failed = rows.filter((r) => r.state === "failed").length;
  const total = rows.reduce((t, r) => t + (r.state === "done" ? 1 : r.fraction), 0) / rows.length;
  const count = `${rows.length} photo${rows.length === 1 ? "" : "s"}`;
  const sizeOk = format !== "jpeg" || size.mode === "full" || (size.px >= MIN_LONG_EDGE && size.px <= MAX_LONG_EDGE);

  return (
    <div className="modal" onClick={running ? undefined : onClose}>
      <div className="modal__box export" onClick={(e) => e.stopPropagation()}>
        <h2>Export {count}</h2>

        {!started && (
          <div className="export__options">
            <div className="export__option">
              <span>Format</span>
              <div className="tabs">
                <button className={format === "tiff" ? "active" : ""} onClick={() => onFormat("tiff")}>
                  TIFF · 16-bit
                </button>
                <button className={format === "jpeg" ? "active" : ""} onClick={() => onFormat("jpeg")}>
                  JPEG · maximum quality
                </button>
              </div>
            </div>
            {format === "jpeg" && (
              <div className="export__option">
                <span>Size</span>
                <div className="export__size">
                  <div className="tabs">
                    <button
                      className={size.mode === "full" ? "active" : ""}
                      onClick={() => onSize({ ...size, mode: "full" })}
                    >
                      Full size
                    </button>
                    <button
                      className={size.mode === "long" ? "active" : ""}
                      onClick={() => onSize({ ...size, mode: "long" })}
                      title="Scale down so the longer side is this many pixels; the other side follows in proportion"
                    >
                      Long edge
                    </button>
                  </div>
                  {size.mode === "long" && (
                    <label className="export__px">
                      <input
                        type="number"
                        min={MIN_LONG_EDGE}
                        max={MAX_LONG_EDGE}
                        step={1}
                        value={size.px}
                        onChange={(e) => onSize({ ...size, px: Math.round(Number(e.target.value)) || 0 })}
                      />
                      px
                    </label>
                  )}
                </div>
              </div>
            )}
            <div className="export__option">
              <span>Save to</span>
              <div className="tabs">
                <button className={folder === null ? "active" : ""} onClick={onSameFolder}>
                  Next to the originals
                </button>
                <button className={folder !== null ? "active" : ""} onClick={onChooseFolder}>
                  <FolderOpen size={14} /> {folder ? "Folder" : "Choose folder…"}
                </button>
              </div>
            </div>
            {folder && <div className="export__folder" title={folder}>{folder}</div>}
            <p className="export__note">
              Saved as <code>name_retouched.{format === "tiff" ? "tif" : "jpg"}</code>; an earlier export with
              the same name is replaced.
              {format === "jpeg" && size.mode === "long" && " Photos already smaller than the long edge stay at full size."} Each photo is exported with its own settings, crop and brush work.
            </p>
          </div>
        )}

        <ul className="export__list">
          {rows.map((r) => (
            <li key={r.path} className={`export__row export__row--${r.state}`}>
              {r.thumb ? (
                <img src={`data:image/jpeg;base64,${r.thumb}`} alt="" />
              ) : (
                <span className="export__thumb" />
              )}
              <div className="export__body">
                <div className="export__name">
                  <span>{name(r.path)}</span>
                  <span className="export__state">
                    {r.state === "done" ? (
                      <Check size={14} />
                    ) : r.state === "failed" ? (
                      <X size={14} />
                    ) : r.state === "exporting" ? (
                      `${Math.round(r.fraction * 100)}%`
                    ) : r.state === "cancelled" ? (
                      "Cancelled"
                    ) : started ? (
                      "Waiting"
                    ) : (
                      ""
                    )}
                  </span>
                </div>
                <div className="export__bar">
                  <div style={{ width: `${(r.state === "done" ? 1 : r.fraction) * 100}%` }} />
                </div>
                {r.message && <div className="export__message" title={r.message}>{r.message}</div>}
              </div>
            </li>
          ))}
        </ul>

        {started && rows.length > 1 && (
          <div className="export__total">
            <span>{finished ? `${done} exported${failed ? `, ${failed} failed` : ""}` : `${Math.round(total * 100)}% of all`}</span>
            <div className="export__bar">
              <div style={{ width: `${total * 100}%` }} />
            </div>
          </div>
        )}

        <div className="modal__buttons">
          {!started && (
            <>
              <button onClick={onClose}>Cancel</button>
              <button className="primary" onClick={onStart} disabled={!sizeOk}>
                Export {count}
              </button>
            </>
          )}
          {running && (
            <button onClick={onCancel} disabled={cancelling} title="Stops after the photo being exported">
              {cancelling ? "Stopping after this photo…" : "Cancel"}
            </button>
          )}
          {finished && (
            <button className="primary" onClick={onClose}>
              Close
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
