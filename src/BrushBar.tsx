import { Check, Minus, Plus, Trash2, Undo2 } from "lucide-react";

/** Floating toolbar over the photo while a correction brush is in use (the
 * subject mask or a Refine area brush): everything the brush needs sits
 * where the user is looking, whether or not its panel is open. */
export default function BrushBar({
  what,
  mode,
  addLabel,
  removeLabel,
  onMode,
  radius,
  min,
  max,
  step,
  onRadius,
  edits,
  pending,
  onUndo,
  onClear,
  onDone,
}: {
  what: string;
  mode: "add" | "remove";
  addLabel: string;
  removeLabel: string;
  onMode: (m: "add" | "remove") => void;
  radius: number;
  min: number;
  max: number;
  step: number;
  onRadius: (r: number) => void;
  edits: number;
  pending: boolean;
  onUndo: () => void;
  onClear: () => void;
  onDone: () => void;
}) {
  return (
    <div className="brushbar">
      <span className="brushbar__what">{what}</span>
      <div className="tabs">
        <button className={mode === "add" ? "active" : ""} onClick={() => onMode("add")} title="X swaps">
          <Plus size={14} /> {addLabel}
        </button>
        <button className={mode === "remove" ? "active" : ""} onClick={() => onMode("remove")} title="X swaps">
          <Minus size={14} /> {removeLabel}
        </button>
      </div>
      <label className="brushbar__size" title="Brush size ([ and ] keys)">
        Size
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={radius}
          onChange={(e) => onRadius(Number(e.target.value))}
        />
      </label>
      <button disabled={edits === 0 || pending} onClick={onUndo} title="Undo the last stroke (Ctrl+Z)">
        <Undo2 size={14} />
      </button>
      <button disabled={edits === 0 || pending} onClick={onClear} title="Clear all your corrections here">
        <Trash2 size={14} />
      </button>
      <button className="brushbar__done" onClick={onDone} title="Back to the photo">
        <Check size={14} /> Done
      </button>
    </div>
  );
}
