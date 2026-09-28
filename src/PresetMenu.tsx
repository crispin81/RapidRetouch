import { useEffect, useRef, useState } from "react";
import { Bookmark, Trash2 } from "lucide-react";

export interface Preset {
  name: string;
  settings: Record<string, unknown>;
}

interface Props {
  presets: Preset[];
  disabled?: boolean;
  /** How many photos a preset would be applied to (the film-strip selection). */
  targets: number;
  onApply: (preset: Preset) => void;
  onSave: (name: string) => void;
  onDelete: (name: string) => void;
}

/**
 * Presets dropdown: click a preset to apply it, save the current settings
 * under a name (the same name replaces it), delete with a second click.
 */
export default function PresetMenu({ presets, disabled, targets, onApply, onSave, onDelete }: Props) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  // Close when clicking anywhere else.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("pointerdown", onDown);
    return () => window.removeEventListener("pointerdown", onDown);
  }, [open]);
  useEffect(() => {
    if (!open) setConfirmDelete(null);
  }, [open]);

  const trimmed = name.trim();
  const replacing = presets.some((p) => p.name.toLowerCase() === trimmed.toLowerCase());
  const save = () => {
    if (!trimmed) return;
    onSave(trimmed);
    setName("");
  };

  return (
    <div className="presets" ref={ref}>
      <button
        className={`sidebar__fold presets__button${open ? " active" : ""}`}
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
        title="Save and apply presets"
      >
        <Bookmark size={16} /> Presets
      </button>
      {open && (
        <div className="presets__menu">
          {presets.length === 0 ? (
            <p className="presets__empty">No presets yet. Save your current settings below.</p>
          ) : (
            <>
              <p className="presets__note">
                Apply to {targets} photo{targets === 1 ? "" : "s"}
              </p>
              <ul className="presets__list">
                {presets.map((p) => (
                  <li key={p.name}>
                    <button
                      className="presets__apply"
                      onClick={() => {
                        onApply(p);
                        setOpen(false);
                      }}
                    >
                      {p.name}
                    </button>
                    <button
                      className={`presets__delete${confirmDelete === p.name ? " presets__delete--confirm" : ""}`}
                      onClick={() => {
                        if (confirmDelete === p.name) {
                          onDelete(p.name);
                          setConfirmDelete(null);
                        } else setConfirmDelete(p.name);
                      }}
                      title={confirmDelete === p.name ? "Click again to delete" : `Delete ${p.name}`}
                    >
                      {confirmDelete === p.name ? "Delete?" : <Trash2 size={13} />}
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}
          <form
            className="presets__save"
            onSubmit={(e) => {
              e.preventDefault();
              save();
            }}
          >
            <input
              type="text"
              placeholder="New preset name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => e.key === "Escape" && setOpen(false)}
            />
            <button type="submit" disabled={!trimmed} title="Save the current settings as a preset">
              {replacing ? "Replace" : "Save"}
            </button>
          </form>
        </div>
      )}
    </div>
  );
}
