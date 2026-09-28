import { useEffect, useRef } from "react";
import { Copy, ClipboardPaste, X } from "lucide-react";

export interface StripItem {
  path: string;
  thumb?: string; // base64 JPEG, loaded lazily
  edited: boolean; // settings differ from the defaults
}

interface Props {
  items: StripItem[];
  active: string | null;
  selected: Set<string>;
  canPaste: boolean;
  onActivate: (path: string) => void;
  onSelect: (selected: Set<string>) => void;
  onRemove: (path: string) => void;
  onCopy: () => void;
  onPaste: () => void;
}

const fileName = (p: string) => p.split("/").pop() ?? p;

/**
 * Photos open in this session. Click to edit one; Ctrl+click adds to or
 * removes from the selection, Shift+click selects a range (like a file
 * manager). The selection is what "Paste to selected" applies to.
 */
export default function FilmStrip({
  items,
  active,
  selected,
  canPaste,
  onActivate,
  onSelect,
  onRemove,
  onCopy,
  onPaste,
}: Props) {
  const anchor = useRef<string | null>(null);
  const activeRef = useRef<HTMLDivElement>(null);

  // Keep the photo being edited in view (e.g. when stepping with the arrow keys).
  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" });
  }, [active]);

  const click = (e: React.MouseEvent, path: string) => {
    if (e.ctrlKey || e.metaKey) {
      const next = new Set(selected);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      anchor.current = path;
      onSelect(next);
    } else if (e.shiftKey && anchor.current) {
      const a = items.findIndex((i) => i.path === anchor.current);
      const b = items.findIndex((i) => i.path === path);
      const [lo, hi] = a < b ? [a, b] : [b, a];
      onSelect(new Set(items.slice(lo, hi + 1).map((i) => i.path)));
    } else {
      anchor.current = path;
      onSelect(new Set([path]));
      if (path !== active) onActivate(path);
    }
  };

  return (
    <div className="filmstrip">
      <div className="filmstrip__bar">
        <span className="filmstrip__count">
          {items.length} photo{items.length === 1 ? "" : "s"}
          {selected.size > 1 ? ` · ${selected.size} selected` : ""}
        </span>
        <button onClick={onCopy} disabled={!active} title="Copy this photo's slider settings (Backdrop and Eyes)">
          <Copy size={14} /> Copy settings
        </button>
        <button
          onClick={onPaste}
          disabled={!canPaste || selected.size === 0}
          title="Apply the copied slider settings to every selected photo. Brush strokes and mask edits aren't pasted: they belong to each photo."
        >
          <ClipboardPaste size={14} /> Paste to selected
          {selected.size > 0 ? ` (${selected.size})` : ""}
        </button>
        <span className="filmstrip__hint">
          Ctrl+click to add to the selection, Shift+click for a range, ← → to step through
        </span>
      </div>
      <div className="filmstrip__items">
        {items.map((item) => (
          <div
            key={item.path}
            ref={item.path === active ? activeRef : undefined}
            className={[
              "filmstrip__item",
              item.path === active ? "filmstrip__item--active" : "",
              selected.has(item.path) ? "filmstrip__item--selected" : "",
            ].join(" ")}
            onClick={(e) => click(e, item.path)}
            title={item.path}
          >
            {item.thumb ? (
              <img src={`data:image/jpeg;base64,${item.thumb}`} alt="" draggable={false} />
            ) : (
              <div className="filmstrip__placeholder" />
            )}
            <span className="filmstrip__name">{fileName(item.path)}</span>
            {item.edited && <span className="filmstrip__edited" title="Settings changed" />}
            <button
              className="filmstrip__remove"
              title="Take out of the strip (the file isn't touched)"
              onClick={(e) => {
                e.stopPropagation();
                onRemove(item.path);
              }}
            >
              <X size={12} />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
