import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

export interface EngineError {
  message: string;
  kind?: "licence";
  model?: string;
  licence?: string;
  licence_url?: string;
}

export interface OpenResult {
  width: number;
  height: number;
  bit_depth: number;
  preview: string; // base64 JPEG
  removals: number; // restored when switching back to a photo
  mask_edits: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  task: string;
  licence: string;
  licence_url: string;
  commercial_use: boolean;
  default: boolean;
  accepted: boolean;
  badges: string[];
  note?: string; // one-line description of the model
}

export interface BackdropParams {
  strength: number;
  smoothness: number;
  evenness: number;
  grain: number;
  edge_protect: number;
  exposure: number;
}

export interface EyesParams {
  dark_circles: number;
  eye_bags: number;
  wrinkles: number;
  whites: number;
  iris: number;
  iris_saturation: number; // -1 muted .. 0 unchanged .. +1 richer
  iris_hue: number; // turns the iris colour around the colour wheel, -1 .. +1
  catchlight: number;
  veins: number;
  lashes: number;
}

export interface SkinRegion {
  acne: number; // spots and acne healed
  blemishes: number; // large pores and small marks evened out
  smooth: number;
  even: number;
  shine: number; // -1 matte .. 0 natural .. +1 gloss
  texture: number; // pore softening
  pores: number; // pores taken out, pits and bumps alike
  // Wrinkles group (Face tab only)
  forehead_lines: number;
  frown_lines: number;
  smile_lines: number;
  chin_lines: number;
  neck_lines: number; // Neck tab only
}

export interface MouthParams {
  lip_saturation: number; // -1 muted .. 0 unchanged .. +1 rich
  lip_hue: number; // -1 cooler, pinker .. 0 unchanged .. +1 warmer, more coral
  lip_smooth: number;
  teeth_whiten: number;
}

export interface SkinParams {
  face: SkinRegion;
  neck: SkinRegion;
  body: SkinRegion;
}

export type EngineEvent =
  | { event: "status"; message: string }
  | { event: "progress"; fraction: number } // how far the export under way is, 0..1
  | { event: "stopped" };

export function call<T>(method: string, params: Record<string, unknown> = {}): Promise<T> {
  return invoke<T>("engine_call", { method, params });
}

export function onEngineEvent(handler: (e: EngineEvent) => void) {
  return listen<EngineEvent>("engine-event", (e) => handler(e.payload));
}

export const jpegSrc = (b64: string) => `data:image/jpeg;base64,${b64}`;
