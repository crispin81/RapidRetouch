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
  catchlight: number;
  veins: number;
}

export type EngineEvent =
  | { event: "status"; message: string }
  | { event: "stopped" };

export function call<T>(method: string, params: Record<string, unknown> = {}): Promise<T> {
  return invoke<T>("engine_call", { method, params });
}

export function onEngineEvent(handler: (e: EngineEvent) => void) {
  return listen<EngineEvent>("engine-event", (e) => handler(e.payload));
}

export const jpegSrc = (b64: string) => `data:image/jpeg;base64,${b64}`;
