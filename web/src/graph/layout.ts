/** Deterministic layout: columns are decision order, rows are conversation turns.
 *
 * Positions are computed from (col, turn) and never recomputed, so nodes do not jump around as
 * the cascade streams in — which matters when this is being screen-recorded. A left gutter holds
 * the turn labels so they never collide with the caller node.
 */

export const COL_W = 214;
export const ROW_H = 166;
export const GUTTER = 168;
export const ORIGIN_X = GUTTER + 14;
export const ORIGIN_Y = 64;
export const MAX_COL = 12;
export const NODE_W = 190;
export const NODE_H = 116;

export function nodePosition(col: number, turn: number): { x: number; y: number } {
  return { x: ORIGIN_X + col * COL_W, y: ORIGIN_Y + (turn - 1) * ROW_H };
}

export function laneRect(turn: number): { x: number; y: number; width: number; height: number } {
  return {
    x: 14,
    y: ORIGIN_Y + (turn - 1) * ROW_H - 26,
    width: ORIGIN_X + MAX_COL * COL_W + NODE_W + 14 - 14,
    height: ROW_H - 28,
  };
}

export function nodeCenter(col: number, turn: number): { x: number; y: number } {
  const p = nodePosition(col, turn);
  return { x: p.x + NODE_W / 2, y: p.y + NODE_H / 2 };
}
