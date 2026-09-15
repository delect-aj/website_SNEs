/**
 * A click on the atlas selects the point drawn under it.
 *
 * The vertex shader draws a point at clip position (pos - view) * zoom *
 * aspect * 2, and the click, hover and drag code has to invert exactly that.
 * It once kept the shader's factor of 2 after leaving clip space, so a click
 * at distance d from the centre picked the point drawn at d / 2: roughly right
 * in the middle of the map, a wrong OTU or nothing towards its edges.
 *
 * Run: cd tests/js && node scatter.test.mjs
 */

import { Scatter } from '../../web/assets/js/scatter.js';

const LEFT = 30;
const TOP = 200;

/** A scatter with its state set directly: no WebGL, no DOM. */
function scatter(width, height, zoom, view) {
  const s = Object.create(Scatter.prototype);
  s.canvas = { getBoundingClientRect: () => ({ left: LEFT, top: TOP, width, height }) };
  s.cssWidth = width;
  s.cssHeight = height;
  s.zoom = zoom;
  s.view = view;
  const side = 40;
  s.count = side * side;
  s.positions = new Float32Array(s.count * 2);
  for (let i = 0; i < s.count; i += 1) {
    s.positions[i * 2] = (i % side) / (side - 1);
    s.positions[i * 2 + 1] = Math.floor(i / side) / (side - 1);
  }
  return s;
}

/** Where the vertex shader draws point i, as a pointer event on the page. */
function drawnAt(s, i) {
  const aspect = s._aspect();
  const clipX = (s.positions[i * 2] - s.view.x) * s.zoom * aspect.x * 2;
  const clipY = (s.positions[i * 2 + 1] - s.view.y) * s.zoom * aspect.y * 2;
  return {
    clientX: LEFT + (clipX + 1) / 2 * s.cssWidth,
    clientY: TOP + (1 - clipY) / 2 * s.cssHeight,
  };
}

const cases = [
  ['wide, whole map', scatter(1000, 600, 1, { x: 0.5, y: 0.5 })],
  ['tall, zoomed', scatter(600, 900, 3.7, { x: 0.31, y: 0.62 })],
  ['square, deep zoom', scatter(400, 400, 12, { x: 0.8, y: 0.2 })],
];

let failures = 0;
for (const [label, s] of cases) {
  let clicked = 0;
  let wrong = 0;
  for (let i = 0; i < s.count; i += 1) {
    const event = drawnAt(s, i);
    const inside = event.clientX >= LEFT && event.clientX <= LEFT + s.cssWidth
      && event.clientY >= TOP && event.clientY <= TOP + s.cssHeight;
    if (!inside) continue;
    clicked += 1;
    if (s._nearest(event) !== i) wrong += 1;
  }
  const ok = clicked > 0 && wrong === 0;
  if (!ok) failures += 1;
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${label.padEnd(18)} ${clicked} visible points clicked, `
    + `${wrong} picked another point or none`);
}
console.log(`checks: ${cases.length}, failures: ${failures}`);
process.exit(failures ? 1 : 0);
