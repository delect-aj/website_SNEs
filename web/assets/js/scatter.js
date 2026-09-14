/**
 * A WebGL scatter for the 14k-point atlas.
 *
 * Written here rather than pulled in: the alternative needs three vendored
 * packages with two different module formats in a site that has no build step,
 * and the drawing this page actually does is one `POINTS` draw call plus a
 * highlighted point on top.
 *
 * Colours arrive pre-resolved as RGB per point, so switching the colour-by
 * field is a buffer update, not a shader change. The canvas is dark for
 * contrast at 14k points; the surrounding page is not.
 */

const VERTEX_SHADER = `
attribute vec2 aPosition;
attribute vec3 aColor;
uniform vec2 uView;
uniform float uZoom;
uniform float uPointSize;
uniform vec2 uAspect;
uniform float uSelected;
varying vec3 vColor;
varying float vSelected;

void main() {
  vec2 p = (aPosition - uView) * uZoom * uAspect * 2.0;
  gl_Position = vec4(p, 0.0, 1.0);
  gl_PointSize = uPointSize;
  vColor = aColor;
  vSelected = uSelected;
}`;

const FRAGMENT_SHADER = `
precision mediump float;
varying vec3 vColor;
varying float vSelected;
uniform float uOpacity;

void main() {
  if (vSelected < 0.5) {
    gl_FragColor = vec4(vColor, uOpacity);
    return;
  }
  // The highlighted point keeps its colour but grows a dark ring, so it stays
  // findable without colour carrying the information.
  vec2 offset = gl_PointCoord - vec2(0.5);
  float radius = length(offset) * 2.0;
  if (radius > 0.95) discard;
  if (radius > 0.55) {
    gl_FragColor = vec4(0.09, 0.09, 0.11, 1.0);
  } else {
    gl_FragColor = vec4(vColor, 1.0);
  }
}`;

export class Scatter {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {{onHover?: Function, onSelect?: Function}} handlers
   */
  constructor(canvas, handlers = {}) {
    this.canvas = canvas;
    this.onHover = handlers.onHover || (() => {});
    this.onSelect = handlers.onSelect || (() => {});

    this.gl = canvas.getContext('webgl', { antialias: true, alpha: false });
    if (!this.gl) throw new Error('WebGL is not available in this browser');

    this.view = { x: 0.5, y: 0.5 };
    this.zoom = 1;
    this.minZoom = 1;
    this.maxZoom = 60;
    this.count = 0;
    this.selected = -1;
    this.hovered = -1;
    this.positions = null;
    this.bounds = { minX: 0, maxX: 1, minY: 0, maxY: 1 };

    this._initGL();
    this._bindEvents();
    this.resize();
    // The well grows with the card beside it, which no window resize reports:
    // without this the backing store keeps its first size, the points stretch
    // and clicks land on the wrong OTU.
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  _initGL() {
    const gl = this.gl;
    const program = gl.createProgram();
    for (const [type, source] of [[gl.VERTEX_SHADER, VERTEX_SHADER],
                                   [gl.FRAGMENT_SHADER, FRAGMENT_SHADER]]) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(shader));
      }
      gl.attachShader(program, shader);
    }
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program));
    }
    gl.useProgram(program);
    this.program = program;

    this.uniforms = {};
    for (const name of ['uView', 'uZoom', 'uPointSize', 'uAspect', 'uOpacity',
                        'uSelected']) {
      this.uniforms[name] = gl.getUniformLocation(program, name);
    }
    this.aPosition = gl.getAttribLocation(program, 'aPosition');
    this.aColor = gl.getAttribLocation(program, 'aColor');

    this.positionBuffer = gl.createBuffer();
    this.colorBuffer = gl.createBuffer();
    this.singleBuffer = gl.createBuffer();

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(0.071, 0.086, 0.106, 1.0);   // --well
  }

  /** Upload point positions, normalized once into a square [0,1] box. */
  setPoints(positions) {
    const gl = this.gl;
    const n = positions.length / 2;

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < n; i += 1) {
      const x = positions[i * 2];
      const y = positions[i * 2 + 1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    // One scale for both axes, so the layout is not stretched, and the larger
    // extent decides the box so nothing is clipped.
    const extent = Math.max(maxX - minX, maxY - minY) || 1;
    const pad = extent * 0.04;
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const size = extent + pad * 2;

    const normalized = new Float32Array(n * 2);
    for (let i = 0; i < n; i += 1) {
      normalized[i * 2] = (positions[i * 2] - cx) / size + 0.5;
      normalized[i * 2 + 1] = (positions[i * 2 + 1] - cy) / size + 0.5;
    }

    this.positions = normalized;
    this.bounds = { size, cx, cy, extent };
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, normalized, gl.STATIC_DRAW);
    this.count = n;
    this.render();
  }

  /** Upload a per-point RGB colour, values in 0..1. */
  setColors(colors) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, colors, gl.DYNAMIC_DRAW);
    this.render();
  }

  /** Move the ring to one point, or clear it with -1. */
  setSelected(index) {
    this.selected = index;
    this.render();
  }

  /** Recentre and rescale so one point fills a comfortable part of the view. */
  focus(index, zoom = 6) {
    if (!this.positions || index < 0 || index >= this.count) return;
    this.view.x = this.positions[index * 2];
    this.view.y = this.positions[index * 2 + 1];
    this.zoom = Math.min(Math.max(zoom, this.minZoom), this.maxZoom);
    this.selected = index;
    this.render();
  }

  reset() {
    this.view = { x: 0.5, y: 0.5 };
    this.zoom = 1;
    this.render();
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = this.canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width * dpr));
    const height = Math.max(1, Math.round(rect.height * dpr));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    this.dpr = dpr;
    this.cssWidth = rect.width;
    this.cssHeight = rect.height;
    this.render();
  }

  /** Normalized-space coordinates of a pointer event. */
  _toWorld(event) {
    const rect = this.canvas.getBoundingClientRect();
    const px = (event.clientX - rect.left) / rect.width;
    const py = (event.clientY - rect.top) / rect.height;
    const aspect = this._aspect();
    return {
      x: (px - 0.5) / (2 * this.zoom * aspect.x) + this.view.x,
      y: (0.5 - py) / (2 * this.zoom * aspect.y) + this.view.y,
    };
  }

  _aspect() {
    const smaller = Math.min(this.cssWidth, this.cssHeight);
    return { x: smaller / this.cssWidth, y: smaller / this.cssHeight };
  }

  /** Index of the point nearest to a pointer event, or -1 beyond `radius` px. */
  _nearest(event, radius = 10) {
    if (!this.positions) return -1;
    const rect = this.canvas.getBoundingClientRect();
    const target = this._toWorld(event);
    const aspect = this._aspect();
    const scale = 2 * this.zoom;
    let best = -1;
    let bestDistance = Infinity;
    for (let i = 0; i < this.count; i += 1) {
      const dx = (this.positions[i * 2] - target.x) * scale * aspect.x
        * rect.width;
      const dy = (this.positions[i * 2 + 1] - target.y) * scale * aspect.y
        * rect.height;
      const distance = dx * dx + dy * dy;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = i;
      }
    }
    return bestDistance <= radius * radius ? best : -1;
  }

  _bindEvents() {
    const canvas = this.canvas;
    let dragging = false;
    let moved = false;
    let last = { x: 0, y: 0 };
    let start = { x: 0, y: 0 };

    canvas.addEventListener('pointerdown', (event) => {
      dragging = true;
      moved = false;
      last = { x: event.clientX, y: event.clientY };
      start = last;
      canvas.setPointerCapture(event.pointerId);
    });

    canvas.addEventListener('pointermove', (event) => {
      if (dragging) {
        // A click is rarely perfectly still. Until the pointer has travelled a
        // few pixels this is still a click, or every trackpad tap would pan by
        // a pixel and select nothing.
        if (!moved && Math.hypot(event.clientX - start.x, event.clientY - start.y) < 4) {
          return;
        }
        const aspect = this._aspect();
        const rect = canvas.getBoundingClientRect();
        this.view.x -= (event.clientX - last.x) / rect.width
          / (2 * this.zoom * aspect.x);
        this.view.y += (event.clientY - last.y) / rect.height
          / (2 * this.zoom * aspect.y);
        last = { x: event.clientX, y: event.clientY };
        moved = true;
        this.render();
        return;
      }
      const index = this._nearest(event);
      if (index !== this.hovered) {
        this.hovered = index;
        canvas.style.cursor = index >= 0 ? 'pointer' : 'grab';
      }
      this.onHover(index, event);
    });

    const release = (event) => {
      dragging = false;
      if (canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
    };
    canvas.addEventListener('pointerup', release);
    canvas.addEventListener('pointercancel', release);

    // Selection rides on the browser's own click, which every mouse, trackpad
    // and touch screen produces, rather than on pointerup, which some of them
    // replace with a cancel. A click that ended a drag is ignored.
    canvas.addEventListener('click', (event) => {
      if (Math.hypot(event.clientX - start.x, event.clientY - start.y) >= 6) return;
      const index = this._nearest(event);
      if (index >= 0) this.onSelect(index);
    });

    canvas.addEventListener('pointerleave', () => {
      this.hovered = -1;
      this.onHover(-1, null);
    });

    canvas.addEventListener('wheel', (event) => {
      event.preventDefault();
      const before = this._toWorld(event);
      const factor = Math.exp(-event.deltaY * 0.0015);
      this.zoom = Math.min(Math.max(this.zoom * factor, this.minZoom),
                           this.maxZoom);
      const after = this._toWorld(event);
      // Keep the point under the cursor where it was.
      this.view.x += before.x - after.x;
      this.view.y += before.y - after.y;
      this.render();
    }, { passive: false });

    canvas.addEventListener('keydown', (event) => {
      const step = 0.04 / this.zoom;
      const moves = {
        ArrowLeft: [-step, 0], ArrowRight: [step, 0],
        ArrowUp: [0, step], ArrowDown: [0, -step],
      };
      if (moves[event.key]) {
        this.view.x += moves[event.key][0];
        this.view.y += moves[event.key][1];
        this.render();
        event.preventDefault();
      } else if (event.key === '+' || event.key === '=') {
        this.zoom = Math.min(this.zoom * 1.2, this.maxZoom);
        this.render();
        event.preventDefault();
      } else if (event.key === '-') {
        this.zoom = Math.max(this.zoom / 1.2, this.minZoom);
        this.render();
        event.preventDefault();
      } else if (event.key === '0') {
        this.reset();
        event.preventDefault();
      }
    });
  }

  render() {
    const gl = this.gl;
    if (!gl || !this.count) {
      gl.clear(gl.COLOR_BUFFER_BIT);
      return;
    }
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const aspect = this._aspect();
    gl.uniform2f(this.uniforms.uView, this.view.x, this.view.y);
    gl.uniform1f(this.uniforms.uZoom, this.zoom);
    gl.uniform2f(this.uniforms.uAspect, aspect.x, aspect.y);
    gl.uniform1f(this.uniforms.uPointSize, Math.max(2, 2.4 * this.dpr));

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(this.aPosition);
    gl.vertexAttribPointer(this.aPosition, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuffer);
    gl.enableVertexAttribArray(this.aColor);
    gl.vertexAttribPointer(this.aColor, 3, gl.FLOAT, false, 0, 0);

    gl.uniform1f(this.uniforms.uSelected, 0.0);
    gl.uniform1f(this.uniforms.uOpacity, 0.85);
    gl.drawArrays(gl.POINTS, 0, this.count);

    if (this.selected >= 0 && this.selected < this.count) {
      const point = new Float32Array([
        this.positions[this.selected * 2],
        this.positions[this.selected * 2 + 1],
      ]);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.singleBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, point, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(this.aPosition);
      gl.vertexAttribPointer(this.aPosition, 2, gl.FLOAT, false, 0, 0);
      // The colour array is still enabled from the cloud above, and an
      // enabled array wins over a constant one: without this the selected
      // core would be drawn in point 0's colour instead of white.
      gl.disableVertexAttribArray(this.aColor);
      gl.vertexAttrib3f(this.aColor, 1.0, 1.0, 1.0);
      gl.uniform1f(this.uniforms.uSelected, 1.0);
      gl.uniform1f(this.uniforms.uPointSize, Math.max(10, 12 * this.dpr));
      gl.drawArrays(gl.POINTS, 0, 1);
    }
  }
}
