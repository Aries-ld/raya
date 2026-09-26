// Original Canvas renderer. Gesture choices select shapes; no hand detector or landmarks.
export class ParticleField {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.mode = "none";
    this.frames = 0;
    this.reduced = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    this.points = Array.from({ length: 1200 }, (_, i) => ({
      x: 0,
      y: 0,
      a: i * 2.399963229728653,
      r: Math.sqrt((i + 0.5) / 1200),
      seed: (Math.sin(i * 127.1) * 43758.5453) % 1,
      layer: i % 4,
    }));
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    this.resize();
    this.last = 0;
    this.time = 0;
    this.animate = this.animate.bind(this);
    this.raf = requestAnimationFrame(this.animate);
  }
  resize() {
    this.width = this.canvas.clientWidth;
    this.height = this.canvas.clientHeight;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(this.width * dpr);
    this.canvas.height = Math.round(this.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  setMode(mode) {
    this.mode = mode;
  }
  target(p, i, t) {
    const a = p.a,
      r = p.r,
      spin = t * 0.2;
    switch (this.mode) {
      case "fist": {
        const radius = 0.08 + r * 0.2;
        return [
          Math.cos(a + t * (1 + r)) * radius,
          Math.sin(a + t * (1 + r)) * radius * 0.83,
        ];
      }
      case "one": {
        const y = (r * 2 - 1) * 0.82;
        const x = Math.cos(a + t * 1.5) * (0.06 + 0.11 * Math.sin(r * Math.PI));
        return [x, y];
      }
      case "two": {
        const side = i % 2 ? 1 : -1;
        const ring = 0.26 + 0.1 * p.seed;
        return [
          side * 0.34 + Math.cos(a + t) * ring,
          Math.sin(a + t) * ring * 1.35,
        ];
      }
      case "three": {
        const theta = a + spin;
        const radius = 0.23 + Math.abs(Math.cos(theta * 1.5)) * r * 0.55;
        return [Math.cos(theta) * radius, Math.sin(theta) * radius * 0.93];
      }
      case "four": {
        const q = (p.layer * Math.PI) / 2 + Math.PI / 4;
        const radius = 0.04 + r * 0.2;
        return [
          Math.cos(q) * 0.48 + Math.cos(a + t * 1.1) * radius,
          Math.sin(q) * 0.48 + Math.sin(a + t * 1.1) * radius,
        ];
      }
      case "five": {
        const ray = ((i % 12) * Math.PI) / 6;
        const theta = ray + p.seed * 0.08 + spin;
        const radius = 0.12 + r * 0.77 + Math.sin(t * 1.3 + a) * 0.045;
        return [Math.cos(theta) * radius, Math.sin(theta) * radius * 0.88];
      }
      default: {
        const theta = a + t * 0.09;
        return [
          Math.cos(theta) * (0.12 + r * 0.52),
          Math.sin(theta) * (0.08 + r * 0.36) + Math.sin(t * 0.25 + a) * 0.055,
        ];
      }
    }
  }
  animate(now) {
    const dt = Math.min((now - (this.last || now)) / 1000, 0.05);
    this.last = now;
    if (!document.hidden) {
      this.time += this.reduced ? dt * 0.12 : dt;
      this.frames++;
      const ctx = this.ctx,
        w = this.width,
        h = this.height,
        cx = w * 0.45,
        cy = h * 0.43,
        scale = Math.min(w * 0.4, h * 0.42);
      ctx.fillStyle = "#0b1410";
      ctx.fillRect(0, 0, w, h);
      const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, scale * 1.3);
      glow.addColorStop(0, this.mode === "none" ? "#1b32232b" : "#34572b38");
      glow.addColorStop(1, "#0b141000");
      ctx.fillStyle = glow;
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = "#7ca18412";
      ctx.lineWidth = 0.65;
      for (const s of [0.48, 0.8, 1.13]) {
        ctx.beginPath();
        ctx.ellipse(cx, cy, scale * s, scale * s * 0.48, -0.16, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.fillStyle = "#6b8a7324";
      for (let x = 22; x < w; x += 36)
        for (let y = 19; y < h; y += 36) ctx.fillRect(x, y, 0.7, 0.7);
      const ease = this.reduced ? 1 : 1 - Math.exp(-dt * 9);
      for (let i = 0; i < this.points.length; i++) {
        const p = this.points[i],
          [tx, ty] = this.target(p, i, this.time);
        p.x += (tx - p.x) * ease;
        p.y += (ty - p.y) * ease;
        const light = 66 + (22 * (p.seed + 1)) / 2,
          alpha = this.mode === "none" ? 0.24 + p.r * 0.42 : 0.36 + p.r * 0.55;
        ctx.fillStyle = `hsla(${105 + p.layer * 11},${this.mode === "none" ? 24 : 49}%,${light}%,${alpha})`;
        ctx.beginPath();
        ctx.arc(
          cx + p.x * scale,
          cy + p.y * scale,
          0.65 + p.r * 0.9,
          0,
          Math.PI * 2,
        );
        ctx.fill();
      }
    }
    this.raf = requestAnimationFrame(this.animate);
  }
  close() {
    cancelAnimationFrame(this.raf);
    this.resizeObserver.disconnect();
  }
}
