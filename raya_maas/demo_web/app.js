import {
  GESTURES,
  gestureById,
  decisionRequest,
  validateDecision,
  appliedDecision,
  numberHeader,
  DecisionLoop,
} from "./core.js";
import { ParticleField } from "./particles.js";
const $ = (id) => document.getElementById(id);
const video = $("camera"),
  field = new ParticleField($("particles"));
const captureCanvas = document.createElement("canvas"),
  captureCtx = captureCanvas.getContext("2d");
const records = [];
let config = null,
  stream = null,
  cameraEpoch = 0,
  opening = false,
  ready = false,
  sampleCount = 0,
  skipped = 0;
const emptyFeed = $("empty-feed");
let healthTimer = null;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function status(text, error = false) {
  $("status").textContent = text;
  $("status").dataset.error = String(error);
}
function setMode(id, subtitle = "") {
  const gesture = gestureById(id) || gestureById("none");
  field.setMode(gesture.id);
  $("world").dataset.mode = gesture.id;
  $("effect-index").textContent = gesture.number;
  $("effect-title").textContent = gesture.effect;
  $("effect-subtitle").textContent =
    subtitle ||
    (id === "none"
      ? "等待清晰的单手手势"
      : `${gesture.label} · ${gesture.effect}`);
  document.querySelectorAll(".gesture").forEach((node) => {
    node.dataset.active = String(node.dataset.gesture === gesture.id);
  });
}
for (const g of GESTURES.slice(0, 6)) {
  const item = el("div", "gesture");
  item.dataset.gesture = g.id;
  item.dataset.active = "false";
  const words = el("div");
  words.append(el("strong", "", g.label), el("small", "", g.effect));
  item.append(el("span", "number", g.number), words);
  $("gesture-legend").append(item);
}
function refreshControls() {
  $("camera-on").disabled = opening || Boolean(stream);
  $("recognize").disabled = !stream || !ready || !config;
  $("recognize").textContent = loop.active ? "暂停识别" : "开始识别";
  $("camera-off").disabled = !stream && !opening;
  $("live-dot").classList.toggle("active", loop.active);
  $("live-label").textContent = loop.active
    ? "LIVE / 1 FRAME PER SECOND"
    : stream
      ? "CAMERA PREVIEW"
      : "READY WHEN YOU ARE";
}
function capture() {
  const capturedAt = performance.now(),
    time = new Date().toLocaleTimeString("zh-CN", { hour12: false });
  const side = Math.min(video.videoWidth, video.videoHeight);
  if (!side) throw new Error("摄像头画面尚未就绪。");
  captureCtx.drawImage(
    video,
    (video.videoWidth - side) / 2,
    (video.videoHeight - side) / 2,
    side,
    side,
    0,
    0,
    config.capture_size,
    config.capture_size,
  );
  return {
    image: captureCanvas.toDataURL("image/jpeg", 0.9),
    capturedAt,
    time,
  };
}
async function request(snapshot, signal) {
  const response = await fetch("/api/systemone", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Raya-Demo": "gesture-lab",
    },
    body: JSON.stringify(decisionRequest(config, snapshot.image)),
    signal,
  });
  let output;
  try {
    output = await response.json();
  } catch {
    throw new Error(`服务返回了非 JSON 响应（${response.status}）。`);
  }
  if (!response.ok)
    throw new Error(output.error?.message || `请求失败（${response.status}）`);
  const answer = validateDecision(output);
  return {
    output,
    answer,
    requestId: response.headers.get("x-request-id"),
    forwardMs: numberHeader(response.headers, "x-raya-forward-ms"),
    processingMs: numberHeader(response.headers, "x-raya-processing-ms"),
    queueMs: numberHeader(response.headers, "x-raya-queue-ms"),
  };
}
const formatMs = (value) => (value === null ? "—" : `${Math.round(value)} ms`);
function addRecord(record) {
  emptyFeed.remove();
  $("feed")
    .querySelectorAll('details[data-auto-open="true"]')
    .forEach((node) => {
      node.open = false;
      node.dataset.autoOpen = "false";
    });
  const card = el("article", "record");
  card.dataset.applied = String(record.applied);
  card.dataset.choice = record.answer.choice;
  const top = el("div", "record-top"),
    intro = el("div");
  intro.append(
    el(
      "div",
      "record-id",
      `#${String(record.index).padStart(3, "0")} · ${record.snapshot.time}`,
    ),
    el("h3", "record-title", gestureById(record.answer.choice).label),
    el("div", "record-action", record.reason),
  );
  const image = el("img", "record-image");
  image.src = record.snapshot.image;
  image.alt = `第${record.index}次送入模型的未镜像图片`;
  top.append(intro, image);
  card.append(top);
  const timings = el("div", "timing-row");
  for (const [name, value] of [
    ["往返 RT", record.wallMs],
    ["模型前向", record.forwardMs],
    ["服务端处理", record.processingMs],
  ]) {
    const cell = el("div");
    cell.append(el("span", "", name), el("strong", "", formatMs(value)));
    timings.append(cell);
  }
  card.append(timings);
  for (const g of GESTURES) {
    const p = record.answer.probabilities[g.id],
      row = el("div", "probability");
    row.dataset.winner = String(g.id === record.answer.choice);
    const track = el("span", "track"),
      fill = el("i");
    fill.style.width = `${p * 100}%`;
    track.append(fill);
    row.append(
      el(
        "span",
        "",
        g.id === "none"
          ? "无手势"
          : g.number === "0"
            ? "拳头"
            : `${g.number} 根手指`,
      ),
      track,
      el("span", "value", `${(p * 100).toFixed(1)}%`),
    );
    card.append(row);
  }
  const details = el("details");
  details.open = true;
  details.dataset.autoOpen = "true";
  details.append(
    el("summary", "", "响应 JSON · 原始返回"),
    el("pre", "", JSON.stringify(record.output, null, 2)),
  );
  card.append(details);
  details.querySelector("summary").addEventListener("click", () => {
    details.dataset.autoOpen = "false";
  });
  $("feed").prepend(card);
  while ($("feed").children.length > 12) $("feed").lastElementChild.remove();
  records.unshift(record);
  records.splice(12);
  $("feed-count").textContent = String(records.length).padStart(2, "0");
  $("export").disabled = false;
}
function onResult(result) {
  const action = appliedDecision(result.answer, result.ageMs, config);
  sampleCount++;
  $("sample-count").textContent = sampleCount;
  $("last-rt").replaceChildren(
    document.createTextNode(String(Math.round(result.wallMs))),
    el("small", "", "ms"),
  );
  $("last-forward").replaceChildren(
    document.createTextNode(
      result.forwardMs === null ? "—" : String(Math.round(result.forwardMs)),
    ),
    el("small", "", "ms"),
  );
  setMode(action.mode, action.applied ? "" : action.reason);
  const record = { ...result, ...action, index: sampleCount };
  addRecord(record);
  status(
    `${gestureById(result.answer.choice).label} · 置信度 ${(result.answer.confidence * 100).toFixed(1)}% · ${action.reason}`,
  );
}
function onError(error) {
  setMode("none", "识别已暂停");
  refreshControls();
  status(`识别已暂停：${error.message}`, true);
  emptyFeed.remove();
  const item = el("div", "error-record", `请求失败：${error.message}`);
  $("feed").prepend(item);
  while ($("feed").children.length > 12) $("feed").lastElementChild.remove();
}
const loop = new DecisionLoop({
  capture,
  request,
  onResult,
  onError,
  onSkip: () => {
    skipped++;
    $("skip-count").textContent = skipped;
  },
  eligible: () => Boolean(stream) && !document.hidden && video.readyState >= 2,
});
function pauseRecognition(message = "识别已暂停，摄像头仍在预览。") {
  loop.stop();
  setMode("none", "识别已暂停");
  refreshControls();
  status(message);
}
async function openCamera() {
  if (stream || opening) return;
  opening = true;
  const ticket = ++cameraEpoch;
  refreshControls();
  status("请允许浏览器访问摄像头，仅请求视频。");
  try {
    if (!navigator.mediaDevices?.getUserMedia)
      throw new Error("摄像头需要 localhost 或 HTTPS 页面。");
    const acquired = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user",
      },
      audio: false,
    });
    if (ticket !== cameraEpoch) {
      acquired.getTracks().forEach((track) => track.stop());
      return;
    }
    stream = acquired;
    video.srcObject = acquired;
    await video.play();
    if (ticket !== cameraEpoch) return;
    acquired.getVideoTracks().forEach((track) =>
      track.addEventListener("ended", () => {
        if (stream === acquired) closeCamera();
      }),
    );
    $("camera-empty").hidden = true;
    $("camera-frame").dataset.active = "true";
    status("摄像头已开启。将一只手放进方框，点击“开始识别”。");
  } catch (error) {
    if (ticket !== cameraEpoch) return;
    stream?.getTracks().forEach((track) => track.stop());
    stream = null;
    video.srcObject = null;
    const messages = {
      NotAllowedError:
        "摄像头权限被拒绝。请在浏览器地址栏或系统设置中允许访问后重试。",
      NotFoundError: "未发现摄像头，请连接摄像头后重试。",
      NotReadableError: "摄像头不可用，可能正被其他应用占用。",
    };
    status(messages[error.name] || `无法开启摄像头：${error.message}`, true);
  } finally {
    if (ticket === cameraEpoch) {
      opening = false;
      refreshControls();
    }
  }
}
function closeCamera() {
  cameraEpoch++;
  opening = false;
  loop.stop();
  const previous = stream;
  stream = null;
  previous?.getTracks().forEach((track) => track.stop());
  video.pause();
  video.srcObject = null;
  captureCtx.clearRect(0, 0, captureCanvas.width, captureCanvas.height);
  $("camera-empty").hidden = false;
  $("camera-frame").dataset.active = "false";
  records.length = 0;
  sampleCount = 0;
  skipped = 0;
  $("feed").replaceChildren(emptyFeed);
  $("feed-count").textContent = "00";
  $("export").disabled = true;
  $("sample-count").textContent = "0";
  $("skip-count").textContent = "0";
  $("last-rt").replaceChildren(
    document.createTextNode("—"),
    el("small", "", "ms"),
  );
  $("last-forward").replaceChildren(
    document.createTextNode("—"),
    el("small", "", "ms"),
  );
  setMode("none", "先开启摄像头，再开始识别");
  refreshControls();
  status("摄像头已关闭，采样图片和页面历史已清空。");
}
async function checkHealth() {
  const wasReady = ready;
  try {
    const response = await fetch("/api/health", {
      signal: AbortSignal.timeout(4000),
    });
    const info = await response.json();
    ready = response.ok && info.ready;
    if (ready && !wasReady && !stream)
      status("模型已就绪。开启摄像头预览后，可开始识别。");
    $("connection").dataset.ready = String(ready);
    $("connection").querySelector("span").textContent = ready
      ? "本机模型已连接"
      : "模型服务未就绪";
    if (!ready && loop.active)
      pauseRecognition(info.message || "模型服务暂不可用。");
    if (!ready && !stream)
      status(info.message || "请先启动 Raya MaaS 服务。", true);
  } catch {
    ready = false;
    $("connection").dataset.ready = "false";
    $("connection").querySelector("span").textContent = "模型连接已断开";
    if (loop.active) pauseRecognition("模型连接已断开。");
  }
  refreshControls();
}
$("camera-on").addEventListener("click", openCamera);
$("camera-off").addEventListener("click", closeCamera);
$("recognize").addEventListener("click", () => {
  if (loop.active) {
    pauseRecognition();
    return;
  }
  if (!stream || !ready || !config) return;
  status("每秒采样一次，正在等待模型判断……");
  loop.start();
  refreshControls();
});
$("export").addEventListener("click", () => {
  const data = records.map(
    ({
      index,
      snapshot,
      wallMs,
      forwardMs,
      processingMs,
      queueMs,
      output,
      reason,
      requestId,
    }) => ({
      index,
      time: snapshot.time,
      rt_ms: wallMs,
      forward_ms: forwardMs,
      processing_ms: processingMs,
      queue_ms: queueMs,
      request_id: requestId,
      action: reason,
      response: output,
    }),
  );
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const a = el("a");
  a.href = url;
  a.download = `raya-gestures-${Date.now()}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden && loop.active)
    pauseRecognition("页面已隐藏，识别自动暂停。返回后可手动继续。");
});
window.addEventListener("pagehide", () => {
  closeCamera();
  field.close();
  clearInterval(healthTimer);
});
// Read-only diagnostics used by UI tests; no fake recognition or manual effect overrides.
Object.defineProperty(window, "rayaGestureDebug", {
  get: () => ({
    active: loop.active,
    busy: loop.busy,
    camera: Boolean(stream),
    mode: field.mode,
    frames: field.frames,
    samples: sampleCount,
    skipped,
    lastSampleAt: loop.lastSampleAt,
    tracks: stream?.getTracks().map((t) => t.readyState) || [],
  }),
});
try {
  const response = await fetch("/api/config");
  if (!response.ok) throw new Error("无法读取演示配置");
  config = await response.json();
  captureCanvas.width = captureCanvas.height = config.capture_size;
  loop.intervalMs = config.sample_interval_ms;
  $("model-name").textContent = config.model;
  await checkHealth();
  healthTimer = setInterval(checkHealth, 5000);
} catch (error) {
  status(`初始化失败：${error.message}。请刷新页面重试。`, true);
}
