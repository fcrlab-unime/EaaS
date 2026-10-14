// ---------------------------------------------------------------------------
// Python file parser — extracts classes and __init__ params entirely in-browser
// ---------------------------------------------------------------------------

function parsePythonFile(src) {
  const classes = [];
  const classRe = /^class\s+(\w+)/gm;
  let cm;
  while ((cm = classRe.exec(src)) !== null) {
    classes.push({ name: cm[1], offset: cm.index });
  }
  return classes.map(cls => ({
    name: cls.name,
    params: extractInitParams(src, cls.offset),
  }));
}

function extractInitParams(src, fromOffset) {
  const marker = 'def __init__(';
  const start = src.indexOf(marker, fromOffset);
  if (start === -1) return {};

  const parenOpen = start + marker.length - 1;
  let depth = 1;
  let i = parenOpen + 1;
  while (i < src.length && depth > 0) {
    if (src[i] === '(') depth++;
    else if (src[i] === ')') depth--;
    i++;
  }
  const paramStr = src.slice(parenOpen + 1, i - 1);
  return parseParamString(paramStr);
}

function parseParamString(paramStr) {
  const parts = splitByComma(paramStr);
  const result = {};
  for (const raw of parts) {
    const p = raw.replace(/#[^\n]*/g, '').trim();
    if (!p) continue;
    if (p === 'self') continue;
    if (p.startsWith('**') || p.startsWith('*')) continue;
    const parsed = parseOneParam(p);
    if (parsed) result[parsed.name] = parsed.value;
  }
  return result;
}

function splitByComma(str) {
  const parts = [];
  let depth = 0, cur = '';
  for (const ch of str) {
    if ('([{'.includes(ch)) depth++;
    else if (')]}'.includes(ch)) depth--;
    else if (ch === ',' && depth === 0) { parts.push(cur); cur = ''; continue; }
    cur += ch;
  }
  if (cur.trim()) parts.push(cur);
  return parts;
}

function parseOneParam(p) {
  let eqIdx = -1;
  let depth = 0;
  for (let i = 0; i < p.length; i++) {
    if ('([{'.includes(p[i])) depth++;
    else if (')]}'.includes(p[i])) depth--;
    else if (p[i] === '=' && depth === 0 && p[i - 1] !== '!' && p[i + 1] !== '=') {
      eqIdx = i; break;
    }
  }

  let name, typeAnnotation = null, defaultRaw = null;

  if (eqIdx !== -1) {
    const before = p.slice(0, eqIdx).trim();
    defaultRaw = p.slice(eqIdx + 1).trim();
    const colonIdx = before.indexOf(':');
    if (colonIdx !== -1) {
      name = before.slice(0, colonIdx).trim();
      typeAnnotation = before.slice(colonIdx + 1).trim();
    } else {
      name = before;
    }
  } else {
    const colonIdx = p.indexOf(':');
    if (colonIdx !== -1) {
      name = p.slice(0, colonIdx).trim();
      typeAnnotation = p.slice(colonIdx + 1).trim();
    } else {
      name = p;
    }
  }

  if (!name || !name.match(/^\w+$/)) return null;

  const value = defaultRaw !== null
    ? pyToJs(defaultRaw)
    : inferFromType(typeAnnotation);

  return { name, value };
}

function pyToJs(val) {
  val = val.trim();
  if (val === 'True')  return true;
  if (val === 'False') return false;
  if (val === 'None')  return null;
  const n = Number(val);
  if (!isNaN(n) && val !== '') return n;
  if (/^(['"]).*\1$/.test(val)) return val.slice(1, -1);
  return val;
}

function inferFromType(typeStr) {
  if (!typeStr) return null;
  const t = typeStr.replace(/\s/g, '').toLowerCase();
  if (t === 'int')   return 0;
  if (t === 'float') return 0.0;
  if (t === 'bool')  return false;
  if (t === 'str')   return '';
  return null;
}

// ---------------------------------------------------------------------------
// UI wiring
// ---------------------------------------------------------------------------

let detectedClasses = [];
let datasetColumns = [];
function updateFeatureCount() {
  const label = document.getElementById('labelSelect').value;
  const count = datasetColumns.filter(c => c !== label).length;
  document.getElementById('featureCount').textContent = datasetColumns.length
    ? `${count} input features: set input_channels to ${count}. UART uses this same column order.`
    : 'Load the CSV to see the available columns';
}
document.getElementById('labelSelect').addEventListener('change', updateFeatureCount);

document.getElementById('modelScriptInput').addEventListener('change', function () {
  const file = this.files[0];
  const nameEl = document.getElementById('modelScriptName');

  if (!file) {
    nameEl.textContent = 'No file selected';
    nameEl.classList.remove('loaded');
    resetModelFields();
    return;
  }

  nameEl.textContent = file.name;
  nameEl.classList.add('loaded');

  const reader = new FileReader();
  reader.onload = ev => applyParsedClasses(parsePythonFile(ev.target.result));
  reader.readAsText(file);
});

// ---------------------------------------------------------------------------
// Dynamic param-form
// ---------------------------------------------------------------------------

const paramRows   = document.getElementById('paramRows');
const addParamBtn = document.getElementById('addParamBtn');

// Column headers (injected once)
const colHeaders = document.createElement('div');
colHeaders.className = 'param-col-headers';
colHeaders.innerHTML = '<span>Key</span><span>Value</span><span></span>';
paramRows.before(colHeaders);
colHeaders.style.display = 'none';

function addParamRow(key = '', value = '', isWarning = false) {
  colHeaders.style.display = '';

  const row = document.createElement('div');
  row.className = 'param-row';

  const keyIn = document.createElement('input');
  keyIn.type = 'text';
  keyIn.placeholder = 'key';
  keyIn.value = key;
  keyIn.autocomplete = 'off';

  const valIn = document.createElement('input');
  valIn.type = 'text';
  valIn.placeholder = 'value';
  valIn.value = value === null ? 'null' : String(value);
  valIn.autocomplete = 'off';
  if (isWarning) valIn.classList.add('warn-value');

  // Remove warning style on edit
  valIn.addEventListener('input', () => {
    valIn.classList.remove('warn-value');
    document.getElementById('modelParamsBadge').classList.remove('visible');
    updateParamsHint();
  });
  keyIn.addEventListener('input', () => {
    document.getElementById('modelParamsBadge').classList.remove('visible');
  });

  const removeBtn = document.createElement('button');
  removeBtn.type = 'button';
  removeBtn.className = 'param-remove-btn';
  removeBtn.title = 'Remove';
  removeBtn.textContent = '×';
  removeBtn.addEventListener('click', () => {
    row.remove();
    if (paramRows.children.length === 0) colHeaders.style.display = 'none';
    updateParamsHint();
  });

  row.appendChild(keyIn);
  row.appendChild(valIn);
  row.appendChild(removeBtn);
  paramRows.appendChild(row);
  updateParamsHint();
}

addParamBtn.addEventListener('click', () => addParamRow());

function getParamsObject() {
  const obj = {};
  for (const row of paramRows.querySelectorAll('.param-row')) {
    const [keyIn, valIn] = row.querySelectorAll('input[type="text"]');
    const k = keyIn.value.trim();
    if (!k) continue;
    obj[k] = parseParamValue(valIn.value.trim());
  }
  return obj;
}

function parseParamValue(raw) {
  if (raw === 'null' || raw === 'None') return null;
  if (raw === 'true'  || raw === 'True')  return true;
  if (raw === 'false' || raw === 'False') return false;
  const n = Number(raw);
  if (!isNaN(n) && raw !== '') return n;
  return raw;
}

function serializeParamsToHidden() {
  document.getElementById('modelParamsInput').value = JSON.stringify(getParamsObject());
}

function updateParamsHint() {
  const hint = document.getElementById('paramsHint');
  const hasWarning = paramRows.querySelector('.warn-value') !== null;
  hint.className = hasWarning ? 'hint warn' : 'hint';
  hint.innerHTML = hasWarning
    ? 'Check values set to <strong>0</strong> or <strong>null</strong>: these are required parameters without defaults.'
    : 'Arguments passed to <code>ModelClass(**model_params)</code>';
}

function setParamRows(params) {
  paramRows.innerHTML = '';
  colHeaders.style.display = 'none';
  for (const [k, v] of Object.entries(params)) {
    const isWarning = v === 0 || v === null || v === '';
    addParamRow(k, v, isWarning);
  }
}

// ---------------------------------------------------------------------------
// Auto-fill from parsed Python
// ---------------------------------------------------------------------------

function applyParsedClasses(classes) {
  detectedClasses = classes;
  const picker      = document.getElementById('classPicker');
  const pickerSel   = document.getElementById('classPickerSelect');
  const nameInput   = document.getElementById('classNameInput');
  const nameBadge   = document.getElementById('classNameBadge');
  const paramsBadge = document.getElementById('modelParamsBadge');

  if (classes.length === 0) { resetModelFields(); return; }

  if (classes.length > 1) {
    pickerSel.innerHTML = classes.map(c => `<option value="${c.name}">${c.name}</option>`).join('');
    picker.classList.add('visible');
    pickerSel.onchange = () => fillFromClass(pickerSel.value);
  } else {
    picker.classList.remove('visible');
  }

  fillFromClass(classes[0].name);

  function fillFromClass(className) {
    const cls = detectedClasses.find(c => c.name === className);
    if (!cls) return;

    nameInput.value = cls.name;
    nameInput.classList.add('autofilled');
    nameBadge.classList.add('visible');

    setParamRows(cls.params);
    paramsBadge.classList.add('visible');
    updateParamsHint();
  }
}

function resetModelFields() {
  document.getElementById('classPicker').classList.remove('visible');
  const nameInput = document.getElementById('classNameInput');
  nameInput.value = '';
  nameInput.classList.remove('autofilled');
  document.getElementById('classNameBadge').classList.remove('visible');
  paramRows.innerHTML = '';
  colHeaders.style.display = 'none';
  document.getElementById('modelParamsInput').value = '{}';
  document.getElementById('modelParamsBadge').classList.remove('visible');
  updateParamsHint();
}

document.getElementById('classNameInput').addEventListener('input', function () {
  this.classList.remove('autofilled');
  document.getElementById('classNameBadge').classList.remove('visible');
});

function bindFileLabel(inputId, nameId) {
  const input = document.getElementById(inputId);
  const nameEl = document.getElementById(nameId);
  input.addEventListener('change', function () {
    if (this.files[0]) {
      nameEl.textContent = this.files[0].name;
      nameEl.classList.add('loaded');
    } else {
      nameEl.textContent = 'No file selected';
      nameEl.classList.remove('loaded');
    }
  });
}
bindFileLabel('datasetInput', 'datasetName');

document.getElementById('datasetInput').addEventListener('change', function () {
  const file = this.files[0];
  const select = document.getElementById('labelSelect');

  if (!file) {
    select.innerHTML = '<option value="">— load the CSV first —</option>';
    select.disabled = true;
    datasetColumns = [];
    updateFeatureCount();
    return;
  }

  const reader = new FileReader();
  reader.onload = ev => {
    const firstLine = ev.target.result.split(/\r?\n/)[0];
    const cols = firstLine.match(/(".*?"|[^,]+)/g)
      ?.map(c => c.replace(/^"|"$/g, '').trim())
      ?? firstLine.split(',').map(c => c.trim());

    if (!cols.length) return;
    select.innerHTML = cols.map(c =>
      `<option value="${c.replace(/"/g, '&quot;')}">${c}</option>`
    ).join('');
    select.disabled = false;
    select.selectedIndex = cols.length - 1;
    datasetColumns = cols;
    updateFeatureCount();
  };
  reader.readAsText(file);
});

// ---------------------------------------------------------------------------
// Form submission
// ---------------------------------------------------------------------------

function isValidJSON(s) { try { JSON.parse(s); return true; } catch { return false; } }

document.getElementById('jobForm').addEventListener('submit', async function (e) {
  e.preventDefault();

  const btn    = document.getElementById('submitBtn');
  const result = document.getElementById('result');

  // Serialize param rows → hidden JSON field before building FormData
  serializeParamsToHidden();

  const modelScript = document.getElementById('modelScriptInput').files[0];
  const dataset     = document.getElementById('datasetInput').files[0];
  const className   = document.getElementById('classNameInput').value.trim();
  const labelColumn = document.getElementById('labelSelect').value;
  const patience    = parseInt(document.querySelector('[name=patience]').value, 10);
  const maxEpochs   = parseInt(document.querySelector('[name=max_epochs]').value, 10);

  if (!modelScript)         { showError(result, 'Select a Python script (.py)'); return; }
  if (!dataset)             { showError(result, 'Select a dataset (.csv)'); return; }
  if (!className)           { showError(result, 'Enter the model class name'); return; }
  if (!labelColumn)         { showError(result, 'Select the label column from the CSV'); return; }
  if (!(patience > 0))      { showError(result, 'Patience must be at least 1'); return; }
  if (!(maxEpochs > 0))     { showError(result, 'Max epochs must be at least 1'); return; }
  if (patience > maxEpochs) { showError(result, 'Patience cannot be greater than Max epochs'); return; }

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Submitting…';
  result.className = 'result hidden';

  try {
    const response = await fetch('/api/submit', { method: 'POST', body: new FormData(this) });
    const data = await response.json();

    if (response.ok) {
      result.className = 'result success';
      result.innerHTML =
        `<strong>Job started successfully!</strong>&nbsp; ` +
        `ID: <code>${data.job_id}</code>`;
      startProgress(data.job_id);
    } else {
      showError(result, data.detail ?? 'Unknown server error');
    }
  } catch (err) {
    showError(result, `Network error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Start';
  }
});

function showError(el, msg) {
  el.className = 'result error';
  el.textContent = `Error: ${msg}`;
}

// ---------------------------------------------------------------------------
// Pipeline progress — SSE client
// ---------------------------------------------------------------------------

const STAGE_ORDER = ['submitted', 'training', 'quantization', 'export', 'flash', 'done'];
let _sse = null;
let _capturedMetrics = null;

function startProgress(jobId) {
  _capturedMetrics = null;
  window.dispatchEvent(new CustomEvent("pipeline-job", {detail: jobId}));

  document.getElementById('progressJobId').textContent = jobId;
  document.getElementById('progressBadge').textContent = 'Waiting';
  document.getElementById('progressBadge').className = 'progress-badge';
  document.getElementById('progressPanel').classList.remove('hidden');
  document.getElementById('epochProgress').classList.add('hidden');
  document.getElementById('metricsBlock').classList.add('hidden');
  document.getElementById('quantMetricsBlock').classList.add('hidden');
  document.getElementById('quantMetricsGrid').innerHTML = '';
  document.getElementById('espOutputBlock').classList.add('hidden');
  document.getElementById('espRawOutput').textContent = '';
  document.getElementById('espDequantBlock').classList.add('hidden');
  document.getElementById('espDequantGrid').innerHTML = '';
  document.getElementById('downloadBlock').classList.add('hidden');
  document.getElementById('downloadBtn').href = '#';
  document.getElementById('logConsole').innerHTML = '';
  document.getElementById('epochBarFill').style.width = '0%';

  _setStepper('submitted');

  if (_sse) { _sse.close(); }
  _sse = new EventSource(`/api/progress/${jobId}`);
  _sse.onmessage = (e) => _handleEvent(JSON.parse(e.data));
  _sse.onerror = () => {
    if (document.getElementById('progressBadge').className.indexOf('done') === -1) {
      _appendLog({ type: 'error', message: 'Connection to server lost.' });
    }
  };
}

function _handleEvent(ev) {
  const { type, stage, message, data } = ev;

  if (stage) _setStepper(stage);

  if (type === 'epoch') {
    const d = data || ev;
    document.getElementById('epochProgress').classList.remove('hidden');
    document.getElementById('epochText').innerHTML = `<strong>Epoch ${d.epoch}</strong> / ${d.max_epochs}`;
    document.getElementById('epochLoss').textContent = `Train: ${d.train_loss.toFixed(4)}  |  Val: ${d.val_loss.toFixed(4)}`;
    const noImp = document.getElementById('epochNoImprove');
    if (d.no_improve > 0) {
      noImp.textContent = `no_improve: ${d.no_improve}/${d.patience}`;
      noImp.style.display = 'inline';
    } else {
      noImp.style.display = 'none';
    }
    document.getElementById('epochBarFill').style.width = `${(d.epoch / d.max_epochs * 100).toFixed(1)}%`;
  }

  if (type === 'stage' && data && data.metrics) {
    _capturedMetrics = data.metrics;
    _showMetrics(data.metrics);
  }

  if (type === 'stage' && data && data.quant_metrics) {
    _showQuantMetrics(data.quant_metrics);
  }

  if (type === 'stage' && data && data.export_archive) {
    _showDownloadButton(ev.job_id);
    _appendLog({ type: 'log', message: `Export archive: exports/${data.export_archive}` });
  }

  if (stage === 'quantization' || stage === 'export' || stage === 'flash') {
    document.getElementById('epochProgress').classList.add('hidden');
  }

  if (type === 'done') {
    window.dispatchEvent(new CustomEvent('pipeline-ready', {detail: ev.job_id}));
    document.getElementById('epochProgress').classList.add('hidden');
    if (data && data.serial_output) {
      _showEspOutput(data.serial_output);
    }
    if (_sse) { _sse.close(); }
  }

  if (type === 'error') {
    document.getElementById('progressBadge').textContent = 'Error';
    document.getElementById('progressBadge').className = 'progress-badge error';
    if (_sse) { _sse.close(); }
  }

  _appendLog(ev);
}

function _setStepper(stage) {
  const idx = STAGE_ORDER.indexOf(stage);
  if (idx === -1) return;

  const badge = document.getElementById('progressBadge');
  if (stage === 'done') {
    badge.textContent = 'Completed';
    badge.className = 'progress-badge done';
  } else {
    const labels = { submitted: 'Waiting', training: 'Training', quantization: 'Quantization', export: 'Export C++', flash: 'Flash ESP32' };
    badge.textContent = labels[stage] || stage;
    badge.className = 'progress-badge running';
  }

  document.querySelectorAll('#stepper .step').forEach((el, i) => {
    el.classList.remove('completed', 'active');
    const icon = el.querySelector('.step-icon');
    if (i < idx) {
      el.classList.add('completed');
      icon.textContent = '✓';
    } else if (i === idx) {
      el.classList.add('active');
      icon.textContent = icon.dataset.num;
    } else {
      icon.textContent = icon.dataset.num;
    }
  });

  document.querySelectorAll('#stepper .step-line').forEach((el, i) => {
    el.classList.toggle('completed', i < idx);
  });
}

function _appendLog(ev) {
  const { type, message } = ev;
  if (!message) return;

  const console_ = document.getElementById('logConsole');
  const ts = new Date().toLocaleTimeString('en-GB', { hour12: false });
  const line = document.createElement('div');

  let cls = '';
  if (type === 'epoch') cls = 'epoch';
  else if (type === 'stage') cls = 'stage';
  else if (type === 'error') cls = 'error';

  line.className = `log-line${cls ? ' ' + cls : ''}`;
  line.textContent = `[${ts}] ${message}`;
  console_.appendChild(line);

  const distFromBottom = console_.scrollHeight - console_.scrollTop - console_.clientHeight;
  if (distFromBottom < 60) console_.scrollTop = console_.scrollHeight;
}

function _showMetrics(metrics) {
  const grid = document.getElementById('metricsGrid');
  grid.innerHTML = [
    ['MAE',  metrics.mae],
    ['MSE',  metrics.mse],
    ['RMSE', metrics.rmse],
    ['R²',   metrics.r2],
  ].map(([name, val]) => `
    <div class="metric-card">
      <div class="metric-name">${name}</div>
      <div class="metric-value">${typeof val === 'number' ? val.toFixed(4) : val}</div>
    </div>`).join('');
  document.getElementById('metricsBlock').classList.remove('hidden');
}

function _showDownloadButton(jobId) {
  const btn = document.getElementById('downloadBtn');
  btn.href = `/api/download/${jobId}`;
  document.getElementById('downloadBlock').classList.remove('hidden');
}

function _showEspOutput(serial_output) {
  const cleaned = serial_output.replace(/\0/g, '').trim();
  document.getElementById('espRawOutput').textContent = cleaned;
  document.getElementById('espOutputBlock').classList.remove('hidden');
  // Firmware emits JSON with SOH already dequantized by the model.
  const results = cleaned.split(/\r?\n/).flatMap(line => {
    try { const value = JSON.parse(line); return value.type === 'inference' ? [value] : []; }
    catch { return []; }
  });
  const grid = document.getElementById('espDequantGrid');
  grid.replaceChildren();
  for (const result of results) {
    const card = document.createElement('div');
    card.className = 'metric-card';
    card.textContent = `SOH: ${result.soh.toFixed(2)}% — ${result.color}`;
    grid.appendChild(card);
  }
  document.getElementById('espDequantBlock').classList.toggle('hidden', !results.length);
}

function _showQuantMetrics(metrics) {
  const grid = document.getElementById('quantMetricsGrid');
  grid.innerHTML = [
    ['MAE',  metrics.mae],
    ['MSE',  metrics.mse],
    ['RMSE', metrics.rmse],
    ['R²',   metrics.r2],
  ].map(([name, val]) => `
    <div class="metric-card">
      <div class="metric-name">${name}</div>
      <div class="metric-value">${typeof val === 'number' ? val.toFixed(4) : val}</div>
    </div>`).join('');
  document.getElementById('quantMetricsBlock').classList.remove('hidden');
}
