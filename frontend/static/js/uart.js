(() => {
  const el = id => document.getElementById(id);
  let metadata = null, board = null, busy = false, replay = false, replayRunning = false;
  const status = text => { el('uartStatus').textContent = text; };
  const log = text => { el('uartLog').textContent = (el('uartLog').textContent + text + '\n').slice(-16000); };
  function controls() {
    const compatible = board && metadata && board.job_id === metadata.job_id && board.features === metadata.feature_names.length;
    el('connectUart').disabled = busy || !!board || replayRunning;
    el('disconnectUart').disabled = busy || !board;
    el('sendCurve').disabled = !compatible || busy || replayRunning;
    el('streamCurves').disabled = !compatible || busy || replayRunning;
    el('stopCurves').disabled = !replay;
    el('loadSamples').disabled = busy || replayRunning;
    el('testSample').disabled = !metadata || busy || replayRunning;
    el('curveFile').disabled = !metadata || busy || replayRunning;
    el('uartValues').disabled = busy || replayRunning;
    el('uartJob').disabled = busy || replayRunning;
  }
  async function request(action, payload) {
    const response = await fetch(`/api/uart/${action}`, {
      method: action === 'status' ? 'GET' : 'POST',
      headers: {'Content-Type': 'application/json'},
      ...(action === 'status' ? {} : {body: JSON.stringify(payload || {})}),
    });
    const data = await response.json();
    if (!response.ok) {
      // Session errors require a fresh HELLO; invalid curves may be retried directly.
      if ([409, 503, 504].includes(response.status)) board = null;
      throw new Error(data.detail || 'Unable to reach the board. Please try again.');
    }
    return data;
  }
  function selectSample() {
    const sample = metadata?.test_samples[Number(el('testSample').value)];
    if (!sample) return;
    el('uartValues').value = sample.values.join(',');
    el('sendCurve').textContent = 'Check battery health';
    el('sampleProvenance').textContent = `Known battery health: ${sample.expected_soh}%. Compare it with the estimate below.`;
  }
  async function loadSamples(jobId) {
    const response = await fetch(`/api/deployment/${encodeURIComponent(jobId)}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Samples are unavailable. Please try again.');
    if (!data.test_samples?.length) throw new Error('No test curves are available for this model.');
    metadata = data;
    el('uartJob').value = data.job_id;
    el('testSample').replaceChildren(...data.test_samples.map((sample, index) => {
      const option = document.createElement('option');
      option.value = index;
      option.textContent = `Curve ${sample.test_index} · Known health: ${sample.expected_soh}%`;
      return option;
    }));
    el('testSample').value = '0';
    selectSample();
    controls();
  }
  el('connectUart').onclick = async () => {
    busy = true; controls(); status('Connecting to your board…');
    try {
      const connection = await request('connect');
      board = connection.board;
      await loadSamples(board.job_id);
      if (board.features !== metadata.feature_names.length) throw new Error('These curves do not match the model on your board.');
      status(`Connected · ${metadata.test_samples.length} curves ready. Choose one and click Check battery health.`);
    } catch (error) { status(error.message); }
    finally { busy = false; controls(); }
  };
  el('disconnectUart').onclick = async () => {
    replay = false; busy = true; controls();
    try { await request('disconnect'); board = null; status('Board disconnected.'); }
    catch (error) { status(error.message); }
    finally { busy = false; controls(); }
  };
  el('testSample').onchange = selectSample;
  el('loadSamples').onclick = async () => {
    if (busy || replayRunning) return;
    busy = true; controls();
    try {
      await loadSamples(el('uartJob').value.trim());
      status(board && board.job_id !== metadata.job_id ? 'This model does not match your board. Use the model ID from your board.' : 'Samples loaded.');
    } catch (error) { status(error.message); }
    finally { busy = false; controls(); }
  };
  async function sendCurve() {
    if (!board || !metadata || board.job_id !== metadata.job_id || board.features !== metadata.feature_names.length)
      throw new Error('Connect your board to load the matching curves.');
    const fields = el('uartValues').value.trim().split(',');
    if (fields.length !== board.features || fields.some(value => !value.trim() || !Number.isFinite(Number(value))))
      throw new Error(`Enter ${board.features} comma-separated numbers for this curve.`);
    const result = await request('send', {job_id: metadata.job_id, values: fields.map(Number)});
    if (result.job_id !== board?.job_id || !Number.isFinite(result.soh)) throw new Error('Unable to read the result. Please try again.');
    el('liveSoh').textContent = `${result.soh.toFixed(2)}% · ${result.soh <= 80 ? 'Low health' : 'Good health'}`;
    el('liveSoh').className = result.soh <= 80 ? 'health-low' : 'health-good';
    log(JSON.stringify(result));
    return result;
  }
  el('sendCurve').onclick = async () => {
    if (busy || replayRunning) return;
    busy = true; controls(); status('Checking battery health…');
    try { await sendCurve(); status('Result ready. You can check another curve.'); }
    catch (error) { status(error.message); }
    finally { busy = false; controls(); }
  };
  el('streamCurves').onclick = async () => {
    if (busy || replayRunning) return;
    replay = true; replayRunning = true; controls(); status('Checking all curves. Click Stop to finish.');
    try {
      let index = 0;
      while (replay && board) {
        el('testSample').value = index++ % metadata.test_samples.length;
        selectSample(); busy = true; controls();
        await sendCurve(); busy = false; controls();
        if (replay) await new Promise(resolve => setTimeout(resolve, 1000));
      }
      status('Repeated checks stopped.');
    } catch (error) { status(error.message); }
    finally { replay = false; replayRunning = false; busy = false; controls(); }
  };
  el('stopCurves').onclick = () => { replay = false; controls(); };
  el('uartValues').oninput = () => {
    el('sendCurve').textContent = 'Check battery health';
    el('sampleProvenance').textContent = 'Custom curve · Known battery health unavailable.';
  };
  el('curveFile').onchange = async () => {
    const file = el('curveFile').files[0];
    if (!file || !metadata) return;
    try {
      const lines = (await file.text()).trim().split(/\r?\n/).filter(line => line.trim());
      let values;
      if (lines.length === 1) values = lines[0].split(',');
      else if (lines.length === 2) {
        const header = lines[0].replace(/^\uFEFF/, '').split(',').map(s => s.trim().replace(/^"|"$/g, ''));
        const row = lines[1].split(',');
        if (header.length !== row.length || metadata.feature_names.some(name => !header.includes(name)))
          throw new Error('This file does not match your model. Check the CSV headers.');
        values = metadata.feature_names.map(name => row[header.indexOf(name)]);
      } else throw new Error('Upload a file with one curve, with an optional header row.');
      if (values.length !== metadata.feature_names.length || values.some(s => !s.trim() || !Number.isFinite(Number(s))))
        throw new Error(`Your file needs ${metadata.feature_names.length} numeric values.`);
      el('uartValues').value = values.map(Number).join(',');
      el('uartValues').oninput();
      status(`Curva ${file.name} caricata. Premi Check battery health.`);
    } catch (error) { status(error.message); }
  };
  el('downloadSamples').onclick = () => {
    if (!metadata) { status('Connect your board or load samples using a model ID.'); return; }
    const url = URL.createObjectURL(new Blob([JSON.stringify(metadata, null, 2)], {type: 'application/json'}));
    const anchor = document.createElement('a');
    anchor.href = url; anchor.download = `test_samples_${metadata.job_id}.json`; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  window.addEventListener('pipeline-job', event => { el('uartJob').value = event.detail; });
  window.addEventListener('pipeline-ready', async event => {
    if (busy || replayRunning) return;
    board = null; busy = true; controls();
    try { await loadSamples(event.detail); status('Your model is ready. Click Connect ESP32 to try a curve.'); }
    catch (error) { status(error.message); }
    finally { busy = false; controls(); }
  });
  controls();
})();
